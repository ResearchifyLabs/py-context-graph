import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from decision_graph.core.config import GraphConfig, build_cluster_metadata_cfg
from decision_graph.core.domain import ClusterMetadataExtract, DecisionCluster, DecisionLink
from decision_graph.core.interfaces import LLMAdapter, MatchScorer
from decision_graph.core.matching import SimpleJaccardScorer
from decision_graph.core.registry import StorageBackend

_logger = logging.getLogger(__name__)

EARLY_EXIT_SCORE = 0.90


class DecisionClusterService:
    def __init__(
        self,
        *,
        backend: StorageBackend,
        executor: LLMAdapter,
        scorer: MatchScorer = None,
        match_threshold: float = 0.40,
        early_exit_score: float = EARLY_EXIT_SCORE,
        config: GraphConfig = None,
    ):
        self._cluster_store = backend.cluster_store()
        self._link_store = backend.link_store()
        self._executor = executor
        self._scorer = scorer or SimpleJaccardScorer()
        self._match_threshold = match_threshold
        self._early_exit_score = early_exit_score
        self._config = config or GraphConfig()
        self._cluster_metadata_cfg = build_cluster_metadata_cfg(self._config.model)

    async def _generate_cluster_metadata(
        self,
        *,
        subject_labels: List[str],
        topics: List[str],
        entities: List[str],
    ) -> ClusterMetadataExtract:
        decision_info = f"""Subject Labels: {', '.join(subject_labels[:10])}
Topics: {', '.join(topics[:15])}
Key Entities: {', '.join(entities[:10])}"""

        result = await self._executor.execute_async(
            self._cluster_metadata_cfg,
            data=decision_info,
        )
        if not result:
            return ClusterMetadataExtract(
                primary_subject=subject_labels[0] if subject_labels else "Unknown",
                rolling_summary="Cluster of related decisions",
            )
        return result

    async def link_decisions_to_cluster(
        self,
        *,
        new_decisions: List[Dict[str, Any]],
        candidate_decisions: List[Dict[str, Any]],
        match_threshold: float = None,
    ) -> Dict[str, Any]:
        if not new_decisions:
            _logger.warning("link_decisions.no_new_decisions")
            return {"linked": 0, "skipped": 0, "new_clusters": 0}

        threshold = match_threshold if match_threshold is not None else self._match_threshold
        now_ts = datetime.now(timezone.utc).timestamp()

        precomputed_cands = [self._scorer.precompute(d) for d in candidate_decisions]

        linked_count = 0
        skipped_count = 0
        new_cluster_ids = []
        affected_cluster_ids = set()
        unmatched = []

        for decision in new_decisions:
            did = decision["decision_id"]

            existing_link = self._link_store.find_by_decision_id(did)
            if existing_link:
                skipped_count += 1
                continue

            precomputed_new = self._scorer.precompute(decision)

            best_score = 0.0
            best_cand_idx = -1
            best_scores = {}
            above_threshold_cands = []

            for j, cand_d in enumerate(precomputed_cands):
                if cand_d.get("decision_id") == did:
                    continue
                scores = self._scorer.score_pair(precomputed_new, cand_d)
                if scores["combined_score"] >= threshold:
                    above_threshold_cands.append((scores["combined_score"], j, scores))
                if scores["combined_score"] > best_score:
                    best_score = scores["combined_score"]
                    best_cand_idx = j
                    best_scores = scores
                    if best_score >= self._early_exit_score:
                        break

            if best_score >= threshold and best_cand_idx >= 0:
                cand_did = candidate_decisions[best_cand_idx]["decision_id"]
                cluster_id = self._link_store.find_cluster_id_for_decision(cand_did)

                if not cluster_id:
                    for alt_score, alt_idx, alt_scores in sorted(above_threshold_cands, key=lambda x: -x[0]):
                        if alt_idx == best_cand_idx:
                            continue
                        alt_did = candidate_decisions[alt_idx]["decision_id"]
                        alt_cluster_id = self._link_store.find_cluster_id_for_decision(alt_did)
                        if alt_cluster_id:
                            cluster_id = alt_cluster_id
                            best_scores = alt_scores
                            break

                if cluster_id:
                    self._add_decision_to_cluster(
                        cluster_id=cluster_id,
                        decision=decision,
                        scores=best_scores,
                        now_ts=now_ts,
                    )
                    affected_cluster_ids.add(cluster_id)
                    linked_count += 1
                else:
                    cluster_id = await self._create_cluster_for_pair(
                        decision_a=decision,
                        decision_b=candidate_decisions[best_cand_idx],
                        scores=best_scores,
                        now_ts=now_ts,
                    )
                    new_cluster_ids.append(cluster_id)
                    linked_count += 1
            else:
                unmatched.append(decision)

        for um_decision in unmatched:
            cluster_id = await self._create_cluster_for_unmatched(
                decisions=[um_decision],
                now_ts=now_ts,
            )
            new_cluster_ids.append(cluster_id)
            linked_count += 1

        _logger.info(
            "link_decisions.done linked=%d skipped=%d new_clusters=%d unmatched=%d",
            linked_count,
            skipped_count,
            len(new_cluster_ids),
            len(unmatched),
        )

        all_cluster_ids = list(affected_cluster_ids) + new_cluster_ids
        return {
            "linked": linked_count,
            "skipped": skipped_count,
            "new_clusters": len(new_cluster_ids),
            "cluster_ids": all_cluster_ids,
        }

    def _add_decision_to_cluster(
        self,
        *,
        cluster_id: str,
        decision: Dict[str, Any],
        scores: Dict[str, float],
        now_ts: float,
    ):
        did = decision["decision_id"]
        cluster = self._cluster_store.find_by_id(cluster_id)
        if not cluster:
            _logger.warning("link_decisions.cluster_not_found cluster_id=%s", cluster_id)
            return

        existing_dids = cluster.get("decision_ids", [])
        if did not in existing_dids:
            updated_dids = existing_dids + [did]
            existing_cids = cluster.get("cids", [])
            cid = decision.get("cid", "")
            updated_cids = existing_cids if cid in existing_cids else existing_cids + [cid]

            self._cluster_store.update(
                cluster_id,
                {
                    "last_updated_at": now_ts,
                    "decision_count": len(updated_dids),
                    "decision_ids": updated_dids,
                    "cids": updated_cids,
                },
            )

        link = DecisionLink(
            decision_id=did,
            cluster_id=cluster_id,
            cid=decision.get("cid", ""),
            gid=decision.get("gid", ""),
            trace_id=decision.get("trace_id", ""),
            match_metadata={**scores, "linked_from": "matched"},
            linked_at=now_ts,
        ).model_dump()
        self._link_store.save_batch([link])

    async def _create_cluster_for_pair(
        self,
        *,
        decision_a: Dict[str, Any],
        decision_b: Dict[str, Any],
        scores: Dict[str, float],
        now_ts: float,
    ) -> str:
        cluster_id = uuid.uuid4().hex

        subjects = [s for s in [decision_a.get("subject_label"), decision_b.get("subject_label")] if s]
        topics = list(set((decision_a.get("topics") or []) + (decision_b.get("topics") or [])))
        entities = list(set((decision_a.get("entities") or []) + (decision_b.get("entities") or [])))

        metadata = await self._generate_cluster_metadata(
            subject_labels=subjects,
            topics=topics,
            entities=entities,
        )

        dids = [decision_a["decision_id"], decision_b["decision_id"]]
        cids = list(set(filter(None, [decision_a.get("cid"), decision_b.get("cid")])))

        cluster_data = DecisionCluster(
            cluster_id=cluster_id,
            primary_subject=metadata.primary_subject,
            rolling_summary=metadata.rolling_summary,
            created_at=now_ts,
            last_updated_at=now_ts,
            decision_count=len(dids),
            cids=cids,
            decision_ids=dids,
        ).model_dump()
        self._cluster_store.create(cluster_data)

        links = []
        for d in [decision_a, decision_b]:
            links.append(
                DecisionLink(
                    decision_id=d["decision_id"],
                    cluster_id=cluster_id,
                    cid=d.get("cid", ""),
                    gid=d.get("gid", ""),
                    trace_id=d.get("trace_id", ""),
                    match_metadata={**scores, "linked_from": "new_cluster_pair"},
                    linked_at=now_ts,
                ).model_dump()
            )
        self._link_store.save_batch(links)

        return cluster_id

    async def _create_cluster_for_unmatched(
        self,
        *,
        decisions: List[Dict[str, Any]],
        now_ts: float,
    ) -> str:
        cluster_id = uuid.uuid4().hex

        subjects = [d.get("subject_label") for d in decisions if d.get("subject_label")]
        topics = list(set(t for d in decisions for t in (d.get("topics") or [])))
        entities = list(set(e for d in decisions for e in (d.get("entities") or [])))

        metadata = await self._generate_cluster_metadata(
            subject_labels=subjects,
            topics=topics,
            entities=entities,
        )

        dids = [d["decision_id"] for d in decisions]
        cids = list(set(d.get("cid", "") for d in decisions if d.get("cid")))

        cluster_data = DecisionCluster(
            cluster_id=cluster_id,
            primary_subject=metadata.primary_subject,
            rolling_summary=metadata.rolling_summary,
            created_at=now_ts,
            last_updated_at=now_ts,
            decision_count=len(dids),
            cids=cids,
            decision_ids=dids,
        ).model_dump()
        self._cluster_store.create(cluster_data)

        links = []
        for d in decisions:
            links.append(
                DecisionLink(
                    decision_id=d["decision_id"],
                    cluster_id=cluster_id,
                    cid=d.get("cid", ""),
                    gid=d.get("gid", ""),
                    trace_id=d.get("trace_id", ""),
                    match_metadata={"linked_from": "new_no_match"},
                    linked_at=now_ts,
                ).model_dump()
            )
        self._link_store.save_batch(links)

        return cluster_id

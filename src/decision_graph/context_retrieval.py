import logging
from typing import Any, Dict, List, Optional

from decision_graph.core.interfaces import MatchScorer, VectorIndex
from decision_graph.core.matching import (
    SimpleJaccardScorer,
    build_conversation_fingerprint,
    calculate_jaccard,
    calculate_max_subject_match,
    calculate_sequence_match,
    normalize_entity,
)
from decision_graph.core.registry import StorageBackend

_logger = logging.getLogger(__name__)


class DecisionContextRetriever:
    def __init__(
        self,
        *,
        gids: List[str],
        backend: StorageBackend,
        vector_index: Optional[VectorIndex] = None,
        scorer: Optional[MatchScorer] = None,
    ):
        self._gids = gids
        self._vector_index = vector_index
        self._enrichment_store = backend.enrichment_store()
        self._projection_store = backend.projection_store()
        self._scorer = scorer or SimpleJaccardScorer()

    async def get_top_k_projections(
        self,
        *,
        summary_text: str,
        top_k: int = 10,
    ):
        query_filter = {"gid": {"$in": self._gids}} if self._gids else None

        df = self._vector_index.get_top_n_matches(
            query=summary_text,
            query_filter=query_filter,
            top_n=top_k,
        )

        if df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            results.append(
                {
                    "pid": row.get("pid"),
                    "score": row.get("score"),
                    "gid": row.get("gid"),
                    "cid": row.get("cid"),
                }
            )

        return results

    async def get_matching_candidates(
        self,
        *,
        summary_text: str,
        top_k: int = 10,
        score_threshold: float = 0.55,
    ):
        projection_results = await self.get_top_k_projections(
            summary_text=summary_text,
            top_k=top_k,
        )

        if not projection_results:
            return []

        pid_score_map = {item["pid"]: (item["score"], item["cid"]) for item in projection_results if item.get("pid")}
        pids = list(pid_score_map.keys())
        projection_docs = self._projection_store.find_by_ids(pids)

        filtered_projections = []
        for pid, data in projection_docs.items():
            if "projection" in data and "Summary" in data["projection"]:
                score_cid_tuple = pid_score_map.get(pid, (0, None))
                score = score_cid_tuple[0] if isinstance(score_cid_tuple, tuple) else 0
                cid = score_cid_tuple[1] if isinstance(score_cid_tuple, tuple) else None
                if score > score_threshold and score < 1:
                    filtered_projections.append({"pid": pid, "cid": cid, "score": score, "data": data})

        filtered_projections.sort(key=lambda x: x["score"], reverse=True)
        if not filtered_projections:
            return []

        cids = [p["cid"] for p in filtered_projections if p["cid"]]
        decision_traces = self._projection_store.find_by_conv_ids(cids, "decision_trace")
        if not decision_traces:
            return []

        decision_ids = [dt["pid"] for dt in decision_traces]
        enrichment_map = await self._enrichment_store.find_by_ids_async(decision_ids)

        matching_candidates = []
        for dt in decision_traces:
            decision_id = dt.get("pid")
            enrichment = enrichment_map.get(decision_id)

            if not enrichment:
                continue

            proj = dt.get("projection", {})

            entities = enrichment.get("entities", [])
            person_entities = []
            non_person_entities = []
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict):
                        name = entity.get("name") or entity.get("value") or ""
                        etype = (entity.get("type") or "").lower()
                        if name:
                            if etype == "person":
                                person_entities.append(name)
                            else:
                                non_person_entities.append(name)
                    elif isinstance(entity, str):
                        person_entities.append(entity)

            key_facts = enrichment.get("key_facts", [])
            key_fact_values = []
            if isinstance(key_facts, list):
                for fact in key_facts:
                    if isinstance(fact, dict) and "v" in fact:
                        key_fact_values.append(fact["v"])
                    elif isinstance(fact, str):
                        key_fact_values.append(fact)

            candidate = {
                "decision_id": decision_id,
                "trace_id": enrichment.get("trace_id"),
                "cid": dt.get("cid"),
                "gid": dt.get("gid"),
                "subject_label": proj.get("subject_label", ""),
                "action_desc": proj.get("action_desc", ""),
                "evidence_span": proj.get("evidence_span", ""),
                "topics": enrichment.get("topics", []),
                "entities": person_entities + non_person_entities,
                "person_entities": person_entities,
                "non_person_entities": non_person_entities,
                "key_facts": key_fact_values,
                "action_params": enrichment.get("action_params") or {},
            }
            matching_candidates.append(candidate)

        return matching_candidates

    async def get_new_decisions_for_cid(
        self,
        *,
        cid: str,
    ) -> List[Dict[str, Any]]:
        decision_traces = self._projection_store.find_by_conv_ids([cid], "decision_trace")
        if not decision_traces:
            return []

        decision_ids = [dt["pid"] for dt in decision_traces]
        enrichment_map = await self._enrichment_store.find_by_ids_async(decision_ids)

        new_decisions = []
        for dt in decision_traces:
            decision_id = dt.get("pid")
            enrichment = enrichment_map.get(decision_id)

            if not enrichment:
                continue

            proj = dt.get("projection", {})

            entities = enrichment.get("entities", [])
            person_entities = []
            non_person_entities = []
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict):
                        name = entity.get("name") or entity.get("value") or ""
                        etype = (entity.get("type") or "").lower()
                        if name:
                            if etype == "person":
                                person_entities.append(name)
                            else:
                                non_person_entities.append(name)
                    elif isinstance(entity, str):
                        person_entities.append(entity)

            key_facts = enrichment.get("key_facts", [])
            key_fact_values = []
            if isinstance(key_facts, list):
                for fact in key_facts:
                    if isinstance(fact, dict) and "v" in fact:
                        key_fact_values.append(fact["v"])
                    elif isinstance(fact, str):
                        key_fact_values.append(fact)

            new_decision = {
                "decision_id": decision_id,
                "trace_id": enrichment.get("trace_id"),
                "cid": dt.get("cid"),
                "gid": dt.get("gid"),
                "subject_label": proj.get("subject_label", ""),
                "action_desc": proj.get("action_desc", ""),
                "evidence_span": proj.get("evidence_span", ""),
                "topics": enrichment.get("topics", []),
                "entities": person_entities + non_person_entities,
                "person_entities": person_entities,
                "non_person_entities": non_person_entities,
                "key_facts": key_fact_values,
                "action_params": enrichment.get("action_params") or {},
            }
            new_decisions.append(new_decision)

        return new_decisions

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        return normalize_entity(entity)

    @staticmethod
    def _calculate_jaccard(set1: List[str], set2: List[str], normalize_fn=None) -> float:
        return calculate_jaccard(set1, set2, normalize_fn=normalize_fn)

    @staticmethod
    def _calculate_sequence_match(str1: str, str2: str) -> float:
        return calculate_sequence_match(str1, str2)

    async def rank_matching_candidates(
        self,
        *,
        new_decision: Dict[str, Any],
        matching_candidates: List[Dict[str, Any]],
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        precomputed_new = self._scorer.precompute(new_decision)
        ranked_results = []

        for candidate in matching_candidates:
            if candidate.get("decision_id") == new_decision.get("decision_id"):
                continue

            precomputed_cand = self._scorer.precompute(candidate)
            scores = self._scorer.score_pair(precomputed_new, precomputed_cand)

            result = {
                **candidate,
                "match_metadata": scores,
            }
            ranked_results.append(result)

        ranked_results.sort(key=lambda x: x["match_metadata"]["combined_score"], reverse=True)

        if score_threshold:
            ranked_results = [r for r in ranked_results if r["match_metadata"]["combined_score"] >= score_threshold]

        return ranked_results

    @staticmethod
    def _build_conversation_fingerprint(
        decisions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return build_conversation_fingerprint(decisions)

    def _calculate_max_subject_match(
        self,
        subject_labels_a: List[str],
        subject_labels_b: List[str],
    ) -> float:
        return calculate_max_subject_match(subject_labels_a, subject_labels_b)

    async def run_clustering_analysis(
        self,
        *,
        summary_text: str,
        cid: str,
        top_k: int = 10,
        vector_score_threshold: float = 0.55,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        new_decisions = await self.get_new_decisions_for_cid(cid=cid)
        if not new_decisions:
            return [], []

        if not self._vector_index:
            return new_decisions, []

        matching_candidates = await self.get_matching_candidates(
            summary_text=summary_text,
            top_k=top_k,
            score_threshold=vector_score_threshold,
        )

        return new_decisions, matching_candidates + new_decisions

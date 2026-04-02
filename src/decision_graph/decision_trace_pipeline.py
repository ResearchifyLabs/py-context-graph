import logging
import os
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from decision_graph.clustering_service import DecisionClusterService
from decision_graph.context_retrieval import DecisionContextRetriever
from decision_graph.core.domain import DecisionUnitRow
from decision_graph.core.interfaces import GraphStore, LLMAdapter, VectorIndex
from decision_graph.core.matching import merge_decision_trace_history
from decision_graph.core.registry import StorageBackend
from decision_graph.enrichment_service import DecisionEnrichmentService
from decision_graph.extraction_service import DecisionExtractionService
from decision_graph.retrieval import DecisionEnrichmentRetrieval

_logger = logging.getLogger(__name__)


class DecisionTracePipeline:
    """
    End-to-end pipeline for decision trace processing:
    extract → persist → deduplicate → enrich → cluster.
    """

    def __init__(
        self,
        *,
        backend: StorageBackend,
        executor: LLMAdapter,
        vector_index: Optional[VectorIndex] = None,
        graph_store: GraphStore,
    ):
        self._backend = backend
        self._executor = executor
        self._enrichment_service = DecisionEnrichmentService(backend=backend, executor=executor)
        self._extraction_service = DecisionExtractionService(executor=executor)
        self._vector_index = vector_index
        self._graph_store = graph_store

    async def run_from_text(
        self,
        *,
        conv_text: str,
        conv_id: str,
        gid: str,
        updated_at: float,
        summary_pid: str,
        query_gids: List[str],
        allowed_decision_types: Optional[List[str]] = None,
        domain_hint: str = "NA",
        industry: str = "generic_b2b",
        on_step: Optional[Callable[[str], None]] = None,
    ):
        """Extract decision items from raw text then run the full pipeline."""
        if on_step:
            on_step("extract")
        decision_items = await self._extraction_service.extract(
            conv_text,
            allowed_decision_types=allowed_decision_types,
            domain_hint=domain_hint,
            industry=industry,
        )
        await self.run(
            decision_items=decision_items,
            summary_text=conv_text,
            conv_id=conv_id,
            gid=gid,
            updated_at=updated_at,
            summary_pid=summary_pid,
            query_gids=query_gids,
            on_step=on_step,
        )
        return decision_items

    async def run(
        self,
        *,
        decision_items: List[dict],
        summary_text: str,
        conv_id: str,
        gid: str,
        updated_at: float,
        summary_pid: str,
        query_gids: List[str],
        on_step: Optional[Callable[[str], None]] = None,
    ):
        if not decision_items:
            return

        projection_store = self._backend.projection_store()
        recorded_at = int(datetime.now(timezone.utc).timestamp())

        if on_step:
            on_step("persist")
        decision_rows = self._build_and_persist_traces(
            decision_items=decision_items,
            conv_id=conv_id,
            gid=gid,
            updated_at=updated_at,
            summary_pid=summary_pid,
            recorded_at=recorded_at,
            projection_store=projection_store,
        )

        if on_step:
            on_step("deduplicate")
        decision_rows = self._deduplicate_traces(decision_rows, gid, conv_id)

        if on_step:
            on_step("enrich")
        enriched = await self._enrichment_service.run_enrichment(
            summary_text=summary_text,
            decision_rows=decision_rows,
            gid=gid,
            cid=conv_id,
        )
        _logger.info(
            "decision_enrichment.enriched gid=%s cid=%s enriched=%s attempted=%s",
            gid,
            conv_id,
            len(enriched),
            len(decision_rows),
        )

        if enriched:
            if on_step:
                on_step("cluster")
            await self._run_clustering(summary_text=summary_text, cid=conv_id, gids=query_gids)

    def _build_and_persist_traces(
        self,
        *,
        decision_items: List[dict],
        conv_id: str,
        gid: str,
        updated_at: float,
        summary_pid: str,
        recorded_at: int,
        projection_store,
    ) -> List[DecisionUnitRow]:
        decision_rows: List[DecisionUnitRow] = []
        _logger.info("decision_trace.extracted gid=%s cid=%s extracted=%s", gid, conv_id, len(decision_items))

        for item in decision_items:
            action_type = (item.get("action") or {}).get("type")
            status_state = (item.get("status") or {}).get("state", "unknown")
            status_blocker = (item.get("status") or {}).get("blocker")

            decision_id = DecisionEnrichmentService.compute_decision_id(
                cid=conv_id,
                decision_type=item.get("decision_type"),
                action_type=action_type,
                subject_label=(item.get("subject") or {}).get("label"),
                initiator_name=(item.get("actors") or {}).get("initiator_name"),
            )

            row_dict: Dict = {
                "decision_id": decision_id,
                "pid": summary_pid,
                "gid": gid,
                "cid": conv_id,
                "updated_at": float(updated_at),
                "recorded_at": recorded_at,
                "decision_type": item.get("decision_type"),
                "decision_subtype": None,
                "initiator_name": (item.get("actors") or {}).get("initiator_name"),
                "initiator_role": (item.get("actors") or {}).get("initiator_role", "unknown"),
                "counterparty_names": (item.get("actors") or {}).get("counterparty_names", []),
                "subject_label": (item.get("subject") or {}).get("label"),
                "action_type": action_type,
                "action_desc": (item.get("action") or {}).get("description"),
                "action_key": None,
                "status_state": status_state,
                "status_blocker": status_blocker,
                "evidence_span": (item.get("evidence") or {}).get("span"),
                "confidence": (item.get("evidence") or {}).get("confidence"),
            }
            row = DecisionUnitRow.model_validate(row_dict)
            decision_rows.append(row)

            projection = row.model_dump()
            projection["proj_id"] = row.decision_id
            event = {
                "updated_at": float(updated_at),
                "recorded_at": recorded_at,
                "action_type": action_type,
                "status_state": status_state,
                "status_blocker": status_blocker,
            }

            is_new = projection_store.save(
                pid=row.decision_id,
                gid=gid,
                cid=conv_id,
                proj_type="decision_trace",
                projection=projection,
                msg_ts=int(updated_at),
            )

            if not is_new:
                existing = projection_store.find_by_id(row.decision_id)
                existing_proj = (existing.get("projection") or {}) if existing else {}
            else:
                existing_proj = {}

            projection = merge_decision_trace_history(
                existing_proj=existing_proj,
                projection=projection,
                event=event,
            )
            projection_store.update(
                pid=row.decision_id,
                projection=projection,
                update_type="automated",
                msg_ts=int(updated_at),
            )

        return decision_rows

    def _deduplicate_traces(
        self,
        decision_rows: List[DecisionUnitRow],
        gid: str,
        conv_id: str,
    ) -> List[DecisionUnitRow]:
        dedupe_threshold = float(os.environ.get("DECISION_TRACE_DEDUPE_THRESHOLD", "0.6"))
        dedupe_jaccard_threshold = float(os.environ.get("DECISION_TRACE_DEDUPE_JACCARD_THRESHOLD", "0.3"))

        retrieval = DecisionEnrichmentRetrieval(backend=self._backend)
        dedupe_result = retrieval.invalidate_duplicate_decision_trace_projections_within_conversation(
            gid=gid,
            cid=conv_id,
            similarity_threshold=dedupe_threshold,
            jaccard_threshold=dedupe_jaccard_threshold,
        )

        invalidated = set(dedupe_result.get("invalidated") or [])
        if invalidated:
            before = len(decision_rows)
            decision_rows = [dr for dr in decision_rows if dr.decision_id not in invalidated]
            _logger.info(
                "decision_trace.dedupe_applied gid=%s cid=%s seq_threshold=%s jac_threshold=%s before=%s after=%s invalidated=%s",
                gid,
                conv_id,
                dedupe_threshold,
                dedupe_jaccard_threshold,
                before,
                len(decision_rows),
                len(invalidated),
            )

        return decision_rows

    async def _run_clustering(self, *, summary_text: str, cid: str, gids: List[str]) -> None:
        try:
            retriever = DecisionContextRetriever(gids=gids, backend=self._backend, vector_index=self._vector_index)
            new_decisions, candidate_decisions = await retriever.run_clustering_analysis(
                summary_text=summary_text,
                cid=cid,
            )
            if not new_decisions:
                return
            cluster_service = DecisionClusterService(backend=self._backend, executor=self._executor)
            retval = await cluster_service.link_decisions_to_cluster(
                new_decisions=new_decisions,
                candidate_decisions=candidate_decisions,
            )
            _logger.info(
                "decision_clustering.result linked=%s new_clusters=%s", retval.get("linked"), retval.get("new_clusters")
            )

            for cluster_id in retval.get("cluster_ids", []):
                self._sync_cluster_to_graph(cluster_id=cluster_id)
        except Exception:
            _logger.exception("decision_clustering.failed gid=%s cid=%s", gids[0] if gids else "", cid)

    def _sync_cluster_to_graph(self, *, cluster_id: str) -> None:
        try:
            from decision_graph.services import DecisionGraphService

            graph_service = DecisionGraphService(backend=self._backend)
            cluster = graph_service.get_cluster_by_id(cluster_id)
            if not cluster:
                _logger.warning("graph_sync.cluster_not_found cluster_id=%s", cluster_id)
                return

            hydrated = graph_service.bulk_hydrate_clusters([cluster])
            if not hydrated:
                return

            self._graph_store.ingest(hydrated)
        except Exception:
            _logger.exception("graph_sync.failed cluster_id=%s", cluster_id)

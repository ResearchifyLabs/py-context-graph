import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decision_graph.core.matching import (
    canonicalize_subject_label as _canonicalize_subject_label,
)
from decision_graph.core.matching import jaccard_similarity as _jaccard_similarity_fn
from decision_graph.core.matching import sequence_similarity as _sequence_similarity_fn
from decision_graph.core.registry import StorageBackend

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryFilter:
    field: str
    op: str
    value: Any


@dataclass(frozen=True)
class QueryOrder:
    field: str
    direction: str = "ASCENDING"


@dataclass(frozen=True)
class QueryPlan:
    filters: Sequence[QueryFilter]
    limit: int = 200
    order_by: Tuple[QueryOrder, ...] = ()


class DecisionEnrichmentRetrieval:
    """
    Read-only retrieval helpers for decision enrichment documents.
    """

    def __init__(self, *, backend: StorageBackend):
        self._enrichment_store = backend.enrichment_store()
        self._projection_store = backend.projection_store()

    @staticmethod
    def _get_by_field_path(data: Dict[str, Any], field_path: str) -> Tuple[bool, Any]:
        if not field_path:
            return False, None
        cur: Any = data
        for part in field_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False, None
            cur = cur[part]
        return True, cur

    @classmethod
    def _matches_filter(cls, row: Dict[str, Any], f: QueryFilter) -> bool:
        exists, field_value = cls._get_by_field_path(row, f.field)
        if not exists:
            return False

        op = f.op
        value = f.value
        if op == "==":
            return field_value == value
        if op == "!=":
            return field_value != value
        if op == "in":
            return field_value in (value or [])
        if op == "array_contains":
            return isinstance(field_value, list) and value in field_value
        if op == ">=":
            return field_value >= value
        if op == ">":
            return field_value > value
        if op == "<=":
            return field_value <= value
        if op == "<":
            return field_value < value
        raise ValueError(f"Unsupported filter op: {op}")

    def list_by_plan(self, plan: QueryPlan) -> List[Dict[str, Any]]:
        rows = self._enrichment_store.query(
            filters=[(f.field, f.op, f.value) for f in plan.filters],
            order_by=[(o.field, o.direction) for o in plan.order_by],
            limit=plan.limit,
        )

        for f in plan.filters:
            rows = [r for r in rows if self._matches_filter(r, f)]

        if plan.order_by:
            for o in reversed(plan.order_by):
                reverse = o.direction.upper() == "DESCENDING"
                rows = sorted(rows, key=lambda r: (r.get(o.field) is None, r.get(o.field)), reverse=reverse)

        return rows[: plan.limit]

    def get_by_decision_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        return self._enrichment_store.find_by_id(decision_id)

    def list_by_gid(self, gid: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self.list_by_plan(plan=QueryPlan(filters=[QueryFilter(field="gid", op="==", value=gid)], limit=limit))

    def list_by_cid(self, cid: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self.list_by_plan(plan=QueryPlan(filters=[QueryFilter(field="cid", op="==", value=cid)], limit=limit))

    def list_enrichments_by_gid_and_decision_type(
        self, *, gid: str, decision_type: str, limit: int = 200, decision_trace_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch decision_type from decision-trace projections, then return corresponding enrichment docs.
        """
        decision_trace_limit = decision_trace_limit or limit

        proj_rows = self._projection_store.query(
            filters=[("gid", "==", gid), ("proj_type", "==", "decision_trace"), ("valid", "==", True)],
            limit=decision_trace_limit,
        )

        proj_rows = [p for p in proj_rows if (p.get("projection") or {}).get("decision_type") == decision_type]
        proj_by_decision_id: Dict[str, Dict[str, Any]] = {}
        decision_ids: List[str] = []
        for p in proj_rows:
            did = p.get("pid") or p.get("_id")
            if did:
                proj_by_decision_id[did] = p
                decision_ids.append(did)

        if not decision_ids:
            return []

        out = self._enrichment_store.query(
            filters=[("decision_id", "in", decision_ids)],
            limit=len(decision_ids),
        )

        out = [r for r in out if r.get("gid") == gid]
        for r in out:
            did = r.get("decision_id") or r.get("_id")
            proj_row = proj_by_decision_id.get(did or "")
            if proj_row:
                r["decision_trace"] = proj_row.get("projection") or {}
                r["decision_trace_pid"] = proj_row.get("pid") or proj_row.get("_id")
                r["decision_type"] = (proj_row.get("projection") or {}).get("decision_type")
        out.sort(key=lambda r: (r.get("recorded_at") is None, r.get("recorded_at")), reverse=True)
        return out[:limit]

    @staticmethod
    def _sequence_similarity(a: str, b: str) -> float:
        return _sequence_similarity_fn(a, b)

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        return _jaccard_similarity_fn(a, b)

    @staticmethod
    def canonicalize_subject_label(subject_label: str | None) -> str:
        return _canonicalize_subject_label(subject_label)

    def invalidate_duplicate_decision_trace_projections_within_conversation(
        self,
        *,
        gid: str,
        cid: str,
        decision_type: Optional[str] = None,
        similarity_threshold: float = 0.6,
        jaccard_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Find near-duplicate decision_trace projections (same gid/cid, optional decision_type) and invalidate duplicates.
        """
        filters = [
            ("gid", "==", gid),
            ("cid", "==", cid),
            ("proj_type", "==", "decision_trace"),
            ("valid", "==", True),
        ]
        if decision_type:
            filters.append(("projection.decision_type", "==", decision_type))

        proj_rows = self._projection_store.query(filters=filters)

        buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for p in proj_rows:
            pr = p.get("projection") or {}
            key = (
                str(pr.get("decision_type") or ""),
                str(pr.get("initiator_name") or ""),
            )
            buckets.setdefault(key, []).append(p)

        invalidated: List[str] = []
        kept: List[str] = []
        clusters: List[Dict[str, Any]] = []
        log_matches = os.environ.get("DECISION_TRACE_DEDUPE_LOG", "false").lower() == "true"

        def _trunc(s: str, n: int = 140) -> str:
            s = s or ""
            return s if len(s) <= n else s[: n - 3] + "..."

        for _, rows in buckets.items():
            rows = sorted(rows, key=lambda r: r.get("updated_at") or 0, reverse=True)
            used = [False] * len(rows)

            for i, base in enumerate(rows):
                if used[i]:
                    continue
                used[i] = True
                base_proj = base.get("projection") or {}
                base_subject = str(base_proj.get("subject_label") or "")
                base_norm = self.canonicalize_subject_label(base_subject)
                cluster = [base]
                for j in range(i + 1, len(rows)):
                    if used[j]:
                        continue
                    cand = rows[j]
                    cand_proj = cand.get("projection") or {}
                    cand_subject = str(cand_proj.get("subject_label") or "")
                    cand_norm = self.canonicalize_subject_label(cand_subject)
                    seq = self._sequence_similarity(base_norm, cand_norm)
                    jac = self._jaccard_similarity(base_norm, cand_norm)
                    if log_matches:
                        base_pid = base.get("pid") or base.get("_id")
                        cand_pid = cand.get("pid") or cand.get("_id")
                        _logger.info(
                            "decision_trace.dedupe_score gid=%s cid=%s base_pid=%s cand_pid=%s seq=%.3f jac=%.3f "
                            "base_norm=%s cand_norm=%s",
                            gid,
                            cid,
                            base_pid,
                            cand_pid,
                            seq,
                            jac,
                            _trunc(base_norm),
                            _trunc(cand_norm),
                        )
                    if seq >= similarity_threshold or jac >= jaccard_threshold:
                        used[j] = True
                        cluster.append(cand)
                        if log_matches:
                            _logger.info(
                                "decision_trace.dedupe_match gid=%s cid=%s base_pid=%s cand_pid=%s seq=%.3f jac=%.3f "
                                "base_subject=%s cand_subject=%s base_norm=%s cand_norm=%s",
                                gid,
                                cid,
                                base_pid,
                                cand_pid,
                                seq,
                                jac,
                                _trunc(base_subject),
                                _trunc(cand_subject),
                                _trunc(base_norm),
                                _trunc(cand_norm),
                            )

                keeper = cluster[0]
                keeper_pid = keeper.get("pid") or keeper.get("_id")
                if keeper_pid:
                    kept.append(keeper_pid)

                dup_pids: List[str] = []
                for dup in cluster[1:]:
                    dup_pid = dup.get("pid") or dup.get("_id")
                    if not dup_pid:
                        continue
                    self._projection_store.invalidate(dup_pid)
                    invalidated.append(dup_pid)
                    dup_pids.append(dup_pid)

                if dup_pids and keeper_pid:
                    clusters.append({"kept": keeper_pid, "invalidated": dup_pids})

        return {"gid": gid, "cid": cid, "invalidated": invalidated, "kept": kept, "clusters": clusters}

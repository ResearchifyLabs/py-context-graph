"""In-memory implementations of the decision graph store protocols."""

import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from decision_graph.core.interfaces import (
    ClusterStore,
    EnrichmentStore,
    LinkStore,
    ProjectionStore,
)
from decision_graph.ingestion import breakdown_hydrated_clusters


def _get_nested(data: dict, field_path: str) -> Tuple[bool, Any]:
    cur = data
    for part in field_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _matches_filter(row: dict, field: str, op: str, value: Any) -> bool:
    exists, field_value = _get_nested(row, field)
    if not exists:
        return False
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


def _apply_query(
    data: Dict[str, dict],
    filters: List[Tuple[str, str, Any]],
    order_by: Optional[List[Tuple[str, str]]] = None,
    limit: int = 200,
) -> List[dict]:
    rows = list(data.values())
    for field, op, value in filters:
        rows = [r for r in rows if _matches_filter(r, field, op, value)]
    if order_by:
        for ob_field, ob_dir in reversed(order_by):
            reverse = ob_dir.upper() == "DESCENDING"
            rows = sorted(
                rows,
                key=lambda r, f=ob_field: (r.get(f) is None, r.get(f)),
                reverse=reverse,
            )
    return rows[:limit]


class InMemoryEnrichmentStore(EnrichmentStore):
    def __init__(self):
        self._data: Dict[str, dict] = {}

    def find_by_id(self, decision_id: str) -> Optional[dict]:
        return self._data.get(decision_id)

    def find_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        return {did: self._data[did] for did in ids if did in self._data}

    async def find_by_ids_async(self, ids: List[str]) -> Dict[str, dict]:
        return self.find_by_ids(ids)

    def save(self, decision_id: str, data: dict) -> None:
        self._data[decision_id] = dict(data)

    def upsert(self, decision_id: str, data: dict) -> None:
        existing = self._data.get(decision_id, {})
        existing.update(data)
        self._data[decision_id] = existing

    def query(
        self,
        filters: List[Tuple[str, str, Any]],
        order_by: Optional[List[Tuple[str, str]]] = None,
        limit: int = 200,
    ) -> List[dict]:
        return _apply_query(self._data, filters, order_by, limit)


class InMemoryProjectionStore(ProjectionStore):
    def __init__(self):
        self._data: Dict[str, dict] = {}

    def find_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        return {pid: self._data[pid] for pid in ids if pid in self._data}

    def find_by_conv_ids(self, cids: List[str], proj_type: str) -> List[dict]:
        cid_set = set(cids)
        return [
            doc
            for doc in self._data.values()
            if doc.get("cid") in cid_set and doc.get("proj_type") == proj_type and doc.get("valid", True)
        ]

    async def find_by_filters(
        self,
        *,
        gids: List[str],
        proj_type: str,
        last_n_days: Optional[int] = None,
        limit: Optional[int] = None,
        before_ts: Optional[float] = None,
    ) -> List[dict]:
        gid_set = set(gids)
        rows = [
            doc
            for doc in self._data.values()
            if doc.get("gid") in gid_set and doc.get("proj_type") == proj_type and doc.get("valid", True)
        ]
        if last_n_days is not None:
            cutoff = time.time() - (last_n_days * 86400)
            rows = [r for r in rows if (r.get("updated_at") or r.get("created_at") or 0) >= cutoff]
        if before_ts is not None:
            rows = [r for r in rows if (r.get("updated_at") or r.get("created_at") or 0) < before_ts]
        rows.sort(key=lambda r: r.get("updated_at") or r.get("created_at") or 0, reverse=True)
        if limit:
            rows = rows[:limit]
        return rows

    def find_by_id(self, pid: str) -> Optional[dict]:
        return self._data.get(pid)

    def query(
        self,
        filters: List[Tuple[str, str, Any]],
        order_by: Optional[List[Tuple[str, str]]] = None,
        limit: int = 200,
    ) -> List[dict]:
        return _apply_query(self._data, filters, order_by, limit)

    def invalidate(self, pid: str) -> None:
        if pid in self._data:
            self._data[pid]["valid"] = False

    def save(self, *, pid: str, gid: str, cid: str, proj_type: str, projection: dict, msg_ts: int) -> bool:
        is_new = pid not in self._data
        if is_new:
            self._data[pid] = {
                "pid": pid,
                "gid": gid,
                "cid": cid,
                "proj_type": proj_type,
                "projection": projection,
                "created_at": msg_ts,
                "updated_at": msg_ts,
                "valid": True,
            }
        return is_new

    def update(self, *, pid: str, projection: dict, update_type: str, msg_ts: int) -> dict:
        if pid in self._data:
            self._data[pid]["projection"] = projection
            self._data[pid]["updated_at"] = msg_ts
        return self._data.get(pid, {})


class InMemoryClusterStore(ClusterStore):
    def __init__(self):
        self._data: Dict[str, dict] = {}

    def create(self, data: dict) -> str:
        cluster_id = data.get("cluster_id", "")
        self._data[cluster_id] = dict(data)
        return cluster_id

    def update(self, cluster_id: str, updates: dict) -> None:
        if cluster_id in self._data:
            self._data[cluster_id].update(updates)

    def find_by_id(self, cluster_id: str) -> Optional[dict]:
        return self._data.get(cluster_id)

    def find_by_ids(self, cluster_ids: List[str]) -> List[dict]:
        return [self._data[cid] for cid in cluster_ids if cid in self._data]


class InMemoryLinkStore(LinkStore):
    def __init__(self):
        self._data: Dict[str, dict] = {}

    def save_batch(self, links: List[dict]) -> int:
        for link in links:
            decision_id = link.get("decision_id", "")
            self._data[decision_id] = dict(link)
        return len(links)

    def find_by_decision_id(self, decision_id: str) -> Optional[dict]:
        return self._data.get(decision_id)

    def find_by_cluster_id(self, cluster_id: str) -> List[dict]:
        return [d for d in self._data.values() if d.get("cluster_id") == cluster_id]

    def find_by_decision_ids(self, decision_ids: List[str]) -> Dict[str, dict]:
        return {did: self._data[did] for did in decision_ids if did in self._data}

    def find_cluster_ids_by_gids(self, gids: List[str]) -> List[str]:
        gid_set = set(gids)
        cluster_ids = set()
        for link in self._data.values():
            if link.get("gid") in gid_set and link.get("cluster_id"):
                cluster_ids.add(link["cluster_id"])
        return list(cluster_ids)

    def find_cluster_id_for_decision(self, decision_id: str) -> Optional[str]:
        link = self._data.get(decision_id)
        return link.get("cluster_id") if link else None


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could i you he she it we they me him her us them "
    "my your his its our their this that these those in on at to for with by from "
    "of and or not no but if so as up out about into over after".split()
)


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP_WORDS and len(w) > 1]


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryVectorIndex:
    """TF-IDF cosine similarity vector index. No external dependencies."""

    def __init__(self):
        self._docs: List[Dict[str, Any]] = []
        self._idf: Dict[str, float] = {}

    def add(self, *, pid: str, text: str, gid: str, cid: str):
        tokens = _tokenize(text)
        self._docs.append({"pid": pid, "tokens": tokens, "gid": gid, "cid": cid})
        self._rebuild_idf()

    def _rebuild_idf(self):
        n = len(self._docs)
        df: Dict[str, int] = {}
        for doc in self._docs:
            for t in set(doc["tokens"]):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def get_top_n_matches(self, *, query: str, query_filter: Optional[dict], top_n: int) -> pd.DataFrame:
        if not self._docs:
            return pd.DataFrame()

        query_tokens = _tokenize(query)
        query_vec = _tfidf_vector(query_tokens, self._idf)

        allowed_gids = None
        if query_filter and "$in" in (query_filter.get("gid") or {}):
            allowed_gids = set(query_filter["gid"]["$in"])

        results = []
        for doc in self._docs:
            if allowed_gids and doc["gid"] not in allowed_gids:
                continue
            doc_vec = _tfidf_vector(doc["tokens"], self._idf)
            score = _cosine(query_vec, doc_vec)
            results.append({"pid": doc["pid"], "score": score, "gid": doc["gid"], "cid": doc["cid"]})

        results.sort(key=lambda x: x["score"], reverse=True)
        return pd.DataFrame(results[:top_n]) if results else pd.DataFrame()


class InMemoryGraphStore:
    """In-memory graph store that materializes hydrated clusters into graph arrays."""

    def __init__(self):
        self._hydrated_clusters: List[Dict[str, Any]] = []
        self._graph_arrays: Dict[str, list] = {}

    def ingest(self, hydrated_clusters: List[Dict[str, Any]]) -> None:
        self._hydrated_clusters.extend(hydrated_clusters)
        arrays = breakdown_hydrated_clusters(hydrated_clusters)
        for key, items in arrays.items():
            self._graph_arrays.setdefault(key, []).extend(items)

    @property
    def hydrated_clusters(self) -> List[Dict[str, Any]]:
        return self._hydrated_clusters

    @property
    def graph_arrays(self) -> Dict[str, list]:
        return self._graph_arrays

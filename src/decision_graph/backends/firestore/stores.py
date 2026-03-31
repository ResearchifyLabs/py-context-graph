"""Simple Firestore implementations of the decision graph store protocols.

No BaseDAO, no Singleton, no encryption. Just a Firestore client and collection names.
Users provide their own ``google.cloud.firestore.Client`` instance.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from decision_graph.core.interfaces import (
    ClusterStore,
    EnrichmentStore,
    LinkStore,
    ProjectionStore,
)


class FirestoreEnrichmentStore(EnrichmentStore):
    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def _col(self):
        return self._client.collection(self._collection_name)

    def find_by_id(self, decision_id: str) -> Optional[dict]:
        snap = self._col().document(decision_id).get()
        return snap.to_dict() if snap.exists else None

    def find_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        if not ids:
            return {}
        refs = [self._col().document(did) for did in ids]
        result = {}
        for snap in self._client.get_all(refs):
            if snap.exists:
                result[snap.id] = snap.to_dict()
        return result

    async def find_by_ids_async(self, ids: List[str]) -> Dict[str, dict]:
        return self.find_by_ids(ids)

    def save(self, decision_id: str, data: dict) -> None:
        self._col().document(decision_id).set(data)

    def upsert(self, decision_id: str, data: dict) -> None:
        self._col().document(decision_id).set(data, merge=True)

    def query(
        self,
        filters: List[Tuple[str, str, Any]],
        order_by: Optional[List[Tuple[str, str]]] = None,
        limit: int = 200,
    ) -> List[dict]:
        in_filters = [(f, v) for f, op, v in filters if op == "in"]
        regular_filters = [(f, op, v) for f, op, v in filters if op != "in"]

        if in_filters:
            all_rows: List[dict] = []
            for field, values in in_filters:
                for i in range(0, len(values), 30):
                    chunk = values[i : i + 30]
                    q = self._col()
                    for rf, rop, rv in regular_filters:
                        q = q.where(rf, rop, rv)
                    q = q.where(field, "in", chunk)
                    if order_by:
                        for ob_field, ob_dir in order_by:
                            q = q.order_by(ob_field, direction=ob_dir)
                    q = q.limit(limit)
                    all_rows.extend(doc.to_dict() for doc in q.stream())
            return all_rows[:limit]

        q = self._col()
        for f, op, v in filters:
            q = q.where(f, op, v)
        if order_by:
            for ob_field, ob_dir in order_by:
                q = q.order_by(ob_field, direction=ob_dir)
        q = q.limit(limit)
        return [doc.to_dict() for doc in q.stream()]


class FirestoreProjectionStore(ProjectionStore):
    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def _col(self):
        return self._client.collection(self._collection_name)

    def find_by_id(self, pid: str) -> Optional[dict]:
        snap = self._col().document(pid).get()
        return snap.to_dict() if snap.exists else None

    def find_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        if not ids:
            return {}
        refs = [self._col().document(pid) for pid in ids]
        result = {}
        for snap in self._client.get_all(refs):
            if snap.exists:
                result[snap.id] = snap.to_dict()
        return result

    def find_by_conv_ids(self, cids: List[str], proj_type: str) -> List[dict]:
        cid_set = set(cids)
        q = self._col().where("proj_type", "==", proj_type).where("valid", "==", True)
        return [doc.to_dict() for doc in q.stream() if doc.to_dict().get("cid") in cid_set]

    async def find_by_filters(
        self,
        *,
        gids: List[str],
        proj_type: str,
        last_n_days: Optional[int] = None,
        limit: Optional[int] = None,
        before_ts: Optional[float] = None,
    ) -> List[dict]:
        q = self._col().where("proj_type", "==", proj_type).where("valid", "==", True)
        rows = []
        gid_set = set(gids)
        for doc in q.stream():
            data = doc.to_dict()
            if data.get("gid") not in gid_set:
                continue
            rows.append(data)

        if last_n_days is not None:
            cutoff = time.time() - (last_n_days * 86400)
            rows = [r for r in rows if (r.get("updated_at") or r.get("created_at") or 0) >= cutoff]
        if before_ts is not None:
            rows = [r for r in rows if (r.get("updated_at") or r.get("created_at") or 0) < before_ts]
        rows.sort(key=lambda r: r.get("updated_at") or r.get("created_at") or 0, reverse=True)
        if limit:
            rows = rows[:limit]
        return rows

    def query(
        self,
        filters: List[Tuple[str, str, Any]],
        order_by: Optional[List[Tuple[str, str]]] = None,
        limit: int = 200,
    ) -> List[dict]:
        q = self._col()
        for f, op, v in filters:
            q = q.where(f, op, v)
        if order_by:
            for ob_field, ob_dir in order_by:
                q = q.order_by(ob_field, direction=ob_dir)
        q = q.limit(limit)
        return [doc.to_dict() for doc in q.stream()]

    def invalidate(self, pid: str) -> None:
        self._col().document(pid).update({"valid": False})

    def save(self, *, pid: str, gid: str, cid: str, proj_type: str, projection: dict, msg_ts: int) -> bool:
        doc_ref = self._col().document(pid)
        snap = doc_ref.get()
        if snap.exists:
            return False
        doc_ref.set(
            {
                "pid": pid,
                "gid": gid,
                "cid": cid,
                "proj_type": proj_type,
                "projection": projection,
                "created_at": msg_ts,
                "updated_at": msg_ts,
                "valid": True,
            }
        )
        return True

    def update(self, *, pid: str, projection: dict, update_type: str, msg_ts: int) -> dict:
        doc_ref = self._col().document(pid)
        doc_ref.update({"projection": projection, "updated_at": msg_ts})
        snap = doc_ref.get()
        return snap.to_dict() if snap.exists else {}


class FirestoreClusterStore(ClusterStore):
    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def _col(self):
        return self._client.collection(self._collection_name)

    def create(self, data: dict) -> str:
        cluster_id = data.get("cluster_id", "")
        self._col().document(cluster_id).set(data)
        return cluster_id

    def update(self, cluster_id: str, updates: dict) -> None:
        self._col().document(cluster_id).update(updates)

    def find_by_id(self, cluster_id: str) -> Optional[dict]:
        snap = self._col().document(cluster_id).get()
        return snap.to_dict() if snap.exists else None

    def find_by_ids(self, cluster_ids: List[str]) -> List[dict]:
        if not cluster_ids:
            return []
        refs = [self._col().document(cid) for cid in cluster_ids]
        return [snap.to_dict() for snap in self._client.get_all(refs) if snap.exists]



class FirestoreLinkStore(LinkStore):
    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def _col(self):
        return self._client.collection(self._collection_name)

    def save_batch(self, links: List[dict]) -> int:
        if not links:
            return 0
        for link in links:
            decision_id = link.get("decision_id", "")
            self._col().document(decision_id).set(link)
        return len(links)

    def find_by_decision_id(self, decision_id: str) -> Optional[dict]:
        snap = self._col().document(decision_id).get()
        return snap.to_dict() if snap.exists else None

    def find_by_cluster_id(self, cluster_id: str) -> List[dict]:
        q = self._col().where("cluster_id", "==", cluster_id)
        return [doc.to_dict() for doc in q.stream()]

    def find_by_decision_ids(self, decision_ids: List[str]) -> Dict[str, dict]:
        if not decision_ids:
            return {}
        refs = [self._col().document(did) for did in decision_ids]
        return {snap.id: snap.to_dict() for snap in self._client.get_all(refs) if snap.exists}

    def find_cluster_ids_by_gids(self, gids: List[str]) -> List[str]:
        if not gids:
            return []
        gid_set = set(gids)
        cluster_ids = set()
        for doc in self._col().stream():
            data = doc.to_dict()
            if data.get("gid") in gid_set and data.get("cluster_id"):
                cluster_ids.add(data["cluster_id"])
        return list(cluster_ids)

    def find_cluster_id_for_decision(self, decision_id: str) -> Optional[str]:
        link = self.find_by_decision_id(decision_id)
        return link.get("cluster_id") if link else None

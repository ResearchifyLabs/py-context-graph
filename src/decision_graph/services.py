"""
Decision Graph Services.

This module provides business logic for retrieving and joining decision enrichments and projections.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from decision_graph.core.registry import StorageBackend

_logger = logging.getLogger(__name__)


class DecisionGraphService:
    """Service for fetching and joining decision enrichments with projections."""

    def __init__(self, *, backend: StorageBackend):
        self._enrichment_store = backend.enrichment_store()
        self._projection_store = backend.projection_store()
        self._cluster_store = backend.cluster_store()
        self._link_store = backend.link_store()

    def _get_enrichments_by_groups(
        self, group_ids: List[str], since_ts: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        filters = [("gid", "in", group_ids)]
        if since_ts:
            filters.append(("recorded_at", ">=", since_ts))
        return self._enrichment_store.query(filters=filters)

    async def _get_projections_by_groups(
        self,
        group_ids: List[str],
        since_ts: Optional[float] = None,
        limit: Optional[int] = None,
        before_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        last_n_days = None
        if since_ts:
            now_ts = datetime.now(timezone.utc).timestamp()
            last_n_days = int((now_ts - since_ts) / 86400) + 1

        projections = await self._projection_store.find_by_filters(
            gids=group_ids,
            proj_type="decision_trace",
            last_n_days=last_n_days,
            limit=limit,
            before_ts=before_ts,
        )

        return projections

    async def get_enrichments_and_projections_joined(
        self,
        group_ids: List[str],
        since_ts: Optional[float] = None,
        limit: Optional[int] = None,
        before_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        _logger.info(
            f"Fetching data for {len(group_ids)} groups, since_ts={since_ts}, limit={limit}, before_ts={before_ts}"
        )

        projections = await self._get_projections_by_groups(
            group_ids=group_ids,
            since_ts=since_ts,
            limit=limit,
            before_ts=before_ts,
        )

        decision_ids = []
        for projection in projections:
            projection_obj = projection.get("projection", {})
            did = projection_obj.get("proj_id") or projection_obj.get("decision_id")
            if did:
                decision_ids.append(did)

        enrichments_lookup = {}
        if decision_ids:
            enrichments_raw = self._enrichment_store.find_by_ids(decision_ids)
            for did, data in enrichments_raw.items():
                if did:
                    enrichments_lookup[did] = {
                        "topics": data.get("topics", []),
                        "entities": data.get("entities", []),
                        "action_params": data.get("action_params", {}),
                        "constraints_text": data.get("constraints_text", []),
                        "key_facts": data.get("key_facts", []),
                        "recorded_at": data.get("recorded_at"),
                        "trace_id": data.get("trace_id"),
                        "gid": data.get("gid"),
                        "cid": data.get("cid"),
                    }

        _logger.info(f"Retrieved {len(projections)} projections, {len(enrichments_lookup)} enrichments")

        enriched_projections = []
        for projection in projections:
            projection_obj = projection.get("projection", {})
            decision_id = projection_obj.get("proj_id") or projection_obj.get("decision_id")

            result = {
                "pid": projection.get("pid") or projection.get("_id"),
                "gid": projection.get("gid"),
                "cid": projection.get("cid"),
                "proj_type": projection.get("proj_type"),
                "valid": projection.get("valid"),
                "created_at": projection.get("created_at"),
                "updated_at": projection.get("updated_at"),
                "projection": projection_obj,
            }

            if decision_id and decision_id in enrichments_lookup:
                result["enrichment"] = enrichments_lookup[decision_id]
                result["has_enrichment"] = True
            else:
                result["has_enrichment"] = False

            enriched_projections.append(result)

        total_joined = len([p for p in enriched_projections if p["has_enrichment"]])

        has_more = False
        if limit is not None and limit > 0 and len(projections) >= limit:
            has_more = True

        _logger.info(f"Joined {total_joined} projections with enrichments")

        return {
            "projections": enriched_projections,
            "total_enrichments": len(enrichments_lookup),
            "total_projections": len(projections),
            "total_joined": total_joined,
            "has_more": has_more,
        }

    def hydrate_decisions_by_ids(self, decision_ids: List[str]) -> List[Dict[str, Any]]:
        if not decision_ids:
            return []

        enrichment_raw = self._enrichment_store.find_by_ids(decision_ids)
        projection_raw = self._projection_store.find_by_ids(decision_ids)

        results = []
        for did in decision_ids:
            proj_data = projection_raw.get(did, {})
            enrich_data = enrichment_raw.get(did, {})

            result = {
                "pid": proj_data.get("pid") or proj_data.get("_id") or did,
                "gid": proj_data.get("gid"),
                "cid": proj_data.get("cid"),
                "proj_type": proj_data.get("proj_type"),
                "valid": proj_data.get("valid"),
                "created_at": proj_data.get("created_at"),
                "updated_at": proj_data.get("updated_at"),
                "projection": proj_data.get("projection", {}),
            }

            if enrich_data:
                result["enrichment"] = {
                    "topics": enrich_data.get("topics", []),
                    "entities": enrich_data.get("entities", []),
                    "action_params": enrich_data.get("action_params", {}),
                    "constraints_text": enrich_data.get("constraints_text", []),
                    "key_facts": enrich_data.get("key_facts", []),
                    "recorded_at": enrich_data.get("recorded_at"),
                    "trace_id": enrich_data.get("trace_id"),
                    "gid": enrich_data.get("gid"),
                    "cid": enrich_data.get("cid"),
                }
                result["has_enrichment"] = True
            else:
                result["has_enrichment"] = False

            results.append(result)

        return results

    def get_clusters_by_gid(
        self,
        gid: str,
        since_ts: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.get_clusters_by_gids(gids=[gid], since_ts=since_ts, limit=limit)

    def get_clusters_by_gids(
        self,
        gids: List[str],
        since_ts: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        cluster_ids = self._link_store.find_cluster_ids_by_gids(gids)
        if not cluster_ids:
            return []

        clusters = self._cluster_store.find_by_ids(cluster_ids)

        if since_ts:
            clusters = [c for c in clusters if c.get("last_updated_at", 0) >= since_ts]

        clusters.sort(key=lambda x: x.get("last_updated_at", 0), reverse=True)
        _logger.info(f"Retrieved {len(clusters)} clusters for {len(gids)} groups")
        return clusters[:limit]

    def get_cluster_by_id(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        return self._cluster_store.find_by_id(cluster_id)

    def get_link_by_decision_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        return self._link_store.find_by_decision_id(decision_id)

    def get_links_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        links = self._link_store.find_by_cluster_id(cluster_id)
        _logger.info(f"Retrieved {len(links)} links for cluster_id={cluster_id}")
        return links

    def bulk_hydrate_clusters(
        self, clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        all_decision_ids = []
        for cluster in clusters:
            all_decision_ids.extend(cluster.get("decision_ids", []))

        if not all_decision_ids:
            return [
                {"cluster": c, "decisions": [], "total_decisions": 0}
                for c in clusters
            ]

        enrichment_raw = self._enrichment_store.find_by_ids(all_decision_ids)
        enrichment_map = {}
        for did, data in enrichment_raw.items():
            enrichment_map[did] = {
                "topics": data.get("topics", []),
                "entities": data.get("entities", []),
                "action_params": data.get("action_params", {}),
                "constraints_text": data.get("constraints_text", []),
                "key_facts": data.get("key_facts", []),
                "recorded_at": data.get("recorded_at"),
                "trace_id": data.get("trace_id"),
            }

        projection_raw = self._projection_store.find_by_ids(all_decision_ids)
        projection_map = {}
        for pid, data in projection_raw.items():
            projection_map[pid] = {
                "pid": pid,
                "gid": data.get("gid"),
                "cid": data.get("cid"),
                "proj_type": data.get("proj_type"),
                "projection": data.get("projection", {}),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }

        link_map = self._link_store.find_by_decision_ids(all_decision_ids)

        hydrated = []
        for cluster in clusters:
            decision_ids = cluster.get("decision_ids", [])
            decisions = []
            for did in decision_ids:
                projection = projection_map.get(did, {})
                enrichment = enrichment_map.get(did, {})
                link = link_map.get(did, {})

                decisions.append(
                    {
                        "decision_id": did,
                        "cid": projection.get("cid") or link.get("cid", ""),
                        "gid": projection.get("gid") or link.get("gid", ""),
                        "trace_id": enrichment.get("trace_id")
                        or link.get("trace_id", ""),
                        "projection": projection.get("projection", {}),
                        "enrichment": enrichment,
                        "link_metadata": link.get("match_metadata", {}),
                        "linked_at": link.get("linked_at"),
                    }
                )

            hydrated.append(
                {
                    "cluster": cluster,
                    "decisions": decisions,
                    "total_decisions": len(decisions),
                }
            )

        _logger.info(
            "Bulk hydrated %d clusters with %d total decisions",
            len(clusters),
            len(all_decision_ids),
        )
        return hydrated

    def get_hydrated_cluster_detail(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        cluster = self.get_cluster_by_id(cluster_id)
        if not cluster:
            return None

        decision_ids = cluster.get("decision_ids", [])
        if not decision_ids:
            return {
                "cluster": cluster,
                "decisions": [],
                "total_decisions": 0,
            }

        enrichment_raw = self._enrichment_store.find_by_ids(decision_ids)
        enrichment_map = {}
        for did, data in enrichment_raw.items():
            enrichment_map[did] = {
                "topics": data.get("topics", []),
                "entities": data.get("entities", []),
                "action_params": data.get("action_params", {}),
                "constraints_text": data.get("constraints_text", []),
                "key_facts": data.get("key_facts", []),
                "recorded_at": data.get("recorded_at"),
                "trace_id": data.get("trace_id"),
            }

        projection_raw = self._projection_store.find_by_ids(decision_ids)
        projection_map = {}
        for pid, data in projection_raw.items():
            projection_map[pid] = {
                "pid": pid,
                "gid": data.get("gid"),
                "cid": data.get("cid"),
                "proj_type": data.get("proj_type"),
                "projection": data.get("projection", {}),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }

        link_map = self._link_store.find_by_decision_ids(decision_ids)

        decisions = []
        for did in decision_ids:
            projection = projection_map.get(did, {})
            enrichment = enrichment_map.get(did, {})
            link = link_map.get(did, {})

            decision = {
                "decision_id": did,
                "cid": projection.get("cid") or link.get("cid", ""),
                "gid": projection.get("gid") or link.get("gid", ""),
                "trace_id": enrichment.get("trace_id") or link.get("trace_id", ""),
                "projection": projection.get("projection", {}),
                "enrichment": enrichment,
                "link_metadata": link.get("match_metadata", {}),
                "linked_at": link.get("linked_at"),
            }
            decisions.append(decision)

        return {
            "cluster": cluster,
            "decisions": decisions,
            "total_decisions": len(decisions),
        }

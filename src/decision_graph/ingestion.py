import logging
from typing import Any, Dict, List

from decision_graph.core.matching import normalize_entity

_logger = logging.getLogger(__name__)


def breakdown_hydrated_clusters(
    hydrated_clusters: List[Dict[str, Any]],
) -> Dict[str, List[Dict]]:
    clusters: List[Dict] = []
    decisions: List[Dict] = []
    decision_cluster: List[Dict] = []
    decision_topics: List[Dict] = []
    decision_constraints: List[Dict] = []
    decision_entities: List[Dict] = []
    decision_facts: List[Dict] = []
    decision_initiators: List[Dict] = []

    for hc in hydrated_clusters:
        cluster = hc.get("cluster", {})
        cluster_id = cluster.get("cluster_id", "")
        cluster_gid = cluster.get("gid", "")

        clusters.append(
            {
                "cluster_id": cluster_id,
                "gid": cluster_gid,
                "primary_subject": cluster.get("primary_subject", ""),
                "created_at": cluster.get("created_at"),
                "last_updated_at": cluster.get("last_updated_at"),
                "rolling_summary": cluster.get("rolling_summary", ""),
                "decision_count": cluster.get("decision_count", 0),
            }
        )

        for dec in hc.get("decisions", []):
            decision_id = dec.get("decision_id", "")
            gid = dec.get("gid", "") or cluster_gid
            link_metadata = dec.get("link_metadata") or {}

            decisions.append(
                {
                    "decision_id": decision_id,
                    "gid": gid,
                    "cid": dec.get("cid", ""),
                    "trace_id": dec.get("trace_id", ""),
                    "linked_at": dec.get("linked_at"),
                    "linked_from": link_metadata.get("linked_from", ""),
                }
            )

            decision_cluster.append(
                {
                    "cluster_id": cluster_id,
                    "decision_id": decision_id,
                }
            )

            enrichment = dec.get("enrichment", {})

            for topic in enrichment.get("topics", []):
                if topic:
                    decision_topics.append({"decision_id": decision_id, "gid": gid, "topic": topic})

            for text in enrichment.get("constraints_text", []):
                if text:
                    decision_constraints.append({"decision_id": decision_id, "gid": gid, "text": text})

            for entity in enrichment.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                etype = entity.get("type", "")
                ename = entity.get("name", "")
                if etype and ename:
                    normalized = normalize_entity(ename)
                    decision_entities.append(
                        {
                            "decision_id": decision_id,
                            "gid": gid,
                            "type": etype,
                            "name": normalized or ename,
                            "display_name": ename,
                        }
                    )

            for kv in enrichment.get("key_facts", []):
                if not isinstance(kv, dict):
                    continue
                k = kv.get("k", "")
                if k:
                    decision_facts.append(
                        {
                            "decision_id": decision_id,
                            "k": k,
                            "v": kv.get("v", ""),
                        }
                    )

            action_params = enrichment.get("action_params", {})
            if isinstance(action_params, dict):
                for k, v in action_params.items():
                    if k:
                        decision_facts.append(
                            {
                                "decision_id": decision_id,
                                "k": k,
                                "v": str(v) if v is not None else "",
                            }
                        )

            projection = dec.get("projection", {})
            initiator_name = projection.get("initiator_name", "")
            if initiator_name:
                normalized_initiator = normalize_entity(initiator_name)
                decision_initiators.append(
                    {
                        "decision_id": decision_id,
                        "gid": gid,
                        "initiator_name": normalized_initiator or initiator_name,
                        "display_name": initiator_name,
                        "initiator_role": projection.get("initiator_role", "unknown"),
                    }
                )

    return {
        "clusters": clusters,
        "decisions": decisions,
        "decision_cluster": decision_cluster,
        "decision_topics": decision_topics,
        "decision_constraints": decision_constraints,
        "decision_entities": decision_entities,
        "decision_facts": decision_facts,
        "decision_initiators": decision_initiators,
    }


def ingest_to_graph(
    hydrated_clusters: List[Dict[str, Any]],
    graph_dao,
) -> Dict[str, int]:
    arrays = breakdown_hydrated_clusters(hydrated_clusters)

    _logger.info(
        "Breakdown: %d clusters, %d decisions, %d topics, %d constraints, " "%d entities, %d facts, %d initiators",
        len(arrays["clusters"]),
        len(arrays["decisions"]),
        len(arrays["decision_topics"]),
        len(arrays["decision_constraints"]),
        len(arrays["decision_entities"]),
        len(arrays["decision_facts"]),
        len(arrays["decision_initiators"]),
    )

    graph_dao.upsert_clusters(arrays["clusters"])
    graph_dao.upsert_decisions(arrays["decisions"])
    graph_dao.link_clusters_decisions(arrays["decision_cluster"])

    decision_ids = [d["decision_id"] for d in arrays["decisions"]]
    graph_dao.delete_enrichment_edges(decision_ids)

    graph_dao.upsert_topics(arrays["decision_topics"])
    graph_dao.upsert_constraints(arrays["decision_constraints"])
    graph_dao.upsert_entities(arrays["decision_entities"])
    graph_dao.upsert_facts(arrays["decision_facts"])
    graph_dao.upsert_initiators(arrays["decision_initiators"])

    counts = {k: len(v) for k, v in arrays.items()}
    _logger.info("Ingestion complete: %s", counts)
    return counts

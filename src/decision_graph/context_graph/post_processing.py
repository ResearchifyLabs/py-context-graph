"""
Post-processing for context graph query results.

- Graph normalization (paths → nodes[] + edges[])
- Caps + deterministic sampling
- Ranking
- Evidence (why_matched) generation
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from decision_graph.context_graph.registry import get_weight

_logger = logging.getLogger(__name__)

DEFAULT_MAX_NODES = 150
DEFAULT_MAX_EDGES = 300

PER_TYPE_QUOTAS = {
    "Decision": 50,
    "Topic": 30,
    "Entity": 40,
    "Constraint": 15,
    "Fact": 15,
}


def _node_key(node) -> str:
    props = dict(node) if hasattr(node, "items") else {}
    return (
        props.get("decision_id")
        or props.get("cluster_id")
        or f"{props.get('gid', '')}:{props.get('name', '')}:{props.get('type', '')}"
    )


def _node_label(node) -> str:
    if hasattr(node, "labels"):
        labels = list(node.labels)
        for preferred in ("Decision", "Topic", "Entity", "Cluster", "Constraint", "Fact"):
            if preferred in labels:
                return preferred
        return labels[0] if labels else "Unknown"
    return "Unknown"


def _node_to_dict(node) -> Dict[str, Any]:
    props = dict(node) if hasattr(node, "items") else {}
    return {
        "id": _node_key(node),
        "type": _node_label(node),
        "properties": props,
    }


def _edge_key(rel) -> Tuple[str, str, str]:
    start_key = _node_key(rel.start_node) if hasattr(rel, "start_node") else ""
    end_key = _node_key(rel.end_node) if hasattr(rel, "end_node") else ""
    return (rel.type if hasattr(rel, "type") else "", start_key, end_key)


def _edge_to_dict(rel) -> Dict[str, Any]:
    start_key = _node_key(rel.start_node) if hasattr(rel, "start_node") else ""
    end_key = _node_key(rel.end_node) if hasattr(rel, "end_node") else ""
    return {
        "type": rel.type if hasattr(rel, "type") else "",
        "from": start_key,
        "to": end_key,
        "properties": dict(rel) if hasattr(rel, "items") else {},
    }


def normalize_graph(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert raw Neo4j records (containing paths) into deduplicated nodes[] + edges[]."""
    nodes_map: Dict[str, Dict[str, Any]] = {}
    edges_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for record in records:
        for value in record.values():
            _extract_from_value(value, nodes_map, edges_map)

    return {
        "nodes": list(nodes_map.values()),
        "edges": list(edges_map.values()),
    }


def _extract_from_value(value, nodes_map, edges_map):
    if value is None:
        return

    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        for node in value.nodes:
            key = _node_key(node)
            if key not in nodes_map:
                nodes_map[key] = _node_to_dict(node)
        for rel in value.relationships:
            ek = _edge_key(rel)
            if ek not in edges_map:
                edges_map[ek] = _edge_to_dict(rel)

    elif isinstance(value, list):
        for item in value:
            _extract_from_value(item, nodes_map, edges_map)

    elif hasattr(value, "labels"):
        key = _node_key(value)
        if key not in nodes_map:
            nodes_map[key] = _node_to_dict(value)


def apply_caps(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    root_ids: List[str],
    result_ids: List[str],
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
) -> Dict[str, Any]:
    """Enforce hard caps with deterministic per-type sampling."""
    sampled = False
    protected_ids = set(root_ids) | set(result_ids)

    if len(nodes) <= max_nodes and len(edges) <= max_edges:
        return {
            "nodes": nodes,
            "edges": edges,
            "caps_applied": {"max_nodes": max_nodes, "max_edges": max_edges, "sampled": False},
        }

    sampled = True
    kept_nodes: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = {}

    for n in nodes:
        nid = n["id"]
        ntype = n["type"]
        if nid in protected_ids:
            kept_nodes[nid] = n
        else:
            by_type.setdefault(ntype, []).append(n)

    budget = max_nodes - len(kept_nodes)
    for ntype, quota in PER_TYPE_QUOTAS.items():
        candidates = by_type.get(ntype, [])
        take = min(len(candidates), quota, budget)
        for c in candidates[:take]:
            kept_nodes[c["id"]] = c
            budget -= 1
        if budget <= 0:
            break

    kept_ids = set(kept_nodes.keys())
    kept_edges = [
        e for e in edges
        if e["from"] in kept_ids and e["to"] in kept_ids
    ][:max_edges]

    return {
        "nodes": list(kept_nodes.values()),
        "edges": kept_edges,
        "caps_applied": {"max_nodes": max_nodes, "max_edges": max_edges, "sampled": sampled},
    }


def rank_decisions(
    decisions: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    rel_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Rank decision nodes by weighted score: recency, degree, overlap, rel weights."""
    degree_map: Dict[str, int] = {}
    for e in graph_edges:
        degree_map[e["from"]] = degree_map.get(e["from"], 0) + 1
        degree_map[e["to"]] = degree_map.get(e["to"], 0) + 1

    now_ts = time.time()
    scored = []
    for d in decisions:
        did = d.get("id", "")
        props = d.get("properties", {})

        linked_at = props.get("linked_at")
        recency = 0.0
        if linked_at:
            try:
                age_days = max((now_ts - float(linked_at)) / 86400, 0.01)
                recency = 1.0 / age_days
            except (ValueError, TypeError):
                pass

        degree = degree_map.get(did, 0)

        rel_score = 0.0
        for e in graph_edges:
            if e["from"] == did or e["to"] == did:
                rel_score += rel_weights.get(e["type"], 0.0)

        score = (recency * 0.3) + (degree * 0.3) + (rel_score * 0.4)
        d["_score"] = round(score, 4)
        scored.append(d)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def generate_evidence(
    decision_id: str,
    edges: List[Dict[str, Any]],
    nodes_map: Dict[str, Dict[str, Any]],
    max_bullets: int = 4,
) -> List[str]:
    """Generate why_matched bullets from traversed edges for a decision."""
    bullets = []
    for e in edges:
        if e["from"] != decision_id and e["to"] != decision_id:
            continue

        other_id = e["to"] if e["from"] == decision_id else e["from"]
        other = nodes_map.get(other_id, {})
        other_name = other.get("properties", {}).get("name", other_id)
        other_type = other.get("type", "")
        rel_type = e["type"]

        if rel_type == "HAS_TOPIC":
            bullets.append(f"Decision HAS_TOPIC {other_name}")
        elif rel_type == "INITIATED_BY":
            bullets.append(f"Initiated by {other_name}")
        elif rel_type == "MENTIONS_ENTITY":
            bullets.append(f"Mentions {other_type}: {other_name}")
        elif rel_type == "HAS_COUNTERPARTY":
            bullets.append(f"Counterparty: {other_name}")
        elif rel_type == "DEPENDS_ON":
            bullets.append(f"Depends on Decision {other_name}")
        elif rel_type == "HAS_CONSTRAINT":
            text = other.get("properties", {}).get("text", other_name)
            bullets.append(f"Constraint: {text}")
        else:
            bullets.append(f"{rel_type} → {other_name}")

        if len(bullets) >= max_bullets:
            break

    return bullets

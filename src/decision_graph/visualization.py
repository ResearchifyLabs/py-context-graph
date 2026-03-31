"""Visualization helpers for decision graph data."""

from typing import Any, Dict, List


def build_vis_graph(graph_arrays: dict, hydrated_clusters: list) -> dict:
    """Convert breakdown arrays into vis.js-compatible nodes and edges with full metadata."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes: set = set()

    decision_detail: Dict[str, dict] = {}
    for hc in hydrated_clusters:
        cluster = hc.get("cluster", {})
        for dec in hc.get("decisions", []):
            decision_detail[dec.get("decision_id", "")] = {
                "projection": dec.get("projection", {}),
                "enrichment": dec.get("enrichment", {}),
                "cluster_subject": cluster.get("primary_subject", ""),
            }

    for c in graph_arrays.get("clusters", []):
        cid = c["cluster_id"]
        if cid not in seen_nodes:
            nodes.append(
                {
                    "id": cid,
                    "label": c.get("primary_subject", "Cluster"),
                    "group": "cluster",
                    "meta": {
                        "primary_subject": c.get("primary_subject", ""),
                        "rolling_summary": c.get("rolling_summary", ""),
                        "decision_count": c.get("decision_count", 0),
                    },
                }
            )
            seen_nodes.add(cid)

    for d in graph_arrays.get("decisions", []):
        did = d["decision_id"]
        if did not in seen_nodes:
            detail = decision_detail.get(did, {})
            proj = detail.get("projection", {})
            nodes.append(
                {
                    "id": did,
                    "label": did[:8],
                    "group": "decision",
                    "meta": {
                        "subject_label": proj.get("subject_label", ""),
                        "decision_type": proj.get("decision_type", ""),
                        "action_type": proj.get("action_type", ""),
                        "action_desc": proj.get("action_desc", ""),
                        "status_state": proj.get("status_state", ""),
                        "status_blocker": proj.get("status_blocker"),
                        "initiator_name": proj.get("initiator_name", ""),
                        "initiator_role": proj.get("initiator_role", ""),
                        "counterparty_names": proj.get("counterparty_names", []),
                        "evidence_span": proj.get("evidence_span", ""),
                        "confidence": proj.get("confidence"),
                        "cid": d.get("cid", ""),
                    },
                }
            )
            seen_nodes.add(did)

    for link in graph_arrays.get("decision_cluster", []):
        edges.append({"from": link["cluster_id"], "to": link["decision_id"], "label": "contains"})

    for t in graph_arrays.get("decision_topics", []):
        tid = f"topic:{t['topic']}"
        if tid not in seen_nodes:
            nodes.append({"id": tid, "label": t["topic"], "group": "topic", "meta": {"topic": t["topic"]}})
            seen_nodes.add(tid)
        edges.append({"from": t["decision_id"], "to": tid})

    for e in graph_arrays.get("decision_entities", []):
        eid = f"entity:{e['type']}:{e['name']}"
        if eid not in seen_nodes:
            nodes.append(
                {
                    "id": eid,
                    "label": e["name"],
                    "group": "entity",
                    "meta": {"name": e["name"], "type": e["type"]},
                }
            )
            seen_nodes.add(eid)
        edges.append({"from": e["decision_id"], "to": eid})

    for i in graph_arrays.get("decision_initiators", []):
        iid = f"person:{i['initiator_name']}"
        if iid not in seen_nodes:
            nodes.append(
                {
                    "id": iid,
                    "label": i["initiator_name"],
                    "group": "person",
                    "meta": {"name": i["initiator_name"], "role": i["initiator_role"]},
                }
            )
            seen_nodes.add(iid)
        edges.append({"from": iid, "to": i["decision_id"], "label": "initiated"})

    for f in graph_arrays.get("decision_facts", []):
        fid = f"fact:{f['decision_id']}:{f['k']}"
        if fid not in seen_nodes:
            nodes.append(
                {
                    "id": fid,
                    "label": f"{f['k']}: {f['v']}",
                    "group": "fact",
                    "meta": {"key": f["k"], "value": f["v"]},
                }
            )
            seen_nodes.add(fid)
        edges.append({"from": f["decision_id"], "to": fid})

    for c in graph_arrays.get("decision_constraints", []):
        cid = f"constraint:{c['decision_id']}:{c['text'][:30]}"
        if cid not in seen_nodes:
            nodes.append(
                {
                    "id": cid,
                    "label": c["text"],
                    "group": "constraint",
                    "meta": {"text": c["text"]},
                }
            )
            seen_nodes.add(cid)
        edges.append({"from": c["decision_id"], "to": cid})

    return {"nodes": nodes, "edges": edges}

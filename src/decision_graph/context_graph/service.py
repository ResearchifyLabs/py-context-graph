"""
Context Graph Service.

Orchestrates: resolve → plan → execute template → post-process → format response.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from decision_graph.context_graph.planner import plan
from decision_graph.context_graph.post_processing import (
    apply_caps,
    generate_evidence,
    normalize_graph,
    rank_decisions,
)
from decision_graph.core.interfaces import GraphReader

_logger = logging.getLogger(__name__)

DEFAULT_TIME_WINDOW_DAYS = 90


def _collect_connected(
    decision_id: str,
    edges: List[Dict[str, Any]],
    nodes_map: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    topics, entities, people, facts = [], [], [], []
    for e in edges:
        if e["from"] != decision_id and e["to"] != decision_id:
            continue
        other_id = e["to"] if e["from"] == decision_id else e["from"]
        other = nodes_map.get(other_id, {})
        other.get("type", "")
        other_props = other.get("properties", {})
        name = other_props.get("name", other_id)

        if e["type"] == "HAS_TOPIC":
            topics.append(name)
        elif e["type"] == "INITIATED_BY":
            people.append(name)
        elif e["type"] == "MENTIONS_ENTITY":
            subtype = other_props.get("type", "")
            if subtype == "Person":
                people.append(name)
            else:
                entities.append(name)
        elif e["type"] == "HAS_FACT":
            k = other_props.get("k", "")
            v = other_props.get("v", "")
            if k and v:
                facts.append(f"{k}: {v}")
        elif e["type"] == "HAS_CONSTRAINT":
            text = other_props.get("text", name)
            if text:
                facts.append(text)

    return {"topics": topics, "entities": entities, "people": people, "facts": facts}


class ContextGraphService:
    def __init__(self, reader: GraphReader, projection_store=None):
        self._reader = reader
        self._projection_store = projection_store

    def resolve(
        self,
        text: str,
        types: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        _logger.info("graph.resolve text=%s types=%s top_k=%d", text, types, top_k)
        result = self._reader.resolve(text, types=types, top_k=top_k)
        self._log_telemetry(
            "graph.resolve",
            seed_type=None,
            latency_ms=result["debug"]["latency_ms"],
            node_count=len(result["candidates"]),
        )
        return result

    def query(
        self,
        intent: str,
        seed: Dict[str, Any],
        scope: Optional[Dict[str, Any]] = None,
        time_window: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limits: Optional[Dict[str, Any]] = None,
        response_mode: str = "CHAT",
    ) -> Dict[str, Any]:
        start = time.time()
        filters = filters or {}
        limits = limits or {}

        seed_type = seed.get("type", "Topic")
        seed_text = seed.get("text", "")
        seed_id = seed.get("id")

        rel_allowlist = filters.get("relationship_allowlist")
        decision_type_filter = filters.get("decision_type_filter")
        plan_result = plan(
            intent=intent,
            seed_type=seed_type,
            response_mode=response_mode,
            rel_allowlist=rel_allowlist,
        )

        resolved_seed = None
        resolved_decision_ids = []
        if not seed_id and seed_text:
            resolve_result = self._reader.resolve(seed_text, types=["Topic", "Entity"], top_k=5)
            if resolve_result["candidates"]:
                non_decision = [c for c in resolve_result["candidates"] if c["type"] != "Decision"]
                decision_hits = [c for c in resolve_result["candidates"] if c["type"] == "Decision"]

                if non_decision:
                    preferred_type = "Entity" if seed_type in ("Person", "Entity") else seed_type
                    match = next((c for c in non_decision if c["type"] == preferred_type), None)
                    resolved_seed = match or non_decision[0]
                    seed_id = resolved_seed["id"]
                elif decision_hits:
                    resolved_seed = decision_hits[0]
                    resolved_decision_ids = [c["id"] for c in decision_hits]

        max_paths = limits.get("max_paths", 200)
        max_nodes = limits.get("max_nodes", 150)
        max_edges = limits.get("max_edges", 300)
        top_k = limits.get("top_k", 20)

        all_nodes = []
        all_edges = []
        debug = {}

        if resolved_decision_ids:
            from decision_graph.context_graph.registry import ALL_REL_TYPES

            for did in resolved_decision_ids[:top_k]:
                raw = self._reader.execute_template(
                    "DECISION_EGO_V1",
                    {
                        "node_id": did,
                        "rel_allowlist": ALL_REL_TYPES,
                    },
                )
                graph = normalize_graph(raw["records"])
                all_nodes.extend(graph["nodes"])
                all_edges.extend(graph["edges"])
                debug = raw["debug"]
        else:
            params = {
                "q": seed_text or seed_id or "",
                "node_id": seed_id,
                "max_paths": max_paths,
                "rel_allowlist": plan_result.rel_allowlist,
            }
            raw = self._reader.execute_template(plan_result.template_id, params)
            graph = normalize_graph(raw["records"])
            all_nodes = graph["nodes"]
            all_edges = graph["edges"]
            debug = raw["debug"]

        seen = set()
        nodes = []
        for n in all_nodes:
            if n["id"] not in seen:
                seen.add(n["id"])
                nodes.append(n)
        edges = all_edges

        decision_nodes = [n for n in nodes if n["type"] == "Decision"]
        decision_ids = resolved_decision_ids or [d["id"] for d in decision_nodes]

        root_ids = resolved_decision_ids or ([seed_id] if seed_id else [])
        capped = apply_caps(
            nodes, edges, root_ids=root_ids, result_ids=decision_ids, max_nodes=max_nodes, max_edges=max_edges
        )

        ranked = rank_decisions(decision_nodes, capped["nodes"], capped["edges"], plan_result.rel_weights)

        nodes_map = {n["id"]: n for n in capped["nodes"]}

        top_decision_ids = [d["id"] for d in ranked[:top_k]]
        projections = {}
        if self._projection_store and top_decision_ids:
            projections = self._projection_store.find_by_ids(top_decision_ids)

        facts_by_decision = self._fetch_facts_batch(top_decision_ids)

        results = []
        for d in ranked[:top_k]:
            did = d["id"]
            props = d.get("properties", {})

            proj_doc = projections.get(did, {})
            proj = proj_doc.get("projection", {}) if proj_doc else {}

            if decision_type_filter and proj.get("decision_type", "") != decision_type_filter:
                continue

            why = generate_evidence(did, capped["edges"], nodes_map)
            connected = _collect_connected(did, capped["edges"], nodes_map)

            graph_facts = connected.get("facts", [])
            batch_facts = facts_by_decision.get(did, [])
            all_facts = list(dict.fromkeys(graph_facts + batch_facts))

            result_entry = {
                "id": did,
                "type": "Decision",
                "subject": proj.get("subject_label", ""),
                "action": proj.get("action_desc", ""),
                "action_type": proj.get("action_type", ""),
                "decision_type": proj.get("decision_type", ""),
                "initiator": proj.get("initiator_name", ""),
                "counterparties": proj.get("counterparty_names", []),
                "status": proj.get("status_state", ""),
                "evidence": proj.get("evidence_span", ""),
                "timestamp": props.get("linked_at"),
                "topics": connected.get("topics", []),
                "entities": connected.get("entities", []),
                "people": connected.get("people", []),
                "facts": all_facts,
                "why_matched": why,
                "open_payload": {"mode": "EGO", "node_id": did, "k": 2},
            }
            results.append(result_entry)

        normalized_info = {
            "intent": intent,
            "seed": resolved_seed or {"id": seed_id, "type": seed_type, "name": seed_text},
            "time_window": time_window or self._default_time_window(),
        }

        latency_ms = int((time.time() - start) * 1000)
        debug["latency_ms"] = latency_ms

        response = {
            "normalized": normalized_info,
            "results": results,
            "debug": debug,
        }

        if response_mode == "UI":
            response["graph"] = {
                "nodes": capped["nodes"],
                "edges": capped["edges"],
                "caps_applied": capped["caps_applied"],
            }

        self._log_telemetry(
            "graph.query",
            intent=intent,
            seed_type=seed_type,
            latency_ms=latency_ms,
            node_count=len(capped["nodes"]),
            edge_count=len(capped["edges"]),
            result_count=len(results),
            sampled=capped["caps_applied"]["sampled"],
        )

        return response

    def open(
        self,
        mode: str,
        node_id: str,
        k: int = 2,
        rel_allowlist: Optional[List[str]] = None,
        caps: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        caps = caps or {}
        max_nodes = caps.get("max_nodes", 150)
        max_edges = caps.get("max_edges", 300)

        if not rel_allowlist:
            from decision_graph.context_graph.registry import ALL_REL_TYPES

            rel_allowlist = ALL_REL_TYPES

        params = {
            "node_id": node_id,
            "rel_allowlist": rel_allowlist,
        }
        raw = self._reader.execute_template("DECISION_EGO_V1", params)
        records = raw["records"]

        graph = normalize_graph(records)
        capped = apply_caps(
            graph["nodes"],
            graph["edges"],
            root_ids=[node_id],
            result_ids=[node_id],
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

        nodes_map = {n["id"]: n for n in capped["nodes"]}
        seed_node = nodes_map.get(node_id, {})

        latency_ms = int((time.time() - start) * 1000)

        response = {
            "summary": {
                "node_id": node_id,
                "type": seed_node.get("type", "Decision"),
                "properties": seed_node.get("properties", {}),
                "neighbor_count": len(capped["nodes"]) - 1,
            },
            "graph": {
                "nodes": capped["nodes"],
                "edges": capped["edges"],
                "caps_applied": capped["caps_applied"],
            },
            "debug": {**raw["debug"], "latency_ms": latency_ms},
        }

        self._log_telemetry(
            "graph.open",
            seed_type="Decision",
            latency_ms=latency_ms,
            node_count=len(capped["nodes"]),
            edge_count=len(capped["edges"]),
            sampled=capped["caps_applied"]["sampled"],
        )

        return response

    def _fetch_facts_batch(self, decision_ids: List[str]) -> Dict[str, List[str]]:
        if not decision_ids:
            return {}
        records = self._reader.run_query(
            'UNWIND $ids AS did '
            'MATCH (d:Decision {decision_id: did})-[:HAS_FACT]->(f:Fact) '
            'RETURN d.decision_id AS decision_id, f.k AS k, f.v AS v',
            {"ids": decision_ids},
        )
        result: Dict[str, List[str]] = {}
        for rec in records:
            did = rec["decision_id"]
            k = rec.get("k", "")
            v = rec.get("v", "")
            if k and v:
                result.setdefault(did, []).append(f"{k}: {v}")
        return result

    def _default_time_window(self) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=DEFAULT_TIME_WINDOW_DAYS)
        return {
            "from": from_date.strftime("%Y-%m-%d"),
            "to": now.strftime("%Y-%m-%d"),
        }

    def _log_telemetry(self, tool: str, **kwargs):
        _logger.info("TELEMETRY tool=%s %s", tool, " ".join(f"{k}={v}" for k, v in kwargs.items()))

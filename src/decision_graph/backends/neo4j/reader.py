"""Neo4j implementation of the GraphReader protocol."""

import logging
import time
from typing import Any, Dict, List, Optional

from decision_graph.context_graph.templates import RESOLVE_BY_KEYWORD, RESOLVE_NODES, TEMPLATES
from decision_graph.core.interfaces import GraphReader

_logger = logging.getLogger(__name__)

# Label preference order when a node carries multiple labels.
_LABEL_PRIORITY = ("Decision", "Cluster", "Topic", "Entity", "Constraint", "Fact")


def _primary_label(labels: List[str], preferred: List[str]) -> str:
    for p in preferred:
        if p in labels:
            return p
    for p in _LABEL_PRIORITY:
        if p in labels:
            return p
    return labels[0] if labels else "Unknown"


def _candidate_name(props: Dict[str, Any], node_type: str) -> str:
    if node_type == "Decision":
        return props.get("subject_label") or props.get("decision_id", "")
    # Topic nodes store the text in "name"; Entity/Person nodes also use "name".
    return props.get("name") or props.get("display_name", "")


def _candidate_id(props: Dict[str, Any]) -> str:
    """Stable node identity used by post-processing and the UI."""
    return (
        props.get("decision_id")
        or props.get("cluster_id")
        or f"{props.get('name', '')}:{props.get('type', '')}"
    )


class Neo4jGraphReader(GraphReader):
    """Neo4j-backed read interface for context-graph queries.

    Implements the three methods required by ``GraphReader``:
      * ``resolve``          — fuzzy node lookup by text (seed resolution)
      * ``execute_template`` — named Cypher template execution
      * ``run_query``        — raw parameterised Cypher (used by ``_fetch_facts_batch``)
    """

    def __init__(self, driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    # ------------------------------------------------------------------
    # GraphReader protocol
    # ------------------------------------------------------------------

    def resolve(
        self,
        text: str,
        types: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Find candidate nodes whose ``name`` contains *text*.

        Falls back to a keyword search over Fact values when the name-match
        returns nothing and ``Decision`` is in *types*.

        Returns
        -------
        {
          "candidates": [{"id": str, "type": str, "name": str, "properties": dict}, …],
          "debug":      {"latency_ms": int, "query": str, "types": list},
        }
        """
        start = time.time()
        if not types:
            types = ["Topic", "Entity", "Decision"]

        candidates: List[Dict[str, Any]] = []
        seen: set = set()

        with self._driver.session(database=self._database) as session:
            result = session.run(
                RESOLVE_NODES,
                {"q": text.lower(), "types": types, "top_k": top_k},
            )
            for record in result:
                node = record["n"]
                props = dict(node)
                node_type = _primary_label(list(record["lbls"]), types)
                cid = _candidate_id(props)
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(
                        {
                            "id": cid,
                            "type": node_type,
                            "name": _candidate_name(props, node_type),
                            "properties": props,
                        }
                    )

            # Fallback: search Fact k/v when name-match produced nothing.
            if not candidates and "Decision" in types:
                result = session.run(
                    RESOLVE_BY_KEYWORD,
                    {"q": text.lower(), "top_k": top_k},
                )
                for record in result:
                    node = record["d"]
                    props = dict(node)
                    cid = props.get("decision_id", "")
                    if cid and cid not in seen:
                        seen.add(cid)
                        candidates.append(
                            {
                                "id": cid,
                                "type": "Decision",
                                "name": _candidate_name(props, "Decision"),
                                "properties": props,
                            }
                        )

        latency_ms = int((time.time() - start) * 1000)
        return {
            "candidates": candidates,
            "debug": {"latency_ms": latency_ms, "query": text, "types": types},
        }

    def execute_template(
        self,
        template_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a named Cypher template.

        Parameters are forwarded verbatim to Neo4j so the caller controls
        ``$q``, ``$node_id``, ``$rel_allowlist``, etc.

        Returns
        -------
        {
          "records": [neo4j.Record, …],   # raw records for normalize_graph()
          "debug":   {"latency_ms": int, "template_id": str},
        }
        """
        query = TEMPLATES.get(template_id)
        if not query:
            raise ValueError(f"Unknown template_id: {template_id!r}")

        start = time.time()
        with self._driver.session(database=self._database) as session:
            records = list(session.run(query, params))

        latency_ms = int((time.time() - start) * 1000)
        _logger.debug(
            "execute_template template_id=%s records=%d latency_ms=%d",
            template_id,
            len(records),
            latency_ms,
        )
        return {
            "records": records,
            "debug": {"latency_ms": latency_ms, "template_id": template_id},
        }

    def run_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute an arbitrary parameterised Cypher query.

        Returns each record as a plain ``dict`` (``record.data()``).
        Used by ``ContextGraphService._fetch_facts_batch``.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

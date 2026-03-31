"""
Relationship planner.

Decides which relationships, node types, hop depth, and Cypher template
to use based on intent + seed type.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlanResult:
    rel_allowlist: List[str]
    node_type_allowlist: List[str]
    k: int
    template_id: str
    rel_weights: Dict[str, float] = field(default_factory=dict)


SEED_BUNDLES: Dict[str, Dict[str, List[str]]] = {
    "Topic": {
        "primary": ["HAS_TOPIC"],
        "enrichment": ["INITIATED_BY", "HAS_COUNTERPARTY", "MENTIONS_ENTITY"],
        "optional": ["DEPENDS_ON"],
    },
    "Person": {
        "primary": ["INITIATED_BY", "HAS_COUNTERPARTY"],
        "enrichment": ["HAS_TOPIC", "MENTIONS_ENTITY", "DEPENDS_ON"],
        "optional": [],
    },
    "Entity": {
        "primary": ["MENTIONS_ENTITY"],
        "enrichment": ["HAS_TOPIC", "INITIATED_BY", "HAS_COUNTERPARTY", "DEPENDS_ON"],
        "optional": [],
    },
    "Decision": {
        "primary": ["HAS_TOPIC", "INITIATED_BY", "HAS_COUNTERPARTY", "MENTIONS_ENTITY"],
        "dependency": ["DEPENDS_ON"],
        "optional": [],
    },
}

SEED_TO_TEMPLATE: Dict[str, str] = {
    "Topic": "TOPIC_RELATED_DECISIONS_V1",
    "Person": "PERSON_RELATED_DECISIONS_V1",
    "Entity": "ENTITY_RELATED_DECISIONS_V1",
    "Decision": "DECISION_EGO_V1",
}


def _collect_rels(seed_type: str) -> List[str]:
    bundle = SEED_BUNDLES.get(seed_type, SEED_BUNDLES["Topic"])
    rels = []
    for key in ("primary", "dependency", "enrichment", "optional"):
        rels.extend(bundle.get(key, []))
    return rels


def plan(
    intent: str,
    seed_type: str,
    response_mode: str = "CHAT",
    rel_allowlist: Optional[List[str]] = None,
) -> PlanResult:
    from decision_graph.context_graph.registry import get_weight

    rels = rel_allowlist if rel_allowlist else _collect_rels(seed_type)
    weights = {r: get_weight(r) for r in rels}

    if intent == "FIND_RELATED":
        k = 2
        template_id = SEED_TO_TEMPLATE.get(seed_type, "TOPIC_RELATED_DECISIONS_V1")
        node_types = ["Decision", "Topic", "Entity", "Constraint", "Fact"]

    elif intent == "FIND_DECISIONS":
        k = 2
        template_id = SEED_TO_TEMPLATE.get(seed_type, "TOPIC_RELATED_DECISIONS_V1")
        node_types = ["Decision"]

    elif intent == "EXPLAIN_CONNECTION":
        k = 5
        template_id = "EXPLAIN_CONNECTION_V1"
        node_types = ["Decision", "Topic", "Entity", "Constraint", "Fact"]

    elif intent == "SUMMARIZE_SCOPE":
        k = 1
        template_id = SEED_TO_TEMPLATE.get(seed_type, "TOPIC_RELATED_DECISIONS_V1")
        node_types = ["Decision", "Topic", "Entity"]

    else:
        k = 1
        template_id = SEED_TO_TEMPLATE.get(seed_type, "TOPIC_RELATED_DECISIONS_V1")
        node_types = ["Decision", "Topic", "Entity", "Constraint", "Fact"]

    return PlanResult(
        rel_allowlist=rels,
        node_type_allowlist=node_types,
        k=k,
        template_id=template_id,
        rel_weights=weights,
    )

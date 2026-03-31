"""
Canonical relationship registry for the context graph.

Single source of truth for relationship types, their endpoints, and weights.
Used by the planner, templates, and UI filters.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RelationshipDef:
    from_types: List[str]
    to_types: List[str]
    weight: float


REL_REGISTRY: Dict[str, RelationshipDef] = {
    "HAS_TOPIC": RelationshipDef(from_types=["Decision"], to_types=["Topic"], weight=0.9),
    "INITIATED_BY": RelationshipDef(from_types=["Decision"], to_types=["Entity"], weight=0.8),
    "HAS_COUNTERPARTY": RelationshipDef(from_types=["Decision"], to_types=["Entity"], weight=0.8),
    "MENTIONS_ENTITY": RelationshipDef(from_types=["Decision"], to_types=["Entity"], weight=0.6),
    "DEPENDS_ON": RelationshipDef(from_types=["Decision"], to_types=["Decision"], weight=0.7),
    "HAS_CONSTRAINT": RelationshipDef(from_types=["Decision"], to_types=["Constraint"], weight=0.5),
    "HAS_FACT": RelationshipDef(from_types=["Decision"], to_types=["Fact"], weight=0.4),
    "HAS_DECISION": RelationshipDef(from_types=["Cluster"], to_types=["Decision"], weight=0.6),
}

ALL_REL_TYPES = list(REL_REGISTRY.keys())


def get_weight(rel_type: str) -> float:
    defn = REL_REGISTRY.get(rel_type)
    return defn.weight if defn else 0.0

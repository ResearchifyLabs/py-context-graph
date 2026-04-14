import re
import uuid
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional
)

from pydantic import AliasChoices
from pydantic import BaseModel as PydanticBaseModel
from pydantic import (
    ConfigDict,
    Field,
    field_validator
)


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class _Debug(BaseModel):
    reasoning: str = Field(
        ...,
        description="Concise and clear justification for extracted field values and validation decisions",
    )


class LLMResponseModel(BaseModel):
    debug: _Debug


InitiatorRole = Literal["client", "internal", "vendor", "unknown"]
EntityType = str

ActionType = Literal[
    "request",
    "approve",
    "confirm",
    "review",
    "fix",
    "discuss",
    "propose",
    "announce_change",
    "share_link",
    "provide_report",
    "status_update",
    "other",
]

_ACTION_TYPE_VALUES = set(ActionType.__args__)

StatusState = Literal["proposed", "in_progress", "blocked", "done", "unknown"]


class Entity(BaseModel):
    type: EntityType
    name: str = Field(validation_alias=AliasChoices("name", "value"))


class KV(BaseModel):
    k: str
    v: str


class EvidenceRef(BaseModel):
    channel: str
    gid: str
    cid: str
    message_id: Optional[str] = None
    ts: Optional[int] = None
    author: Optional[str] = None
    excerpt: Optional[str] = None


class ConstraintOverride(BaseModel):
    by_whom: str
    reason: str


class ConstraintEvaluation(BaseModel):
    constraint_type: str
    constraint_value: Any
    evaluated_on: Dict[str, Any]
    result: Literal["pass", "fail", "overridden", "unknown"]
    override: Optional[ConstraintOverride] = None


class AlternativeConsidered(BaseModel):
    action_type: str
    reason_rejected: str


class SelectedAlternative(BaseModel):
    action_type: str


class DecisionUnitRow(BaseModel):
    decision_id: str
    pid: str
    gid: str
    cid: str
    updated_at: float
    recorded_at: float
    decision_type: str
    decision_subtype: Optional[str] = None

    initiator_name: Optional[str] = None
    initiator_role: InitiatorRole = "unknown"
    counterparty_names: List[str] = Field(default_factory=list)

    subject_label: str

    action_type: ActionType
    action_desc: str
    action_key: Optional[str] = None

    status_state: StatusState = "unknown"
    status_blocker: Optional[str] = None

    evidence_span: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("action_type", mode="before")
    @classmethod
    def _coerce_action_type(cls, v):
        return v if v in _ACTION_TYPE_VALUES else "other"

    @field_validator("counterparty_names", mode="before")
    @classmethod
    def _coerce_counterparties(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r"[;,]", v) if p.strip()]
            return parts
        return v


class DecisionUnitRowList(BaseModel):
    items: List[DecisionUnitRow] = Field(default_factory=list)


class DecisionUnitCoreExtractActors(BaseModel):
    initiator_name: Optional[str]
    initiator_role: InitiatorRole
    counterparty_names: List[str]


class DecisionUnitCoreExtractSubject(BaseModel):
    label: str


class DecisionUnitCoreExtractAction(BaseModel):
    type: ActionType
    description: str

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_action_type(cls, v):
        return v if v in _ACTION_TYPE_VALUES else "other"


class DecisionUnitCoreExtractStatus(BaseModel):
    state: StatusState
    blocker: Optional[str]


class DecisionUnitCoreExtractEvidence(BaseModel):
    span: str
    confidence: float = Field(ge=0.0, le=1.0)


class DecisionUnitCoreExtract(BaseModel):
    decision_type: str
    actors: DecisionUnitCoreExtractActors
    subject: DecisionUnitCoreExtractSubject
    action: DecisionUnitCoreExtractAction
    status: DecisionUnitCoreExtractStatus
    evidence: DecisionUnitCoreExtractEvidence


class DecisionUnitCoreExtractList(BaseModel):
    items: List[DecisionUnitCoreExtract]


class DecisionJoinRow(BaseModel):
    from_decision_id: str
    to_decision_id: str
    join_type: Literal["continuation", "same_intent", "clarification"]
    join_confidence: float = Field(ge=0.0, le=1.0)
    join_reasons: List[str] = Field(default_factory=list)
    created_at: int


class DecisionLineageRow(BaseModel):
    from_decision_id: str
    to_decision_id: str
    edge_type: Literal["depends_on", "supersedes", "confirms", "cancels", "overrides", "conflicts_with"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    created_at: int


class DecisionEnrichmentRow(BaseModel):
    decision_id: str
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    recorded_at: float
    topics: List[str] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)

    action_params: Dict[str, Any] = Field(default_factory=dict)

    constraints_text: List[str] = Field(default_factory=list)
    key_facts: List[KV] = Field(default_factory=list)

    constraint_evaluations: List[ConstraintEvaluation] = Field(default_factory=list)
    alternatives_considered: List[AlternativeConsidered] = Field(default_factory=list)
    selected_alternative: Optional[SelectedAlternative] = None
    rejection_reasons: List[str] = Field(default_factory=list)

    @staticmethod
    def _split_semicolonish(v: str) -> List[str]:
        return [p.strip() for p in re.split(r"[;]", v) if p.strip()]

    @staticmethod
    def _parse_kv_string(v: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in DecisionEnrichmentRow._split_semicolonish(v):
            if "=" not in part:
                continue
            k, val = part.split("=", 1)
            k = k.strip()
            val = val.strip()
            if k:
                out[k] = val
        return out

    @field_validator("topics", mode="before")
    @classmethod
    def _coerce_topics(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return cls._split_semicolonish(v)
        return v

    @field_validator("entities", mode="before")
    @classmethod
    def _coerce_entities(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            items = []
            for part in cls._split_semicolonish(v):
                if ":" not in part:
                    continue
                et, name = part.split(":", 1)
                et = et.strip()
                name = name.strip()
                if et and name:
                    items.append({"type": et, "value": name})
            return items
        return v

    @field_validator("action_params", mode="before")
    @classmethod
    def _coerce_action_params(cls, v):
        if v is None:
            return {}
        if isinstance(v, str):
            return cls._parse_kv_string(v)
        return v

    @field_validator("constraints_text", mode="before")
    @classmethod
    def _coerce_constraints_text(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return cls._split_semicolonish(v)
        return v

    @field_validator("key_facts", mode="before")
    @classmethod
    def _coerce_key_facts(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            kvs = []
            for k, val in cls._parse_kv_string(v).items():
                kvs.append({"k": k, "v": val})
            return kvs
        return v


class DecisionEnrichmentCoreExtract(BaseModel):
    topics: List[str]
    entities: List[Entity]
    action_params: List[KV]
    constraints_text: List[str]
    key_facts: List[KV]


class DecisionCluster(BaseModel):
    cluster_id: str
    gid: str = ""
    primary_subject: str
    rolling_summary: str
    created_at: float
    last_updated_at: float
    decision_count: int = 0
    cids: List[str] = Field(default_factory=list)
    decision_ids: List[str] = Field(default_factory=list)


class DecisionLink(BaseModel):
    decision_id: str
    cluster_id: str
    cid: str
    gid: str
    trace_id: str
    match_metadata: Dict[str, Any] = Field(default_factory=dict)
    linked_at: float


class ClusterMetadataExtract(LLMResponseModel):
    primary_subject: str
    rolling_summary: str

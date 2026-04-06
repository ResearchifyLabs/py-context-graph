from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class EnrichmentStore(Protocol):
    """Domain-oriented store for decision enrichment documents."""

    def find_by_id(self, decision_id: str) -> Optional[dict]:
        ...

    def find_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        ...

    async def find_by_ids_async(self, ids: List[str]) -> Dict[str, dict]:
        ...

    def save(self, decision_id: str, data: dict) -> None:
        ...

    def upsert(self, decision_id: str, data: dict) -> None:
        ...

    def query(
        self,
        filters: List[Tuple[str, str, Any]],
        order_by: Optional[List[Tuple[str, str]]] = None,
        limit: int = 200,
    ) -> List[dict]:
        ...


@runtime_checkable
class ProjectionStore(Protocol):
    """Domain-oriented store for projection documents."""

    def find_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        ...

    def find_by_conv_ids(self, cids: List[str], proj_type: str) -> List[dict]:
        ...

    async def find_by_filters(
        self,
        *,
        gids: List[str],
        proj_type: str,
        last_n_days: Optional[int] = None,
        limit: Optional[int] = None,
        before_ts: Optional[float] = None,
    ) -> List[dict]:
        ...

    def find_by_id(self, pid: str) -> Optional[dict]:
        ...

    def query(
        self,
        filters: List[Tuple[str, str, Any]],
        order_by: Optional[List[Tuple[str, str]]] = None,
        limit: int = 200,
    ) -> List[dict]:
        ...

    def invalidate(self, pid: str) -> None:
        ...

    def save(self, *, pid: str, gid: str, cid: str, proj_type: str, projection: dict, msg_ts: int) -> bool:
        ...

    def update(self, *, pid: str, projection: dict, update_type: str, msg_ts: int) -> dict:
        ...


@runtime_checkable
class ClusterStore(Protocol):
    """Domain-oriented store for decision cluster documents."""

    def create(self, data: dict) -> str:
        ...

    def update(self, cluster_id: str, updates: dict) -> None:
        ...

    def find_by_id(self, cluster_id: str) -> Optional[dict]:
        ...

    def find_by_ids(self, cluster_ids: List[str]) -> List[dict]:
        ...


@runtime_checkable
class LinkStore(Protocol):
    """Domain-oriented store for decision link documents."""

    def save_batch(self, links: List[dict]) -> int:
        ...

    def find_by_decision_id(self, decision_id: str) -> Optional[dict]:
        ...

    def find_by_cluster_id(self, cluster_id: str) -> List[dict]:
        ...

    def find_by_decision_ids(self, decision_ids: List[str]) -> Dict[str, dict]:
        ...

    def find_cluster_ids_by_gids(self, gids: List[str]) -> List[str]:
        ...

    def find_cluster_id_for_decision(self, decision_id: str) -> Optional[str]:
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Write-only store for syncing hydrated cluster data to a graph database."""

    def ingest(self, hydrated_clusters: List[Dict[str, Any]]) -> None:
        ...


class NullGraphStore(GraphStore):
    """No-op graph store. Use when graph materialization is not needed."""

    def ingest(self, hydrated_clusters: List[Dict[str, Any]]) -> None:
        pass


@runtime_checkable
class VectorIndex(Protocol):
    """Domain-oriented interface for vector similarity search."""

    def get_top_n_matches(self, *, query: str, query_filter: Optional[dict], top_n: int) -> Any:
        ...


@runtime_checkable
class LLMAdapter(Protocol):
    """Domain-oriented interface for LLM execution.

    Implementations translate an ``LLMConfig`` into provider-specific calls
    (OpenAI, LiteLLM, etc.) and return parsed results.
    """

    async def execute_async(self, model_config: Any, data: Any, additional_data: Optional[dict] = None) -> Any:
        ...


@runtime_checkable
class MatchScorer(Protocol):
    """Pluggable scoring strategy for decision-pair matching.

    ``precompute`` may enrich a decision dict with cached tokens/embeddings.
    ``score_pair`` returns a dict that **must** contain a ``combined_score``
    float; any extra keys are persisted as match metadata.
    """

    def precompute(self, decision: Dict[str, Any]) -> Dict[str, Any]: ...

    def score_pair(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]: ...


@runtime_checkable
class GraphReader(Protocol):
    """Read-only interface for querying the graph database (e.g. Neo4j)."""

    def resolve(self, text: str, types: Optional[List[str]] = None, top_k: int = 8) -> Dict[str, Any]:
        ...

    def execute_template(self, template_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ...

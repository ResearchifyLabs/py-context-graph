from abc import ABC, abstractmethod

from decision_graph.core.interfaces import (
    ClusterStore,
    EnrichmentStore,
    LinkStore,
    ProjectionStore,
)


class StorageBackend(ABC):
    """
    Groups the four document-store interfaces behind a single swap-point.

    Implement this to use a different storage engine (Postgres, Mongo, etc.).
    Vector search (VectorIndex), graph materialization (GraphStore), and
    LLM execution (LLMAdapter) are injected separately — they are orthogonal
    to document storage.
    """

    @abstractmethod
    def enrichment_store(self) -> EnrichmentStore:
        ...

    @abstractmethod
    def projection_store(self) -> ProjectionStore:
        ...

    @abstractmethod
    def cluster_store(self) -> ClusterStore:
        ...

    @abstractmethod
    def link_store(self) -> LinkStore:
        ...

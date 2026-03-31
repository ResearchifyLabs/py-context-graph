from decision_graph.backends.memory.stores import (
    InMemoryClusterStore,
    InMemoryEnrichmentStore,
    InMemoryLinkStore,
    InMemoryProjectionStore,
    InMemoryVectorIndex,
)
from decision_graph.core.registry import StorageBackend


class InMemoryBackend(StorageBackend):
    """Fully in-memory backend. Useful for testing and as a reference implementation."""

    def __init__(self):
        self._enrichment_store = InMemoryEnrichmentStore()
        self._projection_store = InMemoryProjectionStore()
        self._cluster_store = InMemoryClusterStore()
        self._link_store = InMemoryLinkStore()

    def enrichment_store(self):
        return self._enrichment_store

    def projection_store(self):
        return self._projection_store

    def cluster_store(self):
        return self._cluster_store

    def link_store(self):
        return self._link_store

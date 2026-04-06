from typing import Optional

from decision_graph.core.config import GraphConfig
from decision_graph.core.interfaces import LLMAdapter, VectorIndex
from decision_graph.core.registry import StorageBackend


class DecisionGraph:
    """
    Entry point for the decision graph library.

    Holds a StorageBackend and provides factory methods for query services.

    Usage::

        from decision_graph import DecisionGraph
        from decision_graph.backends.memory import InMemoryBackend
        from decision_graph.llm import LiteLLMAdapter

        dg = DecisionGraph(backend=InMemoryBackend(), executor=LiteLLMAdapter())
        service = dg.graph_service()
    """

    def __init__(self, *, backend: StorageBackend, executor: LLMAdapter, config: Optional[GraphConfig] = None):
        self._backend = backend
        self._executor = executor
        self._config = config or GraphConfig()

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    def graph_service(self):
        from decision_graph.services import DecisionGraphService

        return DecisionGraphService(backend=self._backend)

    def retrieval(self):
        from decision_graph.retrieval import DecisionEnrichmentRetrieval

        return DecisionEnrichmentRetrieval(backend=self._backend)

    def context_retriever(self, *, gids, vector_index: VectorIndex):
        from decision_graph.context_retrieval import DecisionContextRetriever

        return DecisionContextRetriever(
            gids=gids,
            backend=self._backend,
            vector_index=vector_index,
        )

    def cluster_service(self):
        from decision_graph.clustering_service import DecisionClusterService

        return DecisionClusterService(backend=self._backend, executor=self._executor, config=self._config)

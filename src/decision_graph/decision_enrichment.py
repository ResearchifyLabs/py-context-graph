"""
Backward-compatibility facade.

All classes have been split into focused modules:
- DecisionEnrichmentService -> enrichment_service.py
- DecisionContextRetriever  -> context_retrieval.py
- DecisionClusterService    -> clustering_service.py
- DecisionTracePipeline     -> decision_trace_pipeline.py
- Pure algorithms           -> core/matching.py
"""

from decision_graph.clustering_service import DecisionClusterService
from decision_graph.context_retrieval import DecisionContextRetriever
from decision_graph.decision_trace_pipeline import DecisionTracePipeline
from decision_graph.enrichment_service import DecisionEnrichmentService

__all__ = [
    "DecisionEnrichmentService",
    "DecisionContextRetriever",
    "DecisionClusterService",
    "DecisionTracePipeline",
]

"""Simple Firestore backend for decision_graph.

Usage::

    from google.cloud import firestore
    from decision_graph.backends.firestore import FirestoreBackend

    client = firestore.Client()
    backend = FirestoreBackend(client=client, collection_prefix="myapp_")
"""

from decision_graph.backends.firestore.stores import (
    FirestoreClusterStore,
    FirestoreEnrichmentStore,
    FirestoreLinkStore,
    FirestoreProjectionStore,
)
from decision_graph.core.registry import StorageBackend


class FirestoreBackend(StorageBackend):
    """Standalone Firestore backend — requires only a ``google.cloud.firestore.Client``."""

    def __init__(self, *, client, collection_prefix: str = ""):
        self._client = client
        self._prefix = collection_prefix

    def _col(self, name: str) -> str:
        return f"{self._prefix}{name}"

    def enrichment_store(self):
        return FirestoreEnrichmentStore(self._client, self._col("decision_enrichments"))

    def projection_store(self):
        return FirestoreProjectionStore(self._client, self._col("decision_projections"))

    def cluster_store(self):
        return FirestoreClusterStore(self._client, self._col("decision_clusters"))

    def link_store(self):
        return FirestoreLinkStore(self._client, self._col("decision_links"))

"""Tests for the simple Firestore backend (OSS version).

Uses MockFirestore — no real GCP credentials needed.
"""

import unittest

from mockfirestore import MockFirestore

from decision_graph.backends.firestore import FirestoreBackend
from decision_graph.backends.firestore.stores import (
    FirestoreClusterStore,
    FirestoreEnrichmentStore,
    FirestoreLinkStore,
    FirestoreProjectionStore,
)


class TestFirestoreBackendWiring(unittest.TestCase):
    def setUp(self):
        self.client = MockFirestore()

    def tearDown(self):
        self.client.reset()

    def test_returns_correct_store_types(self):
        backend = FirestoreBackend(client=self.client)
        self.assertIsInstance(backend.enrichment_store(), FirestoreEnrichmentStore)
        self.assertIsInstance(backend.projection_store(), FirestoreProjectionStore)
        self.assertIsInstance(backend.cluster_store(), FirestoreClusterStore)
        self.assertIsInstance(backend.link_store(), FirestoreLinkStore)

    def test_collection_prefix_applied(self):
        backend = FirestoreBackend(client=self.client, collection_prefix="test_")
        enrichment = backend.enrichment_store()
        self.assertEqual(enrichment._collection_name, "test_decision_enrichments")

    def test_default_prefix_is_empty(self):
        backend = FirestoreBackend(client=self.client)
        enrichment = backend.enrichment_store()
        self.assertEqual(enrichment._collection_name, "decision_enrichments")


# ── EnrichmentStore ──────────────────────────────────────────────────


class TestFirestoreEnrichmentStore(unittest.TestCase):
    def setUp(self):
        self.client = MockFirestore()
        self.store = FirestoreEnrichmentStore(self.client, "enrichments")

    def tearDown(self):
        self.client.reset()

    def test_save_and_find_by_id(self):
        self.store.save("d1", {"decision_id": "d1", "topics": ["a"]})
        doc = self.store.find_by_id("d1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["topics"], ["a"])

    def test_find_by_id_missing_returns_none(self):
        self.assertIsNone(self.store.find_by_id("nonexistent"))

    def test_find_by_ids(self):
        self.store.save("d1", {"decision_id": "d1"})
        self.store.save("d2", {"decision_id": "d2"})
        result = self.store.find_by_ids(["d1", "d2", "d3"])
        self.assertEqual(len(result), 2)
        self.assertIn("d1", result)
        self.assertIn("d2", result)

    def test_find_by_ids_empty(self):
        self.assertEqual(self.store.find_by_ids([]), {})

    def test_upsert_merges(self):
        self.store.save("d1", {"decision_id": "d1", "topics": ["a"]})
        self.store.upsert("d1", {"topics": ["a", "b"], "entities": ["e1"]})
        doc = self.store.find_by_id("d1")
        self.assertEqual(doc["topics"], ["a", "b"])
        self.assertEqual(doc["entities"], ["e1"])

    def test_query_equality_filter(self):
        self.store.save("d1", {"decision_id": "d1", "gid": "g1"})
        self.store.save("d2", {"decision_id": "d2", "gid": "g2"})
        rows = self.store.query(filters=[("gid", "==", "g1")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gid"], "g1")

    def test_query_in_filter(self):
        self.store.save("d1", {"decision_id": "d1", "gid": "g1"})
        self.store.save("d2", {"decision_id": "d2", "gid": "g2"})
        self.store.save("d3", {"decision_id": "d3", "gid": "g3"})
        rows = self.store.query(filters=[("decision_id", "in", ["d1", "d3"])])
        self.assertEqual(len(rows), 2)

    def test_query_respects_limit(self):
        for i in range(5):
            self.store.save(f"d{i}", {"decision_id": f"d{i}", "gid": "g1"})
        rows = self.store.query(filters=[("gid", "==", "g1")], limit=2)
        self.assertEqual(len(rows), 2)


# ── ProjectionStore ──────────────────────────────────────────────────


class TestFirestoreProjectionStore(unittest.TestCase):
    def setUp(self):
        self.client = MockFirestore()
        self.store = FirestoreProjectionStore(self.client, "projections")

    def tearDown(self):
        self.client.reset()

    def test_save_creates_new_document(self):
        is_new = self.store.save(
            pid="p1",
            gid="g1",
            cid="c1",
            proj_type="decision_trace",
            projection={"subject_label": "test"},
            msg_ts=1000,
        )
        self.assertTrue(is_new)
        doc = self.store.find_by_id("p1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["gid"], "g1")
        self.assertTrue(doc["valid"])
        self.assertEqual(doc["projection"]["subject_label"], "test")

    def test_save_returns_false_if_exists(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=1000)
        is_new = self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=2000)
        self.assertFalse(is_new)

    def test_update_changes_projection(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={"v": 1}, msg_ts=1000)
        self.store.update(pid="p1", projection={"v": 2}, update_type="automated", msg_ts=2000)
        doc = self.store.find_by_id("p1")
        self.assertEqual(doc["projection"]["v"], 2)
        self.assertEqual(doc["updated_at"], 2000)

    def test_invalidate_sets_valid_false(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=1000)
        self.store.invalidate("p1")
        doc = self.store.find_by_id("p1")
        self.assertFalse(doc["valid"])

    def test_find_by_ids(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=1000)
        self.store.save(pid="p2", gid="g1", cid="c2", proj_type="dt", projection={}, msg_ts=1000)
        result = self.store.find_by_ids(["p1", "p2", "p3"])
        self.assertEqual(len(result), 2)

    def test_find_by_conv_ids_filters_proj_type_and_valid(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="decision_trace", projection={}, msg_ts=1000)
        self.store.save(pid="p2", gid="g1", cid="c1", proj_type="summary", projection={}, msg_ts=1000)
        self.store.save(pid="p3", gid="g1", cid="c2", proj_type="decision_trace", projection={}, msg_ts=1000)
        self.store.save(pid="p4", gid="g1", cid="c1", proj_type="decision_trace", projection={}, msg_ts=1000)
        self.store.invalidate("p4")

        rows = self.store.find_by_conv_ids(["c1"], "decision_trace")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pid"], "p1")

    def test_query_with_filters(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=1000)
        self.store.save(pid="p2", gid="g2", cid="c2", proj_type="dt", projection={}, msg_ts=1000)
        rows = self.store.query(filters=[("gid", "==", "g1")])
        self.assertEqual(len(rows), 1)


class TestFirestoreProjectionStoreFindByFilters(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = MockFirestore()
        self.store = FirestoreProjectionStore(self.client, "projections")

    def tearDown(self):
        self.client.reset()

    async def test_find_by_filters_returns_matching(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="decision_trace", projection={}, msg_ts=1000)
        self.store.save(pid="p2", gid="g2", cid="c2", proj_type="decision_trace", projection={}, msg_ts=1000)
        rows = await self.store.find_by_filters(gids=["g1"], proj_type="decision_trace")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gid"], "g1")

    async def test_find_by_filters_excludes_invalid(self):
        self.store.save(pid="p1", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=1000)
        self.store.invalidate("p1")
        rows = await self.store.find_by_filters(gids=["g1"], proj_type="dt")
        self.assertEqual(len(rows), 0)

    async def test_find_by_filters_respects_limit(self):
        for i in range(5):
            self.store.save(pid=f"p{i}", gid="g1", cid="c1", proj_type="dt", projection={}, msg_ts=1000 + i)
        rows = await self.store.find_by_filters(gids=["g1"], proj_type="dt", limit=2)
        self.assertEqual(len(rows), 2)


# ── ClusterStore ─────────────────────────────────────────────────────


class TestFirestoreClusterStore(unittest.TestCase):
    def setUp(self):
        self.client = MockFirestore()
        self.store = FirestoreClusterStore(self.client, "clusters")

    def tearDown(self):
        self.client.reset()

    def _make_cluster(self, cluster_id, last_updated_at=100.0):
        return {
            "cluster_id": cluster_id,
            "primary_subject": "test",
            "rolling_summary": "summary",
            "created_at": 50.0,
            "last_updated_at": last_updated_at,
            "decision_count": 0,
        }

    def test_create_and_find_by_id(self):
        data = self._make_cluster("c1")
        result = self.store.create(data)
        self.assertEqual(result, "c1")
        doc = self.store.find_by_id("c1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["primary_subject"], "test")

    def test_find_by_id_missing(self):
        self.assertIsNone(self.store.find_by_id("nonexistent"))

    def test_update_modifies_fields(self):
        self.store.create(self._make_cluster("c1"))
        self.store.update("c1", {"decision_count": 5, "last_updated_at": 200.0})
        doc = self.store.find_by_id("c1")
        self.assertEqual(doc["decision_count"], 5)
        self.assertEqual(doc["last_updated_at"], 200.0)

    def test_find_by_ids(self):
        self.store.create(self._make_cluster("c1"))
        self.store.create(self._make_cluster("c2"))
        result = self.store.find_by_ids(["c1", "c2", "c3"])
        self.assertEqual(len(result), 2)

    def test_find_by_ids_empty(self):
        self.assertEqual(self.store.find_by_ids([]), [])


# ── LinkStore ────────────────────────────────────────────────────────


class TestFirestoreLinkStore(unittest.TestCase):
    def setUp(self):
        self.client = MockFirestore()
        self.store = FirestoreLinkStore(self.client, "links")

    def tearDown(self):
        self.client.reset()

    def _make_link(self, decision_id, cluster_id="cl1", gid="g1", cid="c1"):
        return {
            "decision_id": decision_id,
            "cluster_id": cluster_id,
            "gid": gid,
            "cid": cid,
            "trace_id": "t1",
            "linked_at": 100.0,
        }

    def test_save_batch_and_find_by_decision_id(self):
        links = [self._make_link("d1"), self._make_link("d2")]
        count = self.store.save_batch(links)
        self.assertEqual(count, 2)
        doc = self.store.find_by_decision_id("d1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["cluster_id"], "cl1")

    def test_save_batch_empty(self):
        self.assertEqual(self.store.save_batch([]), 0)

    def test_find_by_decision_id_missing(self):
        self.assertIsNone(self.store.find_by_decision_id("nonexistent"))

    def test_find_by_cluster_id(self):
        self.store.save_batch(
            [
                self._make_link("d1", cluster_id="cl1"),
                self._make_link("d2", cluster_id="cl1"),
                self._make_link("d3", cluster_id="cl2"),
            ]
        )
        rows = self.store.find_by_cluster_id("cl1")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["cluster_id"] == "cl1" for r in rows))

    def test_find_by_decision_ids(self):
        self.store.save_batch([self._make_link("d1"), self._make_link("d2")])
        result = self.store.find_by_decision_ids(["d1", "d2", "d3"])
        self.assertEqual(len(result), 2)
        self.assertIn("d1", result)
        self.assertIn("d2", result)

    def test_find_by_decision_ids_empty(self):
        self.assertEqual(self.store.find_by_decision_ids([]), {})

    def test_find_cluster_ids_by_gids(self):
        self.store.save_batch(
            [
                self._make_link("d1", cluster_id="cl1", gid="g1"),
                self._make_link("d2", cluster_id="cl1", gid="g1"),
                self._make_link("d3", cluster_id="cl2", gid="g2"),
            ]
        )
        ids = self.store.find_cluster_ids_by_gids(["g1"])
        self.assertEqual(ids, ["cl1"])

    def test_find_cluster_ids_by_gids_empty(self):
        self.assertEqual(self.store.find_cluster_ids_by_gids([]), [])

    def test_find_cluster_id_for_decision(self):
        self.store.save_batch([self._make_link("d1", cluster_id="cl1")])
        self.assertEqual(self.store.find_cluster_id_for_decision("d1"), "cl1")

    def test_find_cluster_id_for_decision_missing(self):
        self.assertIsNone(self.store.find_cluster_id_for_decision("nonexistent"))


# ── Integration: full pipeline round-trip through FirestoreBackend ───


class TestFirestoreBackendRoundTrip(unittest.IsolatedAsyncioTestCase):
    """Exercises the backend through the pipeline's store interactions."""

    def setUp(self):
        self.client = MockFirestore()
        self.backend = FirestoreBackend(client=self.client)

    def tearDown(self):
        self.client.reset()

    async def test_enrichment_save_and_query_round_trip(self):
        store = self.backend.enrichment_store()
        store.save("d1", {"decision_id": "d1", "gid": "g1", "cid": "c1", "topics": ["pricing"]})
        store.save("d2", {"decision_id": "d2", "gid": "g1", "cid": "c2", "topics": ["logistics"]})

        rows = store.query(filters=[("gid", "==", "g1")])
        self.assertEqual(len(rows), 2)

        by_ids = await store.find_by_ids_async(["d1", "d2"])
        self.assertEqual(len(by_ids), 2)

    async def test_projection_lifecycle(self):
        store = self.backend.projection_store()

        store.save(pid="p1", gid="g1", cid="c1", proj_type="decision_trace", projection={"v": 1}, msg_ts=1000)
        store.save(pid="p2", gid="g1", cid="c1", proj_type="decision_trace", projection={"v": 2}, msg_ts=2000)

        projs = await store.find_by_filters(gids=["g1"], proj_type="decision_trace")
        self.assertEqual(len(projs), 2)

        store.invalidate("p1")
        projs = await store.find_by_filters(gids=["g1"], proj_type="decision_trace")
        self.assertEqual(len(projs), 1)
        self.assertEqual(projs[0]["pid"], "p2")

    def test_cluster_and_link_lifecycle(self):
        cluster_store = self.backend.cluster_store()
        link_store = self.backend.link_store()

        cluster_store.create(
            {
                "cluster_id": "cl1",
                "primary_subject": "Widget pricing",
                "rolling_summary": "Pricing discussion",
                "created_at": 100.0,
                "last_updated_at": 200.0,
                "decision_count": 2,
                "decision_ids": ["d1", "d2"],
            }
        )

        link_store.save_batch(
            [
                {
                    "decision_id": "d1",
                    "cluster_id": "cl1",
                    "gid": "g1",
                    "cid": "c1",
                    "trace_id": "t1",
                    "linked_at": 200.0,
                },
                {
                    "decision_id": "d2",
                    "cluster_id": "cl1",
                    "gid": "g1",
                    "cid": "c1",
                    "trace_id": "t2",
                    "linked_at": 200.0,
                },
            ]
        )

        cluster = cluster_store.find_by_id("cl1")
        self.assertEqual(cluster["decision_count"], 2)

        links = link_store.find_by_cluster_id("cl1")
        self.assertEqual(len(links), 2)

        self.assertEqual(link_store.find_cluster_id_for_decision("d1"), "cl1")
        self.assertIsNone(link_store.find_cluster_id_for_decision("d_unknown"))

        cluster_ids = link_store.find_cluster_ids_by_gids(["g1"])
        self.assertEqual(cluster_ids, ["cl1"])

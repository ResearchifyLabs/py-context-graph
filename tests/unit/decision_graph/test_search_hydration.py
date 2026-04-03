import unittest

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.services import DecisionGraphService


class TestHydrateDecisionsByIds(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.service = DecisionGraphService(backend=self.backend)

        self.backend.enrichment_store().save("d1", {
            "topics": ["pricing"],
            "entities": [{"name": "Acme"}],
            "action_params": {"amount": 100},
            "constraints_text": ["budget cap"],
            "key_facts": ["fact1"],
            "recorded_at": 1000.0,
            "trace_id": "t1",
            "gid": "g1",
            "cid": "c1",
        })
        self.backend.projection_store()._data["d1"] = {
            "pid": "d1",
            "gid": "g1",
            "cid": "c1",
            "proj_type": "decision_trace",
            "valid": True,
            "created_at": 900.0,
            "updated_at": 1000.0,
            "projection": {"subject_label": "Pricing discussion", "decision_id": "d1"},
        }

    def test_hydrate_returns_joined_data(self):
        results = self.service.hydrate_decisions_by_ids(["d1"])
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["pid"], "d1")
        self.assertEqual(r["gid"], "g1")
        self.assertTrue(r["has_enrichment"])
        self.assertEqual(r["enrichment"]["topics"], ["pricing"])
        self.assertEqual(r["projection"]["subject_label"], "Pricing discussion")

    def test_hydrate_missing_enrichment(self):
        self.backend.projection_store()._data["d2"] = {
            "pid": "d2",
            "gid": "g1",
            "proj_type": "decision_trace",
            "valid": True,
            "projection": {"subject_label": "No enrichment"},
        }
        results = self.service.hydrate_decisions_by_ids(["d2"])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["has_enrichment"])

    def test_hydrate_empty_ids(self):
        results = self.service.hydrate_decisions_by_ids([])
        self.assertEqual(results, [])

    def test_hydrate_preserves_order(self):
        self.backend.enrichment_store().save("d2", {"topics": ["sales"]})
        self.backend.projection_store()._data["d2"] = {
            "pid": "d2", "gid": "g1", "proj_type": "decision_trace",
            "valid": True, "projection": {"subject_label": "Sales"},
        }
        results = self.service.hydrate_decisions_by_ids(["d2", "d1"])
        self.assertEqual(results[0]["pid"], "d2")
        self.assertEqual(results[1]["pid"], "d1")

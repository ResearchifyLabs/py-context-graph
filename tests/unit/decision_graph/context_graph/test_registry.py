import unittest

from decision_graph.context_graph.registry import (
    ALL_REL_TYPES,
    REL_REGISTRY,
    get_weight,
)


class TestRelRegistry(unittest.TestCase):

    def test_all_entries_have_required_fields(self):
        for name, defn in REL_REGISTRY.items():
            self.assertIsInstance(defn.from_types, list, f"{name} missing from_types")
            self.assertIsInstance(defn.to_types, list, f"{name} missing to_types")
            self.assertIsInstance(defn.weight, float, f"{name} weight not float")
            self.assertGreater(defn.weight, 0, f"{name} weight must be positive")

    def test_all_rel_types_matches_registry_keys(self):
        self.assertEqual(set(ALL_REL_TYPES), set(REL_REGISTRY.keys()))

    def test_get_weight_known(self):
        self.assertAlmostEqual(get_weight("HAS_TOPIC"), 0.9)
        self.assertAlmostEqual(get_weight("INITIATED_BY"), 0.8)

    def test_get_weight_unknown(self):
        self.assertAlmostEqual(get_weight("DOES_NOT_EXIST"), 0.0)

    def test_has_topic_endpoints(self):
        defn = REL_REGISTRY["HAS_TOPIC"]
        self.assertIn("Decision", defn.from_types)
        self.assertIn("Topic", defn.to_types)

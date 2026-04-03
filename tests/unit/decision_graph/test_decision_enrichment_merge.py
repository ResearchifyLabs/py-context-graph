import unittest

from decision_graph.decision_enrichment import DecisionEnrichmentService


class TestDecisionEnrichmentDaoMerge(unittest.TestCase):
    def test_merge_unions_lists_and_merges_action_params(self):
        existing = {
            "decision_id": "d1",
            "trace_id": "t_old",
            "recorded_at": 10.0,
            "topics": ["a"],
            "constraints_text": ["c1"],
            "entities": [{"type": "Company", "name": "EKO"}],
            "key_facts": [{"k": "price", "v": "100"}],
            "action_params": {"k1": "v1"},
        }
        incoming = {
            "decision_id": "d1",
            "trace_id": "t_new",
            "recorded_at": 12.0,
            "topics": ["a", "b"],
            "constraints_text": ["c2"],
            "entities": [{"type": "Company", "name": "EKO"}, {"type": "Product", "name": "X"}],
            "key_facts": [{"k": "price", "v": "100"}, {"k": "qty", "v": "2"}],
            "action_params": {"k2": "v2", "k1": "v1b"},
        }

        merged = DecisionEnrichmentService.merge_enrichment(existing=existing, incoming=incoming)

        self.assertEqual(merged["recorded_at"], 12.0)
        self.assertEqual(merged["trace_id"], "t_new")
        self.assertEqual(set(merged["topics"]), {"a", "b"})
        self.assertEqual(set(merged["constraints_text"]), {"c1", "c2"})
        self.assertEqual(merged["action_params"]["k1"], "v1b")
        self.assertEqual(merged["action_params"]["k2"], "v2")
        self.assertEqual(
            {(e["type"], e["name"]) for e in merged["entities"]},
            {("Company", "EKO"), ("Product", "X")},
        )
        self.assertEqual(
            {(kv["k"], kv["v"]) for kv in merged["key_facts"]},
            {("price", "100"), ("qty", "2")},
        )


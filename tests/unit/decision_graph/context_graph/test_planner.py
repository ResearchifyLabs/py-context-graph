import unittest

from decision_graph.context_graph.planner import (
    SEED_BUNDLES,
    SEED_TO_TEMPLATE,
    PlanResult,
    plan,
)


class TestPlanner(unittest.TestCase):

    def test_find_related_topic_chat(self):
        result = plan(intent="FIND_RELATED", seed_type="Topic", response_mode="CHAT")
        self.assertIsInstance(result, PlanResult)
        self.assertEqual(result.template_id, "TOPIC_RELATED_DECISIONS_V1")
        self.assertEqual(result.k, 2)
        self.assertIn("HAS_TOPIC", result.rel_allowlist)

    def test_find_related_topic_ui(self):
        result = plan(intent="FIND_RELATED", seed_type="Topic", response_mode="UI")
        self.assertEqual(result.k, 2)

    def test_find_decisions_person(self):
        result = plan(intent="FIND_DECISIONS", seed_type="Person")
        self.assertEqual(result.template_id, "PERSON_RELATED_DECISIONS_V1")
        self.assertEqual(result.k, 2)
        self.assertEqual(result.node_type_allowlist, ["Decision"])

    def test_explain_connection(self):
        result = plan(intent="EXPLAIN_CONNECTION", seed_type="Topic")
        self.assertEqual(result.template_id, "EXPLAIN_CONNECTION_V1")
        self.assertEqual(result.k, 5)

    def test_summarize_scope(self):
        result = plan(intent="SUMMARIZE_SCOPE", seed_type="Entity")
        self.assertIn("Decision", result.node_type_allowlist)
        self.assertIn("Topic", result.node_type_allowlist)

    def test_explicit_allowlist_overrides_default(self):
        custom = ["HAS_TOPIC", "DEPENDS_ON"]
        result = plan(intent="FIND_RELATED", seed_type="Topic", rel_allowlist=custom)
        self.assertEqual(result.rel_allowlist, custom)

    def test_weights_populated(self):
        result = plan(intent="FIND_RELATED", seed_type="Topic")
        self.assertIn("HAS_TOPIC", result.rel_weights)
        self.assertGreater(result.rel_weights["HAS_TOPIC"], 0)

    def test_unknown_intent_defaults(self):
        result = plan(intent="UNKNOWN_INTENT", seed_type="Topic")
        self.assertEqual(result.k, 1)
        self.assertIsNotNone(result.template_id)

    def test_all_seed_types_have_bundles(self):
        for seed_type in ("Topic", "Person", "Entity", "Decision"):
            self.assertIn(seed_type, SEED_BUNDLES)
            self.assertIn(seed_type, SEED_TO_TEMPLATE)

import unittest
from unittest.mock import MagicMock, patch

from decision_graph.context_graph.service import ContextGraphService


class TestContextGraphServiceResolve(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.service = ContextGraphService(reader=self.reader)

    def test_resolve_delegates_to_reader(self):
        self.reader.resolve.return_value = {
            "query": "mitti cool",
            "candidates": [{"id": "t1", "type": "Topic", "name": "Mitti Cool"}],
            "debug": {"latency_ms": 10},
        }
        result = self.service.resolve("mitti cool", types=["Topic"], top_k=5)
        self.reader.resolve.assert_called_once_with("mitti cool", types=["Topic"], top_k=5)
        self.assertEqual(len(result["candidates"]), 1)

    def test_resolve_empty(self):
        self.reader.resolve.return_value = {
            "query": "xyz",
            "candidates": [],
            "debug": {"latency_ms": 5},
        }
        result = self.service.resolve("xyz")
        self.assertEqual(result["candidates"], [])


class TestContextGraphServiceQuery(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.service = ContextGraphService(reader=self.reader)

        self.reader.resolve.return_value = {
            "query": "mitti cool",
            "candidates": [{"id": "t1", "type": "Topic", "name": "Mitti Cool", "properties": {}}],
            "debug": {"latency_ms": 5},
        }
        self.reader.execute_template.return_value = {
            "records": [],
            "debug": {"template": "TOPIC_RELATED_DECISIONS_V1", "latency_ms": 20, "cypher": "", "params": {}},
        }

    def test_query_returns_expected_structure(self):
        result = self.service.query(
            intent="FIND_RELATED",
            seed={"type": "Topic", "text": "mitti cool", "id": None},
            response_mode="CHAT",
        )
        self.assertIn("normalized", result)
        self.assertIn("results", result)
        self.assertIn("debug", result)
        self.assertNotIn("graph", result)

    def test_query_ui_mode_includes_graph(self):
        result = self.service.query(
            intent="FIND_RELATED",
            seed={"type": "Topic", "text": "mitti cool", "id": None},
            response_mode="UI",
        )
        self.assertIn("graph", result)
        self.assertIn("nodes", result["graph"])
        self.assertIn("edges", result["graph"])
        self.assertIn("caps_applied", result["graph"])

    def test_query_with_explicit_seed_id_skips_resolve(self):
        result = self.service.query(
            intent="FIND_RELATED",
            seed={"type": "Topic", "text": "mitti cool", "id": "t1"},
        )
        self.reader.resolve.assert_not_called()
        self.reader.execute_template.assert_called_once()

    def test_query_without_seed_id_resolves(self):
        self.service.query(
            intent="FIND_RELATED",
            seed={"type": "Topic", "text": "mitti cool", "id": None},
        )
        self.reader.resolve.assert_called_once()


class TestContextGraphServiceOpen(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.service = ContextGraphService(reader=self.reader)
        self.reader.execute_template.return_value = {
            "records": [],
            "debug": {"template": "DECISION_EGO_V1", "latency_ms": 15, "cypher": "", "params": {}},
        }

    def test_open_returns_expected_structure(self):
        result = self.service.open(mode="EGO", node_id="dec_1", k=2)
        self.assertIn("summary", result)
        self.assertIn("graph", result)
        self.assertIn("debug", result)
        self.assertEqual(result["summary"]["node_id"], "dec_1")

    def test_open_calls_ego_template(self):
        self.service.open(mode="EGO", node_id="dec_1")
        call_args = self.reader.execute_template.call_args
        self.assertEqual(call_args[0][0], "DECISION_EGO_V1")
        self.assertEqual(call_args[0][1]["node_id"], "dec_1")

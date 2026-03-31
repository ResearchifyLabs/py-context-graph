import unittest
from unittest.mock import MagicMock

from decision_graph.context_graph.post_processing import (
    apply_caps,
    generate_evidence,
    normalize_graph,
    rank_decisions,
)


def _make_node(node_id, label, **props):
    node = MagicMock()
    node.labels = {label}
    all_props = {"name": node_id, **props}
    if label == "Decision":
        all_props["decision_id"] = node_id
    node.items.return_value = list(all_props.items())
    node.__iter__ = lambda s: iter(all_props)
    node.__getitem__ = lambda s, k: all_props[k]
    node.keys = lambda: all_props.keys()
    node.values = lambda: all_props.values()
    node.get = lambda k, d=None: all_props.get(k, d)
    return node


def _make_rel(rel_type, start_node, end_node, **props):
    rel = MagicMock()
    rel.type = rel_type
    rel.start_node = start_node
    rel.end_node = end_node
    rel.items.return_value = list(props.items())
    rel.__iter__ = lambda s: iter(props)
    rel.__getitem__ = lambda s, k: props[k]
    rel.keys = lambda: props.keys()
    rel.values = lambda: props.values()
    return rel


def _make_path(nodes, rels):
    path = MagicMock()
    path.nodes = nodes
    path.relationships = rels
    return path


class TestNormalizeGraph(unittest.TestCase):

    def test_empty_records(self):
        result = normalize_graph([])
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])

    def test_single_path(self):
        d = _make_node("dec_1", "Decision")
        t = _make_node("topic_1", "Topic")
        r = _make_rel("HAS_TOPIC", d, t)
        path = _make_path([d, t], [r])

        result = normalize_graph([{"primary_paths": path}])
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)

    def test_deduplication(self):
        d = _make_node("dec_1", "Decision")
        t = _make_node("topic_1", "Topic")
        r = _make_rel("HAS_TOPIC", d, t)
        path1 = _make_path([d, t], [r])
        path2 = _make_path([d, t], [r])

        result = normalize_graph([
            {"p1": path1},
            {"p2": path2},
        ])
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)


class TestApplyCaps(unittest.TestCase):

    def test_no_sampling_when_under_caps(self):
        nodes = [{"id": f"n{i}", "type": "Decision"} for i in range(5)]
        edges = [{"from": "n0", "to": f"n{i}", "type": "HAS_TOPIC"} for i in range(1, 5)]
        result = apply_caps(nodes, edges, root_ids=["n0"], result_ids=["n0"])
        self.assertFalse(result["caps_applied"]["sampled"])
        self.assertEqual(len(result["nodes"]), 5)

    def test_sampling_when_over_cap(self):
        nodes = [{"id": f"n{i}", "type": "Topic"} for i in range(200)]
        edges = []
        result = apply_caps(nodes, edges, root_ids=["n0"], result_ids=["n0"], max_nodes=50)
        self.assertTrue(result["caps_applied"]["sampled"])
        self.assertLessEqual(len(result["nodes"]), 50)

    def test_protected_nodes_always_kept(self):
        nodes = [{"id": f"n{i}", "type": "Topic"} for i in range(200)]
        nodes[0]["type"] = "Decision"
        edges = []
        result = apply_caps(nodes, edges, root_ids=["n0"], result_ids=["n0", "n1"], max_nodes=10)
        kept_ids = {n["id"] for n in result["nodes"]}
        self.assertIn("n0", kept_ids)
        self.assertIn("n1", kept_ids)


class TestRankDecisions(unittest.TestCase):

    def test_ranking_order(self):
        d1 = {"id": "d1", "type": "Decision", "properties": {"decision_id": "d1", "linked_at": "1000"}}
        d2 = {"id": "d2", "type": "Decision", "properties": {"decision_id": "d2", "linked_at": "9999999999"}}
        nodes = [d1, d2]
        edges = [
            {"from": "d2", "to": "t1", "type": "HAS_TOPIC"},
            {"from": "d2", "to": "t2", "type": "HAS_TOPIC"},
        ]
        ranked = rank_decisions([d1, d2], nodes, edges, {"HAS_TOPIC": 0.9})
        self.assertEqual(ranked[0]["id"], "d2")

    def test_empty_decisions(self):
        result = rank_decisions([], [], [], {})
        self.assertEqual(result, [])


class TestGenerateEvidence(unittest.TestCase):

    def test_generates_bullets(self):
        edges = [
            {"from": "d1", "to": "t1", "type": "HAS_TOPIC"},
            {"from": "d1", "to": "p1", "type": "INITIATED_BY"},
        ]
        nodes_map = {
            "t1": {"id": "t1", "type": "Topic", "properties": {"name": "Mitti Cool"}},
            "p1": {"id": "p1", "type": "Entity", "properties": {"name": "Rahul"}},
        }
        bullets = generate_evidence("d1", edges, nodes_map)
        self.assertEqual(len(bullets), 2)
        self.assertIn("HAS_TOPIC", bullets[0])
        self.assertIn("Rahul", bullets[1])

    def test_max_bullets_cap(self):
        edges = [
            {"from": "d1", "to": f"t{i}", "type": "HAS_TOPIC"} for i in range(10)
        ]
        nodes_map = {
            f"t{i}": {"id": f"t{i}", "type": "Topic", "properties": {"name": f"Topic {i}"}}
            for i in range(10)
        }
        bullets = generate_evidence("d1", edges, nodes_map, max_bullets=3)
        self.assertEqual(len(bullets), 3)

    def test_no_edges_for_decision(self):
        edges = [{"from": "d2", "to": "t1", "type": "HAS_TOPIC"}]
        bullets = generate_evidence("d1", edges, {})
        self.assertEqual(bullets, [])

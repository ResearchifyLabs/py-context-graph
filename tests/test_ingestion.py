import unittest
from unittest.mock import MagicMock

from decision_graph.ingestion import breakdown_hydrated_clusters, ingest_to_graph


class TestBreakdownHydratedClusters(unittest.TestCase):
    def _make_hydrated_cluster(self, **overrides):
        base = {
            "cluster": {
                "cluster_id": "cluster_1",
                "gid": "gid_1",
                "primary_subject": "App Submission",
                "created_at": 1770979181.0,
                "last_updated_at": 1770979181.0,
                "rolling_summary": "Summary text",
                "decision_count": 1,
                "decision_ids": ["dec_1"],
            },
            "decisions": [
                {
                    "decision_id": "dec_1",
                    "cid": "conv_1",
                    "gid": "gid_1",
                    "trace_id": "trace_1",
                    "projection": {
                        "initiator_name": "Akshay",
                        "initiator_role": "client",
                    },
                    "enrichment": {
                        "topics": ["App Submission", "iOS App Review"],
                        "entities": [{"type": "Person", "name": "Akshay"}],
                        "constraints_text": ["Review may take 48 hours"],
                        "key_facts": [{"k": "Submission Date", "v": "2026-02-13"}],
                        "action_params": {"Review Duration": "48 hours"},
                    },
                    "link_metadata": {"linked_from": "new_no_match"},
                    "linked_at": 1770979181.0,
                }
            ],
            "total_decisions": 1,
        }
        base.update(overrides)
        return base

    def test_single_cluster_produces_correct_cluster_array(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["clusters"]), 1)
        c = result["clusters"][0]
        self.assertEqual(c["cluster_id"], "cluster_1")
        self.assertEqual(c["gid"], "gid_1")
        self.assertEqual(c["primary_subject"], "App Submission")
        self.assertEqual(c["decision_count"], 1)

    def test_single_cluster_produces_correct_decision_array(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decisions"]), 1)
        d = result["decisions"][0]
        self.assertEqual(d["decision_id"], "dec_1")
        self.assertEqual(d["gid"], "gid_1")
        self.assertEqual(d["cid"], "conv_1")
        self.assertEqual(d["linked_from"], "new_no_match")

    def test_decision_cluster_mapping(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_cluster"]), 1)
        self.assertEqual(result["decision_cluster"][0]["cluster_id"], "cluster_1")
        self.assertEqual(result["decision_cluster"][0]["decision_id"], "dec_1")

    def test_topics_breakdown(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_topics"]), 2)
        topics = {t["topic"] for t in result["decision_topics"]}
        self.assertEqual(topics, {"App Submission", "iOS App Review"})
        for t in result["decision_topics"]:
            self.assertEqual(t["gid"], "gid_1")
            self.assertEqual(t["decision_id"], "dec_1")

    def test_entities_breakdown(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_entities"]), 1)
        self.assertEqual(result["decision_entities"][0]["type"], "Person")
        self.assertEqual(result["decision_entities"][0]["name"], "akshay")
        self.assertEqual(result["decision_entities"][0]["display_name"], "Akshay")
        self.assertEqual(result["decision_entities"][0]["gid"], "gid_1")

    def test_constraints_breakdown(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_constraints"]), 1)
        self.assertEqual(result["decision_constraints"][0]["text"], "Review may take 48 hours")
        self.assertEqual(result["decision_constraints"][0]["gid"], "gid_1")

    def test_facts_from_key_facts_and_action_params(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_facts"]), 2)
        fact_keys = {f["k"] for f in result["decision_facts"]}
        self.assertEqual(fact_keys, {"Submission Date", "Review Duration"})

    def test_initiator_breakdown(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_initiators"]), 1)
        ini = result["decision_initiators"][0]
        self.assertEqual(ini["initiator_name"], "akshay")
        self.assertEqual(ini["display_name"], "Akshay")
        self.assertEqual(ini["initiator_role"], "client")
        self.assertEqual(ini["gid"], "gid_1")
        self.assertEqual(ini["decision_id"], "dec_1")

    def test_empty_enrichment(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"] = {}
        hc["decisions"][0]["projection"] = {}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_topics"]), 0)
        self.assertEqual(len(result["decision_entities"]), 0)
        self.assertEqual(len(result["decision_constraints"]), 0)
        self.assertEqual(len(result["decision_facts"]), 0)
        self.assertEqual(len(result["decision_initiators"]), 0)

    def test_multiple_clusters(self):
        hc1 = self._make_hydrated_cluster()
        hc2 = {
            "cluster": {
                "cluster_id": "cluster_2",
                "gid": "gid_2",
                "primary_subject": "Budget Review",
                "created_at": 1770979182.0,
                "last_updated_at": 1770979182.0,
                "rolling_summary": "Budget summary",
                "decision_count": 1,
                "decision_ids": ["dec_2"],
            },
            "decisions": [
                {
                    "decision_id": "dec_2",
                    "cid": "conv_2",
                    "gid": "gid_2",
                    "trace_id": "trace_2",
                    "projection": {},
                    "enrichment": {
                        "topics": ["Budget"],
                        "entities": [],
                        "constraints_text": [],
                        "key_facts": [],
                        "action_params": {},
                    },
                    "link_metadata": {},
                    "linked_at": 1770979182.0,
                }
            ],
            "total_decisions": 1,
        }
        result = breakdown_hydrated_clusters([hc1, hc2])

        self.assertEqual(len(result["clusters"]), 2)
        self.assertEqual(len(result["decisions"]), 2)
        self.assertEqual(len(result["decision_cluster"]), 2)
        cluster_ids = {c["cluster_id"] for c in result["clusters"]}
        self.assertEqual(cluster_ids, {"cluster_1", "cluster_2"})

    def test_empty_input(self):
        result = breakdown_hydrated_clusters([])

        for key in result:
            self.assertEqual(len(result[key]), 0)

    def test_skips_empty_topic_strings(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["topics"] = [
            "Valid",
            "",
            None,
            "Also Valid",
        ]
        result = breakdown_hydrated_clusters([hc])

        topics = [t["topic"] for t in result["decision_topics"]]
        self.assertEqual(topics, ["Valid", "Also Valid"])

    def test_skips_entities_with_missing_fields(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["entities"] = [
            {"type": "Person", "name": "Akshay"},
            {"type": "", "name": "Bad"},
            {"type": "Org", "name": ""},
        ]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_entities"]), 1)
        self.assertEqual(result["decision_entities"][0]["name"], "akshay")
        self.assertEqual(result["decision_entities"][0]["display_name"], "Akshay")

    def test_skips_facts_with_empty_key(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = [
            {"k": "Good Key", "v": "value"},
            {"k": "", "v": "no key"},
        ]
        hc["decisions"][0]["enrichment"]["action_params"] = {}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_facts"]), 1)
        self.assertEqual(result["decision_facts"][0]["k"], "Good Key")

    def test_no_initiator_when_projection_missing_name(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["projection"] = {"initiator_role": "client"}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_initiators"]), 0)

    def test_decision_gid_falls_back_to_cluster_gid(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["gid"] = ""
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(result["decisions"][0]["gid"], "gid_1")
        self.assertEqual(result["decision_topics"][0]["gid"], "gid_1")

    def test_action_params_values_coerced_to_string(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = []
        hc["decisions"][0]["enrichment"]["action_params"] = {
            "count": 42,
            "flag": True,
            "empty": None,
        }
        result = breakdown_hydrated_clusters([hc])

        facts_by_key = {f["k"]: f["v"] for f in result["decision_facts"]}
        self.assertEqual(facts_by_key["count"], "42")
        self.assertEqual(facts_by_key["flag"], "True")
        self.assertEqual(facts_by_key["empty"], "")

    def test_skips_non_dict_entities(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["entities"] = [
            {"type": "Person", "name": "Valid"},
            "not a dict",
            123,
            None,
        ]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_entities"]), 1)
        self.assertEqual(result["decision_entities"][0]["name"], "valid")
        self.assertEqual(result["decision_entities"][0]["display_name"], "Valid")

    def test_skips_non_dict_key_facts(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = [
            {"k": "Valid", "v": "value"},
            "not a dict",
            [1, 2],
        ]
        hc["decisions"][0]["enrichment"]["action_params"] = {}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_facts"]), 1)
        self.assertEqual(result["decision_facts"][0]["k"], "Valid")

    def test_skips_action_params_when_not_dict(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = []
        hc["decisions"][0]["enrichment"]["action_params"] = ["a", "b"]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_facts"]), 0)

    def test_action_params_none_adds_no_facts(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = []
        hc["decisions"][0]["enrichment"]["action_params"] = None
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_facts"]), 0)

    def test_skips_empty_constraint_strings(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["constraints_text"] = [
            "Valid constraint",
            "",
            None,
            "Also valid",
        ]
        result = breakdown_hydrated_clusters([hc])

        texts = [c["text"] for c in result["decision_constraints"]]
        self.assertEqual(texts, ["Valid constraint", "Also valid"])

    def test_initiator_role_fallback_to_unknown(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["projection"] = {"initiator_name": "Jane"}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_initiators"]), 1)
        self.assertEqual(result["decision_initiators"][0]["initiator_role"], "unknown")

    def test_cluster_missing_cluster_key(self):
        hc = {"decisions": self._make_hydrated_cluster()["decisions"]}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(result["clusters"][0]["cluster_id"], "")
        self.assertEqual(result["clusters"][0]["gid"], "")
        self.assertEqual(result["decision_cluster"][0]["cluster_id"], "")

    def test_cluster_missing_decisions_key(self):
        hc = {"cluster": self._make_hydrated_cluster()["cluster"]}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(len(result["decisions"]), 0)
        self.assertEqual(len(result["decision_cluster"]), 0)

    def test_single_cluster_multiple_decisions(self):
        hc = self._make_hydrated_cluster()
        hc["cluster"]["decision_count"] = 2
        hc["cluster"]["decision_ids"] = ["dec_1", "dec_2"]
        hc["decisions"].append(
            {
                "decision_id": "dec_2",
                "cid": "conv_2",
                "gid": "gid_1",
                "trace_id": "trace_2",
                "projection": {"initiator_name": "Bob", "initiator_role": "admin"},
                "enrichment": {
                    "topics": ["Budget"],
                    "entities": [],
                    "constraints_text": [],
                    "key_facts": [],
                    "action_params": {},
                },
                "link_metadata": {},
                "linked_at": 1770979182.0,
            }
        )
        hc["total_decisions"] = 2
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(len(result["decisions"]), 2)
        self.assertEqual(len(result["decision_cluster"]), 2)
        self.assertEqual(len(result["decision_topics"]), 3)
        self.assertEqual(len(result["decision_initiators"]), 2)
        decision_ids = {d["decision_id"] for d in result["decisions"]}
        self.assertEqual(decision_ids, {"dec_1", "dec_2"})

    def test_link_metadata_none_uses_empty_linked_from(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["link_metadata"] = None
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(result["decisions"][0]["linked_from"], "")

    # ── Ordering ────────────────────────────────────────────────────

    def test_topics_preserve_input_order_after_filtering(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["topics"] = ["C", "", "A", None, "B"]
        result = breakdown_hydrated_clusters([hc])

        topics = [t["topic"] for t in result["decision_topics"]]
        self.assertEqual(topics, ["C", "A", "B"])

    def test_facts_order_key_facts_then_action_params(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = [
            {"k": "kf_1", "v": "a"},
            {"k": "kf_2", "v": "b"},
        ]
        hc["decisions"][0]["enrichment"]["action_params"] = {
            "ap_1": "x",
            "ap_2": "y",
        }
        result = breakdown_hydrated_clusters([hc])

        keys = [f["k"] for f in result["decision_facts"]]
        self.assertEqual(keys, ["kf_1", "kf_2", "ap_1", "ap_2"])

    def test_decision_cluster_order_matches_decision_order(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"].append(
            {
                "decision_id": "dec_2",
                "cid": "conv_2",
                "gid": "gid_1",
                "trace_id": "trace_2",
                "projection": {},
                "enrichment": {
                    "topics": [],
                    "entities": [],
                    "constraints_text": [],
                    "key_facts": [],
                    "action_params": {},
                },
                "link_metadata": {},
                "linked_at": 1770979182.0,
            }
        )
        result = breakdown_hydrated_clusters([hc])

        dc_ids = [dc["decision_id"] for dc in result["decision_cluster"]]
        dec_ids = [d["decision_id"] for d in result["decisions"]]
        self.assertEqual(dc_ids, dec_ids)

    def test_clusters_preserve_input_order(self):
        hc1 = self._make_hydrated_cluster()
        hc2_cluster = dict(hc1["cluster"], cluster_id="cluster_2", gid="gid_2")
        hc2_dec = dict(hc1["decisions"][0], decision_id="dec_2", gid="gid_2")
        hc2 = {"cluster": hc2_cluster, "decisions": [hc2_dec]}
        result = breakdown_hydrated_clusters([hc1, hc2])

        cluster_ids = [c["cluster_id"] for c in result["clusters"]]
        self.assertEqual(cluster_ids, ["cluster_1", "cluster_2"])

    # ── Duplication policy (duplicates are preserved) ───────────────

    def test_duplicate_topics_are_preserved(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["topics"] = ["Same", "Same", "Other"]
        result = breakdown_hydrated_clusters([hc])

        topics = [t["topic"] for t in result["decision_topics"]]
        self.assertEqual(topics, ["Same", "Same", "Other"])

    def test_entity_names_are_normalized(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["entities"] = [
            {"type": "Organization", "name": "researchify.io"},
            {"type": "Organization", "name": "Fly-Flat.com"},
        ]
        result = breakdown_hydrated_clusters([hc])

        names = [e["name"] for e in result["decision_entities"]]
        self.assertEqual(names, ["researchify", "flyflat"])
        display_names = [e["display_name"] for e in result["decision_entities"]]
        self.assertEqual(display_names, ["researchify.io", "Fly-Flat.com"])

    def test_duplicate_entities_are_preserved(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["entities"] = [
            {"type": "Person", "name": "Akshay"},
            {"type": "Person", "name": "Akshay"},
        ]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_entities"]), 2)

    def test_duplicate_constraints_are_preserved(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["constraints_text"] = [
            "48h limit",
            "48h limit",
        ]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_constraints"]), 2)

    def test_duplicate_facts_from_key_facts_are_preserved(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["key_facts"] = [
            {"k": "dup", "v": "val1"},
            {"k": "dup", "v": "val2"},
        ]
        hc["decisions"][0]["enrichment"]["action_params"] = {}
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_facts"]), 2)
        vals = [f["v"] for f in result["decision_facts"]]
        self.assertEqual(vals, ["val1", "val2"])

    # ── Multi-decision / multi-cluster edge cases ───────────────────

    def test_mixed_gid_fallback_across_decisions(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["gid"] = "own_gid"
        hc["decisions"].append(
            {
                "decision_id": "dec_2",
                "cid": "conv_2",
                "gid": "",
                "trace_id": "trace_2",
                "projection": {},
                "enrichment": {
                    "topics": ["T"],
                    "entities": [],
                    "constraints_text": [],
                    "key_facts": [],
                    "action_params": {},
                },
                "link_metadata": {},
                "linked_at": 1770979182.0,
            }
        )
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(result["decisions"][0]["gid"], "own_gid")
        self.assertEqual(result["decisions"][1]["gid"], "gid_1")
        topic_gids = [t["gid"] for t in result["decision_topics"]]
        self.assertIn("own_gid", topic_gids)
        self.assertIn("gid_1", topic_gids)

    def test_mapping_based_on_actual_decisions_not_decision_ids(self):
        hc = self._make_hydrated_cluster()
        hc["cluster"]["decision_ids"] = ["dec_1", "dec_phantom"]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_cluster"]), 1)
        self.assertEqual(result["decision_cluster"][0]["decision_id"], "dec_1")

    def test_decision_gid_differs_from_cluster_gid(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["gid"] = "decision_own_gid"
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(result["decisions"][0]["gid"], "decision_own_gid")
        self.assertEqual(result["clusters"][0]["gid"], "gid_1")
        self.assertEqual(result["decision_topics"][0]["gid"], "decision_own_gid")
        self.assertEqual(result["decision_entities"][0]["gid"], "decision_own_gid")
        self.assertEqual(result["decision_initiators"][0]["gid"], "decision_own_gid")

    # ── Contract: output shape (update when adding new relationships) ─

    EXPECTED_BREAKDOWN_KEYS = {
        "clusters",
        "decisions",
        "decision_cluster",
        "decision_topics",
        "decision_constraints",
        "decision_entities",
        "decision_facts",
        "decision_initiators",
    }

    def test_breakdown_returns_exact_keys(self):
        hc = self._make_hydrated_cluster()
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(set(result.keys()), self.EXPECTED_BREAKDOWN_KEYS)

    def test_breakdown_returns_exact_keys_on_empty_input(self):
        result = breakdown_hydrated_clusters([])

        self.assertEqual(set(result.keys()), self.EXPECTED_BREAKDOWN_KEYS)

    def test_unknown_enrichment_fields_do_not_affect_existing_output(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["enrichment"]["sentiments"] = [{"label": "positive", "score": 0.9}]
        hc["decisions"][0]["enrichment"]["categories"] = ["Tech"]
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(set(result.keys()), self.EXPECTED_BREAKDOWN_KEYS)
        self.assertEqual(len(result["decision_topics"]), 2)
        self.assertEqual(len(result["decision_entities"]), 1)
        self.assertEqual(len(result["decision_constraints"]), 1)
        self.assertEqual(len(result["decision_facts"]), 2)
        self.assertEqual(len(result["decision_initiators"]), 1)

    def test_unknown_projection_fields_do_not_affect_existing_output(self):
        hc = self._make_hydrated_cluster()
        hc["decisions"][0]["projection"]["sentiment"] = "positive"
        hc["decisions"][0]["projection"]["confidence"] = 0.95
        result = breakdown_hydrated_clusters([hc])

        self.assertEqual(len(result["decision_initiators"]), 1)
        self.assertEqual(result["decision_initiators"][0]["initiator_name"], "akshay")
        self.assertEqual(result["decision_initiators"][0]["display_name"], "Akshay")


class TestIngestToGraph(unittest.TestCase):
    def _make_hydrated_cluster(self):
        return {
            "cluster": {
                "cluster_id": "cluster_1",
                "gid": "gid_1",
                "primary_subject": "App Submission",
                "created_at": 1770979181.0,
                "last_updated_at": 1770979181.0,
                "rolling_summary": "Summary",
                "decision_count": 1,
                "decision_ids": ["dec_1"],
            },
            "decisions": [
                {
                    "decision_id": "dec_1",
                    "cid": "conv_1",
                    "gid": "gid_1",
                    "trace_id": "trace_1",
                    "projection": {"initiator_name": "Akshay", "initiator_role": "client"},
                    "enrichment": {
                        "topics": ["App Submission"],
                        "entities": [{"type": "Person", "name": "Akshay"}],
                        "constraints_text": ["Review 48h"],
                        "key_facts": [{"k": "Date", "v": "2026-02-13"}],
                        "action_params": {"Duration": "48h"},
                    },
                    "link_metadata": {"linked_from": "new_no_match"},
                    "linked_at": 1770979181.0,
                }
            ],
            "total_decisions": 1,
        }

    def test_calls_dao_methods_in_order_with_breakdown_data(self):
        dao = MagicMock()
        hc = self._make_hydrated_cluster()
        ingest_to_graph([hc], dao)

        dao.upsert_clusters.assert_called_once()
        dao.upsert_decisions.assert_called_once()
        dao.link_clusters_decisions.assert_called_once()
        dao.delete_enrichment_edges.assert_called_once_with(["dec_1"])
        dao.upsert_topics.assert_called_once()
        dao.upsert_constraints.assert_called_once()
        dao.upsert_entities.assert_called_once()
        dao.upsert_facts.assert_called_once()
        dao.upsert_initiators.assert_called_once()

        calls = [c[0] for c in dao.method_calls]
        upsert_clusters_idx = calls.index("upsert_clusters")
        delete_edges_idx = calls.index("delete_enrichment_edges")
        upsert_topics_idx = calls.index("upsert_topics")
        self.assertLess(upsert_clusters_idx, delete_edges_idx)
        self.assertLess(delete_edges_idx, upsert_topics_idx)

    def test_returns_counts_dict(self):
        dao = MagicMock()
        hc = self._make_hydrated_cluster()
        result = ingest_to_graph([hc], dao)

        self.assertEqual(result["clusters"], 1)
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(result["decision_cluster"], 1)
        self.assertEqual(result["decision_topics"], 1)
        self.assertEqual(result["decision_constraints"], 1)
        self.assertEqual(result["decision_entities"], 1)
        self.assertEqual(result["decision_facts"], 2)
        self.assertEqual(result["decision_initiators"], 1)

    def test_empty_input_calls_dao_with_empty_arrays(self):
        dao = MagicMock()
        result = ingest_to_graph([], dao)

        dao.upsert_clusters.assert_called_once_with([])
        dao.upsert_decisions.assert_called_once_with([])
        dao.link_clusters_decisions.assert_called_once_with([])
        dao.delete_enrichment_edges.assert_called_once_with([])
        dao.upsert_topics.assert_called_once_with([])
        dao.upsert_constraints.assert_called_once_with([])
        dao.upsert_entities.assert_called_once_with([])
        dao.upsert_facts.assert_called_once_with([])
        dao.upsert_initiators.assert_called_once_with([])

        for key in result:
            self.assertEqual(result[key], 0)

    # ── Contract: DAO methods and counts (update when adding new relationships)

    EXPECTED_DAO_METHODS = {
        "upsert_clusters",
        "upsert_decisions",
        "link_clusters_decisions",
        "delete_enrichment_edges",
        "upsert_topics",
        "upsert_constraints",
        "upsert_entities",
        "upsert_facts",
        "upsert_initiators",
    }

    def test_ingest_calls_exact_dao_methods(self):
        dao = MagicMock()
        hc = self._make_hydrated_cluster()
        ingest_to_graph([hc], dao)

        called_methods = {call[0] for call in dao.method_calls}
        self.assertEqual(called_methods, self.EXPECTED_DAO_METHODS)

    def test_counts_keys_match_breakdown_keys(self):
        dao = MagicMock()
        hc = self._make_hydrated_cluster()
        counts = ingest_to_graph([hc], dao)

        self.assertEqual(
            set(counts.keys()),
            TestBreakdownHydratedClusters.EXPECTED_BREAKDOWN_KEYS,
        )

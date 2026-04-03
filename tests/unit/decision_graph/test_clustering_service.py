import asyncio
import unittest
from unittest.mock import AsyncMock

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.clustering_service import DecisionClusterService
from decision_graph.core.domain import ClusterMetadataExtract


def _make_metadata(subject="Test Subject", summary="Test Summary"):
    return ClusterMetadataExtract(
        debug={"reasoning": "test"},
        primary_subject=subject,
        rolling_summary=summary,
    )


def _decision(did, cid, topics, entities=None, subject_label="", gid="g1", trace_id=""):
    ents = entities or []
    return {
        "decision_id": did,
        "cid": cid,
        "gid": gid,
        "trace_id": trace_id,
        "subject_label": subject_label,
        "topics": topics,
        "entities": ents,
        "person_entities": [],
        "non_person_entities": ents,
        "key_facts": [],
        "action_params": {},
    }


def _seed_cluster(backend, cluster_id, decision_ids, cids=None):
    backend.cluster_store().create({
        "cluster_id": cluster_id,
        "primary_subject": "s",
        "rolling_summary": "r",
        "created_at": 1.0,
        "last_updated_at": 1.0,
        "decision_count": len(decision_ids),
        "cids": cids or ["conv_existing"],
        "decision_ids": list(decision_ids),
    })


def _seed_link(backend, did, cluster_id, cid="conv_existing"):
    backend.link_store().save_batch([{
        "decision_id": did, "cluster_id": cluster_id, "cid": cid,
        "gid": "g1", "trace_id": "", "linked_at": 1.0,
    }])


AUTOJOIN_TOPICS = ["autojoin feature", "backend development", "meetings"]
AUTOJOIN_ENTITIES = ["chetto meetbot", "google calendar"]
AUTOJOIN_SUBJECT = "autojoin meeting feature"

LOGO_TOPICS = ["logo design", "branding", "meeto"]
LOGO_ENTITIES = ["meeto", "figma"]
LOGO_SUBJECT = "meeto logo redesign"


class TestEmptyInput(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.executor = AsyncMock()
        self.executor.execute_async = AsyncMock(return_value=_make_metadata())
        self.service = DecisionClusterService(backend=self.backend, executor=self.executor)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_empty_decisions_returns_zeros(self):
        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=[], candidate_decisions=[],
        ))
        self.assertEqual(result["linked"], 0)
        self.assertEqual(result["skipped"], 0)


class TestAlreadyLinkedSkip(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.executor = AsyncMock()
        self.executor.execute_async = AsyncMock(return_value=_make_metadata())
        self.service = DecisionClusterService(backend=self.backend, executor=self.executor)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_already_linked_decision_skipped(self):
        _seed_cluster(self.backend, "c1", ["d1"])
        _seed_link(self.backend, "d1", "c1")

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=[_decision("d1", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)],
            candidate_decisions=[],
        ))
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["linked"], 0)


class TestPerDecisionMatching(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.executor = AsyncMock()
        self.executor.execute_async = AsyncMock(return_value=_make_metadata())
        self.service = DecisionClusterService(backend=self.backend, executor=self.executor)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_decision_matches_candidate_in_existing_cluster(self):
        _seed_cluster(self.backend, "c_autojoin", ["cand_d1"], cids=["conv_cand"])
        _seed_link(self.backend, "cand_d1", "c_autojoin", cid="conv_cand")

        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_cand", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link = self.backend.link_store().find_by_decision_id("d1")
        self.assertIsNotNone(link)
        self.assertEqual(link["cluster_id"], "c_autojoin")
        self.assertEqual(result["linked"], 1)
        self.assertEqual(result["new_clusters"], 0)

    def test_different_decisions_go_to_different_clusters(self):
        _seed_cluster(self.backend, "c_autojoin", ["cand_auto"], cids=["conv_a"])
        _seed_link(self.backend, "cand_auto", "c_autojoin", cid="conv_a")
        _seed_cluster(self.backend, "c_logo", ["cand_logo"], cids=["conv_b"])
        _seed_link(self.backend, "cand_logo", "c_logo", cid="conv_b")

        new = [
            _decision("d_auto", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT),
            _decision("d_logo", "conv_new", LOGO_TOPICS, LOGO_ENTITIES, LOGO_SUBJECT),
        ]
        cands = [
            _decision("cand_auto", "conv_a", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT),
            _decision("cand_logo", "conv_b", LOGO_TOPICS, LOGO_ENTITIES, LOGO_SUBJECT),
        ]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_auto = self.backend.link_store().find_by_decision_id("d_auto")
        link_logo = self.backend.link_store().find_by_decision_id("d_logo")
        self.assertEqual(link_auto["cluster_id"], "c_autojoin")
        self.assertEqual(link_logo["cluster_id"], "c_logo")

    def test_same_cid_candidates_are_matchable(self):
        _seed_cluster(self.backend, "c1", ["cand_d1"], cids=["conv_new"])
        _seed_link(self.backend, "cand_d1", "c1", cid="conv_new")

        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link = self.backend.link_store().find_by_decision_id("d1")
        self.assertEqual(link["cluster_id"], "c1")
        self.assertEqual(result["linked"], 1)
        self.assertEqual(result["new_clusters"], 0)

    def test_no_match_creates_new_cluster(self):
        new = [_decision("d1", "conv_new", ["quantum computing", "physics"], ["cern", "lhc"], "quantum experiments")]
        cands = [_decision("cand_d1", "conv_other", LOGO_TOPICS, LOGO_ENTITIES, LOGO_SUBJECT)]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        self.assertEqual(result["new_clusters"], 1)
        link = self.backend.link_store().find_by_decision_id("d1")
        self.assertIsNotNone(link)
        self.assertEqual(link["match_metadata"]["linked_from"], "new_no_match")

    def test_no_candidates_creates_individual_clusters(self):
        new = [
            _decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT),
            _decision("d2", "conv_new", LOGO_TOPICS, LOGO_ENTITIES, LOGO_SUBJECT),
        ]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=[],
        ))

        self.assertEqual(result["new_clusters"], 2)
        link_d1 = self.backend.link_store().find_by_decision_id("d1")
        link_d2 = self.backend.link_store().find_by_decision_id("d2")
        self.assertIsNotNone(link_d1)
        self.assertIsNotNone(link_d2)
        self.assertNotEqual(link_d1["cluster_id"], link_d2["cluster_id"])

    def test_candidate_without_cluster_creates_pair_cluster(self):
        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_other", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_new = self.backend.link_store().find_by_decision_id("d1")
        link_cand = self.backend.link_store().find_by_decision_id("cand_d1")
        self.assertIsNotNone(link_new)
        self.assertIsNotNone(link_cand)
        self.assertEqual(link_new["cluster_id"], link_cand["cluster_id"])
        self.assertEqual(result["new_clusters"], 1)

    def test_matched_decision_updates_cluster_metadata(self):
        _seed_cluster(self.backend, "c1", ["cand_d1"], cids=["conv_cand"])
        _seed_link(self.backend, "cand_d1", "c1", cid="conv_cand")

        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_cand", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        cluster = self.backend.cluster_store().find_by_id("c1")
        self.assertIn("d1", cluster["decision_ids"])
        self.assertIn("conv_new", cluster["cids"])
        self.assertEqual(cluster["decision_count"], 2)

    def test_mix_of_matched_and_unmatched(self):
        _seed_cluster(self.backend, "c_autojoin", ["cand_auto"], cids=["conv_a"])
        _seed_link(self.backend, "cand_auto", "c_autojoin", cid="conv_a")

        new = [
            _decision("d_auto", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT),
            _decision("d_unrelated", "conv_new", ["quantum physics", "dark matter"], ["cern"], "quantum experiments"),
        ]
        cands = [_decision("cand_auto", "conv_a", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_auto = self.backend.link_store().find_by_decision_id("d_auto")
        link_unrelated = self.backend.link_store().find_by_decision_id("d_unrelated")
        self.assertEqual(link_auto["cluster_id"], "c_autojoin")
        self.assertNotEqual(link_unrelated["cluster_id"], "c_autojoin")

    def test_link_metadata_includes_scores(self):
        _seed_cluster(self.backend, "c1", ["cand_d1"], cids=["conv_cand"])
        _seed_link(self.backend, "cand_d1", "c1", cid="conv_cand")

        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_cand", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link = self.backend.link_store().find_by_decision_id("d1")
        self.assertEqual(link["match_metadata"]["linked_from"], "matched")
        self.assertIn("combined_score", link["match_metadata"])
        self.assertIn("topic_score", link["match_metadata"])

    def test_returns_all_affected_cluster_ids(self):
        _seed_cluster(self.backend, "c1", ["cand_d1"], cids=["conv_cand"])
        _seed_link(self.backend, "cand_d1", "c1", cid="conv_cand")

        new = [
            _decision("d_match", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT),
            _decision("d_new", "conv_new", ["quantum physics"], ["cern"], "quantum"),
        ]
        cands = [_decision("cand_d1", "conv_cand", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        self.assertIn("c1", result["cluster_ids"])
        self.assertTrue(len(result["cluster_ids"]) >= 2)


class TestLLMCalledOnlyForNewClusters(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.executor = AsyncMock()
        self.executor.execute_async = AsyncMock(return_value=_make_metadata())
        self.service = DecisionClusterService(backend=self.backend, executor=self.executor)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_matching_existing_cluster_no_llm(self):
        _seed_cluster(self.backend, "c1", ["cand_d1"], cids=["conv_cand"])
        _seed_link(self.backend, "cand_d1", "c1", cid="conv_cand")

        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_cand", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))
        self.executor.execute_async.assert_not_called()

    def test_new_cluster_calls_llm(self):
        new = [_decision("d1", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=[],
        ))
        self.executor.execute_async.assert_called_once()

    def test_skip_path_no_llm(self):
        _seed_cluster(self.backend, "c1", ["d1"])
        _seed_link(self.backend, "d1", "c1")

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=[_decision("d1", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)],
            candidate_decisions=[],
        ))
        self.executor.execute_async.assert_not_called()


class TestPreferExistingCluster(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.executor = AsyncMock()
        self.executor.execute_async = AsyncMock(return_value=_make_metadata())
        self.service = DecisionClusterService(backend=self.backend, executor=self.executor)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_second_decision_joins_cluster_created_by_first(self):
        """Two new decisions from conv1 each match a different candidate from conv2.
        The first creates a pair. The second should prefer the now-clustered candidate
        over creating a separate pair."""
        new = [
            _decision("d_a", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature setup"),
            _decision("d_c", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature configuration"),
        ]
        cands = [
            _decision("d_b", "conv2", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature setup and process"),
            _decision("d_d", "conv2", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature configuration steps"),
        ]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_a = self.backend.link_store().find_by_decision_id("d_a")
        link_c = self.backend.link_store().find_by_decision_id("d_c")
        link_b = self.backend.link_store().find_by_decision_id("d_b")
        self.assertIsNotNone(link_a)
        self.assertIsNotNone(link_c)
        self.assertIsNotNone(link_b)
        self.assertEqual(link_a["cluster_id"], link_c["cluster_id"],
                         "d_c should join d_a's cluster instead of creating a new pair")
        self.assertEqual(link_a["cluster_id"], link_b["cluster_id"])

    def test_prefers_clustered_even_when_unclustered_scores_higher(self):
        """When best match is unclustered but a clustered candidate is also above threshold,
        the clustered candidate should be preferred."""
        _seed_cluster(self.backend, "c_existing", ["cand_clustered"], cids=["conv_old"])
        _seed_link(self.backend, "cand_clustered", "c_existing", cid="conv_old")

        new = [_decision("d_new", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature")]
        cands = [
            _decision("cand_unclustered", "conv_other", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES,
                       "autojoin meeting feature setup and deployment"),
            _decision("cand_clustered", "conv_old", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES,
                       "autojoin meeting feature process"),
        ]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link = self.backend.link_store().find_by_decision_id("d_new")
        self.assertEqual(link["cluster_id"], "c_existing",
                         "Should join existing cluster instead of creating new pair")

    def test_falls_through_to_new_pair_when_no_clustered_candidate(self):
        """When no candidate is clustered, a new pair cluster should be created as usual."""
        new = [_decision("d_new", "conv_new", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]
        cands = [_decision("cand_d1", "conv_other", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)]

        result = self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_new = self.backend.link_store().find_by_decision_id("d_new")
        link_cand = self.backend.link_store().find_by_decision_id("cand_d1")
        self.assertEqual(link_new["cluster_id"], link_cand["cluster_id"])
        self.assertEqual(result["new_clusters"], 1)


class TestSameCidClustering(unittest.TestCase):

    def setUp(self):
        self.backend = InMemoryBackend()
        self.executor = AsyncMock()
        self.executor.execute_async = AsyncMock(return_value=_make_metadata())
        self.service = DecisionClusterService(backend=self.backend, executor=self.executor)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_same_cid_decisions_can_form_pair(self):
        """When same-CID decisions are included as candidates, similar ones should cluster."""
        d1 = _decision("d1", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature setup")
        d2 = _decision("d2", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature process")
        new = [d1, d2]
        cands = [d1, d2]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_d1 = self.backend.link_store().find_by_decision_id("d1")
        link_d2 = self.backend.link_store().find_by_decision_id("d2")
        self.assertIsNotNone(link_d1)
        self.assertIsNotNone(link_d2)
        self.assertEqual(link_d1["cluster_id"], link_d2["cluster_id"],
                         "Same-CID decisions with matching content should cluster together")

    def test_same_cid_unrelated_decisions_stay_separate(self):
        """Same-CID decisions with different topics should NOT cluster."""
        d1 = _decision("d1", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, AUTOJOIN_SUBJECT)
        d2 = _decision("d2", "conv1", LOGO_TOPICS, LOGO_ENTITIES, LOGO_SUBJECT)
        new = [d1, d2]
        cands = [d1, d2]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_d1 = self.backend.link_store().find_by_decision_id("d1")
        link_d2 = self.backend.link_store().find_by_decision_id("d2")
        self.assertNotEqual(link_d1["cluster_id"], link_d2["cluster_id"],
                            "Unrelated same-CID decisions should stay in separate clusters")

    def test_same_cid_bridges_to_cross_cid_cluster(self):
        """Same-CID decision clusters with a cross-CID candidate via prefer-clustered."""
        _seed_cluster(self.backend, "c_cross", ["cand_cross"], cids=["conv_other"])
        _seed_link(self.backend, "cand_cross", "c_cross", cid="conv_other")

        d1 = _decision("d1", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature")
        d2 = _decision("d2", "conv1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting setup")
        cand_cross = _decision("cand_cross", "conv_other", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES,
                                "autojoin meeting feature process")

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=[d1, d2], candidate_decisions=[d1, d2, cand_cross],
        ))

        link_d1 = self.backend.link_store().find_by_decision_id("d1")
        link_d2 = self.backend.link_store().find_by_decision_id("d2")
        self.assertEqual(link_d1["cluster_id"], "c_cross",
                         "d1 should join cross-CID cluster via cand_cross")
        self.assertEqual(link_d2["cluster_id"], "c_cross",
                         "d2 should join same cluster as d1")

    def test_abcd_two_cids_four_decisions_single_call(self):
        """The A,B,C,D scenario: 2 new decisions from C1, 2 candidates from C2.
        All share topics/entities. Should produce at most 2 clusters, not 3+."""
        new = [
            _decision("A", "C1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature setup"),
            _decision("C", "C1", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature configuration"),
        ]
        cands = [
            _decision("B", "C2", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature setup and testing"),
            _decision("D", "C2", AUTOJOIN_TOPICS, AUTOJOIN_ENTITIES, "autojoin meeting feature configuration review"),
        ]

        self._run(self.service.link_decisions_to_cluster(
            new_decisions=new, candidate_decisions=cands,
        ))

        link_a = self.backend.link_store().find_by_decision_id("A")
        link_c = self.backend.link_store().find_by_decision_id("C")
        link_b = self.backend.link_store().find_by_decision_id("B")
        self.assertIsNotNone(link_a)
        self.assertIsNotNone(link_c)
        self.assertIsNotNone(link_b)
        self.assertEqual(link_a["cluster_id"], link_c["cluster_id"],
                         "A and C should end up in the same cluster via prefer-clustered")
        self.assertEqual(link_a["cluster_id"], link_b["cluster_id"],
                         "B should be in the same cluster as A")

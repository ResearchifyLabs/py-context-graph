import unittest

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.context_retrieval import (
    DecisionContextRetriever as ClusteringContextRetriever,
)
from decision_graph.core.matching import (
    build_conversation_fingerprint,
    calculate_jaccard,
    calculate_max_subject_match,
    calculate_sequence_match,
    canonicalize_subject_label,
    dedupe_list,
    idf_weighted_jaccard,
    jaccard_similarity,
    merge_decision_trace_history,
    normalize_component,
    normalize_entity,
    precompute_decision,
    score_decision_pair,
    sequence_similarity,
    tokenize_specifics,
)
from decision_graph.decision_enrichment import (
    DecisionContextRetriever,
    DecisionEnrichmentService,
)
from decision_graph.retrieval import DecisionEnrichmentRetrieval


class TestCalculateJaccard(unittest.TestCase):
    def test_identical_sets(self):
        score = calculate_jaccard(["a", "b", "c"], ["a", "b", "c"])
        self.assertAlmostEqual(score, 1.0)

    def test_disjoint_sets(self):
        score = calculate_jaccard(["a", "b"], ["c", "d"])
        self.assertAlmostEqual(score, 0.0)

    def test_partial_overlap(self):
        score = calculate_jaccard(["a", "b", "c"], ["b", "c", "d"])
        self.assertAlmostEqual(score, 2 / 4)

    def test_both_empty(self):
        score = calculate_jaccard([], [])
        self.assertAlmostEqual(score, 0.0)

    def test_one_empty(self):
        score = calculate_jaccard(["a"], [])
        self.assertAlmostEqual(score, 0.0)

    def test_case_insensitive(self):
        score = calculate_jaccard(["Apple", "BANANA"], ["apple", "banana"])
        self.assertAlmostEqual(score, 1.0)

    def test_with_custom_normalize_fn(self):
        score = calculate_jaccard(
            ["Fly-Flat", "fly-flat.com"],
            ["flyflat", "FlyFlat"],
            normalize_fn=normalize_entity,
        )
        self.assertAlmostEqual(score, 1.0)

    def test_duplicates_in_input_do_not_inflate(self):
        score = calculate_jaccard(["a", "a", "a"], ["a"])
        self.assertAlmostEqual(score, 1.0)

    def test_backward_compat_via_class(self):
        score = DecisionContextRetriever._calculate_jaccard(["a", "b"], ["b", "c"])
        self.assertAlmostEqual(score, 1 / 3)


class TestCalculateSequenceMatch(unittest.TestCase):
    def test_identical_strings(self):
        score = calculate_sequence_match("hello world", "hello world")
        self.assertAlmostEqual(score, 1.0)

    def test_completely_different(self):
        score = calculate_sequence_match("abc", "xyz")
        self.assertAlmostEqual(score, 0.0)

    def test_similar_strings(self):
        score = calculate_sequence_match("Client Analytics test failure", "Client Analytics test success")
        self.assertGreater(score, 0.7)

    def test_both_empty(self):
        score = calculate_sequence_match("", "")
        self.assertAlmostEqual(score, 0.0)

    def test_one_empty(self):
        score = calculate_sequence_match("hello", "")
        self.assertAlmostEqual(score, 0.0)

    def test_case_insensitive(self):
        score = calculate_sequence_match("Hello World", "hello world")
        self.assertAlmostEqual(score, 1.0)

    def test_backward_compat_via_class(self):
        score = DecisionContextRetriever._calculate_sequence_match("abc", "abc")
        self.assertAlmostEqual(score, 1.0)


class TestNormalizeEntity(unittest.TestCase):
    def test_strips_common_tlds(self):
        self.assertEqual(normalize_entity("fly-flat.com"), "flyflat")

    def test_strips_io_tld(self):
        self.assertEqual(normalize_entity("researchify.io"), "researchify")

    def test_removes_hyphens_underscores_spaces(self):
        self.assertEqual(normalize_entity("Fly-Flat"), "flyflat")
        self.assertEqual(normalize_entity("some_thing"), "something")
        self.assertEqual(normalize_entity("some thing"), "something")

    def test_lowercases(self):
        self.assertEqual(normalize_entity("FlyFlat"), "flyflat")

    def test_empty_and_none(self):
        self.assertEqual(normalize_entity(""), "")
        self.assertEqual(normalize_entity(None), "")

    def test_preserves_non_tld_dots(self):
        result = normalize_entity("v1.2.3")
        self.assertEqual(result, "v1.2.3")

    def test_backward_compat_via_class(self):
        self.assertEqual(DecisionContextRetriever._normalize_entity("fly-flat.com"), "flyflat")


class TestBuildConversationFingerprint(unittest.TestCase):
    def test_empty_decisions(self):
        result = build_conversation_fingerprint([])
        self.assertEqual(result, {})

    def test_single_decision(self):
        decisions = [
            {
                "gid": "g1",
                "cid": "c1",
                "decision_id": "d1",
                "trace_id": "t1",
                "subject_label": "Widget pricing",
                "topics": ["pricing", "negotiation"],
                "entities": ["Acme Corp"],
            }
        ]
        fp = build_conversation_fingerprint(decisions)
        self.assertEqual(fp["gid"], "g1")
        self.assertEqual(fp["cid"], "c1")
        self.assertEqual(fp["decision_count"], 1)
        self.assertIn("pricing", fp["topics"])
        self.assertIn("Acme Corp", fp["entities"])
        self.assertIn("Widget pricing", fp["subject_labels"])
        self.assertEqual(fp["decision_ids"], ["d1"])

    def test_multiple_decisions_deduplicates(self):
        decisions = [
            {
                "gid": "g1",
                "cid": "c1",
                "decision_id": "d1",
                "trace_id": "t1",
                "subject_label": "Widget pricing",
                "topics": ["pricing", "Negotiation"],
                "entities": ["Acme Corp"],
            },
            {
                "gid": "g1",
                "cid": "c1",
                "decision_id": "d2",
                "trace_id": "t2",
                "subject_label": "Widget delivery",
                "topics": ["pricing", "logistics"],
                "entities": ["Acme Corp", "DHL"],
            },
        ]
        fp = build_conversation_fingerprint(decisions)
        self.assertEqual(fp["decision_count"], 2)
        self.assertEqual(len(fp["decision_ids"]), 2)
        self.assertIn("pricing", fp["topics"])
        self.assertIn("negotiation", fp["topics"])
        self.assertIn("logistics", fp["topics"])
        self.assertEqual(len(set(fp["entities"])), 2)

    def test_missing_optional_fields(self):
        decisions = [{"gid": "g1", "cid": "c1"}]
        fp = build_conversation_fingerprint(decisions)
        self.assertEqual(fp["topics"], [])
        self.assertEqual(fp["entities"], [])
        self.assertEqual(fp["subject_labels"], [])
        self.assertEqual(fp["decision_ids"], [])

    def test_backward_compat_via_class(self):
        result = DecisionContextRetriever._build_conversation_fingerprint([])
        self.assertEqual(result, {})


class TestCalculateMaxSubjectMatch(unittest.TestCase):
    def test_identical_labels(self):
        score = calculate_max_subject_match(["hello"], ["hello"])
        self.assertAlmostEqual(score, 1.0)

    def test_best_match_is_returned(self):
        score = calculate_max_subject_match(
            ["Widget pricing discussion"],
            ["Widget pricing discussion", "something completely different"],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_empty_lists(self):
        self.assertAlmostEqual(calculate_max_subject_match([], ["a"]), 0.0)
        self.assertAlmostEqual(calculate_max_subject_match(["a"], []), 0.0)
        self.assertAlmostEqual(calculate_max_subject_match([], []), 0.0)


class TestDedupeList(unittest.TestCase):
    def test_deduplicates_strings(self):
        result = dedupe_list(["a", "b", "a", "c", "b"])
        self.assertEqual(result, ["a", "b", "c"])

    def test_deduplicates_dicts(self):
        result = dedupe_list([{"k": "v"}, {"k": "v"}, {"k": "v2"}])
        self.assertEqual(len(result), 2)

    def test_empty_and_none(self):
        self.assertEqual(dedupe_list([]), [])
        self.assertEqual(dedupe_list(None), [])

    def test_preserves_order(self):
        result = dedupe_list(["c", "a", "b", "a"])
        self.assertEqual(result, ["c", "a", "b"])

    def test_backward_compat_via_class(self):
        result = DecisionEnrichmentService._dedupe_list(["a", "a", "b"])
        self.assertEqual(result, ["a", "b"])


class TestNormalizeComponent(unittest.TestCase):
    def test_lowercases_and_strips(self):
        self.assertEqual(normalize_component("  Hello World  "), "hello world")

    def test_replaces_pipe_with_space(self):
        self.assertEqual(normalize_component("a|b|c"), "a b c")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_component("a   b   c"), "a b c")

    def test_none_becomes_empty(self):
        self.assertEqual(normalize_component(None), "")

    def test_backward_compat_via_class(self):
        self.assertEqual(DecisionEnrichmentService._normalize_component("  A  B  "), "a b")


class TestRetrievalSimilarityFunctions(unittest.TestCase):
    def test_sequence_similarity_identical(self):
        score = sequence_similarity("hello", "hello")
        self.assertAlmostEqual(score, 1.0)

    def test_sequence_similarity_different(self):
        score = sequence_similarity("abc", "xyz")
        self.assertAlmostEqual(score, 0.0)

    def test_jaccard_similarity_identical_tokens(self):
        score = jaccard_similarity("a b c", "a b c")
        self.assertAlmostEqual(score, 1.0)

    def test_jaccard_similarity_disjoint_tokens(self):
        score = jaccard_similarity("a b", "c d")
        self.assertAlmostEqual(score, 0.0)

    def test_jaccard_similarity_both_empty(self):
        score = jaccard_similarity("", "")
        self.assertAlmostEqual(score, 1.0)

    def test_jaccard_similarity_one_empty(self):
        score = jaccard_similarity("a", "")
        self.assertAlmostEqual(score, 0.0)

    def test_jaccard_similarity_partial(self):
        score = jaccard_similarity("a b c", "b c d")
        self.assertAlmostEqual(score, 2 / 4)

    def test_backward_compat_via_retrieval_class(self):
        self.assertAlmostEqual(DecisionEnrichmentRetrieval._sequence_similarity("a", "a"), 1.0)
        self.assertAlmostEqual(DecisionEnrichmentRetrieval._jaccard_similarity("a b", "a b"), 1.0)


class TestCanonicalizeSubjectLabel(unittest.TestCase):
    def test_empty_and_none(self):
        self.assertEqual(canonicalize_subject_label(None), "")
        self.assertEqual(canonicalize_subject_label(""), "")
        self.assertEqual(canonicalize_subject_label("  "), "")

    def test_lowercases(self):
        result = canonicalize_subject_label("BugReport Test Failure")
        self.assertNotIn("B", result)
        self.assertNotIn("R", result)

    def test_removes_stop_words(self):
        result = canonicalize_subject_label("the bug is in the module")
        self.assertNotIn("the", result.split(" "))
        self.assertNotIn("is", result.split(" "))
        self.assertNotIn("in", result.split(" "))

    def test_normalizes_urls(self):
        result = canonicalize_subject_label("test failures in PR https://github.com/org/repo/pull/3226")
        self.assertIn("<url>", result)

    def test_normalizes_pr_numbers(self):
        result = canonicalize_subject_label("fix PR #3226 regression")
        self.assertIn("pr3226", result)

    def test_drops_filler_phrases(self):
        result = canonicalize_subject_label("Fix payment next day")
        self.assertNotIn("next day", result)

    def test_removes_noise_tokens(self):
        result = canonicalize_subject_label("update dashboard module endpoint")
        tokens = result.split(" ")
        self.assertNotIn("dashboard", tokens)
        self.assertNotIn("module", tokens)
        self.assertNotIn("endpoint", tokens)

    def test_preserves_dots_for_test_names(self):
        result = canonicalize_subject_label("TestClientAnalytics.test_action_type failure")
        self.assertIn("testclientanalytics.test_action_type", result)

    def test_backward_compat_via_retrieval_class(self):
        self.assertEqual(
            DecisionEnrichmentRetrieval.canonicalize_subject_label("the bug"),
            canonicalize_subject_label("the bug"),
        )


class TestMergeDecisionTraceHistory(unittest.TestCase):
    def test_new_trace_creates_initial_history(self):
        event = {"updated_at": 100.0, "action_type": "approve", "status_state": "open"}
        result = merge_decision_trace_history(existing_proj={}, projection={}, event=event)
        self.assertEqual(result["status_history"], [event])
        self.assertEqual(result["action_type_history"], ["approve"])

    def test_appends_new_event_to_history(self):
        event1 = {"updated_at": 100.0, "action_type": "approve", "status_state": "open"}
        event2 = {"updated_at": 200.0, "action_type": "reject", "status_state": "closed"}
        existing = {"status_history": [event1], "action_type_history": ["approve"]}
        result = merge_decision_trace_history(existing_proj=existing, projection={}, event=event2)
        self.assertEqual(len(result["status_history"]), 2)
        self.assertEqual(result["status_history"][-1], event2)
        self.assertEqual(result["action_type_history"], ["approve", "reject"])

    def test_duplicate_event_not_appended(self):
        event = {"updated_at": 100.0, "action_type": "approve", "status_state": "open"}
        existing = {"status_history": [event], "action_type_history": ["approve"]}
        result = merge_decision_trace_history(existing_proj=existing, projection={}, event=event)
        self.assertEqual(len(result["status_history"]), 1)

    def test_duplicate_action_type_not_appended(self):
        event1 = {"updated_at": 100.0, "action_type": "approve", "status_state": "open"}
        event2 = {"updated_at": 200.0, "action_type": "approve", "status_state": "closed"}
        existing = {"status_history": [event1], "action_type_history": ["approve"]}
        result = merge_decision_trace_history(existing_proj=existing, projection={}, event=event2)
        self.assertEqual(result["action_type_history"], ["approve"])

    def test_status_history_capped_at_50(self):
        events = [{"updated_at": float(i), "status_state": f"s{i}"} for i in range(55)]
        existing = {"status_history": events}
        new_event = {"updated_at": 999.0, "status_state": "final"}
        result = merge_decision_trace_history(existing_proj=existing, projection={}, event=new_event)
        self.assertEqual(len(result["status_history"]), 50)
        self.assertEqual(result["status_history"][-1], new_event)

    def test_action_type_history_capped_at_20(self):
        types = [f"type_{i}" for i in range(25)]
        existing = {"action_type_history": types}
        event = {"action_type": "brand_new"}
        result = merge_decision_trace_history(existing_proj=existing, projection={}, event=event)
        self.assertEqual(len(result["action_type_history"]), 20)
        self.assertEqual(result["action_type_history"][-1], "brand_new")

    def test_none_action_type_not_added(self):
        event = {"updated_at": 100.0, "action_type": None, "status_state": "open"}
        result = merge_decision_trace_history(existing_proj={}, projection={}, event=event)
        self.assertEqual(result["action_type_history"], [])


class TestTokenizeSpecifics(unittest.TestCase):
    def test_basic_tokenization(self):
        tokens = tokenize_specifics(["PR #3226 merged", "v1.2.3 released"])
        self.assertIn("pr", tokens)
        self.assertIn("3226", tokens)
        self.assertIn("merged", tokens)
        self.assertIn("v1.2.3", tokens)
        self.assertIn("released", tokens)

    def test_filters_stop_words(self):
        tokens = tokenize_specifics(["the fix is in the module"])
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("in", tokens)
        self.assertIn("fix", tokens)
        self.assertIn("module", tokens)

    def test_filters_short_tokens(self):
        tokens = tokenize_specifics(["a b cc dd"])
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
        self.assertIn("cc", tokens)
        self.assertIn("dd", tokens)

    def test_empty_input(self):
        self.assertEqual(tokenize_specifics([]), set())

    def test_non_string_ignored(self):
        tokens = tokenize_specifics([123, None, "hello"])
        self.assertEqual(tokens, {"hello"})

    def test_filters_generic_verbs(self):
        tokens = tokenize_specifics(["confirmed deployment", "shared document"])
        self.assertNotIn("confirmed", tokens)
        self.assertNotIn("shared", tokens)
        self.assertIn("deployment", tokens)
        self.assertIn("document", tokens)


class TestPrecomputeDecision(unittest.TestCase):
    def test_creates_all_precomputed_sets(self):
        d = precompute_decision(
            {
                "topics": ["Code Review", "testing"],
                "person_entities": ["Alice"],
                "non_person_entities": ["ResearchifyLabs"],
                "subject_label": "PR Review",
                "key_facts": ["PR #3226 approved"],
                "action_params": {"target": "main branch"},
            }
        )
        self.assertEqual(d["_topics_set"], {"code", "review", "testing"})
        self.assertIn("alice", d["_person_set"])
        self.assertIn("researchifylabs", d["_non_person_set"])
        self.assertEqual(d["_subject_lower"], "pr review")
        self.assertIsInstance(d["_specifics_set"], set)

    def test_specifics_deduplicated_against_topics(self):
        d = precompute_decision(
            {
                "topics": ["deployment"],
                "person_entities": [],
                "non_person_entities": [],
                "key_facts": ["deployment to staging"],
                "action_params": {},
            }
        )
        self.assertNotIn("deployment", d["_specifics_set"])
        self.assertIn("staging", d["_specifics_set"])

    def test_missing_fields_produce_empty_sets(self):
        d = precompute_decision({})
        self.assertEqual(d["_topics_set"], set())
        self.assertEqual(d["_person_set"], set())
        self.assertEqual(d["_non_person_set"], set())
        self.assertEqual(d["_specifics_set"], set())
        self.assertEqual(d["_subject_lower"], "")


class TestIdfWeightedJaccard(unittest.TestCase):
    def test_plain_jaccard(self):
        score = idf_weighted_jaccard({"a", "b", "c"}, {"b", "c", "d"})
        self.assertAlmostEqual(score, 2 / 4)

    def test_both_empty(self):
        self.assertAlmostEqual(idf_weighted_jaccard(set(), set()), 0.0)

    def test_identical_sets(self):
        self.assertAlmostEqual(idf_weighted_jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets(self):
        self.assertAlmostEqual(idf_weighted_jaccard({"a"}, {"b"}), 0.0)


class TestScoreDecisionPair(unittest.TestCase):
    def _make_decision(self, **kwargs):
        base = {
            "topics": [],
            "person_entities": [],
            "non_person_entities": [],
            "subject_label": "",
            "key_facts": [],
            "action_params": {},
        }
        base.update(kwargs)
        return precompute_decision(base)

    def test_identical_decisions_score_high(self):
        d = self._make_decision(
            topics=["pricing"],
            person_entities=["Alice"],
            non_person_entities=["Acme"],
            subject_label="pricing update",
            key_facts=["Q1 contract"],
        )
        scores = score_decision_pair(d, d)
        self.assertGreater(scores["combined_score"], 0.8)

    def test_completely_different_scores_low(self):
        a = self._make_decision(
            topics=["hiring"],
            person_entities=["Bob"],
            non_person_entities=["Google"],
            subject_label="new hire",
        )
        b = self._make_decision(
            topics=["deployment"],
            person_entities=["Alice"],
            non_person_entities=["AWS"],
            subject_label="prod release",
        )
        scores = score_decision_pair(a, b)
        self.assertLess(scores["combined_score"], 0.1)

    def test_specifics_mismatch_penalty_fires(self):
        a = self._make_decision(
            topics=["code review"],
            subject_label="PR review",
            key_facts=["migration-script v2.1"],
        )
        b = self._make_decision(
            topics=["code review"],
            subject_label="PR review",
            key_facts=["dashboard-widget refactor"],
        )
        scores = score_decision_pair(a, b)
        self.assertEqual(scores["specifics_penalty"], 0.85)

    def test_entity_mismatch_penalty_fires(self):
        a = self._make_decision(
            topics=["meeting"],
            non_person_entities=["ProjectAlpha"],
            subject_label="meeting notes",
        )
        b = self._make_decision(
            topics=["meeting"],
            non_person_entities=["ProjectBeta"],
            subject_label="meeting notes",
        )
        scores = score_decision_pair(a, b)
        self.assertEqual(scores["entity_penalty"], 0.85)

    def test_person_mismatch_penalty_fires(self):
        a = self._make_decision(
            topics=["out of office"],
            person_entities=["Akshay"],
            subject_label="OOO announcement",
        )
        b = self._make_decision(
            topics=["out of office"],
            person_entities=["Neeraj"],
            subject_label="OOO announcement",
        )
        scores = score_decision_pair(a, b)
        self.assertEqual(scores["person_penalty"], 0.85)

    def test_no_penalties_when_entities_match(self):
        a = self._make_decision(
            topics=["sprint planning"],
            person_entities=["Alice"],
            non_person_entities=["ProjectX"],
            subject_label="sprint",
            key_facts=["Sprint 42 goals"],
        )
        b = self._make_decision(
            topics=["sprint planning"],
            person_entities=["Alice"],
            non_person_entities=["ProjectX"],
            subject_label="sprint",
            key_facts=["Sprint 42 review"],
        )
        scores = score_decision_pair(a, b)
        self.assertEqual(scores["specifics_penalty"], 1.0)
        self.assertEqual(scores["entity_penalty"], 1.0)
        self.assertEqual(scores["person_penalty"], 1.0)

    def test_backward_compat_entity_score(self):
        a = self._make_decision(non_person_entities=["Acme"])
        b = self._make_decision(non_person_entities=["Acme"])
        scores = score_decision_pair(a, b)
        self.assertIn("entity_score", scores)

    def test_empty_decisions(self):
        a = self._make_decision()
        b = self._make_decision()
        scores = score_decision_pair(a, b)
        self.assertEqual(scores["combined_score"], 0.0)


def _seed_projection_and_enrichment(backend, decision_id, cid, gid, subject_label, topics, entities):
    backend.projection_store().save(
        pid=decision_id,
        gid=gid,
        cid=cid,
        proj_type="decision_trace",
        projection={"subject_label": subject_label, "Summary": "test"},
        msg_ts=1,
    )
    backend.enrichment_store().save(
        decision_id,
        {
            "decision_id": decision_id,
            "trace_id": f"t_{decision_id}",
            "topics": topics,
            "entities": [{"name": e, "type": "org"} for e in entities],
            "key_facts": [],
            "action_params": {},
        },
    )


class TestRunClusteringAnalysis(unittest.IsolatedAsyncioTestCase):
    async def test_no_decisions_returns_empty(self):
        backend = InMemoryBackend()
        retriever = ClusteringContextRetriever(gids=["g1"], backend=backend)
        new_decisions, candidates = await retriever.run_clustering_analysis(summary_text="test", cid="conv1")
        self.assertEqual(new_decisions, [])
        self.assertEqual(candidates, [])

    async def test_no_vector_index_returns_decisions_only(self):
        backend = InMemoryBackend()
        _seed_projection_and_enrichment(backend, "d1", "conv1", "g1", "autojoin", ["autojoin"], ["chetto"])

        retriever = ClusteringContextRetriever(gids=["g1"], backend=backend, vector_index=None)
        new_decisions, candidates = await retriever.run_clustering_analysis(summary_text="test", cid="conv1")
        self.assertEqual(len(new_decisions), 1)
        self.assertEqual(new_decisions[0]["decision_id"], "d1")
        self.assertEqual(candidates, [])

    async def test_returns_individual_decisions_not_fingerprint(self):
        backend = InMemoryBackend()
        _seed_projection_and_enrichment(backend, "d1", "conv1", "g1", "autojoin", ["autojoin"], ["chetto"])
        _seed_projection_and_enrichment(backend, "d2", "conv1", "g1", "logo design", ["branding"], ["figma"])

        retriever = ClusteringContextRetriever(gids=["g1"], backend=backend, vector_index=None)
        new_decisions, candidates = await retriever.run_clustering_analysis(summary_text="test", cid="conv1")
        self.assertEqual(len(new_decisions), 2)
        dids = {d["decision_id"] for d in new_decisions}
        self.assertEqual(dids, {"d1", "d2"})
        for d in new_decisions:
            self.assertIn("topics", d)
            self.assertIn("subject_label", d)
            self.assertIn("person_entities", d)
            self.assertIn("non_person_entities", d)

    async def test_decisions_have_required_fields(self):
        backend = InMemoryBackend()
        _seed_projection_and_enrichment(backend, "d1", "conv1", "g1", "test", ["topic1"], ["entity1"])

        retriever = ClusteringContextRetriever(gids=["g1"], backend=backend, vector_index=None)
        new_decisions, _ = await retriever.run_clustering_analysis(summary_text="test", cid="conv1")
        d = new_decisions[0]
        required = [
            "decision_id",
            "cid",
            "gid",
            "trace_id",
            "subject_label",
            "topics",
            "entities",
            "person_entities",
            "non_person_entities",
            "key_facts",
            "action_params",
        ]
        for field in required:
            self.assertIn(field, d, f"Missing field: {field}")

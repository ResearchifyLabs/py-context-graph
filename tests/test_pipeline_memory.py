"""
Pipeline test using InMemoryBackend.

Exercises persist → deduplicate → enrich with no external dependencies.
LLM executor, prompt loader, and vector index are all stubbed via patch.
"""

import unittest

import pandas as pd

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.backends.memory.stores import InMemoryGraphStore
from decision_graph.core.domain import KV, DecisionEnrichmentCoreExtract, Entity
from decision_graph.decision_trace_pipeline import DecisionTracePipeline
from decision_graph.retrieval import DecisionEnrichmentRetrieval

_STUB_ENRICHMENT = DecisionEnrichmentCoreExtract(
    topics=["topic_a", "topic_b"],
    entities=[Entity(type="person", name="Alice")],
    action_params=[KV(k="priority", v="high")],
    constraints_text=["must complete by EOD"],
    key_facts=[KV(k="budget", v="10k")],
)


class StubVectorIndex:
    def get_top_n_matches(self, *, query, query_filter, top_n):
        return pd.DataFrame()


def _make_decision_item(
    *,
    decision_type="BugReport",
    initiator_name="Alice",
    subject_label="login broken",
    action_type="fix",
    action_desc="Fix the login bug",
    status_state="proposed",
):
    return {
        "decision_type": decision_type,
        "actors": {
            "initiator_name": initiator_name,
            "initiator_role": "internal",
            "counterparty_names": [],
        },
        "subject": {"label": subject_label},
        "action": {"type": action_type, "description": action_desc},
        "status": {"state": status_state, "blocker": None},
        "evidence": {"span": "Alice said fix login", "confidence": 0.9},
    }


class StubLLMAdapter:
    async def execute_async(self, model_config, data, additional_data=None):
        return _STUB_ENRICHMENT


class TestPipelineWithInMemoryBackend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._executor = StubLLMAdapter()

    def _make_pipeline(self, backend):
        return DecisionTracePipeline(
            backend=backend,
            executor=self._executor,
            vector_index=StubVectorIndex(),
            graph_store=InMemoryGraphStore(),
        )

    async def test_persist_and_enrich_single_decision(self):
        backend = InMemoryBackend()
        pipeline = self._make_pipeline(backend)

        gid, cid = "g_test1", "c_test1"

        await pipeline.run(
            decision_items=[_make_decision_item()],
            summary_text="Alice reported that login is broken and needs fixing.",
            conv_id=cid,
            gid=gid,
            updated_at=1000.0,
            summary_pid="sp_1",
            query_gids=[gid],
        )

        proj_store = backend.projection_store()
        projections = await proj_store.find_by_filters(gids=[gid], proj_type="decision_trace")
        self.assertEqual(len(projections), 1)
        self.assertEqual(projections[0]["gid"], gid)
        self.assertEqual(projections[0]["cid"], cid)
        self.assertTrue(projections[0]["valid"])

        enrich_store = backend.enrichment_store()
        decision_id = projections[0]["pid"]
        enrichment = enrich_store.find_by_id(decision_id)
        self.assertIsNotNone(enrichment)
        self.assertIn("topic_a", enrichment.get("topics", []))
        self.assertEqual(enrichment["gid"], gid)

    async def test_deduplicate_near_duplicate_decisions(self):
        backend = InMemoryBackend()
        pipeline = self._make_pipeline(backend)

        gid, cid = "g_test2", "c_test2"
        items = [
            _make_decision_item(subject_label="client analytics test failure"),
            _make_decision_item(subject_label="client analytics test failure on action types"),
        ]

        await pipeline.run(
            decision_items=items,
            summary_text="Login is broken and needs fixing.",
            conv_id=cid,
            gid=gid,
            updated_at=2000.0,
            summary_pid="sp_2",
            query_gids=[gid],
        )

        proj_store = backend.projection_store()
        all_projs = proj_store.query(filters=[("gid", "==", gid), ("proj_type", "==", "decision_trace")])
        valid_projs = [p for p in all_projs if p.get("valid")]
        invalid_projs = [p for p in all_projs if not p.get("valid")]

        self.assertEqual(len(all_projs), 2, "Both projections should be persisted")
        self.assertEqual(len(valid_projs), 1, "One projection should remain valid after dedup")
        self.assertEqual(len(invalid_projs), 1, "One projection should be invalidated as duplicate")

    async def test_pipeline_direct_construction(self):
        backend = InMemoryBackend()
        pipeline = DecisionTracePipeline(
            backend=backend,
            executor=self._executor,
            vector_index=StubVectorIndex(),
            graph_store=InMemoryGraphStore(),
        )

        gid, cid = "g_test3", "c_test3"

        await pipeline.run(
            decision_items=[_make_decision_item(subject_label="payment failed")],
            summary_text="Payment processing has failed for customer X.",
            conv_id=cid,
            gid=gid,
            updated_at=3000.0,
            summary_pid="sp_3",
            query_gids=[gid],
        )

        proj_store = backend.projection_store()
        projs = await proj_store.find_by_filters(gids=[gid], proj_type="decision_trace")
        self.assertEqual(len(projs), 1)

    async def test_multiple_distinct_decisions_all_persisted(self):
        backend = InMemoryBackend()
        pipeline = self._make_pipeline(backend)

        gid, cid = "g_test4", "c_test4"
        items = [
            _make_decision_item(subject_label="login broken", action_type="fix"),
            _make_decision_item(subject_label="payment failed", action_type="review"),
            _make_decision_item(subject_label="deploy v2.0", action_type="approve"),
        ]

        await pipeline.run(
            decision_items=items,
            summary_text="Multiple issues discussed in standup.",
            conv_id=cid,
            gid=gid,
            updated_at=4000.0,
            summary_pid="sp_4",
            query_gids=[gid],
        )

        proj_store = backend.projection_store()
        projs = await proj_store.find_by_filters(gids=[gid], proj_type="decision_trace")
        valid_projs = [p for p in projs if p.get("valid")]
        self.assertEqual(len(valid_projs), 3, "All 3 distinct decisions should be valid")

        enrich_store = backend.enrichment_store()
        for proj in valid_projs:
            enrichment = enrich_store.find_by_id(proj["pid"])
            self.assertIsNotNone(enrichment, f"Enrichment missing for {proj['pid']}")

    async def test_retrieval_service_with_memory_backend(self):
        backend = InMemoryBackend()
        pipeline = self._make_pipeline(backend)

        gid, cid = "g_test5", "c_test5"

        await pipeline.run(
            decision_items=[_make_decision_item()],
            summary_text="Test summary",
            conv_id=cid,
            gid=gid,
            updated_at=5000.0,
            summary_pid="sp_5",
            query_gids=[gid],
        )

        retrieval = DecisionEnrichmentRetrieval(backend=backend)
        rows = retrieval.list_by_gid(gid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gid"], gid)

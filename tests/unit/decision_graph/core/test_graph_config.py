import unittest
from unittest.mock import AsyncMock, MagicMock

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.backends.memory.stores import InMemoryGraphStore
from decision_graph.clustering_service import DecisionClusterService
from decision_graph.core.config import DEFAULT_MODEL, GraphConfig, build_cluster_metadata_cfg
from decision_graph.decision_trace_pipeline import DecisionTracePipeline
from decision_graph.enrichment_service import DecisionEnrichmentService
from decision_graph.extraction_service import DecisionExtractionService
from decision_graph.graph import DecisionGraph

CUSTOM_MODEL = "anthropic/claude-3.5-sonnet"


class TestGraphConfigDefaults(unittest.TestCase):

    def test_default_model(self):
        config = GraphConfig()
        self.assertEqual(config.model, DEFAULT_MODEL)

    def test_custom_model(self):
        config = GraphConfig(model=CUSTOM_MODEL)
        self.assertEqual(config.model, CUSTOM_MODEL)


class TestBuildClusterMetadataCfg(unittest.TestCase):

    def test_default_model(self):
        cfg = build_cluster_metadata_cfg()
        self.assertEqual(cfg.model_name, DEFAULT_MODEL)

    def test_custom_model(self):
        cfg = build_cluster_metadata_cfg(CUSTOM_MODEL)
        self.assertEqual(cfg.model_name, CUSTOM_MODEL)


class TestConfigPropagationToServices(unittest.TestCase):

    def test_extraction_service_uses_config_model(self):
        config = GraphConfig(model=CUSTOM_MODEL)
        svc = DecisionExtractionService(executor=AsyncMock(), config=config)
        self.assertEqual(svc._config.model, CUSTOM_MODEL)

    def test_extraction_service_defaults_without_config(self):
        svc = DecisionExtractionService(executor=AsyncMock())
        self.assertEqual(svc._config.model, DEFAULT_MODEL)

    def test_enrichment_service_uses_config_model(self):
        config = GraphConfig(model=CUSTOM_MODEL)
        svc = DecisionEnrichmentService(backend=InMemoryBackend(), executor=AsyncMock(), config=config)
        self.assertEqual(svc._config.model, CUSTOM_MODEL)

    def test_cluster_service_uses_config_model(self):
        config = GraphConfig(model=CUSTOM_MODEL)
        svc = DecisionClusterService(backend=InMemoryBackend(), executor=AsyncMock(), config=config)
        self.assertEqual(svc._config.model, CUSTOM_MODEL)
        self.assertEqual(svc._cluster_metadata_cfg.model_name, CUSTOM_MODEL)


class TestPipelineConfigPropagation(unittest.TestCase):

    def test_pipeline_passes_config_to_services(self):
        config = GraphConfig(model=CUSTOM_MODEL)
        pipeline = DecisionTracePipeline(
            backend=InMemoryBackend(),
            executor=AsyncMock(),
            graph_store=InMemoryGraphStore(),
            config=config,
        )
        self.assertIs(pipeline._extraction_service._config, config)
        self.assertIs(pipeline._enrichment_service._config, config)

    def test_pipeline_defaults_without_config(self):
        pipeline = DecisionTracePipeline(
            backend=InMemoryBackend(),
            executor=AsyncMock(),
            graph_store=InMemoryGraphStore(),
        )
        self.assertEqual(pipeline._extraction_service._config.model, DEFAULT_MODEL)
        self.assertEqual(pipeline._enrichment_service._config.model, DEFAULT_MODEL)


class TestDecisionGraphConfigPropagation(unittest.TestCase):

    def test_decision_graph_passes_config_to_cluster_service(self):
        config = GraphConfig(model=CUSTOM_MODEL)
        dg = DecisionGraph(backend=InMemoryBackend(), executor=AsyncMock(), config=config)
        cluster_svc = dg.cluster_service()
        self.assertEqual(cluster_svc._config.model, CUSTOM_MODEL)
        self.assertEqual(cluster_svc._cluster_metadata_cfg.model_name, CUSTOM_MODEL)

    def test_decision_graph_defaults_without_config(self):
        dg = DecisionGraph(backend=InMemoryBackend(), executor=AsyncMock())
        cluster_svc = dg.cluster_service()
        self.assertEqual(cluster_svc._config.model, DEFAULT_MODEL)
        self.assertEqual(cluster_svc._cluster_metadata_cfg.model_name, DEFAULT_MODEL)


class TestModelReachesLLMCall(unittest.IsolatedAsyncioTestCase):

    async def test_extraction_passes_config_model_to_llm_config(self):
        executor = AsyncMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"items": []}
        executor.execute_async = AsyncMock(return_value=mock_response)

        config = GraphConfig(model=CUSTOM_MODEL)
        svc = DecisionExtractionService(executor=executor, config=config)
        await svc.extract("some conversation text")

        call_args = executor.execute_async.call_args
        model_cfg = call_args[0][0]
        self.assertEqual(model_cfg.model_name, CUSTOM_MODEL)

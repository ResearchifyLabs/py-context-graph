import unittest

from decision_graph.core.domain import DecisionEnrichmentCoreExtract


class TestDecisionEnrichmentServiceModels(unittest.TestCase):
    def test_enrichment_schema_builds(self):
        schema = DecisionEnrichmentCoreExtract.model_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("topics", schema.get("properties", {}))

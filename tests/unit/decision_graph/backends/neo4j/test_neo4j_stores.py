import unittest
from unittest.mock import Mock, patch

from decision_graph.backends.neo4j import Neo4jGraphReader, Neo4jGraphStore
from decision_graph.core.interfaces import GraphReader, GraphStore


def _make_driver():
    driver = Mock()
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    driver.session.return_value = session
    return driver, session


def _sample_hydrated_clusters():
    return [
        {
            "cluster": {
                "cluster_id": "cluster-1",
                "gid": "test-gid",
                "primary_subject": "API Architecture",
                "created_at": 1705334400.0,
                "last_updated_at": 1705334400.0,
                "rolling_summary": "Discussion about API architecture",
                "decision_count": 2,
            },
            "decisions": [
                {
                    "decision_id": "decision-1",
                    "gid": "test-gid",
                    "cid": "conv-1",
                    "trace_id": "trace-1",
                    "linked_at": 1705334400.0,
                    "linked_from": "extraction",
                    "enrichment": {
                        "topics": ["GraphQL", "Microservices"],
                        "constraints_text": ["Must be backwards compatible"],
                        "entities": [
                            {"type": "technology", "name": "GraphQL"},
                            {"type": "architecture", "name": "Microservices"},
                        ],
                        "key_facts": [
                            {"k": "complexity", "v": "medium"},
                            {"k": "timeline", "v": "Q1 2024"},
                        ],
                    },
                    "projection": {
                        "initiator_name": "Alice",
                        "initiator_role": "tech-lead",
                    },
                },
                {
                    "decision_id": "decision-2",
                    "gid": "test-gid",
                    "cid": "conv-1",
                    "trace_id": "trace-2",
                    "linked_at": 1705334400.0,
                    "linked_from": "extraction",
                    "enrichment": {
                        "topics": ["REST"],
                        "constraints_text": ["Performance requirements"],
                        "entities": [{"type": "technology", "name": "REST"}],
                        "key_facts": [{"k": "performance", "v": "high"}],
                    },
                    "projection": {
                        "initiator_name": "Bob",
                        "initiator_role": "developer",
                    },
                },
            ],
        }
    ]


# ===========================================================================
# Neo4jGraphStore
# ===========================================================================


class TestNeo4jGraphStoreInit(unittest.TestCase):
    def test_stores_driver_and_database(self):
        driver, _ = _make_driver()
        store = Neo4jGraphStore(driver, database="test-db")
        self.assertIs(store._driver, driver)
        self.assertEqual(store._database, "test-db")

    def test_default_database_is_neo4j(self):
        driver, _ = _make_driver()
        store = Neo4jGraphStore(driver)
        self.assertEqual(store._database, "neo4j")

    def test_satisfies_graphstore_protocol(self):
        driver, _ = _make_driver()
        self.assertIsInstance(Neo4jGraphStore(driver), GraphStore)


class TestNeo4jGraphStoreEnsureSchema(unittest.TestCase):
    def test_ensure_schema_creates_constraints(self):
        driver, session = _make_driver()
        session.run.return_value = Mock()
        store = Neo4jGraphStore(driver)
        store.ensure_schema()
        queries = [str(c[0][0]) for c in session.run.call_args_list]
        self.assertTrue(any("cluster_id_unique" in q for q in queries))
        self.assertTrue(any("decision_id_unique" in q for q in queries))
        self.assertTrue(any("topic_name_unique" in q for q in queries))
        self.assertTrue(any("entity_name_type_unique" in q for q in queries))
        self.assertTrue(any("constraint_unique" in q for q in queries))
        self.assertTrue(any("fact_unique" in q for q in queries))

    def test_ensure_schema_not_called_during_ingest(self):
        driver, session = _make_driver()
        store = Neo4jGraphStore(driver)
        store.ingest(_sample_hydrated_clusters())
        all_queries = " ".join(str(c) for c in session.run.call_args_list)
        self.assertNotIn("CREATE CONSTRAINT", all_queries)

    def test_constraint_errors_do_not_raise(self):
        driver, session = _make_driver()
        session.run.side_effect = Exception("already exists")
        store = Neo4jGraphStore(driver)
        store.ensure_schema()  # must not raise
        self.assertGreaterEqual(session.run.call_count, 6)


class TestNeo4jGraphStoreIngest(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.store = Neo4jGraphStore(self.driver)

    def test_empty_input_makes_no_db_calls(self):
        self.store.ingest([])
        self.session.run.assert_not_called()
        self.session.execute_write.assert_not_called()

    def test_opens_session_with_correct_database(self):
        self.store.ingest(_sample_hydrated_clusters())
        self.driver.session.assert_called_once_with(database="neo4j")

    def test_raises_on_db_error(self):
        self.session.execute_write.side_effect = Exception("connection refused")
        with self.assertRaises(Exception, msg="connection refused"):
            self.store.ingest(_sample_hydrated_clusters())

    def test_uses_execute_write_for_all_dml(self):
        self.store.ingest(_sample_hydrated_clusters())
        self.assertGreater(self.session.execute_write.call_count, 0)
        self.session.run.assert_not_called()

    def test_all_required_relationship_types_written(self):
        captured = []
        def capture(fn):
            captured.append(fn)
        self.session.execute_write.side_effect = capture
        self.store.ingest(_sample_hydrated_clusters())
        # Invoke each captured function with a mock transaction to get the query
        all_queries = ""
        for fn in captured:
            tx = Mock()
            fn(tx)
            if tx.run.call_args:
                all_queries += str(tx.run.call_args[0][0])
        for rel in ("HAS_DECISION", "HAS_TOPIC", "MENTIONS_ENTITY",
                    "INITIATED_BY", "HAS_CONSTRAINT", "HAS_FACT"):
            self.assertIn(rel, all_queries, msg=f"Missing relationship: {rel}")

    def test_legacy_relationship_names_not_written(self):
        captured = []
        self.session.execute_write.side_effect = captured.append
        self.store.ingest(_sample_hydrated_clusters())
        all_queries = ""
        for fn in captured:
            tx = Mock()
            fn(tx)
            if tx.run.call_args:
                all_queries += str(tx.run.call_args[0][0])
        for bad in ("CONTAINS", "HAS_ENTITY", "HAS_INITIATOR"):
            self.assertNotIn(bad, all_queries, msg=f"Legacy rel still present: {bad}")

    @patch("decision_graph.backends.neo4j.stores.breakdown_hydrated_clusters")
    def test_calls_breakdown_hydrated_clusters(self, mock_breakdown):
        mock_breakdown.return_value = {
            "clusters": [], "decisions": [], "decision_cluster": [],
            "decision_topics": [], "decision_constraints": [],
            "decision_entities": [], "decision_facts": [], "decision_initiators": [],
        }
        self.store.ingest(_sample_hydrated_clusters())
        mock_breakdown.assert_called_once_with(_sample_hydrated_clusters())


class TestNeo4jGraphStoreBatching(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.store = Neo4jGraphStore(self.driver)

    def test_large_list_is_chunked(self):
        from decision_graph.backends.neo4j.stores import _CHUNK_SIZE
        decisions = [
            {"decision_id": f"d{i}", "gid": "g", "cid": "c",
             "trace_id": "t", "linked_at": 1.0, "linked_from": "x"}
            for i in range(_CHUNK_SIZE + 1)
        ]
        self.store._upsert_decisions(self.session, decisions)
        self.assertEqual(self.session.execute_write.call_count, 2)

    def test_single_chunk_is_one_call(self):
        decisions = [
            {"decision_id": "d1", "gid": "g", "cid": "c",
             "trace_id": "t", "linked_at": 1.0, "linked_from": "x"}
        ]
        self.store._upsert_decisions(self.session, decisions)
        self.assertEqual(self.session.execute_write.call_count, 1)


class TestNeo4jGraphStoreUpserts(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.store = Neo4jGraphStore(self.driver)

    def _run_captured(self, method, *args):
        captured = []
        self.session.execute_write.side_effect = captured.append
        method(*args)
        queries = []
        kwargs_list = []
        for fn in captured:
            tx = Mock()
            fn(tx)
            if tx.run.call_args:
                queries.append(tx.run.call_args[0][0])
                kwargs_list.append(tx.run.call_args[1])
        return queries, kwargs_list

    def test_upsert_clusters_merges_on_cluster_id(self):
        clusters = [{"cluster_id": "c1", "gid": "g1", "primary_subject": "Test",
                     "created_at": 1.0, "last_updated_at": 1.0,
                     "rolling_summary": "s", "decision_count": 1}]
        queries, _ = self._run_captured(self.store._upsert_clusters, self.session, clusters)
        self.assertTrue(any("MERGE (c:Cluster {cluster_id: cluster.cluster_id})" in q for q in queries))

    def test_upsert_clusters_empty_skips_db(self):
        self.store._upsert_clusters(self.session, [])
        self.session.execute_write.assert_not_called()

    def test_link_clusters_decisions_uses_has_decision(self):
        dc = [{"cluster_id": "c1", "decision_id": "d1"}]
        queries, _ = self._run_captured(self.store._link_clusters_decisions, self.session, dc)
        self.assertTrue(any("MERGE (c)-[:HAS_DECISION]->(d)" in q for q in queries))

    def test_upsert_topics_shared_node_merge_on_name(self):
        topics = [{"decision_id": "d1", "gid": "g1", "topic": "GraphQL"}]
        queries, _ = self._run_captured(self.store._upsert_topics, self.session, topics)
        self.assertTrue(any("MERGE (t:Topic {name: topic.topic})" in q for q in queries))
        self.assertTrue(any("MERGE (d)-[:HAS_TOPIC]->(t)" in q for q in queries))

    def test_upsert_entities_uses_mentions_entity(self):
        entities = [{"decision_id": "d1", "gid": "g1",
                     "type": "technology", "name": "graphql", "display_name": "GraphQL"}]
        queries, _ = self._run_captured(self.store._upsert_entities, self.session, entities)
        self.assertTrue(any("MERGE (d)-[:MENTIONS_ENTITY]->(e)" in q for q in queries))

    def test_upsert_initiators_uses_initiated_by(self):
        initiators = [{"decision_id": "d1", "gid": "g1", "initiator_name": "alice",
                       "display_name": "Alice", "initiator_role": "tech-lead"}]
        queries, _ = self._run_captured(self.store._upsert_initiators, self.session, initiators)
        self.assertTrue(any("MERGE (d)-[:INITIATED_BY]->(e)" in q for q in queries))

    def test_upsert_constraints(self):
        constraints = [{"decision_id": "d1", "gid": "g1", "text": "BC required"}]
        queries, _ = self._run_captured(self.store._upsert_constraints, self.session, constraints)
        self.assertTrue(any("MERGE (d)-[:HAS_CONSTRAINT]->(c)" in q for q in queries))

    def test_upsert_facts(self):
        facts = [{"decision_id": "d1", "k": "timeline", "v": "Q1"}]
        queries, _ = self._run_captured(self.store._upsert_facts, self.session, facts)
        self.assertTrue(any("MERGE (d)-[:HAS_FACT]->(f)" in q for q in queries))


class TestNeo4jGraphStoreDeleteEdges(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.store = Neo4jGraphStore(self.driver)

    def test_empty_ids_skips_db(self):
        self.store._delete_enrichment_edges(self.session, [])
        self.session.execute_write.assert_not_called()

    def test_makes_two_execute_write_calls_per_chunk(self):
        self.store._delete_enrichment_edges(self.session, ["d1"])
        self.assertEqual(self.session.execute_write.call_count, 2)

    def test_shared_rels_deleted_without_node(self):
        captured = []
        self.session.execute_write.side_effect = captured.append
        self.store._delete_enrichment_edges(self.session, ["d1"])
        tx = Mock()
        captured[0](tx)
        q = tx.run.call_args[0][0]
        self.assertIn("HAS_TOPIC", q)
        self.assertIn("MENTIONS_ENTITY", q)
        self.assertIn("INITIATED_BY", q)
        self.assertNotIn("DELETE r, n", q)

    def test_owned_nodes_deleted_with_relationship(self):
        captured = []
        self.session.execute_write.side_effect = captured.append
        self.store._delete_enrichment_edges(self.session, ["d1"])
        tx = Mock()
        captured[1](tx)
        q = tx.run.call_args[0][0]
        self.assertIn("HAS_CONSTRAINT", q)
        self.assertIn("HAS_FACT", q)
        self.assertIn("DELETE r, n", q)


# ===========================================================================
# Neo4jGraphReader
# ===========================================================================


class TestNeo4jGraphReaderInit(unittest.TestCase):
    def test_stores_driver_and_database(self):
        driver, _ = _make_driver()
        reader = Neo4jGraphReader(driver, database="custom")
        self.assertIs(reader._driver, driver)
        self.assertEqual(reader._database, "custom")

    def test_default_database_is_neo4j(self):
        driver, _ = _make_driver()
        reader = Neo4jGraphReader(driver)
        self.assertEqual(reader._database, "neo4j")

    def test_satisfies_graphreader_protocol(self):
        driver, _ = _make_driver()
        self.assertIsInstance(Neo4jGraphReader(driver), GraphReader)


class TestNeo4jGraphReaderResolve(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.reader = Neo4jGraphReader(self.driver)

    def test_opens_correct_session(self):
        self.session.run.return_value = []
        self.reader.resolve("graphql")
        self.driver.session.assert_called_with(database="neo4j")

    def test_returns_candidates_and_debug(self):
        self.session.run.return_value = []
        result = self.reader.resolve("graphql")
        self.assertIn("candidates", result)
        self.assertIn("debug", result)
        self.assertIn("latency_ms", result["debug"])

    def test_fallback_fires_when_fewer_than_top_k(self):
        """Fallback supplements partial results, not only empty results."""
        call_count = {"n": 0}
        def side(query, params):
            call_count["n"] += 1
            return []
        self.session.run.side_effect = side
        # top_k=5 but name-match returns 0 — fallback should fire
        self.reader.resolve("graphql", types=["Decision"], top_k=5)
        self.assertEqual(call_count["n"], 2)

    def test_no_fallback_when_decision_not_in_types(self):
        self.session.run.return_value = []
        self.reader.resolve("graphql", types=["Topic"])
        self.assertEqual(self.session.run.call_count, 1)


class TestNeo4jGraphReaderExecuteTemplate(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.reader = Neo4jGraphReader(self.driver)

    def test_known_template_returns_records_and_debug(self):
        mock_record = Mock()
        mock_record.data.return_value = {"seed": {}, "primary_paths": []}
        self.session.run.return_value = [mock_record]
        result = self.reader.execute_template(
            "DECISION_EGO_V1", {"node_id": "d1", "rel_allowlist": []}
        )
        self.assertEqual(result["records"], [mock_record.data.return_value])
        self.assertEqual(result["debug"]["template_id"], "DECISION_EGO_V1")
        self.assertIn("latency_ms", result["debug"])

    def test_records_are_plain_dicts_not_driver_objects(self):
        mock_record = Mock()
        mock_record.data.return_value = {"k": "v"}
        self.session.run.return_value = [mock_record]
        result = self.reader.execute_template("DECISION_EGO_V1", {"node_id": "d1", "rel_allowlist": []})
        self.assertIsInstance(result["records"][0], dict)

    def test_unknown_template_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.reader.execute_template("DOES_NOT_EXIST", {})

    def test_passes_params_to_session(self):
        self.session.run.return_value = []
        params = {"node_id": "d1", "rel_allowlist": ["HAS_TOPIC"]}
        self.reader.execute_template("DECISION_EGO_V1", params)
        self.assertEqual(self.session.run.call_args[0][1], params)


class TestNeo4jGraphReaderRunQuery(unittest.TestCase):
    def setUp(self):
        self.driver, self.session = _make_driver()
        self.reader = Neo4jGraphReader(self.driver)

    def test_returns_list_of_dicts(self):
        row = {"decision_id": "d1", "k": "timeline", "v": "Q1"}
        self.session.run.return_value = [row]
        result = self.reader.run_query("MATCH (n) RETURN n", {"x": 1})
        self.assertIsInstance(result, list)

    def test_passes_params_to_session(self):
        self.session.run.return_value = []
        self.reader.run_query("MATCH (n) RETURN n LIMIT $lim", {"lim": 5})
        self.assertEqual(self.session.run.call_args[0][1], {"lim": 5})

    def test_defaults_to_empty_params(self):
        self.session.run.return_value = []
        self.reader.run_query("MATCH (n) RETURN n")
        self.assertEqual(self.session.run.call_args[0][1], {})


if __name__ == "__main__":
    unittest.main()

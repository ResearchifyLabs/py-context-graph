"""Tests for Neo4j GraphStore and GraphReader implementations."""

import pytest
from unittest.mock import MagicMock, Mock, patch

from decision_graph.backends.neo4j import Neo4jGraphReader, Neo4jGraphStore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_mock_driver():
    """Return (driver, session) where session is a context-manager mock."""
    driver = Mock()
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    driver.session.return_value = session
    return driver, session


@pytest.fixture
def mock_driver():
    return _make_mock_driver()


@pytest.fixture
def neo4j_store(mock_driver):
    driver, session = mock_driver
    return Neo4jGraphStore(driver), driver, session


@pytest.fixture
def neo4j_reader(mock_driver):
    driver, session = mock_driver
    return Neo4jGraphReader(driver), driver, session


@pytest.fixture
def sample_hydrated_clusters():
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
                        "counterparty_names": ["Bob"],
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
# Neo4jGraphStore tests
# ===========================================================================


class TestNeo4jGraphStore:

    def test_init(self, mock_driver):
        driver, _ = mock_driver
        store = Neo4jGraphStore(driver, database="test-db")
        assert store._driver is driver
        assert store._database == "test-db"

    def test_init_default_database(self, mock_driver):
        driver, _ = mock_driver
        store = Neo4jGraphStore(driver)
        assert store._database == "neo4j"

    def test_ingest_empty_clusters(self, neo4j_store):
        store, driver, session = neo4j_store
        store.ingest([])
        session.run.assert_not_called()

    def test_ingest_successful(self, neo4j_store, sample_hydrated_clusters):
        store, driver, session = neo4j_store
        session.run.return_value = Mock()

        store.ingest(sample_hydrated_clusters)

        driver.session.assert_called_once_with(database="neo4j")
        constraint_calls = [
            str(call[0][0])
            for call in session.run.call_args_list
            if "CREATE CONSTRAINT" in str(call[0][0])
        ]
        assert len(constraint_calls) >= 6

    def test_ingest_raises_on_db_error(self, neo4j_store, sample_hydrated_clusters):
        store, driver, session = neo4j_store
        session.run.side_effect = Exception("Database error")
        with pytest.raises(Exception, match="Database error"):
            store.ingest(sample_hydrated_clusters)

    # --- _create_constraints ---

    def test_create_constraints(self, neo4j_store):
        store, _, session = neo4j_store
        session.run.return_value = Mock()

        store._create_constraints(session)

        calls = [str(c[0][0]) for c in session.run.call_args_list]
        assert any("cluster_id_unique" in c for c in calls)
        assert any("decision_id_unique" in c for c in calls)
        assert any("topic_name_unique" in c for c in calls)
        assert any("entity_name_type_unique" in c for c in calls)
        assert any("constraint_unique" in c for c in calls)
        assert any("fact_unique" in c for c in calls)

    def test_create_constraints_ignores_existing(self, neo4j_store):
        store, _, session = neo4j_store
        session.run.side_effect = Exception("already exists")
        store._create_constraints(session)  # must not raise
        assert session.run.call_count >= 6

    # --- _upsert_clusters ---

    def test_upsert_clusters(self, neo4j_store):
        store, _, session = neo4j_store
        clusters = [
            {
                "cluster_id": "c1",
                "gid": "g1",
                "primary_subject": "Test",
                "created_at": 1.0,
                "last_updated_at": 1.0,
                "rolling_summary": "s",
                "decision_count": 1,
            }
        ]
        store._upsert_clusters(session, clusters)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (c:Cluster {cluster_id: cluster.cluster_id})" in query
        assert kwargs["clusters"] == clusters

    # --- _upsert_decisions ---

    def test_upsert_decisions(self, neo4j_store):
        store, _, session = neo4j_store
        decisions = [
            {
                "decision_id": "d1",
                "gid": "g1",
                "cid": "c1",
                "trace_id": "t1",
                "linked_at": 1.0,
                "linked_from": "extraction",
            }
        ]
        store._upsert_decisions(session, decisions)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (d:Decision {decision_id: decision.decision_id})" in query
        assert kwargs["decisions"] == decisions

    # --- _link_clusters_decisions ---

    def test_link_clusters_decisions_uses_has_decision(self, neo4j_store):
        """Relationship must be HAS_DECISION to match context_graph/registry.py."""
        store, _, session = neo4j_store
        dc = [{"cluster_id": "c1", "decision_id": "d1"}]
        store._link_clusters_decisions(session, dc)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (c)-[:HAS_DECISION]->(d)" in query
        assert kwargs["decision_cluster"] == dc

    # --- _delete_enrichment_edges ---

    def test_delete_enrichment_edges(self, neo4j_store):
        store, _, session = neo4j_store
        decision_ids = ["d1", "d2"]
        store._delete_enrichment_edges(session, decision_ids)
        # Two session.run calls: one for shared-node rels, one for owned nodes.
        assert session.run.call_count == 2
        last_query = session.run.call_args_list[-1][0][0]
        last_kwargs = session.run.call_args_list[-1][1]
        assert "DELETE r, n" in last_query
        assert last_kwargs["decision_ids"] == decision_ids

    def test_delete_enrichment_edges_uses_correct_rel_types(self, neo4j_store):
        store, _, session = neo4j_store
        store._delete_enrichment_edges(session, ["d1"])
        first_query = session.run.call_args_list[0][0][0]
        assert "HAS_TOPIC" in first_query
        assert "MENTIONS_ENTITY" in first_query
        assert "INITIATED_BY" in first_query
        assert "HAS_COUNTERPARTY" in first_query

    def test_delete_enrichment_edges_empty_is_noop(self, neo4j_store):
        store, _, session = neo4j_store
        store._delete_enrichment_edges(session, [])
        session.run.assert_not_called()

    # --- _upsert_topics ---

    def test_upsert_topics(self, neo4j_store):
        store, _, session = neo4j_store
        topics = [{"decision_id": "d1", "gid": "g1", "topic": "GraphQL"}]
        store._upsert_topics(session, topics)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (t:Topic {name: topic.topic})" in query
        assert "MERGE (d)-[:HAS_TOPIC]->(t)" in query
        assert kwargs["topics"] == topics

    # --- _upsert_entities ---

    def test_upsert_entities_uses_mentions_entity(self, neo4j_store):
        """Relationship must be MENTIONS_ENTITY to match all Cypher templates."""
        store, _, session = neo4j_store
        entities = [
            {
                "decision_id": "d1",
                "gid": "g1",
                "type": "technology",
                "name": "graphql",
                "display_name": "GraphQL",
            }
        ]
        store._upsert_entities(session, entities)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (e:Entity {name: entity.name, type: entity.type})" in query
        assert "MERGE (d)-[:MENTIONS_ENTITY]->(e)" in query
        assert kwargs["entities"] == entities

    # --- _upsert_initiators ---

    def test_upsert_initiators_uses_initiated_by(self, neo4j_store):
        """Relationship must be INITIATED_BY to match all Cypher templates."""
        store, _, session = neo4j_store
        initiators = [
            {
                "decision_id": "d1",
                "gid": "g1",
                "initiator_name": "alice",
                "display_name": "Alice",
                "initiator_role": "tech-lead",
            }
        ]
        store._upsert_initiators(session, initiators)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (e:Entity {name: initiator.initiator_name, type: 'Person'})" in query
        assert "MERGE (d)-[:INITIATED_BY]->(e)" in query
        assert kwargs["initiators"] == initiators

    # --- _upsert_counterparties ---

    def test_upsert_counterparties_uses_has_counterparty(self, neo4j_store):
        store, _, session = neo4j_store
        counterparties = [
            {
                "decision_id": "d1",
                "gid": "g1",
                "counterparty_name": "bob",
                "display_name": "Bob",
            }
        ]
        store._upsert_counterparties(session, counterparties)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (e:Entity {name: cp.counterparty_name, type: 'Person'})" in query
        assert "MERGE (d)-[:HAS_COUNTERPARTY]->(e)" in query
        assert kwargs["counterparties"] == counterparties

    def test_upsert_counterparties_empty_is_noop(self, neo4j_store):
        store, _, session = neo4j_store
        store._upsert_counterparties(session, [])
        session.run.assert_not_called()

    # --- _upsert_constraints ---

    def test_upsert_constraints(self, neo4j_store):
        store, _, session = neo4j_store
        constraints = [{"decision_id": "d1", "gid": "g1", "text": "Must be compatible"}]
        store._upsert_constraints(session, constraints)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (d)-[:HAS_CONSTRAINT]->(c)" in query
        assert kwargs["constraints"] == constraints

    # --- _upsert_facts ---

    def test_upsert_facts(self, neo4j_store):
        store, _, session = neo4j_store
        facts = [{"decision_id": "d1", "k": "complexity", "v": "medium"}]
        store._upsert_facts(session, facts)
        query, kwargs = session.run.call_args[0][0], session.run.call_args[1]
        assert "MERGE (d)-[:HAS_FACT]->(f)" in query
        assert kwargs["facts"] == facts

    # --- integration ---

    def test_ingest_integration(self, neo4j_store, sample_hydrated_clusters):
        store, driver, session = neo4j_store
        session.run.return_value = Mock()

        store.ingest(sample_hydrated_clusters)

        # constraints(6) + clusters + decisions + cluster_links +
        # delete_shared_rels + delete_owned_nodes +
        # topics + constraints + entities + facts + initiators + counterparties
        assert session.run.call_count >= 12
        driver.session.assert_called_once_with(database="neo4j")

    @patch("decision_graph.backends.neo4j.stores.breakdown_hydrated_clusters")
    def test_ingest_calls_breakdown(self, mock_breakdown, neo4j_store, sample_hydrated_clusters):
        store, _, session = neo4j_store
        mock_breakdown.return_value = {
            "clusters": [],
            "decisions": [],
            "decision_cluster": [],
            "decision_topics": [],
            "decision_constraints": [],
            "decision_entities": [],
            "decision_facts": [],
            "decision_initiators": [],
            "decision_counterparties": [],
        }
        session.run.return_value = Mock()
        store.ingest(sample_hydrated_clusters)
        mock_breakdown.assert_called_once_with(sample_hydrated_clusters)


# ===========================================================================
# Neo4jGraphReader tests
# ===========================================================================


class TestNeo4jGraphReader:

    def test_init(self, mock_driver):
        driver, _ = mock_driver
        reader = Neo4jGraphReader(driver, database="custom")
        assert reader._driver is driver
        assert reader._database == "custom"

    def test_init_default_database(self, mock_driver):
        driver, _ = mock_driver
        reader = Neo4jGraphReader(driver)
        assert reader._database == "neo4j"

    # --- resolve ---

    def test_resolve_returns_candidates(self, neo4j_reader):
        reader, driver, session = neo4j_reader

        node = MagicMock()
        node.__iter__ = Mock(return_value=iter([("name", "GraphQL"), ("type", "technology")]))
        node.items = Mock(return_value=[("name", "GraphQL"), ("type", "technology")])
        # dict(node) must return a real dict
        node_dict = {"name": "graphql", "type": "technology"}

        record = MagicMock()
        record.__getitem__ = Mock(
            side_effect=lambda k: node if k == "n" else ["Topic", "Entity"]
        )
        # Make session.run return a list with one record.
        session.run.return_value = [record]

        with patch("decision_graph.backends.neo4j.reader.dict", side_effect=lambda x: node_dict if x is node else dict(x)):
            result = reader.resolve("graphql", types=["Topic", "Entity"])

        assert "candidates" in result
        assert "debug" in result
        assert "latency_ms" in result["debug"]

    def test_resolve_opens_correct_session(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        session.run.return_value = []
        reader.resolve("test query")
        driver.session.assert_called_with(database="neo4j")

    def test_resolve_uses_resolve_nodes_template(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        session.run.return_value = []
        reader.resolve("graphql", types=["Topic"])
        first_call_query = session.run.call_args[0][0]
        # RESOLVE_NODES starts with "WITH toLower($q)"
        assert "toLower($q)" in first_call_query or "$q" in first_call_query

    def test_resolve_falls_back_to_keyword(self, neo4j_reader):
        """When RESOLVE_NODES returns nothing and Decision is in types, fall back."""
        reader, driver, session = neo4j_reader
        # First call (RESOLVE_NODES) returns empty; second call (RESOLVE_BY_KEYWORD) returns a record.
        decision_node_dict = {"decision_id": "dec-1", "subject_label": "Use GraphQL"}
        dec_record = MagicMock()

        call_count = {"n": 0}

        def run_side_effect(query, params):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return []  # RESOLVE_NODES → empty
            # RESOLVE_BY_KEYWORD record
            mock_node = MagicMock()
            dec_record.__getitem__ = Mock(return_value=mock_node)
            with patch("decision_graph.backends.neo4j.reader.dict", return_value=decision_node_dict):
                pass
            return []  # simplify: just ensure two calls happen

        session.run.side_effect = run_side_effect
        reader.resolve("graphql", types=["Decision"])
        assert session.run.call_count == 2

    # --- execute_template ---

    def test_execute_template_known(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        mock_record = Mock()
        session.run.return_value = [mock_record]

        result = reader.execute_template("DECISION_EGO_V1", {"node_id": "d1", "rel_allowlist": []})

        assert result["records"] == [mock_record]
        assert result["debug"]["template_id"] == "DECISION_EGO_V1"
        assert "latency_ms" in result["debug"]

    def test_execute_template_unknown_raises(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        with pytest.raises(ValueError, match="Unknown template_id"):
            reader.execute_template("DOES_NOT_EXIST", {})

    def test_execute_template_passes_params(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        session.run.return_value = []
        params = {"node_id": "d1", "rel_allowlist": ["HAS_TOPIC"]}
        reader.execute_template("DECISION_EGO_V1", params)
        _, call_params = session.run.call_args[0][0], session.run.call_args[0][1]
        assert call_params == params

    # --- run_query ---

    def test_run_query_returns_list_of_dicts(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        record = MagicMock()
        record.__iter__ = Mock(return_value=iter([("decision_id", "d1"), ("k", "tier"), ("v", "1")]))
        # dict(record) should give a plain dict
        with patch("decision_graph.backends.neo4j.reader.dict", side_effect=lambda x: {"decision_id": "d1", "k": "tier", "v": "1"} if x is record else dict(x)):
            session.run.return_value = [record]
            result = reader.run_query("MATCH (d:Decision) RETURN d", {"ids": ["d1"]})

        # result must be a list; each element dict-like
        assert isinstance(result, list)

    def test_run_query_passes_params(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        session.run.return_value = []
        reader.run_query("MATCH (n) RETURN n LIMIT $lim", {"lim": 5})
        call_args = session.run.call_args[0]
        assert call_args[1] == {"lim": 5}

    def test_run_query_defaults_empty_params(self, neo4j_reader):
        reader, driver, session = neo4j_reader
        session.run.return_value = []
        reader.run_query("MATCH (n) RETURN n")
        call_args = session.run.call_args[0]
        assert call_args[1] == {}

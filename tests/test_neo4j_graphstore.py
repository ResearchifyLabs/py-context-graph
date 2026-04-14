"""Tests for Neo4j GraphStore implementation."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

from decision_graph.backends.neo4j import Neo4jGraphStore


class TestNeo4jGraphStore:
    """Test suite for Neo4jGraphStore."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        driver.session.return_value = session
        return driver, session

    @pytest.fixture
    def neo4j_store(self, mock_driver):
        """Create Neo4jGraphStore instance with mock driver."""
        driver, session = mock_driver
        store = Neo4jGraphStore(driver)
        return store, driver, session

    @pytest.fixture
    def sample_hydrated_clusters(self) -> List[Dict[str, Any]]:
        """Sample hydrated cluster data for testing."""
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
                                {"type": "architecture", "name": "Microservices"}
                            ],
                            "key_facts": [
                                {"k": "complexity", "v": "medium"},
                                {"k": "timeline", "v": "Q1 2024"}
                            ],
                        },
                        "projection": {
                            "initiator_name": "Alice",
                            "initiator_role": "tech-lead"
                        }
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
                            "entities": [
                                {"type": "technology", "name": "REST"}
                            ],
                            "key_facts": [
                                {"k": "performance", "v": "high"}
                            ],
                        },
                        "projection": {
                            "initiator_name": "Bob",
                            "initiator_role": "developer"
                        }
                    }
                ]
            }
        ]

    def test_init(self, mock_driver):
        """Test Neo4jGraphStore initialization."""
        driver, session = mock_driver
        store = Neo4jGraphStore(driver, database="test-db")
        
        assert store._driver == driver
        assert store._database == "test-db"

    def test_init_default_database(self, mock_driver):
        """Test Neo4jGraphStore initialization with default database."""
        driver, session = mock_driver
        store = Neo4jGraphStore(driver)
        
        assert store._driver == driver
        assert store._database == "neo4j"

    def test_ingest_empty_clusters(self, neo4j_store):
        """Test ingesting empty hydrated clusters."""
        store, driver, session = neo4j_store
        
        store.ingest([])
        
        # Should not call any database operations
        session.run.assert_not_called()

    def test_ingest_successful(self, neo4j_store, sample_hydrated_clusters):
        """Test successful ingestion of hydrated clusters."""
        store, driver, session = neo4j_store
        
        # Mock session.run to return empty results
        session.run.return_value = Mock()
        
        store.ingest(sample_hydrated_clusters)
        
        # Verify session was created
        driver.session.assert_called_once_with(database="neo4j")
        
        # Verify constraints were created
        constraint_calls = [call[0][0] for call in session.run.call_args_list 
                           if "CREATE CONSTRAINT" in str(call)]
        assert len(constraint_calls) >= 7  # Should create all constraints

    def test_ingest_with_database_exception(self, neo4j_store, sample_hydrated_clusters):
        """Test ingestion with database exception."""
        store, driver, session = neo4j_store
        
        # Mock session to raise an exception
        session.run.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            store.ingest(sample_hydrated_clusters)

    def test_create_constraints(self, neo4j_store):
        """Test constraint creation."""
        store, driver, session = neo4j_store
        
        # Mock successful constraint creation
        session.run.return_value = Mock()
        
        store._create_constraints(session)
        
        # Verify all constraint queries were called
        constraint_calls = [call[0][0] for call in session.run.call_args_list]
        assert any("cluster_id_unique" in call for call in constraint_calls)
        assert any("decision_id_unique" in call for call in constraint_calls)
        assert any("topic_unique" in call for call in constraint_calls)
        assert any("constraint_unique" in call for call in constraint_calls)
        assert any("entity_unique" in call for call in constraint_calls)
        assert any("fact_unique" in call for call in constraint_calls)
        assert any("initiator_unique" in call for call in constraint_calls)

    def test_create_constraints_with_existing_errors(self, neo4j_store):
        """Test constraint creation with existing constraint errors."""
        store, driver, session = neo4j_store
        
        # Mock constraint already exists error
        session.run.side_effect = Exception("Constraint already exists")
        
        # Should not raise exception
        store._create_constraints(session)
        
        # Should still attempt all constraints
        assert session.run.call_count >= 7

    def test_upsert_clusters(self, neo4j_store):
        """Test cluster upsertion."""
        store, driver, session = neo4j_store
        
        clusters = [
            {
                "cluster_id": "cluster-1",
                "gid": "test-gid",
                "primary_subject": "Test Subject",
                "created_at": 1705334400.0,
                "last_updated_at": 1705334400.0,
                "rolling_summary": "Test summary",
                "decision_count": 1,
            }
        ]
        
        store._upsert_clusters(session, clusters)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (c:Cluster {cluster_id: cluster.cluster_id})" in query
        assert params["clusters"] == clusters

    def test_upsert_decisions(self, neo4j_store):
        """Test decision upsertion."""
        store, driver, session = neo4j_store
        
        decisions = [
            {
                "decision_id": "decision-1",
                "gid": "test-gid",
                "cid": "conv-1",
                "trace_id": "trace-1",
                "linked_at": 1705334400.0,
                "linked_from": "extraction",
            }
        ]
        
        store._upsert_decisions(session, decisions)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (d:Decision {decision_id: decision.decision_id})" in query
        assert params["decisions"] == decisions

    def test_link_clusters_decisions(self, neo4j_store):
        """Test linking clusters to decisions."""
        store, driver, session = neo4j_store
        
        decision_cluster = [
            {"cluster_id": "cluster-1", "decision_id": "decision-1"}
        ]
        
        store._link_clusters_decisions(session, decision_cluster)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (c)-[:CONTAINS]->(d)" in query
        assert params["decision_cluster"] == decision_cluster

    def test_delete_enrichment_edges(self, neo4j_store):
        """Test deletion of enrichment edges."""
        store, driver, session = neo4j_store
        
        decision_ids = ["decision-1", "decision-2"]
        
        store._delete_enrichment_edges(session, decision_ids)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "DELETE r, n" in query
        assert params["decision_ids"] == decision_ids

    def test_delete_enrichment_edges_empty(self, neo4j_store):
        """Test deletion of enrichment edges with empty list."""
        store, driver, session = neo4j_store
        
        store._delete_enrichment_edges(session, [])
        
        # Should not call session.run
        session.run.assert_not_called()

    def test_upsert_topics(self, neo4j_store):
        """Test topic upsertion."""
        store, driver, session = neo4j_store
        
        topics = [
            {"decision_id": "decision-1", "gid": "test-gid", "topic": "GraphQL"}
        ]
        
        store._upsert_topics(session, topics)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (d)-[:HAS_TOPIC]->(t:Topic" in query
        assert params["topics"] == topics

    def test_upsert_constraints(self, neo4j_store):
        """Test constraint upsertion."""
        store, driver, session = neo4j_store
        
        constraints = [
            {"decision_id": "decision-1", "gid": "test-gid", "text": "Must be compatible"}
        ]
        
        store._upsert_constraints(session, constraints)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (d)-[:HAS_CONSTRAINT]->(c:Constraint" in query
        assert params["constraints"] == constraints

    def test_upsert_entities(self, neo4j_store):
        """Test entity upsertion."""
        store, driver, session = neo4j_store
        
        entities = [
            {
                "decision_id": "decision-1",
                "gid": "test-gid",
                "type": "technology",
                "name": "GraphQL",
                "display_name": "GraphQL"
            }
        ]
        
        store._upsert_entities(session, entities)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (d)-[:HAS_ENTITY]->(e:Entity" in query
        assert params["entities"] == entities

    def test_upsert_facts(self, neo4j_store):
        """Test fact upsertion."""
        store, driver, session = neo4j_store
        
        facts = [
            {"decision_id": "decision-1", "k": "complexity", "v": "medium"}
        ]
        
        store._upsert_facts(session, facts)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (d)-[:HAS_FACT]->(f:Fact" in query
        assert params["facts"] == facts

    def test_upsert_initiators(self, neo4j_store):
        """Test initiator upsertion."""
        store, driver, session = neo4j_store
        
        initiators = [
            {
                "decision_id": "decision-1",
                "gid": "test-gid",
                "initiator_name": "Alice",
                "display_name": "Alice",
                "initiator_role": "tech-lead"
            }
        ]
        
        store._upsert_initiators(session, initiators)
        
        # Verify the correct query was called
        session.run.assert_called()
        call_args = session.run.call_args
        query = call_args[0][0]
        params = call_args[1]
        
        assert "MERGE (d)-[:HAS_INITIATOR]->(i:Initiator" in query
        assert params["initiators"] == initiators

    def test_ingest_integration(self, neo4j_store, sample_hydrated_clusters):
        """Test full ingestion integration."""
        store, driver, session = neo4j_store
        
        # Mock successful operations
        session.run.return_value = Mock()
        
        store.ingest(sample_hydrated_clusters)
        
        # Verify all expected operations were called
        call_count = session.run.call_count
        
        # Should call: constraints + clusters + decisions + links + delete_edges + 
        # topics + constraints + entities + facts + initiators
        assert call_count >= 10
        
        # Verify session was properly managed
        driver.session.assert_called_once_with(database="neo4j")

    @patch('decision_graph.backends.neo4j.stores.breakdown_hydrated_clusters')
    def test_ingest_uses_breakdown_function(self, mock_breakdown, neo4j_store, sample_hydrated_clusters):
        """Test that ingest uses breakdown_hydrated_clusters function."""
        store, driver, session = neo4j_store
        
        # Mock breakdown to return empty arrays
        mock_breakdown.return_value = {
            "clusters": [],
            "decisions": [],
            "decision_cluster": [],
            "decision_topics": [],
            "decision_constraints": [],
            "decision_entities": [],
            "decision_facts": [],
            "decision_initiators": [],
        }
        
        session.run.return_value = Mock()
        
        store.ingest(sample_hydrated_clusters)
        
        # Verify breakdown function was called
        mock_breakdown.assert_called_once_with(sample_hydrated_clusters)

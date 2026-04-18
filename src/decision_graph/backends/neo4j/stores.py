"""Neo4j implementation of the GraphStore protocol."""

import logging
from typing import Any, Dict, List, Optional

from decision_graph.core.interfaces import GraphStore
from decision_graph.ingestion import breakdown_hydrated_clusters

_logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph store for decision data materialization."""

    def __init__(self, driver, database: str = "neo4j"):
        """
        Initialize Neo4j graph store.
        
        Args:
            driver: Neo4j driver instance
            database: Neo4j database name (default: "neo4j")
        """
        self._driver = driver
        self._database = database

    def ingest(self, hydrated_clusters: List[Dict[str, Any]]) -> None:
        """
        Ingest hydrated clusters into Neo4j graph database.
        
        Args:
            hydrated_clusters: List of hydrated cluster dictionaries
        """
        if not hydrated_clusters:
            _logger.info("No hydrated clusters to ingest")
            return

        try:
            arrays = breakdown_hydrated_clusters(hydrated_clusters)
            
            _logger.info(
                "Neo4j ingestion: %d clusters, %d decisions, %d topics, %d constraints, "
                "%d entities, %d facts, %d initiators",
                len(arrays["clusters"]),
                len(arrays["decisions"]),
                len(arrays["decision_topics"]),
                len(arrays["decision_constraints"]),
                len(arrays["decision_entities"]),
                len(arrays["decision_facts"]),
                len(arrays["decision_initiators"]),
            )

            with self._driver.session(database=self._database) as session:
                # Create constraints for uniqueness
                self._create_constraints(session)
                
                # Upsert nodes and relationships
                self._upsert_clusters(session, arrays["clusters"])
                self._upsert_decisions(session, arrays["decisions"])
                self._link_clusters_decisions(session, arrays["decision_cluster"])
                
                # Delete existing enrichment edges for these decisions
                decision_ids = [d["decision_id"] for d in arrays["decisions"]]
                self._delete_enrichment_edges(session, decision_ids)
                
                # Create enrichment relationships
                self._upsert_topics(session, arrays["decision_topics"])
                self._upsert_constraints(session, arrays["decision_constraints"])
                self._upsert_entities(session, arrays["decision_entities"])
                self._upsert_facts(session, arrays["decision_facts"])
                self._upsert_initiators(session, arrays["decision_initiators"])

            counts = {k: len(v) for k, v in arrays.items()}
            _logger.info("Neo4j ingestion complete: %s", counts)

        except Exception as e:
            _logger.exception("Neo4j ingestion failed: %s", str(e))
            raise

    def _create_constraints(self, session) -> None:
        """Create uniqueness constraints for nodes."""
        constraints = [
            "CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE",
            "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE",
            "CREATE CONSTRAINT topic_unique IF NOT EXISTS FOR (t:Topic) REQUIRE (t.decision_id, t.topic) IS NODE KEY",
            "CREATE CONSTRAINT constraint_unique IF NOT EXISTS FOR (c:Constraint) REQUIRE (c.decision_id, c.text) IS NODE KEY",
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.decision_id, e.name, e.type) IS NODE KEY",
            "CREATE CONSTRAINT fact_unique IF NOT EXISTS FOR (f:Fact) REQUIRE (f.decision_id, f.k) IS NODE KEY",
            "CREATE CONSTRAINT initiator_unique IF NOT EXISTS FOR (i:Initiator) REQUIRE (i.decision_id, i.initiator_name) IS NODE KEY",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                _logger.debug("Constraint creation failed (may already exist): %s", str(e))

    def _upsert_clusters(self, session, clusters: List[Dict]) -> None:
        """Upsert cluster nodes."""
        query = """
        UNWIND $clusters AS cluster
        MERGE (c:Cluster {cluster_id: cluster.cluster_id})
        SET c.gid = cluster.gid,
            c.primary_subject = cluster.primary_subject,
            c.created_at = cluster.created_at,
            c.last_updated_at = cluster.last_updated_at,
            c.rolling_summary = cluster.rolling_summary,
            c.decision_count = cluster.decision_count
        """
        session.run(query, clusters=clusters)

    def _upsert_decisions(self, session, decisions: List[Dict]) -> None:
        """Upsert decision nodes."""
        query = """
        UNWIND $decisions AS decision
        MERGE (d:Decision {decision_id: decision.decision_id})
        SET d.gid = decision.gid,
            d.cid = decision.cid,
            d.trace_id = decision.trace_id,
            d.linked_at = decision.linked_at,
            d.linked_from = decision.linked_from
        """
        session.run(query, decisions=decisions)

    def _link_clusters_decisions(self, session, decision_cluster: List[Dict]) -> None:
        """Create CONTAINS relationships between clusters and decisions."""
        query = """
        UNWIND $decision_cluster AS dc
        MATCH (c:Cluster {cluster_id: dc.cluster_id})
        MATCH (d:Decision {decision_id: dc.decision_id})
        MERGE (c)-[:CONTAINS]->(d)
        """
        session.run(query, decision_cluster=decision_cluster)

    def _delete_enrichment_edges(self, session, decision_ids: List[str]) -> None:
        """Delete existing enrichment edges for given decisions."""
        if not decision_ids:
            return
            
        query = """
        UNWIND $decision_ids AS did
        MATCH (d:Decision {decision_id: did})
        OPTIONAL MATCH (d)-[r]->(n)
        WHERE type(r) IN ['HAS_TOPIC', 'HAS_CONSTRAINT', 'HAS_ENTITY', 'HAS_FACT', 'HAS_INITIATOR']
        DELETE r, n
        """
        session.run(query, decision_ids=decision_ids)

    def _upsert_topics(self, session, topics: List[Dict]) -> None:
        """Upsert topic nodes and relationships."""
        query = """
        UNWIND $topics AS topic
        MATCH (d:Decision {decision_id: topic.decision_id})
        MERGE (d)-[:HAS_TOPIC]->(t:Topic {decision_id: topic.decision_id, topic: topic.topic})
        SET t.gid = topic.gid
        """
        session.run(query, topics=topics)

    def _upsert_constraints(self, session, constraints: List[Dict]) -> None:
        """Upsert constraint nodes and relationships."""
        query = """
        UNWIND $constraints AS constraint
        MATCH (d:Decision {decision_id: constraint.decision_id})
        MERGE (d)-[:HAS_CONSTRAINT]->(c:Constraint {decision_id: constraint.decision_id, text: constraint.text})
        SET c.gid = constraint.gid
        """
        session.run(query, constraints=constraints)

    def _upsert_entities(self, session, entities: List[Dict]) -> None:
        """Upsert entity nodes and relationships."""
        query = """
        UNWIND $entities AS entity
        MATCH (d:Decision {decision_id: entity.decision_id})
        MERGE (d)-[:HAS_ENTITY]->(e:Entity {decision_id: entity.decision_id, name: entity.name, type: entity.type})
        SET e.gid = entity.gid,
            e.display_name = entity.display_name
        """
        session.run(query, entities=entities)

    def _upsert_facts(self, session, facts: List[Dict]) -> None:
        """Upsert fact nodes and relationships."""
        query = """
        UNWIND $facts AS fact
        MATCH (d:Decision {decision_id: fact.decision_id})
        MERGE (d)-[:HAS_FACT]->(f:Fact {decision_id: fact.decision_id, k: fact.k})
        SET f.v = fact.v
        """
        session.run(query, facts=facts)

    def _upsert_initiators(self, session, initiators: List[Dict]) -> None:
        """Upsert initiator nodes and relationships."""
        query = """
        UNWIND $initiators AS initiator
        MATCH (d:Decision {decision_id: initiator.decision_id})
        MERGE (d)-[:HAS_INITIATOR]->(i:Initiator {decision_id: initiator.decision_id, initiator_name: initiator.initiator_name})
        SET i.gid = initiator.gid,
            i.display_name = initiator.display_name,
            i.initiator_role = initiator.initiator_role
        """
        session.run(query, initiators=initiators)

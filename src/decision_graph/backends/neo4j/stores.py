"""Neo4j implementation of the GraphStore protocol."""

import logging
from typing import Any, Dict, List

from decision_graph.core.interfaces import GraphStore
from decision_graph.ingestion import breakdown_hydrated_clusters

_logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed GraphStore.

    Materialises hydrated clusters into a property graph using the relationship
    vocabulary defined in ``context_graph/registry.py`` so that writes are
    immediately queryable by ``Neo4jGraphReader`` and ``ContextGraphService``.

    Node model
    ----------
    Shared concept nodes (MERGE on semantic identity):
      * Topic  {name}         — normalised topic text
      * Entity {name, type}   — technology / person / org …
        - Initiators → (d)-[:INITIATED_BY]->(Entity {type:'Person'})
        - Mentions   → (d)-[:MENTIONS_ENTITY]->(Entity)

    Per-decision nodes (owned by exactly one Decision):
      * Constraint {decision_id, text}
      * Fact       {decision_id, k}
    """

    def __init__(self, driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    # ------------------------------------------------------------------
    # Public protocol method
    # ------------------------------------------------------------------

    def ingest(self, hydrated_clusters: List[Dict[str, Any]]) -> None:
        if not hydrated_clusters:
            _logger.info("No hydrated clusters to ingest")
            return

        try:
            arrays = breakdown_hydrated_clusters(hydrated_clusters)

            _logger.info(
                "Neo4j ingestion: %d clusters, %d decisions, %d topics, "
                "%d constraints, %d entities, %d facts, %d initiators",
                len(arrays["clusters"]),
                len(arrays["decisions"]),
                len(arrays["decision_topics"]),
                len(arrays["decision_constraints"]),
                len(arrays["decision_entities"]),
                len(arrays["decision_facts"]),
                len(arrays["decision_initiators"]),
            )

            with self._driver.session(database=self._database) as session:
                self._create_constraints(session)

                self._upsert_clusters(session, arrays["clusters"])
                self._upsert_decisions(session, arrays["decisions"])
                self._link_clusters_decisions(session, arrays["decision_cluster"])

                decision_ids = [d["decision_id"] for d in arrays["decisions"]]
                self._delete_enrichment_edges(session, decision_ids)

                self._upsert_topics(session, arrays["decision_topics"])
                self._upsert_constraints(session, arrays["decision_constraints"])
                self._upsert_entities(session, arrays["decision_entities"])
                self._upsert_facts(session, arrays["decision_facts"])
                self._upsert_initiators(session, arrays["decision_initiators"])

            counts = {k: len(v) for k, v in arrays.items()}
            _logger.info("Neo4j ingestion complete: %s", counts)

        except Exception:
            _logger.exception("Neo4j ingestion failed")
            raise

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _create_constraints(self, session) -> None:
        constraints = [
            "CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS "
            "FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE",
            "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS "
            "FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE",
            "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS "
            "FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY",
            "CREATE CONSTRAINT constraint_unique IF NOT EXISTS "
            "FOR (c:Constraint) REQUIRE (c.decision_id, c.text) IS NODE KEY",
            "CREATE CONSTRAINT fact_unique IF NOT EXISTS "
            "FOR (f:Fact) REQUIRE (f.decision_id, f.k) IS NODE KEY",
        ]
        for cypher in constraints:
            try:
                session.run(cypher)
            except Exception as exc:
                _logger.debug("Constraint skipped (may already exist): %s", exc)

    # ------------------------------------------------------------------
    # Core graph nodes
    # ------------------------------------------------------------------

    def _upsert_clusters(self, session, clusters: List[Dict]) -> None:
        if not clusters:
            return
        session.run(
            """
            UNWIND $clusters AS cluster
            MERGE (c:Cluster {cluster_id: cluster.cluster_id})
            SET c.gid             = cluster.gid,
                c.primary_subject = cluster.primary_subject,
                c.created_at      = cluster.created_at,
                c.last_updated_at = cluster.last_updated_at,
                c.rolling_summary = cluster.rolling_summary,
                c.decision_count  = cluster.decision_count
            """,
            clusters=clusters,
        )

    def _upsert_decisions(self, session, decisions: List[Dict]) -> None:
        if not decisions:
            return
        session.run(
            """
            UNWIND $decisions AS decision
            MERGE (d:Decision {decision_id: decision.decision_id})
            SET d.gid         = decision.gid,
                d.cid         = decision.cid,
                d.trace_id    = decision.trace_id,
                d.linked_at   = decision.linked_at,
                d.linked_from = decision.linked_from
            """,
            decisions=decisions,
        )

    def _link_clusters_decisions(self, session, decision_cluster: List[Dict]) -> None:
        if not decision_cluster:
            return
        session.run(
            """
            UNWIND $decision_cluster AS dc
            MATCH (c:Cluster  {cluster_id:  dc.cluster_id})
            MATCH (d:Decision {decision_id: dc.decision_id})
            MERGE (c)-[:HAS_DECISION]->(d)
            """,
            decision_cluster=decision_cluster,
        )

    # ------------------------------------------------------------------
    # Enrichment edge cleanup
    # ------------------------------------------------------------------

    def _delete_enrichment_edges(self, session, decision_ids: List[str]) -> None:
        if not decision_ids:
            return
        session.run(
            """
            UNWIND $decision_ids AS did
            MATCH (d:Decision {decision_id: did})
            OPTIONAL MATCH (d)-[r:HAS_TOPIC|MENTIONS_ENTITY|INITIATED_BY]->()
            DELETE r
            """,
            decision_ids=decision_ids,
        )
        session.run(
            """
            UNWIND $decision_ids AS did
            MATCH (d:Decision {decision_id: did})
            OPTIONAL MATCH (d)-[r:HAS_CONSTRAINT|HAS_FACT]->(n)
            DELETE r, n
            """,
            decision_ids=decision_ids,
        )

    # ------------------------------------------------------------------
    # Enrichment upserts — shared concept nodes
    # ------------------------------------------------------------------

    def _upsert_topics(self, session, topics: List[Dict]) -> None:
        if not topics:
            return
        session.run(
            """
            UNWIND $topics AS topic
            MATCH (d:Decision {decision_id: topic.decision_id})
            MERGE (t:Topic {name: topic.topic})
            MERGE (d)-[:HAS_TOPIC]->(t)
            """,
            topics=topics,
        )

    def _upsert_entities(self, session, entities: List[Dict]) -> None:
        if not entities:
            return
        session.run(
            """
            UNWIND $entities AS entity
            MATCH (d:Decision {decision_id: entity.decision_id})
            MERGE (e:Entity {name: entity.name, type: entity.type})
            SET e.display_name = entity.display_name
            MERGE (d)-[:MENTIONS_ENTITY]->(e)
            """,
            entities=entities,
        )

    def _upsert_initiators(self, session, initiators: List[Dict]) -> None:
        if not initiators:
            return
        session.run(
            """
            UNWIND $initiators AS initiator
            MATCH (d:Decision {decision_id: initiator.decision_id})
            MERGE (e:Entity {name: initiator.initiator_name, type: 'Person'})
            SET e.display_name = initiator.display_name,
                e.role         = initiator.initiator_role
            MERGE (d)-[:INITIATED_BY]->(e)
            """,
            initiators=initiators,
        )

    # ------------------------------------------------------------------
    # Enrichment upserts — per-decision owned nodes
    # ------------------------------------------------------------------

    def _upsert_constraints(self, session, constraints: List[Dict]) -> None:
        if not constraints:
            return
        session.run(
            """
            UNWIND $constraints AS constraint
            MATCH (d:Decision {decision_id: constraint.decision_id})
            MERGE (c:Constraint {decision_id: constraint.decision_id,
                                  text:        constraint.text})
            SET c.gid = constraint.gid
            MERGE (d)-[:HAS_CONSTRAINT]->(c)
            """,
            constraints=constraints,
        )

    def _upsert_facts(self, session, facts: List[Dict]) -> None:
        if not facts:
            return
        session.run(
            """
            UNWIND $facts AS fact
            MATCH (d:Decision {decision_id: fact.decision_id})
            MERGE (f:Fact {decision_id: fact.decision_id, k: fact.k})
            SET f.v = fact.v
            MERGE (d)-[:HAS_FACT]->(f)
            """,
            facts=facts,
        )

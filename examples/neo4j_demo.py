"""
Neo4j GraphStore Demo

Shows how to use the Neo4j backend with the decision graph pipeline.
Requires a running Neo4j instance.
"""

import asyncio
import logging
from neo4j import GraphDatabase

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.backends.memory.stores import InMemoryVectorIndex
from decision_graph.backends.neo4j import Neo4jGraphStore
from decision_graph.decision_trace_pipeline import DecisionTracePipeline
from decision_graph.llm import LiteLLMAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_neo4j_driver(uri="bolt://localhost:7687", user="neo4j", password="password"):
    """Create Neo4j driver connection."""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return None


async def demo_neo4j_graphstore():
    """Demo Neo4j GraphStore integration."""
    
    # Create Neo4j driver (adjust credentials as needed)
    driver = create_neo4j_driver()
    if not driver:
        logger.error("Cannot proceed without Neo4j connection")
        return
    
    try:
        # Setup backend and pipeline
        backend = InMemoryBackend()
        neo4j_graph_store = Neo4jGraphStore(driver)
        
        pipeline = DecisionTracePipeline(
            backend=backend,
            executor=LiteLLMAdapter(),
            vector_index=InMemoryVectorIndex(),
            graph_store=neo4j_graph_store,
        )
        
        # Sample conversation text
        conv_text = """
        Alice: We need to decide on the API architecture for the new service.
        Bob: I think we should go with GraphQL instead of REST.
        Alice: Good point. GraphQL will give us better flexibility.
        Bob: Let's also consider using microservices.
        Alice: Agreed. We'll implement GraphQL with microservices architecture.
        """
        
        # Run the pipeline
        decisions = await pipeline.run_from_text(
            conv_text=conv_text,
            conv_id="architecture-discussion",
            gid="engineering-team",
            updated_at=1705334400.0,
            summary_pid="summary_architecture",
            query_gids=["engineering-team"],
        )
        
        logger.info(f"Extracted {len(decisions)} decisions")
        logger.info("Data successfully ingested into Neo4j!")
        
        # Query Neo4j to verify data
        with driver.session() as session:
            result = session.run("MATCH (c:Cluster) RETURN count(c) as cluster_count")
            cluster_count = result.single()["cluster_count"]
            logger.info(f"Clusters in Neo4j: {cluster_count}")
            
            result = session.run("MATCH (d:Decision) RETURN count(d) as decision_count")
            decision_count = result.single()["decision_count"]
            logger.info(f"Decisions in Neo4j: {decision_count}")
            
    finally:
        driver.close()


if __name__ == "__main__":
    asyncio.run(demo_neo4j_graphstore())

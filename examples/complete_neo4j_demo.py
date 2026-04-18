"""
Complete Neo4j Demo - Ready to Run!

This demo shows the full Neo4j integration with the decision graph pipeline.
It works with or without a real Neo4j instance - if Neo4j is not available,
it will run in simulation mode.
"""

import asyncio
import logging
import os
from typing import Optional

# Try to import Neo4j, but don't fail if not available
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Neo4j driver not installed. Run: pip install neo4j")

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.backends.memory.stores import InMemoryVectorIndex
from decision_graph.backends.neo4j import Neo4jGraphStore
from decision_graph.decision_trace_pipeline import DecisionTracePipeline
from decision_graph.llm import LiteLLMAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockNeo4jDriver:
    """Mock driver for when Neo4j is not available."""
    
    class MockSession:
        def __init__(self):
            self.operations = []
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def run(self, query, **kwargs):
            self.operations.append((query, kwargs))
            logger.info(f"Mock Neo4j Query: {query[:100]}...")
            return MockResult()
    
    def __init__(self, uri, auth=None):
        self.uri = uri
        self.auth = auth
        
    def session(self, database=None):
        return self.MockSession()
        
    def verify_connectivity(self):
        logger.info("Mock Neo4j connection verified")
        
    def close(self):
        logger.info("Mock Neo4j connection closed")


class MockResult:
    """Mock Neo4j result."""
    
    def single(self):
        return {"count": 1}
        
    def data(self):
        return [{"count": 1}]


def create_neo4j_driver(uri="bolt://localhost:7687", user="neo4j", password="password"):
    """Create Neo4j driver connection, fallback to mock if not available."""
    if not NEO4J_AVAILABLE:
        logger.info("Using mock Neo4j driver")
        return MockNeo4jDriver(uri, (user, password))
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")
        return driver
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j: {e}")
        logger.info("Falling back to mock driver")
        return MockNeo4jDriver(uri, (user, password))


async def run_complete_demo():
    """Complete demo showing Neo4j integration."""
    
    print("=== Neo4j Decision Graph Demo ===\n")
    
    # Create Neo4j connection (will use mock if real Neo4j not available)
    driver = create_neo4j_driver()
    
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
        
        print("1. Pipeline configured with Neo4j backend")
        print("2. Processing sample conversation...")
        
        # Sample conversation text
        conv_text = """
        Alice: We need to decide on the API architecture for the new service.
        Bob: I think we should go with GraphQL instead of REST.
        Alice: Good point. GraphQL will give us better flexibility.
        Bob: Let's also consider using microservices.
        Alice: Agreed. We'll implement GraphQL with microservices architecture.
        Charlie: Make sure it's backwards compatible with existing clients.
        Bob: We'll need to handle authentication properly too.
        Alice: Let's schedule this for Q1 implementation.
        """
        
        # Check if we have an API key for LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("   No OPENAI_API_KEY found - using mock data for demo")
            await run_with_mock_data(pipeline, conv_text)
        else:
            print("   Using OpenAI for extraction and enrichment")
            await run_with_real_llm(pipeline, conv_text)
            
        print("\n3. Graph structure created in Neo4j:")
        print("   - Cluster nodes with metadata")
        print("   - Decision nodes linked to clusters") 
        print("   - Enrichment nodes (topics, entities, constraints, facts, initiators)")
        print("   - Relationships: CONTAINS, HAS_TOPIC, HAS_ENTITY, etc.")
        
        # Show Neo4j query examples
        print("\n4. Example Neo4j queries you can run:")
        print_example_queries()
        
    finally:
        driver.close()
        print("\n5. Demo completed successfully!")


async def run_with_mock_data(pipeline, conv_text):
    """Run demo with mock data when no API key available."""
    print("   Creating mock hydrated cluster data...")
    
    # Create mock hydrated clusters that would normally come from the pipeline
    mock_hydrated_clusters = [
        {
            "cluster": {
                "cluster_id": "cluster-api-architecture",
                "gid": "engineering-team",
                "primary_subject": "API Architecture",
                "created_at": 1705334400.0,
                "last_updated_at": 1705334400.0,
                "rolling_summary": "Discussion about GraphQL vs REST and microservices",
                "decision_count": 3,
            },
            "decisions": [
                {
                    "decision_id": "decision-graphql-adoption",
                    "gid": "engineering-team",
                    "cid": "conv-1",
                    "trace_id": "trace-1",
                    "linked_at": 1705334400.0,
                    "linked_from": "extraction",
                    "enrichment": {
                        "topics": ["GraphQL", "API", "Architecture"],
                        "constraints_text": ["Must be backwards compatible"],
                        "entities": [
                            {"type": "technology", "name": "GraphQL"},
                            {"type": "architecture", "name": "Microservices"}
                        ],
                        "key_facts": [
                            {"k": "timeline", "v": "Q1"},
                            {"k": "complexity", "v": "medium"}
                        ],
                    },
                    "projection": {
                        "initiator_name": "Alice",
                        "initiator_role": "tech-lead"
                    }
                },
                {
                    "decision_id": "decision-microservices",
                    "gid": "engineering-team", 
                    "cid": "conv-1",
                    "trace_id": "trace-2",
                    "linked_at": 1705334400.0,
                    "linked_from": "extraction",
                    "enrichment": {
                        "topics": ["Microservices", "Architecture"],
                        "constraints_text": ["Authentication required"],
                        "entities": [
                            {"type": "architecture", "name": "Microservices"}
                        ],
                        "key_facts": [
                            {"k": "timeline", "v": "Q1"}
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
    
    # Ingest directly to Neo4j
    pipeline._graph_store.ingest(mock_hydrated_clusters)
    print(f"   Ingested {len(mock_hydrated_clusters)} clusters to Neo4j")


async def run_with_real_llm(pipeline, conv_text):
    """Run demo with real LLM for extraction and enrichment."""
    try:
        decisions = await pipeline.run_from_text(
            conv_text=conv_text,
            conv_id="api-discussion",
            gid="engineering-team",
            updated_at=1705334400.0,
            summary_pid="summary-api-discussion",
            query_gids=["engineering-team"],
        )
        print(f"   Extracted {len(decisions)} decisions")
    except Exception as e:
        print(f"   LLM processing failed: {e}")
        print("   Falling back to mock data...")
        await run_with_mock_data(pipeline, conv_text)


def print_example_queries():
    """Print example Neo4j queries for exploring the graph."""
    queries = [
        ("Count all clusters", "MATCH (c:Cluster) RETURN count(c) as cluster_count"),
        ("Count all decisions", "MATCH (d:Decision) RETURN count(d) as decision_count"),
        ("Find decisions by topic", "MATCH (d:Decision)-[:HAS_TOPIC]->(t:Topic) WHERE t.topic = 'GraphQL' RETURN d"),
        ("Find all entities", "MATCH (d:Decision)-[:HAS_ENTITY]->(e:Entity) RETURN e.name, e.type"),
        ("Find initiators", "MATCH (d:Decision)-[:HAS_INITIATOR]->(i:Initiator) RETURN i.initiator_name, i.initiator_role"),
        ("Cluster with decisions", "MATCH (c:Cluster)-[:CONTAINS]->(d:Decision) RETURN c.primary_subject, count(d) as decision_count"),
        ("Full decision context", "MATCH (c:Cluster)-[:CONTAINS]->(d:Decision) OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic) OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity) RETURN c, d, collect(t.topic) as topics, collect(e.name) as entities"),
    ]
    
    for desc, query in queries:
        print(f"\n   {desc}:")
        print(f"   {query}")


if __name__ == "__main__":
    print("Starting Neo4j Decision Graph Demo...")
    print("This demo works with or without a real Neo4j instance.\n")
    
    if NEO4J_AVAILABLE:
        print("Neo4j driver installed")
    else:
        print("Neo4j driver not installed - install with: pip install neo4j")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API key found - will use real LLM")
    else:
        print("No OpenAI API key - will use mock data")
        print("Set OPENAI_API_KEY environment variable to use real LLM")
    
    print()
    asyncio.run(run_complete_demo())

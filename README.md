# py-context-graph

Extract, enrich, cluster, and query decisions from unstructured conversations using LLMs.

[![Tests](https://github.com/ResearchifyLabs/py-context-graph/actions/workflows/test.yml/badge.svg)](https://github.com/ResearchifyLabs/py-context-graph/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/py-context-graph)](https://pypi.org/project/py-context-graph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

**py-context-graph** turns messy conversation text (meeting notes, Slack threads, standups) into a structured decision graph. It uses LLMs to:

1. **Extract** decision items from text (what was decided, by whom, about what)
2. **Deduplicate** near-identical decisions across conversations
3. **Enrich** each decision with structured metadata (topics, entities, constraints, key facts)
4. **Cluster** related decisions across conversations into coherent themes
5. **Materialize** the result into a queryable graph

```
Text → Extract (LLM) → Persist → Deduplicate → Enrich (LLM) → Cluster → Graph
```

## Install

```bash
pip install py-context-graph
```

With optional backends:

```bash
pip install py-context-graph[all]       # LiteLLM + Firestore + in-memory vector index
pip install py-context-graph[llm]       # LiteLLM adapter only
pip install py-context-graph[firestore] # Google Cloud Firestore backend
pip install py-context-graph[memory]    # In-memory TF-IDF vector index (pandas)
```

## Quick start

```python
import asyncio
from decision_graph import DecisionGraph, LiteLLMAdapter
from decision_graph.backends.memory import InMemoryBackend
from decision_graph.backends.memory.stores import InMemoryGraphStore, InMemoryVectorIndex
from decision_graph.decision_trace_pipeline import DecisionTracePipeline

backend = InMemoryBackend()
pipeline = DecisionTracePipeline(
    backend=backend,
    executor=LiteLLMAdapter(),
    vector_index=InMemoryVectorIndex(),
    graph_store=InMemoryGraphStore(),
)

async def main():
    # Process a conversation
    decisions = await pipeline.run_from_text(
        conv_text="Alice: We decided to switch from REST to GraphQL for the new API...",
        conv_id="standup-2024-01-15",
        gid="engineering-team",
        updated_at=1705334400.0,
        summary_pid="summary_standup-2024-01-15",
        query_gids=["engineering-team"],
    )

    # Query the results
    dg = DecisionGraph(backend=backend, executor=LiteLLMAdapter())
    service = dg.graph_service()
    result = await service.get_enrichments_and_projections_joined(
        group_ids=["engineering-team"]
    )
    print(f"Found {result['total_joined']} enriched decisions")

asyncio.run(main())
```

## Key concepts

### The four protocols

py-context-graph is built around pluggable interfaces. You only implement what you need:

| Protocol | Purpose | Bundled implementations |
|----------|---------|------------------------|
| **`StorageBackend`** | Groups 4 document stores (enrichments, projections, clusters, links) | `InMemoryBackend`, `FirestoreBackend` |
| **`LLMAdapter`** | Executes LLM calls for extraction and enrichment | `LiteLLMAdapter` (supports OpenAI, Anthropic, and any LiteLLM provider) |
| **`VectorIndex`** | Similarity search for cross-conversation clustering | `InMemoryVectorIndex` (TF-IDF + cosine) |
| **`GraphStore`** | Write-only sync of hydrated clusters to a graph DB | `InMemoryGraphStore`, `NullGraphStore` |

### DecisionGraph facade

The main entry point. Wire a backend and LLM adapter, then access services:

```python
from decision_graph import DecisionGraph

dg = DecisionGraph(backend=my_backend, executor=my_llm)
service = dg.graph_service()        # query enrichments, projections, clusters
retrieval = dg.retrieval()          # filtered queries over enrichments
clusterer = dg.cluster_service()    # cluster management
```

### DecisionTracePipeline

The end-to-end processing pipeline. Feed it text, get structured decisions:

```python
from decision_graph.decision_trace_pipeline import DecisionTracePipeline

pipeline = DecisionTracePipeline(
    backend=backend,
    executor=llm_adapter,
    vector_index=vector_index,    # optional, enables cross-conversation clustering
    graph_store=graph_store,      # use NullGraphStore() to skip graph materialization
)

# From raw text
decisions = await pipeline.run_from_text(conv_text=text, conv_id="c1", gid="g1", ...)

# From pre-extracted decision items
decisions = await pipeline.run(decision_items=[...], conv_id="c1", gid="g1", ...)
```

### Context Graph (query layer)

For querying the materialized graph (requires a `GraphReader` implementation, e.g. Neo4j):

```python
from decision_graph.context_graph.service import ContextGraphService

ctx = ContextGraphService(reader=my_graph_reader)
result = await ctx.query(text="What decisions were made about the API?", mode="chat")
```

## Bring your own backend

Implement `StorageBackend` to use any database:

```python
from decision_graph.core.registry import StorageBackend
from decision_graph.core.interfaces import EnrichmentStore, ProjectionStore, ClusterStore, LinkStore

class PostgresBackend(StorageBackend):
    def enrichment_store(self) -> EnrichmentStore: ...
    def projection_store(self) -> ProjectionStore: ...
    def cluster_store(self) -> ClusterStore: ...
    def link_store(self) -> LinkStore: ...
```

Each store protocol is defined in `decision_graph.core.interfaces` with clear method signatures.

## Bring your own LLM

Implement the `LLMAdapter` protocol:

```python
from decision_graph.core.interfaces import LLMAdapter

class MyLLMAdapter(LLMAdapter):
    async def execute_async(self, model_config, data, additional_data=None):
        # Call your LLM, return parsed result
        ...
```

## Examples

See the [`examples/`](examples/) directory for a complete demo that:
- Processes sample conversation files through the full pipeline
- Shows live pipeline progress in the browser as conversations are processed
- Generates an interactive HTML viewer with Insights dashboard, Cluster Board, Timeline, Person x Cluster matrix, and Explore (force-directed graph) views

```bash
cd examples
pip install py-context-graph[all]
export OPENAI_API_KEY=sk-...   # or any LiteLLM-supported provider
python run.py                  # opens browser automatically
```

The viewer opens immediately and shows pipeline progress in real time. When processing completes, the dashboard appears with all visualizations.

Options:
- `python run.py --port 9000` — use a different port
- `python run.py --no-browser` — don't auto-open the browser
- `python run.py my_notes.txt` — process your own conversation files

## Project structure

```
src/decision_graph/
├── __init__.py                  # Public API: DecisionGraph, LLMAdapter, LLMConfig, LiteLLMAdapter
├── graph.py                     # DecisionGraph facade
├── decision_trace_pipeline.py   # End-to-end pipeline
├── extraction_service.py        # LLM-based decision extraction
├── enrichment_service.py        # LLM-based decision enrichment
├── clustering_service.py        # Decision clustering
├── retrieval.py                 # Query/filter over enrichments
├── context_retrieval.py         # Vector-based context retrieval
├── services.py                  # DecisionGraphService (joins, hydration)
├── ingestion.py                 # Graph materialization helpers
├── visualization.py             # vis.js graph builder
├── markdown_chunker.py          # Split markdown by headings
├── core/
│   ├── interfaces.py            # Protocol definitions
│   ├── registry.py              # StorageBackend ABC
│   ├── domain.py                # Pydantic models
│   ├── config.py                # LLMConfig
│   └── matching.py              # Dedup, scoring, similarity
├── llm/
│   └── litellm_adapter.py       # LiteLLM-based LLMAdapter
├── backends/
│   ├── memory/                  # In-memory stores + TF-IDF vector index
│   └── firestore/               # Google Cloud Firestore stores
├── context_graph/               # Graph query layer (planner, templates, post-processing)
└── prompts/                     # LLM prompt templates
```

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/ResearchifyLabs/py-context-graph.git
cd py-context-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src:tests python -m unittest discover -s tests -p 'test_*.py'
```

## License

[MIT](LICENSE)

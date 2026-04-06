# Examples

Standalone demo that processes sample conversations through the full decision graph pipeline and visualizes the results as interactive cards and a force-directed graph.

## What it does

1. Reads multiple conversation text files
2. Extracts structured decision items from each conversation via LLM
3. Runs the full pipeline: **persist -> deduplicate -> enrich -> cluster**
4. Uses an in-memory TF-IDF vector index for cross-conversation similarity matching
5. Outputs decision cards and an interactive graph visualization

## Prerequisites

- Python 3.10+
- `pip install py-context-graph[all]` (or install from repo root: `pip install -e ".[all]"`)
- An LLM API key for your chosen provider (defaults to OpenAI)

## Run

```bash
export OPENAI_API_KEY=sk-...
cd examples
python run.py
```

Or with custom files:

```bash
python run.py path/to/conversation1.txt path/to/conversation2.txt
```

## Using a different model

The demo defaults to `gpt-4.1-mini` (OpenAI). You can switch to any LiteLLM-supported provider via the `--model` flag or the `DECISION_GRAPH_MODEL` environment variable:

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --model anthropic/claude-3.5-sonnet

# Google Vertex AI
export GEMINI_API_KEY=...
python run.py --model gemini/gemini-2.0-flash

# Or set the model via environment variable
export DECISION_GRAPH_MODEL=anthropic/claude-3.5-sonnet
python run.py
```

The model string follows [LiteLLM's naming convention](https://docs.litellm.ai/docs/providers) (`provider/model_name`). OpenAI models can omit the prefix.

This processes all `sample_conversation_*.txt` files sequentially, stores results in the configured backends (in-memory by default), and serves them via API endpoints that `viewer.html` fetches from.

## View

The viewer opens automatically at http://localhost:8888/viewer.html when you run `python run.py`. It fetches data from the backend via `/api/output` and `/api/graph-data` endpoints, and polls `/api/status` for live pipeline progress.

### Decisions tab
Card view of all extracted decisions showing subject, type, status, initiator, topics, entities, key facts, and constraints.

### Graph tab
Interactive force-directed graph (vis.js) with color-coded node types:

| Node | Shape | Color |
|------|-------|-------|
| Cluster | Diamond | Blue |
| Decision | Dot | Orange |
| Person | Star | Purple |
| Topic | Triangle | Green |
| Entity | Square | Red |
| Fact | Box | Yellow |
| Constraint | Hexagon | Teal |

Click any node to see full details in the right panel. Connected nodes are clickable for navigation.

## Sample conversations

Three conversations simulate a team standup over multiple days with overlapping topics (login bugs, dashboard analytics, bulk export feature, payment service monitoring). This ensures cross-conversation clustering produces meaningful results.

| File | Content |
|------|---------|
| `sample_conversation_1.txt` | Initial standup: bugs reported, PR created, feature requested |
| `sample_conversation_2.txt` | Follow-up: login fixed, PR reviewed, requirements doc done |
| `sample_conversation_3.txt` | Later follow-up: demo scheduled, API designed, latency root cause found |

## Adding your own conversations

Drop any `.txt` file named `sample_conversation_*.txt` in this directory and re-run. Files are processed in alphabetical order. Conversations with overlapping topics will be clustered together.

You can also pass `.md` files -- they will be chunked by `##` headings, with each section processed as a separate conversation.

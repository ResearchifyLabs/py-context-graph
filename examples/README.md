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
- An OpenAI-compatible LLM service (set `OPENAI_API_KEY` or configure LiteLLM)

## Run

```bash
cd examples
python run.py
```

Or with custom files:

```bash
python run.py path/to/conversation1.txt path/to/conversation2.txt
```

This processes all `sample_conversation_*.txt` files sequentially and writes into `generated/`:
- `generated/output.json` -- projections + enrichments joined
- `generated/graph_data.json` -- nodes and edges for graph visualization

## View

```bash
cd examples
python -m http.server 8888
```

Open http://localhost:8888/viewer.html

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

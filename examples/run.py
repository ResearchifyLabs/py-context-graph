"""
Decision Graph Demo

Reads sample conversations or markdown documents, extracts decisions via LLM,
runs them through the full decision_graph pipeline (extract -> persist -> dedupe
-> enrich -> cluster), and writes output.json + graph_data.json for the HTML viewer.

Supports two input modes:
  - .txt files: each file is one conversation
  - .md files: chunked by ## headings, each section processed independently

Usage:
    python run.py [-h] [FILE ...]

If no files are provided, uses sample_conversation_*.txt from the demo directory.
Then open viewer.html in a browser (serve with python -m http.server from the demo dir).
"""

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import List

from decision_graph.backends.memory import InMemoryBackend
from decision_graph.backends.memory.stores import InMemoryGraphStore, InMemoryVectorIndex
from decision_graph.decision_trace_pipeline import DecisionTracePipeline
from decision_graph.graph import DecisionGraph
from decision_graph.llm import LiteLLMAdapter
from decision_graph.markdown_chunker import chunk_markdown
from decision_graph.visualization import build_vis_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_logger = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).parent


async def _process_text(pipeline, backend, vector_index, *, conv_text, cid, gid):
    summary_pid = f"summary_{cid}"
    backend.projection_store().save(
        pid=summary_pid,
        gid=gid,
        cid=cid,
        proj_type="summary",
        projection={"Summary": conv_text, "proj_id": summary_pid},
        msg_ts=int(time.time()),
    )
    vector_index.add(pid=summary_pid, text=conv_text, gid=gid, cid=cid)

    _logger.info("Running pipeline (extract -> persist -> dedupe -> enrich -> cluster)...")
    decision_items = await pipeline.run_from_text(
        conv_text=conv_text,
        conv_id=cid,
        gid=gid,
        updated_at=time.time(),
        summary_pid=summary_pid,
        query_gids=[gid],
    )
    for i, item in enumerate(decision_items, 1):
        _logger.info("  [%d] %s - %s", i, item.get("decision_type"), item.get("subject", {}).get("label"))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


async def run_demo(conv_files: List[Path]):
    if not conv_files:
        conv_files = sorted(DEMO_DIR.glob("sample_conversation_*.txt"))
    if not conv_files:
        conv_files = [DEMO_DIR / "sample_conversation.txt"]

    backend = InMemoryBackend()
    vector_index = InMemoryVectorIndex()
    graph_store = InMemoryGraphStore()
    llm_adapter = LiteLLMAdapter()
    dg = DecisionGraph(backend=backend, executor=llm_adapter)
    pipeline = DecisionTracePipeline(
        backend=backend,
        executor=llm_adapter,
        vector_index=vector_index,
        graph_store=graph_store,
    )

    gid = "demo_group"

    for conv_file in conv_files:
        raw_text = conv_file.read_text(encoding="utf-8")

        if conv_file.suffix == ".md":
            chunks = chunk_markdown(raw_text)
            _logger.info("--- Processing %s (%d chars, %d sections) ---", conv_file.name, len(raw_text), len(chunks))
            for chunk in chunks:
                cid = f"{conv_file.stem}_{_slugify(chunk['section'])}"
                _logger.info("  Section: %s (%d chars)", chunk["section"], len(chunk["text"]))
                await _process_text(pipeline, backend, vector_index, conv_text=chunk["text"], cid=cid, gid=gid)
        else:
            cid = conv_file.stem
            _logger.info("--- Processing %s (%d chars) ---", conv_file.name, len(raw_text))
            await _process_text(pipeline, backend, vector_index, conv_text=raw_text, cid=cid, gid=gid)

    service = dg.graph_service()
    projections_result = await service.get_enrichments_and_projections_joined(group_ids=[gid])

    hydrated_clusters = graph_store.hydrated_clusters
    graph_arrays = graph_store.graph_arrays

    return projections_result, hydrated_clusters, graph_arrays


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Decision Graph demo pipeline on conversation or markdown files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Paths to .txt or .md files. If omitted, uses sample conversations from the demo directory.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    projections_result, hydrated_clusters, graph_arrays = await run_demo(args.files)

    out_dir = DEMO_DIR / "generated"
    out_dir.mkdir(exist_ok=True)

    output_file = out_dir / "output.json"
    output_file.write_text(json.dumps(projections_result, indent=2, default=str), encoding="utf-8")

    graph_data = build_vis_graph(graph_arrays, hydrated_clusters)
    graph_file = out_dir / "graph_data.json"
    graph_file.write_text(json.dumps(graph_data, indent=2, default=str), encoding="utf-8")

    _logger.info(
        "Done. %d projections, %d enriched, %d clusters, graph: %d nodes / %d edges.",
        projections_result["total_projections"],
        projections_result["total_joined"],
        len(hydrated_clusters),
        len(graph_data["nodes"]),
        len(graph_data["edges"]),
    )
    _logger.info("Serve and open viewer.html:")
    _logger.info("  cd %s && python -m http.server 8888", DEMO_DIR)


if __name__ == "__main__":
    asyncio.run(main())

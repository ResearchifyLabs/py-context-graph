"""
Decision Graph Demo

Reads sample conversations or markdown documents, extracts decisions via LLM,
runs them through the full decision_graph pipeline (extract -> persist -> dedupe
-> enrich -> cluster), and serves results via API for the HTML viewer.

Supports two input modes:
  - .txt files: each file is one conversation
  - .md files: chunked by ## headings, each section processed independently

Usage:
    python run.py [-h] [--port PORT] [--no-browser] [--model MODEL] [FILE ...]

Runs the pipeline, serves results at http://localhost:<port>/viewer.html,
and opens the browser automatically.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
import webbrowser
from pathlib import Path
from typing import List

from aiohttp import web

from decision_graph import GraphConfig
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

_state = {
    "status": {"step": "idle"},
    "backend": None,
    "vector_index": None,
    "graph_store": None,
    "dg": None,
    "config": None,
    "gid": "demo_group",
}


def _write_status(step: str, conversation: str = "", conversation_progress: str = ""):
    _state["status"] = {"step": step, "conversation": conversation, "conversation_progress": conversation_progress}


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

async def handle_status(request):
    return web.json_response(_state["status"])


async def handle_output(request):
    if _state["status"]["step"] != "done":
        return web.json_response(
            {"projections": [], "total_projections": 0, "total_joined": 0, "total_enrichments": 0, "has_more": False}
        )
    service = _state["dg"].graph_service()
    result = await service.get_enrichments_and_projections_joined(group_ids=[_state["gid"]])
    return web.json_response(result, dumps=lambda x: json.dumps(x, default=str))


async def handle_graph_data(request):
    if _state["status"]["step"] != "done":
        return web.json_response({"nodes": [], "edges": []})
    gs = _state["graph_store"]
    graph_data = build_vis_graph(gs.graph_arrays, gs.hydrated_clusters)
    return web.json_response(graph_data, dumps=lambda x: json.dumps(x, default=str))


async def handle_run_demo(request):
    if _state["status"]["step"] not in ("idle", "done"):
        return web.json_response({"error": "Pipeline already running"}, status=409)

    selected = []
    try:
        payload = await request.json()
        if isinstance(payload, dict) and payload.get("files"):
            selected = [f for f in payload.get("files", []) if isinstance(f, str)]
    except Exception:
        selected = []

    sample_files = sorted(DEMO_DIR.glob("sample_conversation_*.txt"))
    if selected:
        conv_files = [DEMO_DIR / file_name for file_name in selected if (DEMO_DIR / file_name).is_file()]
        if not conv_files:
            return web.json_response({"error": "No valid sample files selected"}, status=400)
    else:
        conv_files = sample_files

    asyncio.create_task(run_demo(conv_files, _state["config"] or GraphConfig()))
    return web.json_response({"started": True})


async def handle_upload(request):
    if _state["status"]["step"] not in ("idle", "done"):
        return web.json_response({"error": "Pipeline already running"}, status=409)

    reader = await request.multipart()
    field = await reader.next()
    while field is not None and field.name != "file":
        field = await reader.next()

    if field is None or not field.filename:
        return web.json_response({"error": "No file uploaded"}, status=400)

    filename = field.filename
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md"}:
        return web.json_response({"error": "Only .txt and .md files are supported"}, status=400)

    tmp_path = Path(tempfile.gettempdir()) / f"decision_graph_upload_{uuid.uuid4().hex}{suffix}"
    with tmp_path.open("wb") as tmp_file:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            tmp_file.write(chunk)

    async def upload_task():
        try:
            await run_demo([tmp_path], _state["config"] or GraphConfig())
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    asyncio.create_task(upload_task())
    return web.json_response({"started": True})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def _process_text(pipeline, backend, vector_index, *, conv_text, cid, gid, conv_label, conv_progress):
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

    def on_step(step):
        _write_status(step, conv_label, conv_progress)

    _logger.info("Running pipeline (extract -> persist -> dedupe -> enrich -> cluster)...")
    decision_items = await pipeline.run_from_text(
        conv_text=conv_text,
        conv_id=cid,
        gid=gid,
        updated_at=time.time(),
        summary_pid=summary_pid,
        query_gids=[gid],
        on_step=on_step,
    )
    for i, item in enumerate(decision_items, 1):
        _logger.info("  [%d] %s - %s", i, item.get("decision_type"), item.get("subject", {}).get("label"))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


async def run_demo(conv_files: List[Path], config: GraphConfig):
    if not conv_files:
        conv_files = sorted(DEMO_DIR.glob("sample_conversation_*.txt"))
    if not conv_files:
        conv_files = [DEMO_DIR / "sample_conversation.txt"]

    _write_status("extract", "Initializing...")

    if _state["backend"] and _state["graph_store"] and _state["vector_index"]:
        backend = _state["backend"]
        vector_index = _state["vector_index"]
        graph_store = _state["graph_store"]
    else:
        backend = InMemoryBackend()
        vector_index = InMemoryVectorIndex()
        graph_store = InMemoryGraphStore()
        _state["backend"] = backend
        _state["vector_index"] = vector_index
        _state["graph_store"] = graph_store

    llm_adapter = LiteLLMAdapter()
    dg = DecisionGraph(backend=backend, executor=llm_adapter, config=config)
    pipeline = DecisionTracePipeline(
        backend=backend,
        executor=llm_adapter,
        vector_index=vector_index,
        graph_store=graph_store,
        config=config,
    )

    _state["dg"] = dg

    gid = _state["gid"]
    total_files = len(conv_files)

    for file_idx, conv_file in enumerate(conv_files, 1):
        raw_text = conv_file.read_text(encoding="utf-8")
        conv_label = conv_file.name
        conv_progress = f"{file_idx}/{total_files}"

        if conv_file.suffix == ".md":
            chunks = chunk_markdown(raw_text)
            _logger.info("--- Processing %s (%d chars, %d sections) ---", conv_file.name, len(raw_text), len(chunks))
            for chunk_idx, chunk in enumerate(chunks, 1):
                cid = f"{conv_file.stem}_{_slugify(chunk['section'])}"
                section_progress = f"{file_idx}/{total_files} — section {chunk_idx}/{len(chunks)}"
                _logger.info("  Section: %s (%d chars)", chunk["section"], len(chunk["text"]))
                await _process_text(
                    pipeline, backend, vector_index,
                    conv_text=chunk["text"], cid=cid, gid=gid,
                    conv_label=conv_label, conv_progress=section_progress,
                )
        else:
            cid = conv_file.stem
            _logger.info("--- Processing %s (%d chars) ---", conv_file.name, len(raw_text))
            await _process_text(
                pipeline, backend, vector_index,
                conv_text=raw_text, cid=cid, gid=gid,
                conv_label=conv_label, conv_progress=conv_progress,
            )

    _write_status("done")
    _logger.info("Pipeline finished. Current status step set to done.")


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
    parser.add_argument(
        "--port", type=int, default=8888,
        help="Port for the local viewer server (default: 8888).",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the browser.",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("DECISION_GRAPH_MODEL", "gpt-4.1-mini"),
        help="LiteLLM model string (default: gpt-4.1-mini). E.g. anthropic/claude-3.5-sonnet",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    config = GraphConfig(model=args.model)
    _state["config"] = config
    _logger.info("Using model: %s", config.model)

    app = web.Application()
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/api/output', handle_output)
    app.router.add_get('/api/graph-data', handle_graph_data)
    app.router.add_post('/api/run-demo', handle_run_demo)
    app.router.add_post('/api/upload', handle_upload)
    app.router.add_static('/', DEMO_DIR)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', args.port)
    await site.start()

    url = f"http://localhost:{args.port}/viewer.html"
    _logger.info("Serving at %s", url)

    if not args.no_browser:
        webbrowser.open(url)

    if args.files:
        asyncio.create_task(run_demo(args.files, config))

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _logger.info("Shutting down server.")

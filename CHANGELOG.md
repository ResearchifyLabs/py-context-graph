# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Makefile with targets for venv setup, install, test, coverage, build, clean, and publish
- Development section in README with Makefile usage and coverage instructions
- `pytest-cov` and `build` as dev dependencies
- `GraphConfig` dataclass for library-level configuration, starting with LLM model selection
- `--model` CLI flag and `DECISION_GRAPH_MODEL` env var support in `examples/run.py`
- API endpoints in `examples/run.py` (`/api/status`, `/api/output`, `/api/graph-data`) — viewer now fetches data from backends via HTTP instead of static JSON files
- `aiohttp` added to `[all]` and `[dev]` optional dependencies

### Changed
- `DecisionGraph`, `DecisionTracePipeline`, and all services now accept an optional `config: GraphConfig` parameter
- Replaced hardcoded `gpt-4.1-mini` model name in services with configurable value from `GraphConfig`
- Users can now choose any LiteLLM-supported provider (e.g. `anthropic/claude-3.5-sonnet`) without editing library code
- Replaced sync `http.server` with async `aiohttp` in `examples/run.py` — single event loop, no threading
- `viewer.html` fetches from `/api/*` endpoints instead of reading `generated/*.json` files
- Pipeline results stay in the configured backends and are served on demand via API

## [0.1.1] - 2026-04-03

### Changed
- Refactored decision-pair matching into a pluggable `MatchScorer` protocol with a default `SimpleJaccardScorer` implementation, replacing the previous hard-coded scoring functions
- Moved stop words from inline constant to external `core/stop_words.txt` file
- Reorganized test directory structure under `tests/unit/` to mirror `src/` layout (`core/`, `llm/`, `backends/firestore/`, `context_graph/`)
- Simplified Pydantic imports in `core/domain.py`

### Added
- `MatchScorer` protocol in `core/interfaces.py` for pluggable scoring strategies
- `SimpleJaccardScorer` in `core/matching.py` — a lightweight scorer using Jaccard similarity on subject and topic tokens

## [0.1.0] - 2026-04-02

### Added
- Decision extraction from unstructured conversation text via LLMs
- Decision deduplication with configurable similarity scoring
- LLM-based enrichment (topics, entities, constraints, key facts)
- Cross-conversation decision clustering
- Graph materialization with hydrated cluster output
- Interactive HTML viewer with Insights dashboard, Cluster Board, Timeline, Person x Cluster matrix, and Explore (force-directed graph) views
- Live pipeline progress in the viewer when running `python run.py`
- In-memory backend (stores, TF-IDF vector index, graph store)
- Google Cloud Firestore backend
- LiteLLM adapter supporting OpenAI, Anthropic, and any LiteLLM provider
- Context Graph query layer for natural language graph queries
- Pluggable interfaces: `StorageBackend`, `LLMAdapter`, `VectorIndex`, `GraphStore`


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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


# Contributing to py-context-graph

Thanks for your interest in contributing! Here's how to get started.

## Getting started

```bash
git clone https://github.com/ResearchifyLabs/py-context-graph.git
cd py-context-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests -p 'test_*.py'
```

To run a specific test file:

```bash
PYTHONPATH=src:tests python -m unittest tests/test_example.py
```

## Project layout

- `src/decision_graph/` — the library source
- `tests/` — unit tests (stdlib `unittest`, not pytest)
- `examples/` — runnable demos

## How to contribute

1. **Open an issue first** — describe the bug or feature so we can discuss before you write code.
2. **Fork and branch** — create a feature branch from `main`.
3. **Keep changes focused** — one PR per concern. Small PRs get reviewed faster.
4. **Add tests** — if you're adding a feature or fixing a bug, add a test that covers it.
5. **Make sure tests pass** — run the full test suite before opening a PR.

## Code style

- Keep it simple. No speculative abstractions.
- Match the style of the surrounding code.
- Avoid unnecessary comments — code should be self-documenting.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

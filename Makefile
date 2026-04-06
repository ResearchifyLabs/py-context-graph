VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install install-dev install-all test test-verbose test-cov build clean clean-all

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv ## Install the package in venv
	$(PIP) install -e .

install-dev: venv ## Install with dev dependencies in venv
	$(PIP) install -e ".[dev]"

install-all: venv ## Install with all optional dependencies in venv
	$(PIP) install -e ".[all]"

test: ## Run tests
	$(PYTHON) -m pytest tests/

test-verbose: ## Run tests with verbose output
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=decision_graph --cov-report=term-missing

build: ## Build distribution packages
	$(PYTHON) -m build

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

clean-all: clean ## Remove build artifacts, caches, and venv
	rm -rf $(VENV)

.DEFAULT_GOAL := help

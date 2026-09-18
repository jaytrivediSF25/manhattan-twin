.PHONY: help sync data figures experiments test lint clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

sync: ## Install dependencies
	uv sync

data: ## Pull every source (cached per month; re-runs are no-ops)
	uv run python -m src.mtwin.data bus
	uv run python -m src.mtwin.data crz
	uv run python -m src.mtwin.data bt
	uv run python -m src.mtwin.data dot
	uv run python -m src.mtwin.data subway
	uv run python -m src.mtwin.data weather
	uv run python -c "from src.mtwin.data import tlc; tlc.pull_yellow(zones_filter=None, subdir=tlc.SUBDIR_CITY)"

experiments: ## Twin, ablation, frozen-physics, k_jam sweep
	uv run python -c "import json; from src.mtwin.models.experiments import run_experiments; print(json.dumps(run_experiments(), indent=2, default=str))"

figures: ## Regenerate the four publication figures
	uv run python -m src.mtwin.figures.make_figures

test: ## Run the regression tests
	uv run pytest tests/ -q

lint: ## Lint
	uvx ruff check src tests

clean: ## Remove caches (keeps downloaded data)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

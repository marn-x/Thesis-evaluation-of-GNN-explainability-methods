.PHONY: install lint format typecheck test train explain report all clean

install:          ## create the venv and install everything
	uv sync --extra dev

lint:             ## ruff check
	uv run ruff check src tests

format:           ## ruff format + import sort
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:        ## mypy over the package
	uv run mypy

test:             ## unit tests
	uv run pytest

train:            ## fit and validate the backbone classifier
	uv run sq4 train

explain:          ## run every explainer over the sampled nodes
	uv run sq4 explain

report:           ## summarise runtime, fidelity and consistency
	uv run sq4 report

all: lint typecheck test train explain report

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache

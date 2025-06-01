.PHONY: install test lint format type-check docs clean run benchmark

install:
	poetry install

test:
	poetry run pytest -q

test-verbose:
	poetry run pytest -v

lint:
	poetry run ruff check app tests

format:
	poetry run black app tests

type-check:
	poetry run mypy app

docs:
	poetry run pdoc --html --output-dir docs app

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage coverage.xml htmlcov

run:
	poetry run llm-cockpit

run-debug:
	poetry run llm-cockpit --debug

run-public:
	poetry run llm-cockpit --public

benchmark:
	poetry run python scripts/bench.py

all: format lint type-check test 
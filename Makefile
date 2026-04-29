.PHONY: test test-cluster lint format check

test:
	pytest src/tests/unit

test-integration:
	python src/tests/integration/test_local_cluster.py

lint:
	ruff check --fix .
	ruff format .

format:
	ruff format .

# used for CI - should be synced with .pre-commit-config.yaml,
# and lint and format
check:
	ruff check .
	ruff format --check .

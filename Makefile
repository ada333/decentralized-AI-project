.PHONY: test lint format check

test:
	pytest src/tests

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

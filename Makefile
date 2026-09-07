# NIDRA backend — see docs/PROMPTBOOK.md for the build order.
PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup lint test up down fixtures demo e2e-fast

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check .
	$(BIN)/black --check .

test:
	$(BIN)/pytest -q -m "not e2e"

# The whole serving plane. `web` is the one service still behind a profile — its build
# context belongs to the frontend build and is not on this branch.
up:
	docker compose up -d
	$(BIN)/alembic upgrade head

down:
	docker compose down

# Regenerate the prepared capture. Only needed after changing window_delta, context_L,
# horizon_K or the fixture's own shape — the file is committed.
fixtures:
	$(BIN)/python -m tests.fixtures.replay_csv

# One command: seed demo@nidra.local, upload the prepared capture, replay it at 60x.
# Assumes the stack is up (`make up`).
demo: fixtures
	$(BIN)/python scripts/demo.py

# The CI variant of the same replay, at demo.fast_speed (600x).
e2e-fast: fixtures
	$(BIN)/python scripts/demo.py --fast

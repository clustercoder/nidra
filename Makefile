# NIDRA backend — see docs/PROMPTBOOK.md for the build order.
PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup lint test up down demo

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check .
	$(BIN)/black --check .

test:
	$(BIN)/pytest -q -m "not e2e"

# Datastores only; the application services sit behind the `full` compose profile.
up:
	docker compose up -d redis postgres
	$(BIN)/alembic upgrade head

down:
	docker compose down

demo:
	@echo "demo target lands with the end-to-end replay step (P12)"

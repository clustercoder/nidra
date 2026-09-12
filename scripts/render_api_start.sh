#!/usr/bin/env bash
# Entrypoint for the Render "nidra-api" web service.
#
# docker-compose runs `api` and `ingest` as separate containers sharing a named volume
# (`uploads`) — `api/ingest.py` writes an upload to disk and `services/ingest/worker.py`
# reads it back from the same path. Render web services and background workers are
# separate containers with no shared filesystem (a Render Disk attaches to exactly one
# service), so that split does not carry over directly. Rather than route uploads
# through an external object store, this keeps `api` and `ingest` co-located in the one
# Render service that has a disk: `ingest` runs in the background, `uvicorn` runs in the
# foreground so Render's health check and $PORT binding land on the process it expects.
#
# `features`, `inference`, and `persister` touch only Redis/Postgres/the baked-in model
# weights, so they stay separate Render background workers — see render.yaml.
set -euo pipefail

python -m services.ingest &
INGEST_PID=$!
trap 'kill "$INGEST_PID" 2>/dev/null || true' EXIT

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"

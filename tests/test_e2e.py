"""End-to-end: one capture through the whole serving plane, asserted at every seam.

Runs against the *live* compose stack (`make up`), not the in-process suite: upload over
HTTP, replay paced by the ingest worker, windows built by the features worker, forecasts
from the inference worker, rows from the persister — and a WebSocket client listening the
whole time, exactly as the console would. The suite's Redis isolation (tests/conftest.py)
deliberately does not apply to the stream inspection here: the point is to look at the
bus the containers actually used, so the check reads compose database 0 directly.

A fresh tenant is registered per run. That keeps reruns independent (the forecast and
episode tables are asserted per-tenant) and keeps the prod-mode ingest rate limit
(5 uploads/hour/tenant) from ever biting a second invocation.

What is asserted, in pipeline order:
  1. the job reaches `complete`;
  2. `state_vectors` carried schema-valid 45-feature vectors for this tenant;
  3. the escalating host's final forecast originates at the capture's last window —
     which only exists if the features watchdog closed a window no later event pushed;
  4. the causality contract on every stored forecast: six horizons, k = 1..6, every
     horizon timestamp strictly after `origin_ts`;
  5. the escalating host's peak projected risk separates from both benign hosts';
  6. the WebSocket client received the escalating host's forecasts, all of them this
     tenant's;
  7. exactly one episode opened, and only for the escalating host.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import redis.asyncio as aioredis
import websockets
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from nidra_common.config import get_config
from nidra_common.events import JOB_COMPLETE, JOB_ERROR
from nidra_common.schemas import Forecast, StateVector
from tests.conftest import redis_url_on_db
from tests.fixtures.replay_csv import BENIGN_HOSTS, DEFAULT_PATH, ESCALATING_HOST

pytestmark = pytest.mark.e2e

#: Generous ceilings, not expectations: the capture replays in seconds at fast_speed.
JOB_DEADLINE_S = 240.0
FORECAST_DEADLINE_S = 90.0
POLL_S = 2.0
STACK_DEADLINE_S = 120.0


def api_base_url() -> str:
    configured = os.environ.get("NIDRA_API_URL")
    if configured:
        return configured.rstrip("/")
    return f"http://localhost:{os.environ.get('NIDRA_API_PORT', '8000')}"


def last_window_ts(fixture: Path, host: str, delta_s: int) -> datetime:
    """The capture's final window for `host`: max timestamp, floored to the window grid.

    Read from the file rather than the generator's constants so the assertion tracks
    whatever was actually replayed.
    """
    latest: datetime | None = None
    with fixture.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        columns = {name.strip(): name for name in reader.fieldnames}
        for row in reader:
            if row[columns["Source IP"]].strip() != host:
                continue
            stamp = datetime.strptime(  # noqa: DTZ007 — CIC timestamps are naive UTC
                row[columns["Timestamp"]].strip(), "%d/%m/%Y %H:%M:%S"
            ).replace(tzinfo=UTC)
            if latest is None or stamp > latest:
                latest = stamp
    assert latest is not None, f"{host} never appears in {fixture}"
    return datetime.fromtimestamp(math.floor(latest.timestamp() / delta_s) * delta_s, tz=UTC)


async def collect_ws(url: str, sink: list[Forecast]) -> None:
    """The console's half of the socket: answer pings, keep every forecast frame."""
    async with websockets.connect(url) as socket:
        async for frame in socket:
            body = json.loads(frame)
            if body.get("type") == "ping":
                await socket.send(json.dumps({"type": "pong"}))
                continue
            if "type" in body:  # other control frames (filter echoes)
                continue
            sink.append(Forecast.model_validate(body))


async def wait_for_job(client: httpx.AsyncClient, headers: dict[str, str], job_id: str) -> None:
    deadline = time.monotonic() + JOB_DEADLINE_S
    while time.monotonic() < deadline:
        body = (await client.get(f"/api/v1/ingest/{job_id}", headers=headers)).json()
        if body["status"] == JOB_ERROR:
            pytest.fail(f"ingest job failed: {body.get('error')}")
        if body["status"] == JOB_COMPLETE:
            return
        await asyncio.sleep(POLL_S)
    pytest.fail(f"job {job_id} not complete within {JOB_DEADLINE_S:.0f}s")


async def wait_for_final_forecast(
    client: httpx.AsyncClient, headers: dict[str, str], host: str, origin_ts: datetime
) -> list[dict[str, Any]]:
    """Poll history until the forecast anchored at the capture's last window lands."""
    deadline = time.monotonic() + FORECAST_DEADLINE_S
    items: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        page = (
            await client.get(
                "/api/v1/forecasts",
                headers=headers,
                params={"host": host, "limit": 200, "order": "asc"},
            )
        ).json()
        items = page["items"]
        stamps = {datetime.fromisoformat(item["origin_ts"]) for item in items}
        if origin_ts in stamps:
            return items
        await asyncio.sleep(POLL_S)
    pytest.fail(
        f"no forecast with origin_ts={origin_ts.isoformat()} for {host} within "
        f"{FORECAST_DEADLINE_S:.0f}s ({len(items)} forecasts seen)"
    )


async def test_full_replay_pipeline() -> None:
    cfg = get_config()
    delta_s = int(cfg["window_delta"])
    demo_cfg = dict(cfg.get("demo", {}))
    speed = float(demo_cfg.get("fast_speed", 600))
    fixture = DEFAULT_PATH
    assert fixture.is_file(), "run `make fixtures` first"
    expected_final = last_window_ts(fixture, ESCALATING_HOST, delta_s)

    base = api_base_url()
    async with httpx.AsyncClient(base_url=base, timeout=30.0) as client:
        # A rebuild (`make e2e`, verify.sh) reaches here seconds after `compose up`;
        # wait for the api rather than racing its healthcheck.
        deadline = time.monotonic() + STACK_DEADLINE_S
        while True:
            try:
                (await client.get("/health")).raise_for_status()
                break
            except httpx.HTTPError as exc:  # pragma: no cover — stack still starting
                if time.monotonic() >= deadline:
                    pytest.fail(f"api not reachable at {base} ({exc}); run `make up` first")
                await asyncio.sleep(POLL_S)

        email = f"e2e-{uuid.uuid4().hex[:8]}@nidra.local"
        password = "e2e-password-1"
        register = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "org_name": "NIDRA e2e"},
        )
        register.raise_for_status()
        tenant_id = register.json()["tenant_id"]
        login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
        login.raise_for_status()
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        ws_url = f"{base.replace('http', 'ws', 1)}/api/v1/stream/forecast?token={token}"
        streamed: list[Forecast] = []
        listener = asyncio.create_task(collect_ws(ws_url, streamed))
        await asyncio.sleep(1.0)  # socket accepted before the first window can close

        try:
            with fixture.open("rb") as handle:
                upload = await client.post(
                    "/api/v1/ingest",
                    headers=headers,
                    files={"file": (fixture.name, handle, "text/csv")},
                    data={"speed": str(speed)},
                )
            upload.raise_for_status()
            job_id = upload.json()["job_id"]

            await wait_for_job(client, headers, job_id)
            items = await wait_for_final_forecast(client, headers, ESCALATING_HOST, expected_final)
            await asyncio.sleep(2.0)  # let the socket drain what the api already has
        finally:
            listener.cancel()

        # --- 4. causality on every stored forecast for the escalating host -----------
        horizon_k = int(cfg["horizon_K"])
        for item in items:
            detail = await client.get(
                f"/api/v1/forecasts/{ESCALATING_HOST}/{item['origin_ts']}", headers=headers
            )
            detail.raise_for_status()
            forecast = Forecast.model_validate(detail.json())
            assert [point.k for point in forecast.horizons] == list(range(1, horizon_k + 1))
            assert all(point.ts > forecast.origin_ts for point in forecast.horizons)

        # --- 5. the escalating host separates from the benign ones --------------------
        hosts = (await client.get("/api/v1/hosts", headers=headers)).json()["hosts"]
        peak = {row["host_id"]: row["max_p_compromise"] for row in hosts}
        assert ESCALATING_HOST in peak
        for benign in BENIGN_HOSTS:
            assert benign in peak, f"benign host {benign} produced no forecasts"
            assert peak[ESCALATING_HOST] > peak[benign], (
                f"escalating host peak {peak[ESCALATING_HOST]} did not separate from "
                f"{benign} at {peak[benign]}"
            )

    # --- 6. the WebSocket saw it happen, and only for this tenant ---------------------
    assert any(
        forecast.host_id == ESCALATING_HOST for forecast in streamed
    ), "websocket client received no forecasts for the escalating host"
    assert all(forecast.tenant_id == tenant_id for forecast in streamed)

    # --- 2. the bus carried schema-valid state vectors for this tenant ----------------
    compose_bus = aioredis.from_url(redis_url_on_db(str(cfg["redis"]["url"]), 0))
    try:
        entries = await compose_bus.xrevrange(cfg["streams"]["state_vectors"], count=5000)
        ours = [
            StateVector.model_validate_json(fields[b"payload"])
            for _, fields in entries
            if json.loads(fields[b"payload"])["tenant_id"] == tenant_id
        ]
    finally:
        await compose_bus.aclose()
    assert {vector.host_id for vector in ours} >= {ESCALATING_HOST, *BENIGN_HOSTS}

    # --- 7. exactly one episode, and only where the risk was --------------------------
    engine = create_async_engine(str(cfg["postgres"]["url"]))
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT host_id, started_at, peak_risk FROM episodes"
                        " WHERE tenant_id = :tenant_id"
                    ),
                    {"tenant_id": tenant_id},
                )
            ).all()
    finally:
        await engine.dispose()
    assert [row.host_id for row in rows] == [
        ESCALATING_HOST
    ], f"expected exactly one episode for {ESCALATING_HOST}, got {rows}"
    assert rows[0].peak_risk >= float(cfg["risk_threshold"])

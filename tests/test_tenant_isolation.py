"""P11: tenant B must see nothing of tenant A, on every read surface there is.

Treated as release-blocking rather than as one more test. A tenant leak found live during
judging is unrecoverable, and the failure mode is quiet: every one of these endpoints
returns a perfectly well-formed response whether or not its query carries the tenant
predicate. Nothing else in the suite would notice.

So this file is deliberately exhaustive rather than representative. Tenant A gets a row
in every table a read endpoint touches — an ingest job, state vectors, forecasts, an open
episode — and then tenant B asks for all of it: the history, one forecast by name, the
host list, its jobs, an explanation, a counterfactual, the benchmarks, and the live
socket. The expected answer is 404 or empty, in every single case.

The socket is included because it is the one surface where the tenant filter is not SQL:
the fan-out matches on the `tenant_id` inside the forecast (`api/ws.py`), and a bug there
would leak a live stream rather than a stored row.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import uvicorn
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from websockets.asyncio.client import connect

from api.main import create_app
from api.ws import ForecastFanout
from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.bus import Bus, create_redis
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.schemas import Forecast, HorizonPoint, SignalAttribution, StateVector
from services.features.store import StateVectorStore
from services.persister.worker import PersisterWorker

PASSWORD = "correct-horse-battery"

BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

MODEL_VERSION = "nidra-0.1.0-stub"

#: Tenant A's host. Tenant B never creates data, so any appearance of this string in a
#: response to B is the leak this file exists to catch.
HOST = "192.168.10.50"

#: Above `risk_threshold`, so the forecasts open an episode as well as a forecast row.
PEAK = 0.92

RECV_TIMEOUT = 5.0
SILENCE_TIMEOUT = 1.5
TEST_BLOCK_MS = 100


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def cfg() -> dict:
    return get_config()


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    factory = get_sessionmaker()
    try:
        yield factory
    finally:
        await dispose_engine()


async def _register(org: str) -> dict[str, str]:
    """Register and log in one tenant through the real endpoints."""
    email = f"p11-{uuid.uuid4().hex[:12]}@nidra.test"
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        created = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": PASSWORD, "org_name": org},
        )
        assert created.status_code == 201, created.text
        tokens = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert tokens.status_code == 200, tokens.text
    return {
        "tenant_id": created.json()["tenant_id"],
        "token": tokens.json()["access_token"],
        "email": email,
    }


async def _purge(tenant_id: str) -> None:
    async with get_sessionmaker()() as session:
        for statement in (
            "DELETE FROM episodes WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM forecasts WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM state_vectors WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM ingest_jobs WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM users WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM tenants WHERE id = CAST(:tid AS uuid)",
        ):
            await session.execute(text(statement), {"tid": tenant_id})
        await session.commit()
    await dispose_engine()


@pytest.fixture
async def tenant_a(redis_client: Redis) -> AsyncIterator[dict[str, str]]:
    account = await _register("P11 Tenant A")
    yield account
    keys = [key async for key in redis_client.scan_iter(match=f"ep:{account['tenant_id']}:*")]
    if keys:
        await redis_client.delete(*keys)
    await _purge(account["tenant_id"])


@pytest.fixture
async def tenant_b() -> AsyncIterator[dict[str, str]]:
    account = await _register("P11 Tenant B")
    yield account
    await _purge(account["tenant_id"])


def _client(account: dict[str, str], app=None) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    """An authenticated ASGI client for one tenant."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app if app is not None else create_app()),
        base_url="http://api.test",
        headers={"Authorization": f"Bearer {account['token']}"},
    )


# ---------------------------------------------------------------------- tenant A data


def build_forecast(tenant_id: str, *, window: int, cfg: dict) -> Forecast:
    """A schema-valid forecast well above the risk threshold."""
    horizon_k = int(cfg["horizon_K"])
    window_delta = int(cfg["window_delta"])
    origin = BASE + timedelta(seconds=window_delta * window)
    return Forecast(
        tenant_id=tenant_id,
        host_id=HOST,
        origin_ts=origin,
        horizons=[
            HorizonPoint(
                k=k,
                ts=origin + timedelta(seconds=window_delta * k),
                p_compromise=PEAK * k / horizon_k,
                ci_low=max(0.0, PEAK * k / horizon_k - 0.05),
                ci_high=min(1.0, PEAK * k / horizon_k + 0.05),
                stage_dist=dict.fromkeys(STAGES, 1.0 / len(STAGES)),
                predicted_features=dict.fromkeys(FEATURE_ORDER, 0.0),
            )
            for k in range(1, horizon_k + 1)
        ],
        lead_time_s=float(window_delta * 2),
        observed_stage="recon",
        observed_risk=0.4,
        top_signals=[
            SignalAttribution(
                name="syn_ratio", shap_value=0.31, direction="up", display="SYN ratio rising"
            )
        ],
        driving_window=3,
        model_version=MODEL_VERSION,
    )


@pytest.fixture
async def tenant_a_data(
    tenant_a: dict[str, str],
    redis_client: Redis,
    sessions: async_sessionmaker[AsyncSession],
    cfg: dict,
) -> dict[str, object]:
    """Everything tenant A owns: a job, state vectors, forecasts, and an open episode.

    Written through the real writers — the ingest endpoint, `StateVectorStore`,
    `PersisterWorker` — so the rows are shaped exactly as the pipeline shapes them and
    the reads under test are the reads the console makes.
    """
    tenant_id = tenant_a["tenant_id"]
    window_delta = int(cfg["window_delta"])

    async with _client(tenant_a) as client:
        created = await client.post(
            "/api/v1/ingest",
            files={"file": ("tenant-a.csv", b"Source IP,Destination IP,Timestamp\n", "text/csv")},
            data={"speed": "0"},
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]

    store = StateVectorStore(sessions)
    context_l = int(cfg["context_L"])
    for index in range(context_l):
        await store.save(
            StateVector(
                tenant_id=tenant_id,
                host_id=HOST,
                window_ts=BASE + timedelta(seconds=window_delta * index),
                features=dict.fromkeys(FEATURE_ORDER, 0.1),
            )
        )

    worker = PersisterWorker(redis_client, sessions, cfg=cfg)
    forecasts = [build_forecast(tenant_id, window=window, cfg=cfg) for window in (0, 1, 2)]
    for forecast in forecasts:
        await worker.process(forecast)

    async with get_sessionmaker()() as session:
        episodes = int(
            (
                await session.execute(
                    text("SELECT count(*) FROM episodes WHERE tenant_id = CAST(:tid AS uuid)"),
                    {"tid": tenant_id},
                )
            ).scalar_one()
        )
    assert episodes == 1, "the fixture is meant to leave tenant A holding an episode"

    return {
        "job_id": job_id,
        "forecasts": forecasts,
        "context_ts": BASE + timedelta(seconds=window_delta * (context_l - 1)),
    }


# ------------------------------------------------------------------------ read surfaces


async def test_tenant_b_sees_no_forecasts_of_tenant_a(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    forecasts: list[Forecast] = tenant_a_data["forecasts"]  # type: ignore[assignment]
    async with _client(tenant_b) as client:
        listing = await client.get("/api/v1/forecasts")
        assert listing.status_code == 200, listing.text
        body = listing.json()
        assert body["total"] == 0
        assert body["items"] == []
        assert HOST not in listing.text

        stored = forecasts[-1]
        detail = await client.get(
            f"/api/v1/forecasts/{stored.host_id}/{stored.origin_ts.isoformat()}"
        )
        assert detail.status_code == 404, "another tenant's forecast must not be readable"


async def test_tenant_b_sees_no_hosts_of_tenant_a(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    async with _client(tenant_b) as client:
        response = await client.get("/api/v1/hosts")
        assert response.status_code == 200, response.text
        assert response.json()["hosts"] == []
        assert HOST not in response.text


async def test_tenant_b_sees_no_jobs_of_tenant_a(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    job_id = tenant_a_data["job_id"]
    async with _client(tenant_b) as client:
        listing = await client.get("/api/v1/ingest")
        assert listing.status_code == 200, listing.text
        assert listing.json()["jobs"] == []

        detail = await client.get(f"/api/v1/ingest/{job_id}")
        assert detail.status_code == 404, "another tenant's job must not be readable"


async def test_tenant_b_cannot_explain_a_host_of_tenant_a(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    """The explanation reads the state vectors, which are a table of their own."""
    ts: datetime = tenant_a_data["context_ts"]  # type: ignore[assignment]
    async with _client(tenant_b) as client:
        response = await client.get(f"/api/v1/explain/{HOST}/{ts.isoformat()}")
        assert response.status_code == 404, response.text


async def test_tenant_b_cannot_run_a_counterfactual_on_a_host_of_tenant_a(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    ts: datetime = tenant_a_data["context_ts"]  # type: ignore[assignment]
    async with _client(tenant_b) as client:
        response = await client.post(
            "/api/v1/counterfactual",
            json={
                "host": HOST,
                "ts": ts.isoformat(),
                "feature": "syn_ratio",
                "clamp_value": 0.0,
            },
        )
        assert response.status_code == 404, response.text


async def test_benchmarks_carry_no_tenant_data(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    """`/benchmarks` and `/model` are evaluation surfaces: same answer for every tenant.

    They are in this file because "not tenant-scoped" has to be a checked property rather
    than an assumption — an endpoint that quietly grew a per-tenant read would leak here.
    """
    async with _client(tenant_b) as client:
        benchmarks = await client.get("/api/v1/benchmarks")
        assert benchmarks.status_code == 200, benchmarks.text
        assert HOST not in benchmarks.text
        assert MODEL_VERSION not in benchmarks.text

        model = await client.get("/api/v1/model")
        assert model.status_code == 200, model.text
        assert HOST not in model.text

    async with _client(tenant_b) as anonymous:
        del anonymous.headers["authorization"]
        assert (await anonymous.get("/api/v1/benchmarks")).status_code == 401


async def test_tenant_a_still_reads_its_own_data(
    tenant_a: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    """The control: every 404 above has to be about tenancy, not about a broken fixture."""
    forecasts: list[Forecast] = tenant_a_data["forecasts"]  # type: ignore[assignment]
    ts: datetime = tenant_a_data["context_ts"]  # type: ignore[assignment]
    stored = forecasts[-1]
    async with _client(tenant_a) as client:
        assert (await client.get("/api/v1/forecasts")).json()["total"] == len(forecasts)
        assert [h["host_id"] for h in (await client.get("/api/v1/hosts")).json()["hosts"]] == [HOST]
        assert len((await client.get("/api/v1/ingest")).json()["jobs"]) == 1
        assert (await client.get(f"/api/v1/ingest/{tenant_a_data['job_id']}")).status_code == 200
        detail = await client.get(
            f"/api/v1/forecasts/{stored.host_id}/{stored.origin_ts.isoformat()}"
        )
        assert detail.status_code == 200, detail.text
        assert (await client.get(f"/api/v1/explain/{HOST}/{ts.isoformat()}")).status_code == 200


async def test_the_episode_belongs_to_tenant_a_alone(
    tenant_b: dict[str, str], tenant_a_data: dict[str, object]
) -> None:
    """Episodes have no read endpoint yet; the row itself is checked instead."""
    async with get_sessionmaker()() as session:
        count = int(
            (
                await session.execute(
                    text("SELECT count(*) FROM episodes WHERE tenant_id = CAST(:tid AS uuid)"),
                    {"tid": tenant_b["tenant_id"]},
                )
            ).scalar_one()
        )
    await dispose_engine()
    assert count == 0


# ------------------------------------------------------------------------- the socket


class _Server(uvicorn.Server):
    """uvicorn without signal handlers: pytest's event loop is not ours to reconfigure."""

    def install_signal_handlers(self) -> None:
        return


@contextlib.asynccontextmanager
async def running(app) -> AsyncIterator[str]:  # type: ignore[no-untyped-def]
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    server = _Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():  # pragma: no cover — surfaces a startup failure as itself
                await task
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


@pytest.fixture
async def forecasts_stream(redis_client: Redis) -> AsyncIterator[str]:
    """A private `forecasts`-shaped stream, deleted afterwards."""
    name = f"test:forecasts:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


async def test_tenant_b_never_receives_tenant_a_forecasts_on_the_socket(
    redis_client: Redis,
    forecasts_stream: str,
    tenant_a: dict[str, str],
    tenant_b: dict[str, str],
    cfg: dict,
) -> None:
    """The one filter that is not a SQL predicate: the fan-out's tenant match.

    Both tenants hold a socket open on the same stream. Tenant A's forecast must reach
    A's socket and must not reach B's — and B's socket has to stay silent rather than
    merely receive it late, so A's arrival is awaited first and B is then given a
    generous window to be wrong in.
    """
    app = create_app()
    app.state.fanout = ForecastFanout(
        app.state.connections, stream=forecasts_stream, block_ms=TEST_BLOCK_MS
    )
    publisher = Bus(redis_client, forecasts_stream, group="test", consumer="test")

    async with running(app) as base:
        async with (
            connect(f"{base}/api/v1/stream/forecast?token={tenant_a['token']}") as socket_a,
            connect(f"{base}/api/v1/stream/forecast?token={tenant_b['token']}") as socket_b,
        ):
            await publisher.publish(build_forecast(tenant_a["tenant_id"], window=0, cfg=cfg))

            received = json.loads(await asyncio.wait_for(socket_a.recv(), RECV_TIMEOUT))
            assert received["tenant_id"] == tenant_a["tenant_id"]
            assert received["host_id"] == HOST

            with pytest.raises(TimeoutError):
                await asyncio.wait_for(socket_b.recv(), SILENCE_TIMEOUT)

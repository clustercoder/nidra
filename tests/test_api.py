"""P9: the read API over the forecasts the persister wrote.

Against the real Postgres from `docker compose up -d postgres`, because two of the four
properties under test are SQL rather than Python: `DISTINCT ON` picking the newest row
per host, and the tenant predicate on every statement. A mocked session would assert that
the code calls a query, which is not the thing that can be wrong here.

The rows are inserted through `PersisterWorker`, not by hand, so the promoted columns and
the JSONB payload are exactly what the pipeline produces — a fixture that wrote its own
rows could pass while the real writer and the real reader disagreed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.forecasts import HOST_SORT_COLUMNS, max_page_limit, page_limit, resolved_limit
from api.main import create_app
from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.bus import create_redis
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.schemas import Forecast, HorizonPoint, SignalAttribution
from services.persister.worker import PersisterWorker

PASSWORD = "correct-horse-battery"

#: Aligned to an absolute multiple of the 30 s window, as the features service emits them.
BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

MODEL_VERSION = "nidra-0.1.0-stub"

QUIET = "192.168.10.50"
NOISY = "192.168.10.51"


# ------------------------------------------------------------------------- fixtures


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
    """Session factory, with the engine disposed after each test's event loop closes."""
    factory = get_sessionmaker()
    try:
        yield factory
    finally:
        await dispose_engine()


@pytest.fixture
async def account() -> AsyncIterator[dict[str, str]]:
    """A registered tenant with a live access token, purged afterwards."""
    email = f"p9-{uuid.uuid4().hex[:12]}@nidra.test"
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        created = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": PASSWORD, "org_name": "P9 SOC"},
        )
        assert created.status_code == 201, created.text
        tokens = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert tokens.status_code == 200, tokens.text
        tenant_id = created.json()["tenant_id"]
        yield {"tenant_id": tenant_id, "token": tokens.json()["access_token"], "email": email}

    async with get_sessionmaker()() as session:
        for statement in (
            "DELETE FROM episodes WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM forecasts WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM users WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM tenants WHERE id = CAST(:tid AS uuid)",
        ):
            await session.execute(text(statement), {"tid": tenant_id})
        await session.commit()
    await dispose_engine()


@pytest.fixture
async def other_account() -> AsyncIterator[dict[str, str]]:
    """A second tenant, so 'filtered by tenant' can be asserted rather than assumed."""
    email = f"p9-other-{uuid.uuid4().hex[:12]}@nidra.test"
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        created = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": PASSWORD, "org_name": "P9 Other SOC"},
        )
        assert created.status_code == 201, created.text
        tokens = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        tenant_id = created.json()["tenant_id"]
        yield {"tenant_id": tenant_id, "token": tokens.json()["access_token"], "email": email}

    async with get_sessionmaker()() as session:
        for statement in (
            "DELETE FROM episodes WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM forecasts WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM users WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM tenants WHERE id = CAST(:tid AS uuid)",
        ):
            await session.execute(text(statement), {"tid": tenant_id})
        await session.commit()
    await dispose_engine()


@pytest.fixture
async def api(account: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    """An authenticated client for the read endpoints."""
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://api.test",
        headers={"Authorization": f"Bearer {account['token']}"},
    ) as client:
        yield client


@pytest.fixture(autouse=True)
async def episode_state(redis_client: Redis, account: dict[str, str]) -> AsyncIterator[None]:
    """The persister's per-host Redis state, removed with the tenant that owns it."""
    yield
    keys = [key async for key in redis_client.scan_iter(match=f"ep:{account['tenant_id']}:*")]
    if keys:
        await redis_client.delete(*keys)


# ------------------------------------------------------------------ synthetic forecasts


def build_forecast(
    tenant_id: str,
    *,
    host: str,
    window: int,
    cfg: dict,
    peak: float,
    observed_risk: float = 0.2,
    stage: str = "recon",
) -> Forecast:
    """A schema-valid `Forecast` whose trajectory tops out at `peak`."""
    horizon_k = int(cfg["horizon_K"])
    window_delta = int(cfg["window_delta"])
    origin = BASE + timedelta(seconds=window_delta * window)
    return Forecast(
        tenant_id=tenant_id,
        host_id=host,
        origin_ts=origin,
        horizons=[
            HorizonPoint(
                k=k,
                ts=origin + timedelta(seconds=window_delta * k),
                p_compromise=peak * k / horizon_k,
                ci_low=max(0.0, peak * k / horizon_k - 0.05),
                ci_high=min(1.0, peak * k / horizon_k + 0.05),
                stage_dist=dict.fromkeys(STAGES, 1.0 / len(STAGES)),
                predicted_features=dict.fromkeys(FEATURE_ORDER, 0.0),
            )
            for k in range(1, horizon_k + 1)
        ],
        lead_time_s=float(window_delta * 2),
        observed_stage=stage,
        observed_risk=observed_risk,
        top_signals=[
            SignalAttribution(
                name="syn_ratio", shap_value=0.31, direction="up", display="SYN ratio rising"
            )
        ],
        driving_window=3,
        model_version=MODEL_VERSION,
    )


@pytest.fixture
async def history(
    account: dict[str, str],
    redis_client: Redis,
    sessions: async_sessionmaker[AsyncSession],
    cfg: dict,
) -> list[Forecast]:
    """Six forecasts for this tenant: a quiet host and one whose trajectory climbs."""
    worker = PersisterWorker(redis_client, sessions, cfg=cfg)
    written: list[Forecast] = []
    for window in range(3):
        for host, peak in ((QUIET, 0.10 + 0.01 * window), (NOISY, 0.50 + 0.15 * window)):
            forecast = build_forecast(
                account["tenant_id"],
                host=host,
                window=window,
                cfg=cfg,
                peak=peak,
                observed_risk=0.05 * (window + 1) if host == QUIET else 0.3 + 0.1 * window,
            )
            await worker.process(forecast)
            written.append(forecast)
    return written


# ------------------------------------------------------------------------- pagination


async def test_the_history_is_paginated_newest_first(
    api: httpx.AsyncClient, history: list[Forecast]
) -> None:
    first = await api.get("/api/v1/forecasts", params={"limit": 2})
    assert first.status_code == 200, first.text
    body = first.json()

    assert body["total"] == len(history) == 6
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2

    second = (await api.get("/api/v1/forecasts", params={"limit": 2, "offset": 2})).json()
    third = (await api.get("/api/v1/forecasts", params={"limit": 2, "offset": 4})).json()

    seen = [
        (item["host_id"], item["origin_ts"])
        for page in (body, second, third)
        for item in page["items"]
    ]
    assert len(set(seen)) == 6, "pages overlapped or repeated a row"
    timestamps = [ts for _, ts in seen]
    assert timestamps == sorted(timestamps, reverse=True), "history is not newest-first"


async def test_a_page_past_the_end_is_empty_not_an_error(
    api: httpx.AsyncClient, history: list[Forecast]
) -> None:
    body = (await api.get("/api/v1/forecasts", params={"offset": 100})).json()
    assert body["items"] == []
    assert body["total"] == 6


async def test_the_history_filters_by_host_and_time(
    api: httpx.AsyncClient, history: list[Forecast], cfg: dict
) -> None:
    by_host = (await api.get("/api/v1/forecasts", params={"host": NOISY})).json()
    assert by_host["total"] == 3
    assert {item["host_id"] for item in by_host["items"]} == {NOISY}

    window_delta = int(cfg["window_delta"])
    since = BASE + timedelta(seconds=window_delta)
    until = BASE + timedelta(seconds=window_delta * 2)
    windowed = (
        await api.get(
            "/api/v1/forecasts",
            params={"host": NOISY, "since": since.isoformat(), "until": until.isoformat()},
        )
    ).json()
    # Half-open [since, until): exactly the middle window.
    assert windowed["total"] == 1
    assert windowed["items"][0]["origin_ts"].startswith(since.isoformat()[:19])


async def test_the_page_limit_is_clamped_to_the_configured_ceiling(cfg: dict) -> None:
    assert resolved_limit(None) == page_limit() == int(cfg["api"]["page_limit"])
    assert resolved_limit(max_page_limit() + 1000) == max_page_limit()
    assert resolved_limit(1) == 1


# ------------------------------------------------------------------------ full detail


async def test_the_detail_endpoint_returns_the_whole_forecast(
    api: httpx.AsyncClient, history: list[Forecast], cfg: dict
) -> None:
    stored = history[-1]
    response = await api.get(f"/api/v1/forecasts/{stored.host_id}/{stored.origin_ts.isoformat()}")
    assert response.status_code == 200, response.text

    returned = Forecast.model_validate(response.json())
    assert returned.host_id == stored.host_id
    assert returned.origin_ts == stored.origin_ts
    # The parts the list view does not carry, and the console cannot draw without.
    assert len(returned.horizons) == int(cfg["horizon_K"])
    assert [h.k for h in returned.horizons] == [h.k for h in stored.horizons]
    assert returned.top_signals[0].name == "syn_ratio"
    assert set(returned.horizons[0].predicted_features) == set(FEATURE_ORDER)


async def test_an_unknown_forecast_is_a_404(
    api: httpx.AsyncClient, history: list[Forecast]
) -> None:
    missing = (BASE - timedelta(days=1)).isoformat()
    assert (await api.get(f"/api/v1/forecasts/{QUIET}/{missing}")).status_code == 404


# -------------------------------------------------------------------------- host view


async def test_hosts_shows_the_latest_forecast_per_host(
    api: httpx.AsyncClient, history: list[Forecast], cfg: dict
) -> None:
    body = (await api.get("/api/v1/hosts")).json()
    hosts = {item["host_id"]: item for item in body["hosts"]}

    assert set(hosts) == {QUIET, NOISY}, "one row per host, not one per forecast"
    window_delta = int(cfg["window_delta"])
    latest = (BASE + timedelta(seconds=window_delta * 2)).isoformat()
    assert hosts[NOISY]["origin_ts"].startswith(latest[:19])
    # The most recent NOISY forecast tops out at 0.50 + 0.15 * 2.
    assert hosts[NOISY]["max_p_compromise"] == pytest.approx(0.8, abs=1e-5)
    assert hosts[NOISY]["model_version"] == MODEL_VERSION


async def test_hosts_is_sortable_in_both_directions(
    api: httpx.AsyncClient, history: list[Forecast]
) -> None:
    by_risk = (await api.get("/api/v1/hosts", params={"sort": "risk"})).json()["hosts"]
    assert [item["host_id"] for item in by_risk] == [NOISY, QUIET]

    ascending = (await api.get("/api/v1/hosts", params={"sort": "risk", "order": "asc"})).json()[
        "hosts"
    ]
    assert [item["host_id"] for item in ascending] == [QUIET, NOISY]

    by_host = (await api.get("/api/v1/hosts", params={"sort": "host", "order": "asc"})).json()
    assert [item["host_id"] for item in by_host["hosts"]] == sorted([QUIET, NOISY])


async def test_an_unknown_sort_key_is_refused(api: httpx.AsyncClient) -> None:
    """The sort key indexes a whitelist; anything else must not reach the SQL."""
    assert "; DROP TABLE forecasts" not in HOST_SORT_COLUMNS
    response = await api.get("/api/v1/hosts", params={"sort": "max_p_compromise DESC; DROP TABLE"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------- tenancy


async def test_every_read_is_filtered_by_tenant(
    history: list[Forecast], other_account: dict[str, str]
) -> None:
    """Tenant B sees none of tenant A's rows, and 404s on the one it asks for by name."""
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://api.test",
        headers={"Authorization": f"Bearer {other_account['token']}"},
    ) as client:
        listing = (await client.get("/api/v1/forecasts")).json()
        assert listing["total"] == 0
        assert listing["items"] == []

        assert (await client.get("/api/v1/hosts")).json()["hosts"] == []

        stored = history[-1]
        detail = await client.get(
            f"/api/v1/forecasts/{stored.host_id}/{stored.origin_ts.isoformat()}"
        )
        assert detail.status_code == 404, "another tenant's forecast must not be readable"


async def test_the_read_endpoints_require_a_token(history: list[Forecast]) -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        assert (await client.get("/api/v1/forecasts")).status_code == 401
        assert (await client.get("/api/v1/hosts")).status_code == 401
        stored = history[-1]
        detail = await client.get(
            f"/api/v1/forecasts/{stored.host_id}/{stored.origin_ts.isoformat()}"
        )
        assert detail.status_code == 401


# ------------------------------------------------------------------------ ops surface


async def test_health_reports_liveness_and_per_stream_lag(api: httpx.AsyncClient) -> None:
    body = (await api.get("/health")).json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["streams"]) >= {"raw_events", "state_vectors", "forecasts"}
    assert all(isinstance(value, int) and value >= 0 for value in body["streams"].values())


async def test_metrics_is_prometheus_text(api: httpx.AsyncClient) -> None:
    await api.get("/api/v1/hosts")
    response = await api.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# TYPE nidra_http_requests_total counter" in body
    assert 'nidra_http_requests_total{method="GET",status="200"}' in body
    assert "nidra_websocket_connections 0" in body
    assert 'nidra_stream_pending{stream="forecasts"}' in body
    assert body.endswith("\n")

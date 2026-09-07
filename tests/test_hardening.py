"""P11: the ceilings, the backpressure signal, and the capture `make demo` replays.

The rate limiter is exercised at both levels. `hit()` is checked directly against Redis,
because the property that matters — a window that slides rather than resetting on a
boundary, and a refusal that does not consume a slot — is in the sorted-set arithmetic and
not in the middleware. The middleware is then checked through the real app, because the
other half of the property is *which* request lands in *which* bucket: the health probe in
none of them, an upload in two, and one tenant never in another's.

The ceilings under test are built here rather than read from config: the suite runs as
`NIDRA_ENV=dev`, where the configured limits are multiplied out of reach on purpose, and a
test that had to send 100 requests to prove a limit of 100 would be a slow test proving
the arithmetic of `range`. What the config actually says is asserted once, separately.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from redis.asyncio import Redis
from starlette.requests import Request

from api.auth import create_access_token
from api.main import create_app
from api.metrics import BacklogMonitor, backlog_settings
from api.ratelimit import (
    GENERAL_SCOPE,
    INGEST_SCOPE,
    KEY_PREFIX,
    RETRY_AFTER_HEADER,
    Limit,
    RateLimitSettings,
    caller_key,
    hit,
    now_ms,
    rate_limit_settings,
)
from nidra_common.bus import create_redis
from nidra_common.config import get_config
from tests.fixtures.replay_csv import (
    BENIGN_HOSTS,
    CSV_HEADER,
    DEFAULT_PATH,
    ESCALATING_HOST,
    window_count,
)

#: A read endpoint that needs a token. Unauthenticated it answers 401 — which is the point:
#: the limiter runs before routing, so a caller cannot spend someone else's budget by
#: sending requests it was never going to be allowed to make.
GUARDED_PATH = "/api/v1/hosts"

CSV_UPLOAD = {"file": ("flows.csv", b"Source IP,Destination IP,Timestamp\n", "text/csv")}


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


def tight(*, general: int, ingest: int = 1000, window_s: float = 60.0) -> RateLimitSettings:
    """Reachable ceilings on scopes no other test shares.

    The scope is part of the Redis key, so a unique suffix per call keeps two tests — or
    two runs of the suite against the same Redis — out of each other's windows without
    anything having to be cleaned up.
    """
    suffix = uuid.uuid4().hex[:8]
    return RateLimitSettings(
        enabled=True,
        general=Limit(scope=f"{GENERAL_SCOPE}-{suffix}", limit=general, window_s=window_s),
        ingest=Limit(scope=f"{INGEST_SCOPE}-{suffix}", limit=ingest, window_s=window_s),
    )


def limited_client(settings: RateLimitSettings) -> httpx.AsyncClient:
    """The real app with its limiter's ceilings replaced."""
    app = create_app()
    app.state.rate_limiter.settings = settings
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api.test")


def bearer(tenant_id: str) -> dict[str, str]:
    """A structurally valid access token for `tenant_id`, minted without the database.

    The request it carries is not expected to succeed — only to be *counted* under that
    tenant, which is decided before any route runs.
    """
    token = create_access_token(user_id=str(uuid.uuid4()), tenant_id=tenant_id)
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------------ the window


async def test_the_window_slides_instead_of_resetting_on_a_boundary(
    redis_client: Redis,
) -> None:
    limit = Limit(scope=f"test-{uuid.uuid4().hex[:8]}", limit=2, window_s=1.0)
    caller = f"ip:{uuid.uuid4().hex[:8]}"
    at = now_ms()

    assert (await hit(redis_client, caller, limit, at)).allowed
    assert (await hit(redis_client, caller, limit, at + 100)).allowed

    refused = await hit(redis_client, caller, limit, at + 200)
    assert not refused.allowed
    assert refused.retry_after >= 1
    # The refusal removed its own entry: being throttled must not push the retry out.
    assert await redis_client.zcard(f"{KEY_PREFIX}:{limit.scope}:{caller}") == 2

    # Both entries have aged out of the one-second window by now, so the caller is served
    # again — a fixed per-window counter would still be refusing until the boundary.
    assert (await hit(redis_client, caller, limit, at + 1500)).allowed


def test_the_caller_is_its_tenant_when_it_has_one_and_its_address_before_that() -> None:
    tenant_id = str(uuid.uuid4())

    def request(headers: dict[str, str]) -> Request:
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": GUARDED_PATH,
                "query_string": b"",
                "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                "client": ("203.0.113.9", 51234),
            }
        )

    assert caller_key(request(bearer(tenant_id))) == f"tenant:{tenant_id}"
    assert caller_key(request({})) == "ip:203.0.113.9"
    # A token that does not verify is an anonymous caller, never a tenant of its choosing.
    assert caller_key(request({"Authorization": "Bearer not-a-token"})) == "ip:203.0.113.9"


# -------------------------------------------------------------------- the middleware


async def test_the_general_ceiling_answers_429_with_a_retry_after() -> None:
    settings = tight(general=3)
    async with limited_client(settings) as client:
        for _ in range(3):
            assert (await client.get(GUARDED_PATH)).status_code == 401

        refused = await client.get(GUARDED_PATH)
        assert refused.status_code == 429
        assert 1 <= int(refused.headers[RETRY_AFTER_HEADER]) <= settings.general.window_s
        assert "rate limit exceeded" in refused.json()["detail"]


async def test_the_health_probe_and_the_scrape_are_never_throttled() -> None:
    """Compose polls `/health` every few seconds; a throttled probe would kill the api."""
    async with limited_client(tight(general=1)) as client:
        assert (await client.get(GUARDED_PATH)).status_code == 401
        assert (await client.get(GUARDED_PATH)).status_code == 429

        for _ in range(4):
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/metrics")).status_code == 200


async def test_uploads_carry_a_tighter_ceiling_than_reads() -> None:
    async with limited_client(tight(general=100, ingest=1)) as client:
        assert (await client.post("/api/v1/ingest", files=CSV_UPLOAD)).status_code == 401

        refused = await client.post("/api/v1/ingest", files=CSV_UPLOAD)
        assert refused.status_code == 429
        assert refused.headers[RETRY_AFTER_HEADER]

        # The ingest ceiling is scoped to the upload: reads are still served.
        assert (await client.get(GUARDED_PATH)).status_code == 401


async def test_two_tenants_do_not_share_a_bucket() -> None:
    """One tenant exhausting its budget must not be able to throttle another."""
    noisy, quiet = bearer(str(uuid.uuid4())), bearer(str(uuid.uuid4()))
    async with limited_client(tight(general=1)) as client:
        assert (await client.get(GUARDED_PATH, headers=noisy)).status_code != 429
        assert (await client.get(GUARDED_PATH, headers=noisy)).status_code == 429
        assert (await client.get(GUARDED_PATH, headers=quiet)).status_code != 429


def test_the_configured_ceilings_are_the_ones_the_spec_asks_for() -> None:
    """§8: 100 requests a minute in general, 5 uploads an hour — relaxed only in dev."""
    cfg = get_config()
    multiplier = int(cfg["api"]["rate_limit"]["dev_multiplier"])

    prod = rate_limit_settings({**cfg, "env": "prod"})
    assert (prod.general.limit, prod.general.window_s) == (100, 60)
    assert (prod.ingest.limit, prod.ingest.window_s) == (5, 3600)

    dev = rate_limit_settings({**cfg, "env": "dev"})
    assert dev.general.limit == prod.general.limit * multiplier
    assert dev.ingest.limit == prod.ingest.limit * multiplier
    assert dev.general.window_s == prod.general.window_s


# ------------------------------------------------------------------- backpressure


def test_a_high_but_steady_backlog_is_not_backpressure() -> None:
    """A burst leaves a big pending count. It is the growth that means falling behind."""
    monitor = BacklogMonitor(warn_after_s=30.0)
    for now in range(0, 300, 5):
        assert monitor.observe({"forecasts": 500}, float(now)) == []
    assert monitor.growing == []


def test_a_backlog_that_only_grows_warns_and_stops_when_it_recovers() -> None:
    monitor = BacklogMonitor(warn_after_s=30.0)
    for step, now in enumerate(range(0, 30, 5)):
        assert monitor.observe({"raw_events": 10 + step}, float(now)) == []

    assert monitor.observe({"raw_events": 40}, 30.0) == ["raw_events"]
    assert monitor.growing == ["raw_events"]

    # A count that holds ends the streak: the stage caught up, so the warning stops.
    assert monitor.observe({"raw_events": 40}, 35.0) == []
    assert monitor.growing == []


def test_the_backlog_thresholds_come_from_config() -> None:
    poll_s, warn_after_s = backlog_settings()
    api = get_config()["api"]
    assert (poll_s, warn_after_s) == (api["backlog_poll_s"], api["backlog_warn_after_s"])
    assert warn_after_s == 30  # §7: half a minute of monotone growth


async def test_health_names_the_groups_that_are_falling_behind() -> None:
    app = create_app()
    for step, now in enumerate(range(0, 30, 5)):
        app.state.backlog.observe({"state_vectors": 10 + step}, float(now))
    app.state.backlog.observe({"state_vectors": 99}, 30.0)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        body = (await client.get("/health")).json()
    assert body["backpressure"] == ["state_vectors"]


# ----------------------------------------------------------------- the demo capture


def test_the_demo_capture_is_long_enough_to_forecast_from() -> None:
    """A fixture shorter than L + K windows replays in full and forecasts nothing."""
    cfg = get_config()
    lines = DEFAULT_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0] == CSV_HEADER

    hosts = {line.split(",")[1] for line in lines[1:]}
    assert ESCALATING_HOST in hosts
    assert set(BENIGN_HOSTS) <= hosts

    windows = window_count(cfg)
    assert windows >= int(cfg["context_L"]) + int(cfg["horizon_K"])
    timestamps = {line.split(",")[6] for line in lines[1:]}
    assert len(timestamps) >= windows

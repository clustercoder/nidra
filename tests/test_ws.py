"""P9: the live forecast socket, and the fan-out bug the doc's design would have shipped.

Run against a real uvicorn server and a real websocket client rather than an ASGI test
transport, because the property under test is concurrency: two sockets open at once, both
fed from one Redis consumer group. `IMPLEMENTATION-Backend.md` §8 gives each socket its
own consumer *name* inside that group, which splits the stream between them — with two
browser tabs open, each sees about half the forecasts. The first test here is that
regression, written so it fails against the doc's design.

The fan-out reads a stream private to each test, so nothing here depends on — or
disturbs — whatever else is on `forecasts`.
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
from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy import text
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from api.main import create_app
from api.ws import ConnectionManager, ForecastFanout, parse_hosts, ws_settings
from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.bus import Bus, create_redis
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.schemas import Forecast, HorizonPoint

PASSWORD = "correct-horse-battery"

BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

MODEL_VERSION = "nidra-0.1.0-stub"

HOST_A = "192.168.10.50"
HOST_B = "192.168.10.51"

#: Long enough for a message to cross Redis and the socket, short enough that a broken
#: fan-out fails the run in seconds rather than minutes.
RECV_TIMEOUT = 5.0

#: How long "nothing arrives" is given to be wrong.
SILENCE_TIMEOUT = 1.5

#: Bus block interval in tests: shutdown waits at most this long for the read to return.
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
async def forecasts_stream(redis_client: Redis) -> AsyncIterator[str]:
    """A private `forecasts`-shaped stream, deleted afterwards."""
    name = f"test:forecasts:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


@pytest.fixture
async def publisher(redis_client: Redis, forecasts_stream: str) -> Bus:
    """Stands in for the inference worker: publishes onto the private stream."""
    return Bus(redis_client, forecasts_stream, group="test", consumer="test")


async def _register(org: str) -> dict[str, str]:
    email = f"p9ws-{uuid.uuid4().hex[:12]}@nidra.test"
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ws.test"
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
            "DELETE FROM users WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM tenants WHERE id = CAST(:tid AS uuid)",
        ):
            await session.execute(text(statement), {"tid": tenant_id})
        await session.commit()
    await dispose_engine()


@pytest.fixture
async def tenant_a() -> AsyncIterator[dict[str, str]]:
    account = await _register("P9 SOC A")
    yield account
    await _purge(account["tenant_id"])


@pytest.fixture
async def tenant_b() -> AsyncIterator[dict[str, str]]:
    account = await _register("P9 SOC B")
    yield account
    await _purge(account["tenant_id"])


class _Server(uvicorn.Server):
    """uvicorn without signal handlers: pytest's event loop is not ours to reconfigure."""

    def install_signal_handlers(self) -> None:
        return


@contextlib.asynccontextmanager
async def running(app) -> AsyncIterator[str]:  # type: ignore[no-untyped-def]
    """Serve `app` on an ephemeral port; yield its `ws://` base URL."""
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
async def base_url(forecasts_stream: str) -> AsyncIterator[str]:
    """A live api process whose fan-out reads this test's private stream."""
    app = create_app()
    app.state.fanout = ForecastFanout(
        app.state.connections, stream=forecasts_stream, block_ms=TEST_BLOCK_MS
    )
    async with running(app) as url:
        yield url


def socket_url(base: str, account: dict[str, str], **params: str) -> str:
    query = "".join(f"&{key}={value}" for key, value in params.items())
    return f"{base}/api/v1/stream/forecast?token={account['token']}{query}"


# ------------------------------------------------------------------ synthetic forecasts


def build_forecast(tenant_id: str, *, host: str, window: int, cfg: dict, peak: float) -> Forecast:
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
        observed_stage="recon",
        observed_risk=0.3,
        top_signals=[],
        driving_window=2,
        model_version=MODEL_VERSION,
    )


# ------------------------------------------------------------------------- socket i/o


async def next_forecast(ws, timeout: float = RECV_TIMEOUT) -> dict:  # type: ignore[no-untyped-def]
    """The next forecast frame, skipping the server's control frames."""
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if message.get("type") in {"ping", "filter"}:
            continue
        return message


async def next_control(ws, kind: str, timeout: float = RECV_TIMEOUT) -> dict:  # type: ignore[no-untyped-def]
    """The next control frame of `kind`. Used to know a filter update has taken effect."""
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if message.get("type") == kind:
            return message


async def assert_silent(ws, timeout: float = SILENCE_TIMEOUT) -> None:  # type: ignore[no-untyped-def]
    """Nothing but heartbeats arrives within `timeout`."""
    with pytest.raises(TimeoutError):
        await next_forecast(ws, timeout=timeout)


# ------------------------------------------------------------------------- the fan-out


async def test_two_clients_on_one_tenant_both_receive_the_same_forecast(
    base_url: str, tenant_a: dict[str, str], publisher: Bus, cfg: dict
) -> None:
    """The regression the doc's one-consumer-per-socket design fails.

    Consumers inside a Redis consumer group split the stream, so under that design each
    of these two sockets would receive one of the two forecasts and neither would receive
    both. One consumer plus in-process fan-out is what makes this pass.
    """
    async with (
        connect(socket_url(base_url, tenant_a)) as first,
        connect(socket_url(base_url, tenant_a)) as second,
    ):
        for window in range(2):
            await publisher.publish(
                build_forecast(tenant_a["tenant_id"], host=HOST_A, window=window, cfg=cfg, peak=0.8)
            )

        for socket in (first, second):
            received = [Forecast.model_validate(await next_forecast(socket)) for _ in range(2)]
            assert [item.origin_ts for item in received] == [
                BASE + timedelta(seconds=int(cfg["window_delta"]) * w) for w in range(2)
            ]
            assert {item.host_id for item in received} == {HOST_A}


async def test_the_forecast_arrives_verbatim(
    base_url: str, tenant_a: dict[str, str], publisher: Bus, cfg: dict
) -> None:
    """What the browser parses is what the inference worker produced, byte for byte."""
    published = build_forecast(tenant_a["tenant_id"], host=HOST_A, window=0, cfg=cfg, peak=0.9)
    async with connect(socket_url(base_url, tenant_a)) as ws:
        await publisher.publish(published)
        received = Forecast.model_validate(await next_forecast(ws))

    assert received == published


# ----------------------------------------------------------------------- host filtering


async def test_a_mid_stream_filter_update_takes_effect(
    base_url: str, tenant_a: dict[str, str], publisher: Bus, cfg: dict
) -> None:
    """The doc's `hosts = set(...)` inside a nested function never leaves that function.

    Here the filter lives on the subscription both socket tasks hold, so the update is
    real — and the server acknowledges it, which is how this test knows when to publish.
    """
    async with connect(socket_url(base_url, tenant_a)) as ws:
        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_B, window=0, cfg=cfg, peak=0.4)
        )
        assert (await next_forecast(ws))["host_id"] == HOST_B, "unfiltered socket sees every host"

        await ws.send(json.dumps({"type": "filter", "hosts": [HOST_A]}))
        acknowledged = await next_control(ws, "filter")
        assert acknowledged["hosts"] == [HOST_A]

        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_B, window=1, cfg=cfg, peak=0.4)
        )
        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_A, window=2, cfg=cfg, peak=0.9)
        )

        received = await next_forecast(ws)
        assert received["host_id"] == HOST_A, "the filtered-out host was delivered"
        await assert_silent(ws)


async def test_a_filter_given_at_connect_time_is_applied(
    base_url: str, tenant_a: dict[str, str], publisher: Bus, cfg: dict
) -> None:
    async with connect(socket_url(base_url, tenant_a, hosts=HOST_A)) as ws:
        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_B, window=0, cfg=cfg, peak=0.4)
        )
        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_A, window=1, cfg=cfg, peak=0.9)
        )
        assert (await next_forecast(ws))["host_id"] == HOST_A
        await assert_silent(ws)


async def test_an_empty_filter_restores_every_host(
    base_url: str, tenant_a: dict[str, str], publisher: Bus, cfg: dict
) -> None:
    async with connect(socket_url(base_url, tenant_a, hosts=HOST_A)) as ws:
        await ws.send(json.dumps({"type": "filter", "hosts": []}))
        assert (await next_control(ws, "filter"))["hosts"] == []

        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_B, window=0, cfg=cfg, peak=0.4)
        )
        assert (await next_forecast(ws))["host_id"] == HOST_B


# ---------------------------------------------------------------------------- tenancy


async def test_a_socket_never_sees_another_tenants_forecast(
    base_url: str,
    tenant_a: dict[str, str],
    tenant_b: dict[str, str],
    publisher: Bus,
    cfg: dict,
) -> None:
    """B's socket is silent through A's forecast, then receives its own.

    Publishing A's first makes the assertion ordered rather than timing-based: if the
    tenant filter leaked, the leaked message would arrive before B's own.
    """
    async with connect(socket_url(base_url, tenant_b)) as ws:
        await publisher.publish(
            build_forecast(tenant_a["tenant_id"], host=HOST_A, window=0, cfg=cfg, peak=0.9)
        )
        await publisher.publish(
            build_forecast(tenant_b["tenant_id"], host=HOST_B, window=1, cfg=cfg, peak=0.2)
        )

        received = await next_forecast(ws)
        assert received["tenant_id"] == tenant_b["tenant_id"]
        assert received["host_id"] == HOST_B
        await assert_silent(ws)


async def test_an_invalid_token_is_refused_at_the_handshake(base_url: str) -> None:
    for token in ("not-a-jwt", ""):
        with pytest.raises(InvalidStatus) as raised:
            async with connect(f"{base_url}/api/v1/stream/forecast?token={token}"):
                pass
        assert raised.value.response.status_code == 403


async def test_a_refresh_token_does_not_open_a_socket(
    base_url: str, tenant_a: dict[str, str]
) -> None:
    """Same secret, different type: a 7-day refresh token is not a socket credential."""
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ws.test"
    ) as client:
        tokens = await client.post(
            "/api/v1/auth/login", json={"email": tenant_a["email"], "password": PASSWORD}
        )
    refresh = tokens.json()["refresh_token"]

    with pytest.raises(InvalidStatus) as raised:
        async with connect(f"{base_url}/api/v1/stream/forecast?token={refresh}"):
            pass
    assert raised.value.response.status_code == 403


# -------------------------------------------------------------------------- heartbeat


async def test_a_silent_socket_is_pinged_and_dropped_after_two_missed_pongs(
    forecasts_stream: str, tenant_a: dict[str, str]
) -> None:
    """The 30 s heartbeat, run at 200 ms so the test is a test rather than a wait."""
    app = create_app()
    app.state.ws_settings.ping_interval_s = 0.2
    app.state.ws_settings.max_missed_pongs = 2
    app.state.fanout = ForecastFanout(
        app.state.connections, stream=forecasts_stream, block_ms=TEST_BLOCK_MS
    )
    async with running(app) as base:
        async with connect(socket_url(base, tenant_a)) as ws:
            assert (await next_control(ws, "ping"))["type"] == "ping"
            assert (await next_control(ws, "ping"))["type"] == "ping"
            # Two pings unanswered: the third interval closes the socket instead.
            with pytest.raises(ConnectionClosed) as raised:
                await asyncio.wait_for(ws.recv(), RECV_TIMEOUT)
            assert raised.value.rcvd is not None
            assert raised.value.rcvd.code == 1001


async def test_a_pong_keeps_the_socket_open(
    forecasts_stream: str, tenant_a: dict[str, str], redis_client: Redis, cfg: dict
) -> None:
    """Answering the heartbeat resets the counter, so a quiet console is not disconnected."""
    app = create_app()
    app.state.ws_settings.ping_interval_s = 0.2
    app.state.fanout = ForecastFanout(
        app.state.connections, stream=forecasts_stream, block_ms=TEST_BLOCK_MS
    )
    publisher = Bus(redis_client, forecasts_stream, group="test", consumer="test")
    async with running(app) as base:
        async with connect(socket_url(base, tenant_a)) as ws:
            for _ in range(4):
                await next_control(ws, "ping")
                await ws.send(json.dumps({"type": "pong"}))

            await publisher.publish(
                build_forecast(tenant_a["tenant_id"], host=HOST_A, window=0, cfg=cfg, peak=0.9)
            )
            assert (await next_forecast(ws))["host_id"] == HOST_A


# ------------------------------------------------------------- fan-out, without a socket


def test_the_queue_drops_its_oldest_entry_when_full() -> None:
    """Overflow policy: the newest forecast is the one the chart is about to draw."""
    manager = ConnectionManager(queue_size=2)
    sub = manager.subscribe("tenant", None)

    assert manager.broadcast("tenant", HOST_A, b"one") == 1
    assert manager.broadcast("tenant", HOST_A, b"two") == 1
    assert manager.broadcast("tenant", HOST_A, b"three") == 1

    assert sub.dropped == 1
    assert [sub.queue.get_nowait()[1] for _ in range(2)] == [b"two", b"three"]


def test_broadcast_reaches_every_socket_of_one_tenant_and_no_other() -> None:
    manager = ConnectionManager(queue_size=4)
    first = manager.subscribe("tenant-a", None)
    second = manager.subscribe("tenant-a", [HOST_B])
    other = manager.subscribe("tenant-b", None)

    assert manager.broadcast("tenant-a", HOST_A, b"payload") == 2
    assert manager.connection_count == 3
    # Queued for both sockets of tenant A; the host filter is applied as each drains.
    assert first.queue.qsize() == second.queue.qsize() == 1
    assert other.queue.qsize() == 0
    assert first.wants(HOST_A) and not second.wants(HOST_A)

    manager.unsubscribe(second)
    assert manager.broadcast("tenant-a", HOST_A, b"payload") == 1


def test_the_host_filter_parses_repeated_and_comma_separated_values() -> None:
    assert parse_hosts(None) is None
    assert parse_hosts([]) is None
    assert parse_hosts([""]) is None
    assert parse_hosts([f"{HOST_A},{HOST_B}"]) == {HOST_A, HOST_B}
    assert parse_hosts([HOST_A, HOST_B]) == {HOST_A, HOST_B}


def test_socket_settings_come_from_the_config(cfg: dict) -> None:
    settings = ws_settings(cfg)
    assert settings.queue_size == int(cfg["api"]["ws_queue_size"])
    assert settings.ping_interval_s == float(cfg["api"]["ws_ping_interval_s"])
    assert settings.max_missed_pongs == int(cfg["api"]["ws_max_missed_pongs"])


async def test_the_fanout_validates_what_it_forwards(cfg: dict) -> None:
    """A malformed forecast is refused at this boundary and left pending, never forwarded."""
    manager = ConnectionManager(queue_size=4)
    sub = manager.subscribe("tenant", None)
    fanout = ForecastFanout(manager, stream="unused", cfg=cfg)

    with pytest.raises(ValidationError):
        await fanout.handle(b'{"tenant_id": "tenant", "host_id": "h"}')
    assert sub.queue.qsize() == 0

"""P8: durable forecasts that survive redelivery, and episodes that carry lead time.

Against the real Redis and Postgres from `docker compose up -d redis postgres` (with
`alembic upgrade head` applied), because two of the four properties under test *are*
datastore behaviour: `ON CONFLICT DO NOTHING` needs the real unique constraint, and the
"not acked" assertion needs a real consumer group's pending list.

What is worth a test here:

* **idempotency** — at-least-once delivery guarantees duplicates, so the second copy of a
  forecast must produce neither a second row nor a second step of the episode lifecycle;
* **causality** — a horizon at or before its own `origin_ts` is refused and left pending,
  because coercing it would put a number in the database that the project's central
  claim rests on being impossible;
* **the episode arc** — one crossing produces exactly one episode, with the interval and
  the peak the console will render as "warned N seconds before onset".
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.bus import Bus, create_redis
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.schemas import Forecast, HorizonPoint
from nidra_common.worker import run_consumer
from services.persister.episodes import (
    EpisodeRules,
    ForecastSummary,
    HostEpisodeState,
    advance,
    episode_key,
    read_state,
)
from services.persister.worker import (
    PersisterWorker,
    assert_causal,
    max_p_compromise,
    tenant_uuid,
)

#: Aligned to an absolute multiple of the 30 s window, as the features service emits them.
BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

HOST = "192.168.10.50"

MODEL_VERSION = "nidra-0.1.0-stub"

TEST_BLOCK_MS = 200


# ------------------------------------------------------------------------- fixtures


@pytest.fixture
def cfg() -> dict:
    return get_config()


@pytest.fixture
def horizon_k(cfg: dict) -> int:
    return int(cfg["horizon_K"])


@pytest.fixture
def window_delta(cfg: dict) -> int:
    return int(cfg["window_delta"])


@pytest.fixture
def rules(cfg: dict) -> EpisodeRules:
    return EpisodeRules(
        risk_threshold=float(cfg["risk_threshold"]),
        close_after=int(cfg["episode_close_after"]),
        lead_time_m=int(cfg["lead_time_m"]),
    )


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def tenant_id() -> str:
    """A real UUID: the persister writes it into a UUID column, as the API's tokens do."""
    return str(uuid.uuid4())


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory, with the engine disposed after each test's event loop closes."""
    factory = get_sessionmaker()
    try:
        yield factory
    finally:
        await dispose_engine()


@pytest.fixture(autouse=True)
async def clean_up(
    redis_client: Redis, sessions: async_sessionmaker[AsyncSession], tenant_id: str
) -> AsyncIterator[None]:
    """Each test owns a fresh tenant, so cleanup is a delete on that tenant alone."""
    yield
    keys = [key async for key in redis_client.scan_iter(match=f"ep:{tenant_id}:*")]
    if keys:
        await redis_client.delete(*keys)
    async with sessions() as session:
        for table in ("forecasts", "episodes"):
            await session.execute(
                text(f"DELETE FROM {table} WHERE tenant_id = :t"), {"t": uuid.UUID(tenant_id)}
            )
        await session.commit()


@pytest.fixture
async def forecasts_stream(redis_client: Redis) -> AsyncIterator[str]:
    """A private `forecasts`-shaped stream, deleted afterwards."""
    name = f"test:forecasts:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


@pytest.fixture
def worker(
    redis_client: Redis, sessions: async_sessionmaker[AsyncSession], cfg: dict
) -> PersisterWorker:
    return PersisterWorker(redis_client, sessions, cfg=cfg)


# ------------------------------------------------------------------ synthetic forecasts


def horizon_point(
    k: int, last_k: int, origin: datetime, window_delta: int, peak: float, *, rise: float
) -> HorizonPoint:
    """One projected step. `peak` is reached at the last k, so the arc has a defined top."""
    p = max(0.0, min(1.0, peak - rise * (1 - k / last_k)))
    return HorizonPoint(
        k=k,
        ts=origin + timedelta(seconds=window_delta * k),
        p_compromise=p,
        ci_low=max(0.0, p - 0.05 * k),
        ci_high=min(1.0, p + 0.05 * k),
        stage_dist=dict.fromkeys(STAGES, 1.0 / len(STAGES)),
        predicted_features=dict.fromkeys(FEATURE_ORDER, 0.0),
    )


def forecast(
    tenant_id: str,
    *,
    window: int,
    horizon_k: int,
    window_delta: int,
    peak: float = 0.9,
    stage: str = "recon",
    host: str = HOST,
) -> Forecast:
    """A schema-valid `Forecast` whose trajectory tops out at `peak`."""
    origin = BASE + timedelta(seconds=window_delta * window)
    return Forecast(
        tenant_id=tenant_id,
        host_id=host,
        origin_ts=origin,
        horizons=[
            horizon_point(k, horizon_k, origin, window_delta, peak, rise=0.2)
            for k in range(1, horizon_k + 1)
        ],
        lead_time_s=float(window_delta * 3),
        observed_stage=stage,
        observed_risk=min(peak, 0.4),
        top_signals=[],
        driving_window=0,
        model_version=MODEL_VERSION,
    )


def acausal(valid: Forecast) -> Forecast:
    """The same forecast with `k=1` moved back onto `origin_ts` — the thing that must fail.

    Built with `model_copy`, which does not revalidate, because `Forecast` refuses to
    construct one. That is the point: only a producer that bypassed the schema could put
    this on the stream, and the persister still has to catch it.
    """
    first = valid.horizons[0].model_copy(update={"ts": valid.origin_ts})
    return valid.model_copy(update={"horizons": [first, *valid.horizons[1:]]})


async def row_count(sessions: async_sessionmaker[AsyncSession], tenant_id: str) -> int:
    async with sessions() as session:
        result = await session.execute(
            text("SELECT count(*) FROM forecasts WHERE tenant_id = :t"),
            {"t": uuid.UUID(tenant_id)},
        )
        return int(result.scalar_one())


async def episode_rows(
    sessions: async_sessionmaker[AsyncSession], tenant_id: str
) -> list[dict[str, object]]:
    async with sessions() as session:
        result = await session.execute(
            text(
                "SELECT id, host_id, started_at, ended_at, peak_risk, stages, first_alert_at "
                "FROM episodes WHERE tenant_id = :t ORDER BY started_at"
            ),
            {"t": uuid.UUID(tenant_id)},
        )
        return [dict(row) for row in result.mappings()]


# ------------------------------------------------------------------- durable forecasts


async def test_three_forecasts_become_three_rows(
    worker: PersisterWorker,
    sessions: async_sessionmaker[AsyncSession],
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    for window in range(3):
        outcome = await worker.process(
            forecast(
                tenant_id,
                window=window,
                horizon_k=horizon_k,
                window_delta=window_delta,
                peak=0.2,
            )
        )
        assert outcome.inserted is True

    assert await row_count(sessions, tenant_id) == 3


async def test_a_redelivered_forecast_does_not_write_a_second_row(
    worker: PersisterWorker,
    sessions: async_sessionmaker[AsyncSession],
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    """At-least-once delivery makes this a routine event, not an edge case."""
    forecasts = [
        forecast(tenant_id, window=window, horizon_k=horizon_k, window_delta=window_delta, peak=0.2)
        for window in range(3)
    ]
    for item in forecasts:
        await worker.process(item)

    repeat = await worker.process(forecasts[1])

    assert repeat.inserted is False
    assert await row_count(sessions, tenant_id) == 3


async def test_the_promoted_columns_and_the_jsonb_payload_agree(
    worker: PersisterWorker,
    sessions: async_sessionmaker[AsyncSession],
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    """The columns exist so the console's queries stay indexed; they must not drift."""
    item = forecast(tenant_id, window=0, horizon_k=horizon_k, window_delta=window_delta, peak=0.83)
    await worker.process(item)

    async with sessions() as session:
        result = await session.execute(
            text(
                "SELECT host_id, origin_ts, observed_risk, observed_stage, max_p_compromise, "
                "lead_time_s, model_version, payload FROM forecasts WHERE tenant_id = :t"
            ),
            {"t": uuid.UUID(tenant_id)},
        )
        row = result.mappings().one()

    assert row["host_id"] == HOST
    assert row["origin_ts"] == item.origin_ts
    assert row["observed_stage"] == item.observed_stage
    assert row["model_version"] == MODEL_VERSION
    assert row["lead_time_s"] == pytest.approx(item.lead_time_s)
    assert row["max_p_compromise"] == pytest.approx(max_p_compromise(item), abs=1e-6)
    assert Forecast.model_validate(row["payload"]) == item


# ------------------------------------------------------------------------- causality


def test_assert_causal_rejects_a_horizon_at_the_origin(
    tenant_id: str, horizon_k: int, window_delta: int
) -> None:
    valid = forecast(tenant_id, window=0, horizon_k=horizon_k, window_delta=window_delta)

    with pytest.raises(ValueError, match="causality"):
        assert_causal(acausal(valid))


async def test_an_acausal_forecast_is_rejected_and_left_pending(
    worker: PersisterWorker,
    redis_client: Redis,
    sessions: async_sessionmaker[AsyncSession],
    forecasts_stream: str,
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    """Refused, unacked, and no row: the message stays for a reclaim rather than vanishing."""
    bus = Bus(redis_client, forecasts_stream, group="persister", consumer="worker-1")
    await bus.ensure_group()
    valid = forecast(tenant_id, window=0, horizon_k=horizon_k, window_delta=window_delta)
    await redis_client.xadd(forecasts_stream, {"payload": acausal(valid).model_dump_json()})

    stop = asyncio.Event()
    attempts: list[bytes] = []

    async def handler(payload: bytes) -> None:
        attempts.append(payload)
        stop.set()
        await worker.handle(payload)

    await asyncio.wait_for(
        run_consumer(bus, handler, block_ms=TEST_BLOCK_MS, stop=stop, install_signals=False),
        timeout=10,
    )

    assert len(attempts) == 1
    assert await bus.pending_count() == 1
    assert await row_count(sessions, tenant_id) == 0


def test_a_non_uuid_tenant_is_refused() -> None:
    """The tenant claim reaches a UUID column; anything else did not come from our tokens."""
    with pytest.raises(ValueError, match="non-UUID tenant_id"):
        tenant_uuid("p7-not-a-uuid")


# -------------------------------------------------------------------------- episodes


async def test_a_risk_arc_over_then_under_produces_exactly_one_episode(
    worker: PersisterWorker,
    sessions: async_sessionmaker[AsyncSession],
    redis_client: Redis,
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
    rules: EpisodeRules,
) -> None:
    """The product fact: one crossing, one episode, with the interval the console renders."""
    quiet_before = 2
    peaks = [0.10, 0.20, 0.80, 0.92, 0.86, 0.40, 0.30, 0.20, 0.10, 0.10]
    stages = [
        "benign",
        "benign",
        "recon",
        "recon",
        "initial_access",
        "initial_access",
        "benign",
        "benign",
        "benign",
        "benign",
    ]

    for window, (peak, stage) in enumerate(zip(peaks, stages, strict=True)):
        await worker.process(
            forecast(
                tenant_id,
                window=window,
                horizon_k=horizon_k,
                window_delta=window_delta,
                peak=peak,
                stage=stage,
            )
        )

    rows = await episode_rows(sessions, tenant_id)
    assert len(rows) == 1
    episode = rows[0]

    def origin(window: int) -> datetime:
        return BASE + timedelta(seconds=window_delta * window)

    first_above = quiet_before
    last_above = 4
    closed_at = last_above + rules.close_after

    assert episode["host_id"] == HOST
    assert episode["started_at"] == origin(first_above)
    assert episode["ended_at"] == origin(closed_at)
    assert episode["peak_risk"] == pytest.approx(max(peaks), abs=1e-6)
    # `lead_time_m` consecutive windows above the line before the alert is raised.
    assert episode["first_alert_at"] == origin(first_above + rules.lead_time_m - 1)
    assert episode["stages"] == ["recon", "initial_access", "benign"]

    # Closed: the aggregate is gone from Redis, the watermark that blocks a redelivery stays.
    state = await read_state(redis_client, tenant_id, HOST)
    assert state.episode is None
    assert state.last_ts == origin(len(peaks) - 1)


async def test_an_open_episode_has_no_ended_at_until_it_closes(
    worker: PersisterWorker,
    sessions: async_sessionmaker[AsyncSession],
    redis_client: Redis,
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
    rules: EpisodeRules,
) -> None:
    for window, peak in enumerate([0.90, 0.95, 0.30]):
        await worker.process(
            forecast(
                tenant_id,
                window=window,
                horizon_k=horizon_k,
                window_delta=window_delta,
                peak=peak,
            )
        )

    rows = await episode_rows(sessions, tenant_id)
    assert len(rows) == 1
    assert rows[0]["ended_at"] is None
    assert rows[0]["peak_risk"] == pytest.approx(0.95, abs=1e-6)

    state = await read_state(redis_client, tenant_id, HOST)
    assert state.episode is not None
    assert state.episode.below == 1
    assert state.episode.id == rows[0]["id"]


async def test_a_redelivered_forecast_does_not_advance_the_episode(
    worker: PersisterWorker,
    redis_client: Redis,
    sessions: async_sessionmaker[AsyncSession],
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    """A duplicate below-threshold forecast must not count twice toward the close."""
    opening = forecast(
        tenant_id, window=0, horizon_k=horizon_k, window_delta=window_delta, peak=0.90
    )
    cooling = forecast(
        tenant_id, window=1, horizon_k=horizon_k, window_delta=window_delta, peak=0.20
    )
    await worker.process(opening)
    await worker.process(cooling)

    repeat = await worker.process(cooling)

    assert repeat.transition is None
    state = await read_state(redis_client, tenant_id, HOST)
    assert state.episode is not None
    assert state.episode.below == 1
    assert (await episode_rows(sessions, tenant_id))[0]["ended_at"] is None


async def test_hosts_keep_independent_episodes(
    worker: PersisterWorker,
    sessions: async_sessionmaker[AsyncSession],
    redis_client: Redis,
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    other = "192.168.10.51"
    for window in range(2):
        await worker.process(
            forecast(
                tenant_id,
                window=window,
                horizon_k=horizon_k,
                window_delta=window_delta,
                peak=0.90,
            )
        )
        await worker.process(
            forecast(
                tenant_id,
                window=window,
                horizon_k=horizon_k,
                window_delta=window_delta,
                peak=0.10,
                host=other,
            )
        )

    rows = await episode_rows(sessions, tenant_id)
    assert [row["host_id"] for row in rows] == [HOST]
    assert await redis_client.hget(episode_key(tenant_id, other), "id") is None


async def test_the_bus_handler_stores_what_it_consumed(
    worker: PersisterWorker,
    redis_client: Redis,
    sessions: async_sessionmaker[AsyncSession],
    tenant_id: str,
    horizon_k: int,
    window_delta: int,
) -> None:
    """The path the consumer loop actually takes: raw JSON bytes in, a row out."""
    item = forecast(tenant_id, window=0, horizon_k=horizon_k, window_delta=window_delta)
    await worker.handle(item.model_dump_json().encode())

    assert await row_count(sessions, tenant_id) == 1


# ------------------------------------------------------- lifecycle, without a datastore


def test_a_single_spike_never_raises_an_alert(rules: EpisodeRules, window_delta: int) -> None:
    """One window over the line opens an episode but does not warn — that is `lead_time_m`."""
    state = HostEpisodeState()
    opened = advance(
        state, ForecastSummary(origin_ts=BASE, max_p_compromise=0.9, observed_stage="recon"), rules
    )

    assert opened is not None
    assert opened.opened is True
    assert opened.episode is not None
    assert opened.episode.first_alert_at is None

    closing = opened.state
    for step in range(1, rules.close_after + 1):
        transition = advance(
            closing,
            ForecastSummary(
                origin_ts=BASE + timedelta(seconds=window_delta * step),
                max_p_compromise=0.1,
                observed_stage="benign",
            ),
            rules,
        )
        assert transition is not None
        closing = transition.state

    assert transition.closed is True
    assert closing.episode is None
    assert transition.episode is not None
    assert transition.episode.first_alert_at is None


def test_a_forecast_older_than_the_watermark_is_skipped(rules: EpisodeRules) -> None:
    state = HostEpisodeState(last_ts=BASE)
    assert advance(state, ForecastSummary(origin_ts=BASE, max_p_compromise=0.9), rules) is None
    assert (
        advance(
            state,
            ForecastSummary(origin_ts=BASE - timedelta(seconds=30), max_p_compromise=0.9),
            rules,
        )
        is None
    )

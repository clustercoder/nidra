"""P7: the sequence buffer, the stateless worker, and the stub predictor.

Against the real Redis from `docker compose up -d redis postgres`, because the claim
under test is *where the state lives*. A mocked Redis would prove the mock is shared;
only a real one proves two worker instances see the same buffer.

The three properties worth having a test for:

* **L-window gating** — nothing is forecast until the context is full, and then exactly
  one forecast per window;
* **idempotency** — a redelivered `StateVector` neither lengthens the buffer nor produces
  a second forecast, which at-least-once delivery makes a routine event rather than an
  edge case;
* **statelessness** — a second worker instance, sharing nothing but Redis, continues a
  sequence the first one started. That is the scale-out claim, stated as an assertion.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from redis.asyncio import Redis

from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.bus import Bus, create_redis
from nidra_common.config import get_config
from nidra_common.schemas import Forecast, StateVector
from services.inference.predictor_loader import (
    Predictor,
    configure_torch_threads,
    load_predictor,
)
from services.inference.stub_predictor import (
    COUNTERFACTUAL_LABEL,
    DRIVERS,
    FEATURE_INDEX,
    MODEL_VERSION,
    StubPredictor,
    recent_slope,
    stage_distribution,
)
from services.inference.worker import InferenceWorker, seq_key, states_array

#: Aligned to an absolute multiple of the 30 s window, as the feature service emits them.
BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

HOST = "192.168.10.50"

#: The stub is arithmetic, so the 300 ms serving target should be met with room to spare.
LATENCY_BUDGET_S = 0.3


# ------------------------------------------------------------------------- fixtures


@pytest.fixture
def cfg() -> dict:
    return get_config()


@pytest.fixture
def context_l(cfg: dict) -> int:
    return int(cfg["context_L"])


@pytest.fixture
def horizon_k(cfg: dict) -> int:
    return int(cfg["horizon_K"])


@pytest.fixture
def window_delta(cfg: dict) -> int:
    return int(cfg["window_delta"])


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def tenant_id() -> str:
    return f"p7-{uuid.uuid4().hex[:12]}"


@pytest.fixture(autouse=True)
async def clean_sequence_buffers(redis_client: Redis, tenant_id: str) -> AsyncIterator[None]:
    yield
    keys = [key async for key in redis_client.scan_iter(match=f"seq:{tenant_id}:*")]
    if keys:
        await redis_client.delete(*keys)


@pytest.fixture
async def forecasts_stream(redis_client: Redis) -> AsyncIterator[str]:
    """A private `forecasts`-shaped stream, deleted afterwards."""
    name = f"test:forecasts:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


@pytest.fixture
def predictor(cfg: dict) -> StubPredictor:
    return StubPredictor(cfg=cfg)


@pytest.fixture
def context(tenant_id: str, context_l: int, window_delta: int) -> list[StateVector]:
    """L windows of one escalating host — the shape the world model is meant to read."""
    return escalating_context(tenant_id, length=context_l, window_delta=window_delta)


@pytest.fixture
def states(context: list[StateVector]) -> np.ndarray:
    return states_array(context)


def build_worker(
    redis_client: Redis, forecasts_stream: str, predictor: Predictor, cfg: dict
) -> InferenceWorker:
    """A worker instance. Everything it knows about a host is in Redis, so these are
    interchangeable — which is exactly what the statelessness test exercises."""
    bus = Bus(redis_client, forecasts_stream, group="persister", consumer="test")
    return InferenceWorker(redis_client, bus, predictor, cfg=cfg)


@pytest.fixture
def worker(
    redis_client: Redis, forecasts_stream: str, predictor: StubPredictor, cfg: dict
) -> InferenceWorker:
    return build_worker(redis_client, forecasts_stream, predictor, cfg)


# ------------------------------------------------------------------ synthetic context


def state_vector(
    tenant_id: str,
    *,
    window: int,
    window_delta: int,
    escalating: bool = True,
    host: str = HOST,
) -> StateVector:
    """One window of a host either escalating or sitting quiet.

    The escalating shape mirrors the PRD's worked example — SYN ratio climbing, ports and
    peers fanning out — so the surrogate risk has something to lean on.
    """
    features = dict.fromkeys(FEATURE_ORDER, 0.0)
    features["is_active"] = 1.0
    if escalating:
        step = window + 1
        features["syn_ratio"] = min(0.95, 0.03 * step)
        features["dst_port_entropy"] = 0.15 * step
        features["new_peer_count"] = 2.0 * step
        features["out_degree"] = 2.0 * step
        features["d_out_degree"] = 2.0
        features["bytes_total"] = 1000.0 * step
    return StateVector(
        tenant_id=tenant_id,
        host_id=host,
        window_ts=BASE + timedelta(seconds=window * window_delta),
        features=features,
    )


def escalating_context(tenant_id: str, *, length: int, window_delta: int) -> list[StateVector]:
    return [
        state_vector(tenant_id, window=window, window_delta=window_delta)
        for window in range(length)
    ]


async def published(redis_client: Redis, stream: str) -> list[Forecast]:
    entries = await redis_client.xrange(stream)
    return [Forecast.model_validate_json(fields[b"payload"]) for _, fields in entries]


# ----------------------------------------------------------------- the stub predictor


def test_stub_states_validation_fails_loudly_on_a_schema_mismatch(
    predictor: StubPredictor, context_l: int
) -> None:
    with pytest.raises(ValueError, match="FEATURE_ORDER"):
        predictor.forecast(np.zeros((context_l, len(FEATURE_ORDER) - 1)), HOST, BASE)
    with pytest.raises(ValueError, match=r"\[L, F\]"):
        predictor.forecast(np.zeros(len(FEATURE_ORDER)), HOST, BASE)


def test_recent_slope_is_backward_looking_and_zero_padded() -> None:
    assert recent_slope(np.array([1.0])) == 0.0
    assert recent_slope(np.array([1.0, 2.0])) == 0.0
    assert recent_slope(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert recent_slope(np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    # Only the last three windows count; the distant past cannot move the slope.
    assert recent_slope(np.array([99.0, 0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_stage_distribution_shifts_benign_to_recon_to_initial_access() -> None:
    for risk in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert sum(stage_distribution(risk).values()) == pytest.approx(1.0, abs=1e-5)
        assert set(stage_distribution(risk)) == set(STAGES)

    quiet, middling, hot = (stage_distribution(risk) for risk in (0.05, 0.5, 0.95))
    assert quiet["benign"] > quiet["recon"] > quiet["initial_access"]
    assert middling["recon"] > middling["benign"]
    assert hot["initial_access"] > hot["recon"] > hot["benign"]


def test_stub_forecast_is_schema_valid_and_deterministic(
    predictor: StubPredictor,
    states: np.ndarray,
    tenant_id: str,
    context_l: int,
    window_delta: int,
    horizon_k: int,
) -> None:
    origin = BASE + timedelta(seconds=(context_l - 1) * window_delta)

    result = predictor.forecast(states, HOST, origin)
    forecast = Forecast(tenant_id=tenant_id, **result)

    assert forecast.model_version == MODEL_VERSION
    assert [point.k for point in forecast.horizons] == list(range(1, horizon_k + 1))
    assert forecast.horizons[-1].ts == origin + timedelta(seconds=horizon_k * window_delta)
    assert predictor.forecast(states, HOST, origin) == result


def test_stub_risk_rises_with_the_horizon_and_the_band_widens(
    predictor: StubPredictor, states: np.ndarray
) -> None:
    result = predictor.forecast(states, HOST, BASE)

    risks = [point["p_compromise"] for point in result["horizons"]]
    widths = [point["ci_high"] - point["ci_low"] for point in result["horizons"]]

    assert risks == sorted(risks) and risks[0] < risks[-1]
    assert widths == sorted(widths) and widths[0] < widths[-1]
    for point in result["horizons"]:
        assert 0.0 <= point["ci_low"] <= point["p_compromise"] <= point["ci_high"] <= 1.0


def test_stub_lead_time_needs_consecutive_windows_over_the_threshold(
    predictor: StubPredictor, states: np.ndarray, cfg: dict, window_delta: int
) -> None:
    result = predictor.forecast(states, HOST, BASE)

    risks = [point["p_compromise"] for point in result["horizons"]]
    threshold = float(cfg["risk_threshold"])
    run = int(cfg["lead_time_m"])

    crossings = [
        index
        for index in range(len(risks) - run + 1)
        if all(risk >= threshold for risk in risks[index : index + run])
    ]
    # The fixture is built to cross, so this test cannot pass vacuously.
    assert crossings, f"escalating fixture never reaches {threshold}: {risks}"
    assert result["lead_time_s"] == pytest.approx((crossings[0] + 1) * window_delta)
    # A single window over the line is not a crossing: the run has to be sustained.
    assert risks[crossings[0]] >= threshold and risks[crossings[0] + run - 1] >= threshold
    assert all(risk < threshold for risk in risks[: crossings[0]])


def test_stub_lead_time_is_none_for_a_quiet_host(
    predictor: StubPredictor, tenant_id: str, context_l: int, window_delta: int
) -> None:
    quiet = [
        state_vector(tenant_id, window=window, window_delta=window_delta, escalating=False)
        for window in range(context_l)
    ]
    result = predictor.forecast(states_array(quiet), HOST, BASE)

    assert result["lead_time_s"] is None
    assert result["observed_stage"] == "benign"
    assert result["observed_risk"] < 0.5


def test_stub_top_signals_and_driving_window_are_plausible(
    predictor: StubPredictor, states: np.ndarray, context_l: int
) -> None:
    result = predictor.forecast(states, HOST, BASE)

    signals = result["top_signals"]
    assert [signal["name"] for signal in signals] == sorted(
        [signal["name"] for signal in signals],
        key=lambda name: -abs(next(s["shap_value"] for s in signals if s["name"] == name)),
    )
    assert {signal["name"] for signal in signals} <= set(FEATURE_ORDER)
    assert all(signal["direction"] in {"up", "down"} for signal in signals)
    assert all(signal["display"] for signal in signals)
    # Three drivers climb in this fixture; `out_degree` grows linearly, so its backward
    # difference is constant and the copy says steady rather than inventing a trend.
    by_name = {signal["name"]: signal for signal in signals}
    for name in ("syn_ratio", "dst_port_entropy", "new_peer_count"):
        assert by_name[name]["direction"] == "up"
        assert by_name[name]["display"].endswith("rising")
    assert by_name["d_out_degree"]["display"].endswith("steady")
    assert 0 <= result["driving_window"] < context_l


def test_stub_predicted_features_extrapolate_only_the_drivers(
    predictor: StubPredictor, context: list[StateVector], states: np.ndarray
) -> None:
    result = predictor.forecast(states, HOST, BASE)
    last = context[-1].features

    for point in result["horizons"]:
        projected = point["predicted_features"]
        assert set(projected) == set(FEATURE_ORDER)
        for name in FEATURE_ORDER:
            if name not in DRIVERS:
                assert projected[name] == pytest.approx(last[name])
    # Rising drivers keep rising across the horizon.
    trajectory = [point["predicted_features"]["new_peer_count"] for point in result["horizons"]]
    assert trajectory == sorted(trajectory) and trajectory[0] > last["new_peer_count"]
    # `syn_ratio` is a ratio and stays one, however long the extrapolation runs.
    for point in result["horizons"]:
        assert 0.0 <= point["predicted_features"]["syn_ratio"] <= 1.0


def test_stub_counterfactual_is_labelled_and_bends_the_curve(
    predictor: StubPredictor, states: np.ndarray
) -> None:
    result = predictor.counterfactual(states, "syn_ratio", 0.0)

    assert result["label"] == COUNTERFACTUAL_LABEL
    assert result["feature"] == "syn_ratio"
    original = [point["p_compromise"] for point in result["original"]]
    modified = [point["p_compromise"] for point in result["counterfactual"]]
    assert len(original) == len(modified)
    assert all(new < old for new, old in zip(modified, original, strict=True))


def test_stub_counterfactual_rejects_an_unknown_feature(
    predictor: StubPredictor, states: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="FEATURE_ORDER"):
        predictor.counterfactual(states, "not_a_feature", 0.0)


def test_stub_explain_returns_window_importance_over_the_context(
    predictor: StubPredictor, states: np.ndarray, context_l: int, horizon_k: int
) -> None:
    result = predictor.explain(states, horizon_k)

    assert result["horizon_k"] == horizon_k
    assert len(result["window_importance"]) == context_l
    assert sum(result["window_importance"]) == pytest.approx(1.0, abs=1e-4)
    assert result["window_importance"][0] == 0.0  # no predecessor inside the context
    assert 0 <= result["driving_window"] < context_l

    with pytest.raises(ValueError, match="horizon_k"):
        predictor.explain(states, horizon_k + 1)


def test_stub_forecast_latency_is_under_the_serving_budget(
    predictor: StubPredictor, states: np.ndarray
) -> None:
    started = time.perf_counter()
    predictor.forecast(states, HOST, BASE)
    elapsed = time.perf_counter() - started

    assert elapsed < LATENCY_BUDGET_S


# --------------------------------------------------------------------- predictor load


def test_loader_returns_the_stub_under_the_default_config(cfg: dict) -> None:
    predictor = load_predictor(cfg)
    assert isinstance(predictor, StubPredictor)
    assert isinstance(predictor, Predictor)


def test_loader_rejects_an_unknown_impl(cfg: dict) -> None:
    with pytest.raises(ValueError, match="predictor.impl"):
        load_predictor({**cfg, "predictor": {"impl": "magic"}})


def test_torch_thread_cap_is_guarded_by_import_availability() -> None:
    # Either torch is installed and the cap applied, or it is not and this is a no-op.
    assert configure_torch_threads() in {True, False}


# ------------------------------------------------------------------ the worker buffer


async def test_no_forecast_until_the_context_is_full(
    worker: InferenceWorker,
    redis_client: Redis,
    forecasts_stream: str,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    context = escalating_context(tenant_id, length=context_l, window_delta=window_delta)

    for vector in context[:-1]:
        assert await worker.process(vector) is None

    assert await published(redis_client, forecasts_stream) == []
    assert await redis_client.llen(seq_key(tenant_id, HOST)) == context_l - 1


async def test_the_lth_window_produces_exactly_one_forecast_anchored_at_it(
    worker: InferenceWorker,
    redis_client: Redis,
    forecasts_stream: str,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    context = escalating_context(tenant_id, length=context_l, window_delta=window_delta)
    for vector in context[:-1]:
        await worker.process(vector)

    forecast = await worker.process(context[-1])

    assert forecast is not None
    stream = await published(redis_client, forecasts_stream)
    assert len(stream) == 1
    assert stream[0].tenant_id == tenant_id
    assert stream[0].host_id == HOST
    assert stream[0].origin_ts == context[-1].window_ts
    # The causality contract: every projected window is strictly after the origin.
    assert all(point.ts > stream[0].origin_ts for point in stream[0].horizons)


async def test_a_redelivered_vector_neither_grows_the_buffer_nor_forecasts_twice(
    worker: InferenceWorker,
    redis_client: Redis,
    forecasts_stream: str,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    """At-least-once delivery makes this routine, not exotic (Standing Rules correction 4)."""
    context = escalating_context(tenant_id, length=context_l, window_delta=window_delta)
    for vector in context:
        await worker.process(vector)

    length_before = await redis_client.llen(seq_key(tenant_id, HOST))

    assert await worker.process(context[-1]) is None

    assert await redis_client.llen(seq_key(tenant_id, HOST)) == length_before
    assert len(await published(redis_client, forecasts_stream)) == 1


async def test_the_buffer_is_trimmed_to_the_context_length(
    worker: InferenceWorker,
    redis_client: Redis,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    extra = 5
    context = escalating_context(tenant_id, length=context_l + extra, window_delta=window_delta)
    for vector in context:
        await worker.process(vector)

    key = seq_key(tenant_id, HOST)
    assert await redis_client.llen(key) == context_l
    raw_buffer = await redis_client.lrange(key, 0, -1)
    buffered = [StateVector.model_validate_json(raw) for raw in raw_buffer]
    assert [vector.window_ts for vector in buffered] == [
        vector.window_ts for vector in context[-context_l:]
    ]
    assert 0 < await redis_client.ttl(key) <= 3600


async def test_a_second_worker_continues_the_sequence_from_redis(
    redis_client: Redis,
    forecasts_stream: str,
    predictor: StubPredictor,
    cfg: dict,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    """The scale-out claim: workers share nothing but Redis, so either can take the next
    window of a host the other has been handling."""
    first = build_worker(redis_client, forecasts_stream, predictor, cfg)
    second = build_worker(redis_client, forecasts_stream, StubPredictor(cfg=cfg), cfg)

    context = escalating_context(tenant_id, length=context_l, window_delta=window_delta)
    for vector in context[:-1]:
        assert await first.process(vector) is None

    forecast = await second.process(context[-1])

    assert forecast is not None
    assert forecast.origin_ts == context[-1].window_ts
    assert len(await published(redis_client, forecasts_stream)) == 1
    # And the dedupe is in Redis too, so the first worker sees the redelivery as well.
    assert await first.process(context[-1]) is None
    assert len(await published(redis_client, forecasts_stream)) == 1


async def test_workers_hold_nothing_about_a_host_between_messages(
    redis_client: Redis,
    forecasts_stream: str,
    predictor: StubPredictor,
    cfg: dict,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    """Alternating workers window by window must produce what one worker alone would."""
    context = escalating_context(tenant_id, length=context_l, window_delta=window_delta)
    workers = [
        build_worker(redis_client, forecasts_stream, predictor, cfg),
        build_worker(redis_client, forecasts_stream, StubPredictor(cfg=cfg), cfg),
    ]

    alternating: Forecast | None = None
    for index, vector in enumerate(context):
        alternating = await workers[index % 2].process(vector) or alternating

    solo = predictor.forecast(states_array(context), HOST, context[-1].window_ts)

    assert alternating is not None
    assert alternating.model_dump(mode="json") == Forecast(tenant_id=tenant_id, **solo).model_dump(
        mode="json"
    )


async def test_the_bus_handler_publishes_a_schema_valid_forecast(
    worker: InferenceWorker,
    redis_client: Redis,
    forecasts_stream: str,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    """The path the consumer loop actually takes: raw JSON bytes in, `Forecast` out."""
    for vector in escalating_context(tenant_id, length=context_l, window_delta=window_delta):
        await worker.handle(vector.model_dump_json().encode())

    stream = await published(redis_client, forecasts_stream)
    assert len(stream) == 1
    assert stream[0].model_version == MODEL_VERSION
    assert set(stream[0].horizons[0].predicted_features) == set(FEATURE_ORDER)


async def test_hosts_are_buffered_independently(
    worker: InferenceWorker,
    redis_client: Redis,
    forecasts_stream: str,
    tenant_id: str,
    context_l: int,
    window_delta: int,
) -> None:
    other = "192.168.10.51"
    for window in range(context_l):
        await worker.process(state_vector(tenant_id, window=window, window_delta=window_delta))
        if window < context_l - 1:
            await worker.process(
                state_vector(tenant_id, window=window, window_delta=window_delta, host=other)
            )

    stream = await published(redis_client, forecasts_stream)
    assert [forecast.host_id for forecast in stream] == [HOST]
    assert await redis_client.llen(seq_key(tenant_id, other)) == context_l - 1


def test_states_array_is_built_in_feature_order(tenant_id: str, window_delta: int) -> None:
    context = escalating_context(tenant_id, length=3, window_delta=window_delta)
    states = states_array(context)

    assert states.shape == (3, len(FEATURE_ORDER))
    for name, index in FEATURE_INDEX.items():
        assert states[-1, index] == pytest.approx(context[-1].features[name])

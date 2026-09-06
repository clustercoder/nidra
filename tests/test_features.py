"""P6: windowing, the 45 features, the watchdog, and silent windows.

Against the real Redis from `docker compose up -d redis postgres`, because the window
state is the thing under test and a mocked Redis would only prove the mock works.

The escalating-scan fixture mirrors the PRD's worked example: a host that starts with a
couple of destinations on a couple of ports and ends fanning out across many, its SYN
ratio climbing and its inter-arrival variance collapsing as the scan becomes mechanical.
The assertions are on *trends* rather than values — the point of the feature set is that
the trajectory is legible, and a test pinned to exact numbers would have to be rewritten
every time a denominator is reconsidered.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis

from nidra.data.schema import FEATURE_ORDER
from nidra_common.bus import Bus, create_redis
from nidra_common.config import get_config
from nidra_common.events import RawEvent
from nidra_common.schemas import StateVector
from services.features.compute import (
    WindowAccumulator,
    WindowGraph,
    ols_slope,
    percentile,
    safe_div,
    shannon_entropy,
    validate_features,
    variance,
)
from services.features.state import align, window_key
from services.features.worker import FeaturesWorker

#: Aligned to an absolute multiple of the 30 s window, as the ML doc requires.
BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

SCAN_HOST = "192.168.10.50"
WINDOWS = 6
FLOWS_PER_WINDOW = 8
PACKETS_PER_FLOW = 10.0


# ------------------------------------------------------------------------- fixtures


class FakeWall:
    """Wall clock the test advances by hand, so the watchdog does not cost real seconds."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def tenant_id() -> str:
    return f"p6-{uuid.uuid4().hex[:12]}"


@pytest.fixture
async def state_vectors_stream(redis_client: Redis) -> AsyncIterator[str]:
    """A private `state_vectors`-shaped stream, deleted afterwards."""
    name = f"test:state_vectors:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


@pytest.fixture
async def worker(
    redis_client: Redis, state_vectors_stream: str, tenant_id: str
) -> AsyncIterator[FeaturesWorker]:
    """A worker on a fake wall clock, with every `feat:*` key it wrote removed after."""
    bus = Bus(redis_client, state_vectors_stream, group="inference", consumer="test")
    instance = FeaturesWorker(redis_client, bus, cfg=get_config(), now=FakeWall())
    try:
        yield instance
    finally:
        keys = [key async for key in redis_client.scan_iter(match=f"feat:*{tenant_id}*")]
        if keys:
            await redis_client.delete(*keys)


def wall(worker: FeaturesWorker) -> FakeWall:
    clock = worker._now
    assert isinstance(clock, FakeWall)
    return clock


# ----------------------------------------------------------------- synthetic capture


def flow_event(
    tenant_id: str,
    *,
    ts: datetime,
    src: str,
    dst: str,
    dst_port: int,
    syn_count: float,
    iat_mean: float,
) -> RawEvent:
    """One CIC-shaped flow, in the normalised form ingest publishes."""
    return RawEvent(
        tenant_id=tenant_id,
        job_id="p6",
        kind="flow",
        ts=ts,
        src_ip=src,
        dst_ip=dst,
        src_port=50000,
        dst_port=dst_port,
        protocol=6,
        fields={
            "duration": 1.5,
            "bytes_fwd": 720.0,
            "bytes_bwd": 480.0,
            "pkts_fwd": 6.0,
            "pkts_bwd": 4.0,
            "syn_count": syn_count,
            "ack_count": 1.0,
            "rst_count": 0.0,
            "fin_count": 0.0,
            "psh_count": 0.0,
            "urg_count": 0.0,
            "iat_mean": iat_mean,
            "iat_std": 0.0,
            "iat_max": 0.4,
        },
    )


def escalating_scan(tenant_id: str) -> list[RawEvent]:
    """Six windows of a host escalating from a couple of peers to a spreading scan.

    Per window `w`: `w + 2` destination ports, `w + 1` destinations never contacted
    before, a SYN ratio of `(w + 1) / 10`, and per-flow inter-arrival times converging
    on a constant — the mechanical regularity that separates a scan from a person.
    """
    events: list[RawEvent] = []
    peer = 0
    for window in range(WINDOWS):
        ports = [80 + index for index in range(window + 2)]
        destinations = []
        for _ in range(window + 1):
            peer += 1
            destinations.append(f"10.0.0.{peer}")
        spread = 0.4 / (window + 1)
        for index in range(FLOWS_PER_WINDOW):
            events.append(
                flow_event(
                    tenant_id,
                    ts=BASE + timedelta(seconds=window * 30 + index),
                    src=SCAN_HOST,
                    dst=destinations[index % len(destinations)],
                    dst_port=ports[index % len(ports)],
                    syn_count=float(window + 1),
                    iat_mean=0.1 + index * spread,
                )
            )
    return events


async def drain(worker: FeaturesWorker, events: list[RawEvent]) -> list[StateVector]:
    """Feed every event, then let the watchdog close the final window."""
    emitted: list[StateVector] = []
    for event in events:
        emitted.extend(await worker.ingest_event(event))
    wall(worker).advance(worker.grace_s + 1)
    emitted.extend(await worker.watchdog_tick())
    return emitted


# ------------------------------------------------------------------ pure computation


def test_safe_div_returns_zero_rather_than_nan() -> None:
    assert safe_div(1.0, 0.0) == 0.0
    assert safe_div(0.0, 0.0) == 0.0
    assert safe_div(3.0, 4.0) == pytest.approx(0.75)


def test_variance_is_clamped_at_zero_on_a_constant_series() -> None:
    assert variance(total=30.0, total_sq=300.0, count=3) == 0.0
    assert variance(total=0.0, total_sq=0.0, count=0) == 0.0


def test_entropy_is_natural_log_and_zero_for_a_single_destination() -> None:
    assert shannon_entropy([7.0]) == 0.0
    assert shannon_entropy([1.0, 1.0]) == pytest.approx(0.6931471805599453)
    assert shannon_entropy([]) == 0.0


def test_percentile_reads_a_value_count_distribution() -> None:
    counts = {"0": 90.0, "1460": 10.0}
    assert percentile(counts, 0.95) == 1460.0
    assert percentile(counts, 0.5) == 0.0
    assert percentile({}, 0.95) == 0.0


def test_ols_slope_zero_pads_below_three_points() -> None:
    assert ols_slope([]) == 0.0
    assert ols_slope([1.0]) == 0.0
    assert ols_slope([1.0, 5.0]) == 0.0
    assert ols_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert ols_slope([3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_validate_features_fails_loudly_on_a_missing_key() -> None:
    features = dict.fromkeys(FEATURE_ORDER, 0.0)
    validate_features(features)
    del features["syn_ratio"]
    with pytest.raises(ValueError, match="syn_ratio"):
        validate_features(features)


def test_window_graph_scalars() -> None:
    # a -> b, a -> c, b -> a, b -> c: `a` and `b` are mutual, `c` only receives.
    graph = WindowGraph.from_edges([("a", "b"), ("a", "c"), ("b", "a"), ("b", "c")])
    assert graph.out_degree("a") == 2
    assert graph.in_degree("c") == 2
    assert graph.out_degree("c") == 0
    assert graph.reciprocity("a") == pytest.approx(0.5)
    assert graph.reciprocity("c") == 0.0
    # `a`'s neighbourhood is {b, c}, and b–c are linked: the one possible pair exists.
    assert graph.clustering("a", degree_cap=200) == pytest.approx(1.0)
    assert graph.clustering("c", degree_cap=200) == pytest.approx(1.0)


def test_window_accumulator_splits_scalars_from_distributions() -> None:
    accumulator = WindowAccumulator.from_hash(
        {"n_flows": "3", "pkt_count": "30", "dp:80": "2", "dp:443": "1"}
    )
    assert accumulator.get("n_flows") == 3.0
    assert accumulator.get("never_written") == 0.0
    assert accumulator.dist("dp") == {"80": 2.0, "443": 1.0}


# ------------------------------------------------------------------- escalating scan


async def test_escalating_scan_emits_one_vector_per_window(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    vectors = await drain(worker, escalating_scan(tenant_id))

    assert len(vectors) == WINDOWS
    assert [vector.host_id for vector in vectors] == [SCAN_HOST] * WINDOWS
    assert [vector.window_ts for vector in vectors] == [
        datetime.fromtimestamp(align(BASE, 30) + window * 30, tz=UTC) for window in range(WINDOWS)
    ]
    assert all(vector.features["is_active"] == 1.0 for vector in vectors)


async def test_every_emitted_vector_is_schema_valid_on_the_stream(
    worker: FeaturesWorker, redis_client: Redis, state_vectors_stream: str, tenant_id: str
) -> None:
    await drain(worker, escalating_scan(tenant_id))

    entries = await redis_client.xrange(state_vectors_stream)
    published = [StateVector.model_validate_json(fields[b"payload"]) for _, fields in entries]

    assert len(published) == WINDOWS
    for vector in published:
        assert set(vector.features) == set(FEATURE_ORDER)
        assert vector.tenant_id == tenant_id
        assert all(isinstance(value, float) for value in vector.features.values())


async def test_escalating_scan_trends_are_monotonic(worker: FeaturesWorker, tenant_id: str) -> None:
    vectors = await drain(worker, escalating_scan(tenant_id))

    def series(name: str) -> list[float]:
        return [vector.features[name] for vector in vectors]

    syn = series("syn_ratio")
    ports = series("dst_port_entropy")
    peers = series("new_peer_count")
    fan_out = series("out_degree")
    iat = series("iat_var")

    assert syn == sorted(syn) and syn[0] < syn[-1]
    assert ports == sorted(ports) and ports[0] < ports[-1]
    assert peers == sorted(peers) and peers[0] < peers[-1]
    assert fan_out == sorted(fan_out)
    # The scan becomes mechanical: inter-arrival variance collapses toward zero.
    assert iat == sorted(iat, reverse=True) and iat[-1] < iat[0]

    # Rising quantities have positive deltas and slopes; the collapsing one is negative.
    assert all(vector.features["d_syn_ratio"] > 0 for vector in vectors[1:])
    assert all(vector.features["d_new_peer_count"] >= 0 for vector in vectors[1:])
    assert all(vector.features["slope3_dst_port_entropy"] > 0 for vector in vectors[2:])
    assert all(vector.features["d_iat_var"] < 0 for vector in vectors[1:])


async def test_first_window_has_zero_deltas_and_slopes(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    vectors = await drain(worker, escalating_scan(tenant_id))
    first = vectors[0].features

    dynamics = [name for name in FEATURE_ORDER if name.startswith(("d_", "slope3_"))]
    assert dynamics, "the dynamics block is part of FEATURE_ORDER"
    assert all(first[name] == 0.0 for name in dynamics)
    # A slope needs three points, so the second window is still zero-padded.
    assert all(vectors[1].features[name] == 0.0 for name in dynamics if name.startswith("slope3_"))


async def test_new_peer_count_counts_only_first_contacts(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    """A destination already in the host's peer set is not new the second time round."""
    events = [
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=window * 30 + index),
            src=SCAN_HOST,
            dst="10.0.0.9",
            dst_port=80,
            syn_count=1.0,
            iat_mean=0.2,
        )
        for window in range(2)
        for index in range(2)
    ]
    vectors = await drain(worker, events)

    assert [vector.features["new_peer_count"] for vector in vectors] == [1.0, 0.0]


async def test_graph_scalars_use_the_whole_tenant_window(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    """`in_degree` is a property of the window graph, not of the row's own events."""
    events = [
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=1),
            src=SCAN_HOST,
            dst="192.168.10.51",
            dst_port=80,
            syn_count=1.0,
            iat_mean=0.2,
        ),
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=2),
            src="192.168.10.51",
            dst=SCAN_HOST,
            dst_port=443,
            syn_count=1.0,
            iat_mean=0.2,
        ),
    ]
    vectors = await drain(worker, events)
    by_host = {vector.host_id: vector.features for vector in vectors}

    assert set(by_host) == {SCAN_HOST, "192.168.10.51"}
    assert by_host[SCAN_HOST]["out_degree"] == 1.0
    assert by_host[SCAN_HOST]["in_degree"] == 1.0
    assert by_host[SCAN_HOST]["reciprocity"] == pytest.approx(1.0)


# --------------------------------------------------------------------- silent windows


async def test_silent_windows_are_emitted_with_is_active_zero(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    """Events in w1 and w4 leave w2 and w3 silent — and silence is state, not absence."""
    events = [
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=offset),
            src=SCAN_HOST,
            dst="10.0.0.7",
            dst_port=80,
            syn_count=1.0,
            iat_mean=0.2,
        )
        # w1 at BASE, w4 at BASE + 90 s: two whole windows with nothing in them.
        for offset in (0, 90)
    ]
    vectors = await drain(worker, events)

    assert [vector.window_ts.timestamp() - BASE.timestamp() for vector in vectors] == [
        0.0,
        30.0,
        60.0,
        90.0,
    ]
    silent = vectors[1:3]
    assert [vector.features["is_active"] for vector in vectors] == [1.0, 0.0, 0.0, 1.0]
    for vector in silent:
        assert set(vector.features) == set(FEATURE_ORDER)
        assert all(value == 0.0 for value in vector.features.values())


async def test_a_gap_beyond_max_gap_windows_restarts_the_sequence(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    """A day of silence is a restarted stream, not 2880 windows of zeros."""
    gap = (worker.max_gap_windows + 5) * worker.window_delta
    events = [
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=offset),
            src=SCAN_HOST,
            dst="10.0.0.7",
            dst_port=80,
            syn_count=1.0,
            iat_mean=0.2,
        )
        for offset in (0, gap)
    ]
    vectors = await drain(worker, events)

    assert len(vectors) == 2
    assert [vector.features["is_active"] for vector in vectors] == [1.0, 1.0]
    # History was dropped with the gap, so the second window is zero-padded like a first.
    assert vectors[1].features["d_syn_ratio"] == 0.0


# -------------------------------------------------------------------------- watchdog


async def test_watchdog_closes_the_final_window_with_no_successor_event(
    worker: FeaturesWorker, redis_client: Redis, tenant_id: str
) -> None:
    """The last window of a capture has no later event. Without the timer it never fires."""
    event = flow_event(
        tenant_id,
        ts=BASE + timedelta(seconds=3),
        src=SCAN_HOST,
        dst="10.0.0.7",
        dst_port=80,
        syn_count=1.0,
        iat_mean=0.2,
    )
    assert await worker.ingest_event(event) == []

    # Inside the grace period the window stays open: a brief stall is not a closed window.
    wall(worker).advance(worker.grace_s / 2)
    assert await worker.watchdog_tick() == []
    assert await redis_client.exists(window_key(tenant_id, SCAN_HOST, align(BASE, 30))) == 1

    wall(worker).advance(worker.grace_s)
    vectors = await worker.watchdog_tick()

    assert [vector.host_id for vector in vectors] == [SCAN_HOST]
    assert vectors[0].features["is_active"] == 1.0
    # A closed window leaves no accumulator behind, and closing twice emits nothing.
    assert await redis_client.exists(window_key(tenant_id, SCAN_HOST, align(BASE, 30))) == 0
    assert await worker.watchdog_tick() == []


async def test_late_events_are_dropped_rather_than_reopening_a_closed_window(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    """A window that has already been published cannot be amended after the fact."""
    late = flow_event(
        tenant_id,
        ts=BASE + timedelta(seconds=1),
        src=SCAN_HOST,
        dst="10.0.0.7",
        dst_port=80,
        syn_count=1.0,
        iat_mean=0.2,
    )
    later = flow_event(
        tenant_id,
        ts=BASE + timedelta(seconds=31),
        src=SCAN_HOST,
        dst="10.0.0.8",
        dst_port=80,
        syn_count=1.0,
        iat_mean=0.2,
    )
    await worker.ingest_event(late)
    closed = await worker.ingest_event(later)
    assert [vector.window_ts for vector in closed] == [datetime.fromtimestamp(align(BASE, 30), UTC)]

    assert await worker.ingest_event(late) == []
    wall(worker).advance(worker.grace_s + 1)
    vectors = await worker.watchdog_tick()
    assert [vector.features["out_degree"] for vector in vectors] == [1.0]


async def test_neighbour_risk_fraction_uses_the_previous_window_fan_out(
    worker: FeaturesWorker, tenant_id: str
) -> None:
    """The heuristic prior from IMPLEMENTATION-ML.md §2.6 — peers, last window, no model."""
    events: list[RawEvent] = []
    # Window 1: `10.0.0.1` fans out past the elevated threshold; `10.0.0.2` does not.
    for index in range(worker.elevated_fanout):
        events.append(
            flow_event(
                tenant_id,
                ts=BASE + timedelta(seconds=index),
                src="10.0.0.1",
                dst=f"10.0.1.{index}",
                dst_port=80,
                syn_count=1.0,
                iat_mean=0.2,
            )
        )
    events.append(
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=10),
            src="10.0.0.2",
            dst="10.0.1.99",
            dst_port=80,
            syn_count=1.0,
            iat_mean=0.2,
        )
    )
    # Window 2: the scan host talks to both of them, one risky by the prior, one not.
    events += [
        flow_event(
            tenant_id,
            ts=BASE + timedelta(seconds=30 + offset),
            src=SCAN_HOST,
            dst=peer,
            dst_port=80,
            syn_count=1.0,
            iat_mean=0.2,
        )
        for offset, peer in enumerate(("10.0.0.1", "10.0.0.2"))
    ]

    vectors = await drain(worker, events)
    second = {
        vector.host_id: vector.features
        for vector in vectors
        if vector.window_ts.timestamp() == align(BASE, 30) + 30
    }

    assert second[SCAN_HOST]["neighbour_risk_fraction"] == pytest.approx(0.5)
    # The first window had no prior to read, so nothing counted as elevated.
    first = {
        vector.host_id: vector.features
        for vector in vectors
        if vector.window_ts.timestamp() == align(BASE, 30)
    }
    assert first["10.0.0.1"]["neighbour_risk_fraction"] == 0.0

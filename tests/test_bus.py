"""P4: the bus wrapper, against the real Redis from `docker compose up -d redis`.

Mocking Redis here would test our understanding of consumer groups rather than Redis's
implementation of them, and the two things this module exists to guarantee — that an
unacked message really is recoverable, and that two groups really do each see every
message — are exactly the ones a mock would confirm regardless of the truth.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis

from nidra.data.schema import FEATURE_ORDER
from nidra_common.bus import Bus, create_redis
from nidra_common.schemas import StateVector
from nidra_common.worker import run_consumer

WINDOW_DELTA = 30
ORIGIN = datetime(2017, 7, 5, 14, 32, 0, tzinfo=UTC)

# Short enough that a worker test finishes in well under a second; the production
# default of 2 s only costs shutdown latency, never correctness.
TEST_BLOCK_MS = 50


def _state_vector(index: int) -> StateVector:
    """A schema-valid StateVector, distinguishable by `window_ts` and `syn_ratio`."""
    features = {name: 0.0 for name in FEATURE_ORDER}
    features["syn_ratio"] = index / 10.0
    features["is_active"] = 1.0
    return StateVector(
        tenant_id="demo",
        host_id="192.168.10.50",
        window_ts=ORIGIN + timedelta(seconds=WINDOW_DELTA * index),
        features=features,
    )


def _syn_ratios(payloads: list[bytes]) -> list[float]:
    return [StateVector.model_validate_json(p).features["syn_ratio"] for p in payloads]


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def stream(redis_client: Redis) -> AsyncIterator[str]:
    """A stream name unique to this test, deleted afterwards along with its groups."""
    name = f"test:bus:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


async def _drain(bus: Bus, expected: int, block_ms: int = TEST_BLOCK_MS) -> list[tuple[str, bytes]]:
    """Consume exactly `expected` messages through the public async-generator API."""
    collected: list[tuple[str, bytes]] = []
    async for msg_id, payload in bus.consume(block_ms=block_ms):
        collected.append((msg_id, payload))
        if len(collected) == expected:
            break
    return collected


async def test_publish_consume_ack_clears_pending(redis_client: Redis, stream: str) -> None:
    bus = Bus(redis_client, stream, group="features", consumer="worker-1")
    await bus.ensure_group()

    for i in range(5):
        await bus.publish(_state_vector(i))

    received = await _drain(bus, 5)
    assert _syn_ratios([p for _, p in received]) == [0.0, 0.1, 0.2, 0.3, 0.4]

    # Delivered but unacked messages are pending — that is the whole guarantee.
    assert await bus.pending_count() == 5
    for msg_id, _ in received:
        assert await bus.ack(msg_id) == 1
    assert await bus.pending_count() == 0


async def test_unacked_message_is_reclaimed_by_another_consumer(
    redis_client: Redis, stream: str
) -> None:
    """A worker that dies mid-handler must not take its message with it."""
    dead = Bus(redis_client, stream, group="inference", consumer="worker-dead")
    await dead.ensure_group()
    await dead.publish(_state_vector(7))

    received = await _drain(dead, 1)
    assert len(received) == 1  # read, never acked — worker-dead "crashes" here
    assert await dead.pending_count() == 1

    alive = Bus(redis_client, stream, group="inference", consumer="worker-alive")
    reclaimed = [item async for item in alive.reclaim(min_idle_ms=0)]

    assert _syn_ratios([p for _, p in reclaimed]) == [0.7]
    assert reclaimed[0][0] == received[0][0]

    await alive.ack(reclaimed[0][0])
    assert await alive.pending_count() == 0


async def test_reclaim_yields_nothing_when_messages_are_not_idle(
    redis_client: Redis, stream: str
) -> None:
    """The idle threshold is what stops two live workers stealing each other's work."""
    bus = Bus(redis_client, stream, group="inference", consumer="worker-1")
    await bus.ensure_group()
    await bus.publish(_state_vector(1))
    await _drain(bus, 1)

    other = Bus(redis_client, stream, group="inference", consumer="worker-2")
    assert [item async for item in other.reclaim(min_idle_ms=60_000)] == []
    assert await bus.pending_count() == 1


async def test_ensure_group_is_idempotent(redis_client: Redis, stream: str) -> None:
    """Every service calls this at startup; restarts must not need a try/except."""
    bus = Bus(redis_client, stream, group="persister", consumer="worker-1")
    await bus.ensure_group()
    await bus.ensure_group()

    groups = await redis_client.xinfo_groups(stream)
    assert [g["name"] for g in groups] == [b"persister"]


async def test_group_created_after_publish_still_sees_earlier_messages(
    redis_client: Redis, stream: str
) -> None:
    """`id="0"` — otherwise workers idle next to a stream that already has entries."""
    producer = Bus(redis_client, stream, group="producer-only", consumer="p")
    await producer.ensure_group()
    for i in range(3):
        await producer.publish(_state_vector(i))

    late = Bus(redis_client, stream, group="late", consumer="worker-1")
    await late.ensure_group()
    assert _syn_ratios([p for _, p in await _drain(late, 3)]) == [0.0, 0.1, 0.2]


async def test_two_groups_each_receive_every_message(redis_client: Redis, stream: str) -> None:
    """`api` and `persister` both read `forecasts` in full. P9's fan-out rests on this."""
    api = Bus(redis_client, stream, group="api", consumer="api-1")
    persister = Bus(redis_client, stream, group="persister", consumer="persister-1")
    await api.ensure_group()
    await persister.ensure_group()

    for i in range(3):
        await api.publish(_state_vector(i))

    assert _syn_ratios([p for _, p in await _drain(api, 3)]) == [0.0, 0.1, 0.2]
    assert _syn_ratios([p for _, p in await _drain(persister, 3)]) == [0.0, 0.1, 0.2]


async def test_two_consumers_in_one_group_split_the_stream(
    redis_client: Redis, stream: str
) -> None:
    """The other half of the fan-out property: scaling `inference` must not duplicate work."""
    one = Bus(redis_client, stream, group="inference", consumer="worker-1")
    two = Bus(redis_client, stream, group="inference", consumer="worker-2")
    await one.ensure_group()

    for i in range(4):
        await one.publish(_state_vector(i))

    first = await one.read(count=2, block_ms=TEST_BLOCK_MS)
    second = await two.read(count=2, block_ms=TEST_BLOCK_MS)

    assert _syn_ratios([p for _, p in first]) == [0.0, 0.1]
    assert _syn_ratios([p for _, p in second]) == [0.2, 0.3]


async def test_run_consumer_acks_only_what_the_handler_processed(
    redis_client: Redis, stream: str
) -> None:
    bus = Bus(redis_client, stream, group="features", consumer="worker-1")
    await bus.publish(_state_vector(0))
    await bus.publish(_state_vector(1))
    await bus.publish(_state_vector(2))

    stop = asyncio.Event()
    seen: list[float] = []

    async def handler(payload: bytes) -> None:
        seen.append(StateVector.model_validate_json(payload).features["syn_ratio"])
        if len(seen) == 3:
            stop.set()

    await asyncio.wait_for(
        run_consumer(bus, handler, block_ms=TEST_BLOCK_MS, stop=stop, install_signals=False),
        timeout=10,
    )

    assert seen == [0.0, 0.1, 0.2]
    assert await bus.pending_count() == 0


async def test_run_consumer_leaves_a_failed_message_pending(
    redis_client: Redis, stream: str
) -> None:
    """A raising handler must not ack. Losing the message would be a silent data loss."""
    bus = Bus(redis_client, stream, group="features", consumer="worker-1")
    await bus.publish(_state_vector(0))

    stop = asyncio.Event()
    attempts: list[bytes] = []

    async def handler(payload: bytes) -> None:
        attempts.append(payload)
        stop.set()
        raise RuntimeError("handler blew up")

    await asyncio.wait_for(
        run_consumer(bus, handler, block_ms=TEST_BLOCK_MS, stop=stop, install_signals=False),
        timeout=10,
    )

    assert len(attempts) == 1
    assert await bus.pending_count() == 1

    # And a later reclaim gets it back, which is the point of not acking.
    retry = Bus(redis_client, stream, group="features", consumer="worker-2")
    assert _syn_ratios([p async for _, p in retry.reclaim(min_idle_ms=0)]) == [0.0]

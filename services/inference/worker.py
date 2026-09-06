"""The `state_vectors` consumer: buffer L windows in Redis, forecast, publish.

This is the scale-out unit, and the only thing that makes it one is where the sequence
buffer lives. `seq:{tenant}:{host}` is a Redis list, so any worker can handle any host's
next window without having seen the previous ones, replicas can be added mid-replay, and
a worker that dies takes no context with it. Nothing about a host survives in process
memory between two messages — not a cache, not a last-seen timestamp. That is the whole
basis of the scalability claim and it is one attribute away from being false.

At-least-once delivery makes the dedupe on the way in load-bearing rather than defensive.
A redelivered `StateVector` — a worker that crashed after `RPUSH` but before its ack, a
reclaim after a network stall — would be pushed a second time, and the L-window context
would then contain the same window twice and be silently one window short of real
history. The forecast would still be schema-valid. So the incoming `window_ts` is
compared against the last buffered one and a repeat is dropped before it is pushed
(PROMPTBOOK Standing Rules correction 4).

The predictor is loaded once by the caller and handed in. Loading per message would put
model construction on the critical path of a 300 ms budget.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from redis.asyncio import Redis

from nidra.data.schema import FEATURE_ORDER
from nidra_common.bus import Bus
from nidra_common.config import get_config
from nidra_common.schemas import Forecast, StateVector
from services.inference.predictor_loader import Predictor

logger = logging.getLogger(__name__)

#: Long enough that a pause in the replay does not drop a host's context, short enough
#: that a finished job's buffers do not outlive the demo.
SEQ_TTL_S = 3600


def seq_key(tenant_id: str, host_id: str) -> str:
    """The one piece of per-host state, and it is in Redis rather than in this process."""
    return f"seq:{tenant_id}:{host_id}"


def states_array(vectors: list[StateVector]) -> np.ndarray:
    """`[L, 45]` in `FEATURE_ORDER`, oldest first.

    Built by indexing `FEATURE_ORDER`, never by iterating the dict: a feature order that
    depends on insertion order is a column mismatch waiting for the first service that
    serialises its features differently.
    """
    return np.array(
        [[vector.features[name] for name in FEATURE_ORDER] for vector in vectors],
        dtype=float,
    )


class InferenceWorker:
    """One `state_vectors` consumer. Stateless between messages, by construction."""

    def __init__(
        self,
        redis: Redis,
        bus: Bus,
        predictor: Predictor,
        *,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.redis = redis
        self.bus = bus
        self.predictor = predictor
        self.cfg = cfg if cfg is not None else get_config()

    # ------------------------------------------------------------------ configuration

    @property
    def context_l(self) -> int:
        """Windows of history the encoder consumes. Below this there is no forecast."""
        return int(self.cfg["context_L"])

    # ------------------------------------------------------------------------ handler

    async def handle(self, payload: bytes) -> None:
        """Bus entry point: one `state_vectors` message.

        Raising leaves the message unacked and pending for a later `XAUTOCLAIM`, which is
        the behaviour we want — the dedupe above makes the retry harmless.
        """
        await self.process(StateVector.model_validate_json(payload))

    async def process(self, vector: StateVector) -> Forecast | None:
        """Buffer one window and forecast if the context is now full.

        Returns the published `Forecast`, or `None` when the window was a duplicate or the
        buffer is still short of L. The return value exists so a test can drive the worker
        without polling the stream.
        """
        key = seq_key(vector.tenant_id, vector.host_id)

        if await self._is_duplicate(key, vector):
            logger.debug(
                "tenant %s host %s: dropping redelivered window %s",
                vector.tenant_id,
                vector.host_id,
                vector.window_ts.isoformat(),
            )
            return None

        buffered = await self._push(key, vector)
        if len(buffered) < self.context_l:
            logger.debug(
                "tenant %s host %s: %d/%d windows buffered",
                vector.tenant_id,
                vector.host_id,
                len(buffered),
                self.context_l,
            )
            return None

        result = self.predictor.forecast(states_array(buffered), vector.host_id, vector.window_ts)
        # `tenant_id` is the caller's: the ML interface is deliberately tenant-blind.
        forecast = Forecast(tenant_id=vector.tenant_id, **result)
        await self.bus.publish(forecast)
        logger.debug(
            "tenant %s host %s: forecast at origin %s",
            vector.tenant_id,
            vector.host_id,
            forecast.origin_ts.isoformat(),
        )
        return forecast

    # ------------------------------------------------------------------- redis buffer

    async def _is_duplicate(self, key: str, vector: StateVector) -> bool:
        """True when this exact window is already the tail of the buffer.

        At-least-once means the same `StateVector` can arrive twice; pushing it twice
        would corrupt the L-window context with no visible symptom.
        """
        tail = await self.redis.lindex(key, -1)
        if tail is None:
            return False
        return StateVector.model_validate_json(tail).window_ts == vector.window_ts

    async def _push(self, key: str, vector: StateVector) -> list[StateVector]:
        """Append, trim to the last L windows, refresh the TTL, and read the buffer back."""
        pipeline = self.redis.pipeline()
        pipeline.rpush(key, vector.model_dump_json())
        pipeline.ltrim(key, -self.context_l, -1)
        pipeline.expire(key, SEQ_TTL_S)
        pipeline.lrange(key, 0, -1)
        *_, raw = await pipeline.execute()
        return [StateVector.model_validate_json(entry) for entry in raw]

"""`python -m services.inference` — the stateless forecasting worker.

One consumer in the `inference` group on `state_vectors`, publishing `Forecast`s to
`forecasts`. Scale it with `docker compose up -d --scale inference=3`: consumers within
one group split the stream, and because every sequence buffer is in Redis, which replica
picks up a given host's window does not matter.

The predictor is built once here, before the consume loop starts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.worker import run_consumer
from services.inference.predictor_loader import load_predictor
from services.inference.worker import InferenceWorker

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "inference"


def consumer_name() -> str:
    """Identifies this process within the group. Distinct per container and per replica."""
    return f"{socket.gethostname()}-{os.getpid()}"


async def main() -> None:
    """Load the predictor, then consume until SIGTERM."""
    logging.basicConfig(
        level=os.environ.get("NIDRA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = get_config()
    predictor = load_predictor(cfg)
    redis = create_redis(cfg)
    vectors_bus = Bus(
        redis,
        stream_name("state_vectors", cfg),
        group=CONSUMER_GROUP,
        consumer=consumer_name(),
    )
    forecasts_bus = Bus(
        redis,
        stream_name("forecasts", cfg),
        group=CONSUMER_GROUP,  # publish-only; the group is never read from here
        consumer=consumer_name(),
    )
    worker = InferenceWorker(redis, forecasts_bus, predictor, cfg=cfg)

    try:
        await run_consumer(vectors_bus, worker.handle)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())

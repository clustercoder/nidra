"""`python -m services.persister` — the durable end of the pipeline.

One consumer in the `persister` group on `forecasts`. The `api` process reads the same
stream under its own group, so both see every forecast: consumers within a group split
the stream, groups do not. Nothing is published from here — this is where the pipeline
stops being a stream and starts being a table.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.worker import run_consumer
from services.persister.worker import PersisterWorker

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "persister"


def consumer_name() -> str:
    """Identifies this process within the group. Distinct per container and per replica."""
    return f"{socket.gethostname()}-{os.getpid()}"


async def main() -> None:
    """Consume `forecasts` until SIGTERM."""
    logging.basicConfig(
        level=os.environ.get("NIDRA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = get_config()
    redis = create_redis(cfg)
    forecasts_bus = Bus(
        redis,
        stream_name("forecasts", cfg),
        group=CONSUMER_GROUP,
        consumer=consumer_name(),
    )
    worker = PersisterWorker(redis, get_sessionmaker(), cfg=cfg)

    try:
        await run_consumer(forecasts_bus, worker.handle)
    finally:
        await redis.aclose()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())

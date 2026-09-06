"""`python -m services.features` — the windowing worker process.

One consumer in the `features` group on `raw_events`, publishing `StateVector`s to
`state_vectors`, with the watchdog running alongside the consume loop. Both stop on the
same event, so SIGTERM drains the in-flight message and leaves anything unprocessed
unacked for another worker to reclaim.

Every published vector is also written to `state_vectors` in Postgres first, which is
what `GET /api/v1/explain/{host}/{ts}` reads back to rebuild the context a forecast was
produced from.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.worker import install_shutdown_handlers, run_consumer
from services.features.store import StateVectorStore
from services.features.worker import FeaturesWorker

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "features"


def consumer_name() -> str:
    """Identifies this process within the group. Distinct per container and per replica."""
    return f"{socket.gethostname()}-{os.getpid()}"


async def main() -> None:
    """Run the consumer and the watchdog until SIGTERM."""
    logging.basicConfig(
        level=os.environ.get("NIDRA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = get_config()
    redis = create_redis(cfg)
    events_bus = Bus(
        redis,
        stream_name("raw_events", cfg),
        group=CONSUMER_GROUP,
        consumer=consumer_name(),
    )
    vectors_bus = Bus(
        redis,
        stream_name("state_vectors", cfg),
        group=CONSUMER_GROUP,  # publish-only; the group is never read from here
        consumer=consumer_name(),
    )
    worker = FeaturesWorker(redis, vectors_bus, cfg=cfg, store=StateVectorStore(get_sessionmaker()))

    stop = asyncio.Event()
    install_shutdown_handlers(stop)
    try:
        await asyncio.gather(
            run_consumer(events_bus, worker.handle, stop=stop, install_signals=False),
            worker.run_watchdog(stop),
        )
    finally:
        await redis.aclose()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())

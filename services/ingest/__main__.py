"""`python -m services.ingest` — the ingest worker process.

One consumer in the `ingest` group on the `ingest_jobs` stream, publishing to
`raw_events`. Shutdown is cooperative (SIGTERM from `docker compose stop`): the job in
flight finishes its current message before the loop exits, and an unfinished job stays
unacked so another worker reclaims it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.db import dispose_engine
from nidra_common.worker import run_consumer
from services.ingest.worker import IngestWorker

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "ingest"


def consumer_name() -> str:
    """Identifies this process within the group. Distinct per container and per replica."""
    return f"{socket.gethostname()}-{os.getpid()}"


async def main() -> None:
    """Run the worker until SIGTERM."""
    logging.basicConfig(
        level=os.environ.get("NIDRA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = get_config()
    redis = create_redis(cfg)
    jobs_bus = Bus(
        redis,
        stream_name("ingest_jobs", cfg),
        group=CONSUMER_GROUP,
        consumer=consumer_name(),
    )
    events_bus = Bus(
        redis,
        stream_name("raw_events", cfg),
        group=CONSUMER_GROUP,  # publish-only; the group is never read from here
        consumer=consumer_name(),
    )
    worker = IngestWorker(redis, events_bus, cfg=cfg)

    try:
        await run_consumer(jobs_bus, worker.handle)
    finally:
        await dispose_engine()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())

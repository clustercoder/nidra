"""The `ingest_jobs` consumer: parse an uploaded file and replay it onto `raw_events`.

The worker is the only place capture time becomes wall-clock time. Everything downstream
— window close, watchdog grace, the cone opening on screen — is paced by what this loop
publishes, which is why the `ReplayClock` lives at the producer rather than being faked
in the frontend.

**Timestamp order is a correctness property, not a nicety.** The features service assigns
each event to an absolute 30 s window and closes a window when a later one arrives; an
event published out of order arrives after its window has closed and is lost. CSV chunks
are therefore sorted before publication, and a chunk boundary that regresses is logged
loudly rather than passed on in silence.

Failure handling splits in two. A file that cannot be parsed will fail identically on
every redelivery, so the job is marked `error` and the message is acked. Anything else —
Redis gone, Postgres gone — propagates, leaving the message pending for `XAUTOCLAIM`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nidra_common.bus import Bus
from nidra_common.config import get_config
from nidra_common.db import get_sessionmaker
from nidra_common.events import (
    JOB_COMPLETE,
    JOB_ERROR,
    JOB_RUNNING,
    IngestJobMessage,
    RawEvent,
)
from services.ingest.jobs import progress_fraction, set_progress
from services.ingest.parsers import (
    IngestError,
    count_csv_rows,
    iter_csv_events,
    iter_pcap_events,
)
from services.ingest.replay import ReplayClock, Sleeper, as_utc

logger = logging.getLogger(__name__)

DEFAULT_CSV_CHUNK_ROWS = 5000

UPDATE_JOB = text(
    "UPDATE ingest_jobs SET status = :status, progress = :progress, error = :error "
    "WHERE id = CAST(:job_id AS uuid) AND tenant_id = CAST(:tenant_id AS uuid)"
)


@dataclass(slots=True)
class _Published:
    """How many events have reached the bus so far, readable from the failure path."""

    count: int = 0


@dataclass(frozen=True, slots=True)
class JobResult:
    """What one job did: how many events reached `raw_events`, and how it ended."""

    job_id: str
    status: str
    published: int
    total: int
    error: str = ""


class IngestWorker:
    """Consumes `ingest_jobs`, publishes `RawEvent`s to `raw_events`.

    `sleep` is injectable so a test can assert on pacing without waiting for it; the
    default is `asyncio.sleep` via :class:`ReplayClock`.
    """

    def __init__(
        self,
        redis: Redis,
        bus: Bus,
        *,
        cfg: dict[str, Any] | None = None,
        sessions: async_sessionmaker[AsyncSession] | None = None,
        sleep: Sleeper | None = None,
    ) -> None:
        self.redis = redis
        self.bus = bus
        self.cfg = cfg if cfg is not None else get_config()
        self._sessions = sessions
        self._sleep = sleep

    # ------------------------------------------------------------------ configuration

    @property
    def csv_chunk_rows(self) -> int:
        return int(self.cfg.get("ingest", {}).get("csv_chunk_rows", DEFAULT_CSV_CHUNK_ROWS))

    @property
    def tshark_bin(self) -> str:
        return str(self.cfg.get("ingest", {}).get("tshark_bin", "tshark"))

    # ------------------------------------------------------------------------ handler

    async def handle(self, payload: bytes) -> None:
        """Bus entry point: one `ingest_jobs` message."""
        job = IngestJobMessage.model_validate_json(payload)
        await self.run_job(job)

    async def run_job(self, job: IngestJobMessage) -> JobResult:
        """Parse and replay one job, reporting progress throughout."""
        path = Path(job.path)
        logger.info(
            "ingest job %s starting: %s (%s, speed=%s)", job.job_id, path, job.kind, job.speed
        )
        total = 0
        published = _Published()
        try:
            if not path.is_file():
                raise IngestError(f"uploaded file is gone: {path}")
            if job.kind == "csv":
                total = count_csv_rows(path)
            await self._report(job, status=JOB_RUNNING, processed=0, total=total)
            await self._replay(job, path, total, published)
        except IngestError as exc:
            # Deterministic: retrying the same file fails the same way. Record and ack.
            # `published` is what actually reached the bus before it failed, not zero.
            logger.error(
                "ingest job %s failed after %d event(s): %s", job.job_id, published.count, exc
            )
            await self._report(
                job, status=JOB_ERROR, processed=published.count, total=total, error=str(exc)
            )
            return JobResult(job.job_id, JOB_ERROR, published.count, total, str(exc))

        await self._report(job, status=JOB_COMPLETE, processed=published.count, total=total)
        logger.info("ingest job %s complete: %d event(s) published", job.job_id, published.count)
        return JobResult(job.job_id, JOB_COMPLETE, published.count, total)

    # -------------------------------------------------------------------------- replay

    def _source(self, job: IngestJobMessage, path: Path) -> AsyncIterator[list[RawEvent]]:
        if job.kind == "csv":
            return iter_csv_events(
                path, tenant_id=job.tenant_id, job_id=job.job_id, chunk_rows=self.csv_chunk_rows
            )
        return iter_pcap_events(
            path, tenant_id=job.tenant_id, job_id=job.job_id, binary=self.tshark_bin
        )

    async def _replay(
        self, job: IngestJobMessage, path: Path, total: int, published: _Published
    ) -> None:
        """Publish every event, paced, in timestamp order, counting into `published`."""
        clock: ReplayClock | None = None
        last_ts: datetime | None = None

        async for chunk in self._source(job, path):
            for event in chunk:
                event_ts = as_utc(event.ts)
                if clock is None:
                    clock = self._clock(event_ts, job.speed)
                if last_ts is not None and event_ts < last_ts:
                    # Sorted within a chunk, so this is a chunk-boundary regression:
                    # the window it belongs to may already have closed downstream.
                    logger.warning(
                        "job %s: event at %s precedes the previous event at %s",
                        job.job_id,
                        event_ts.isoformat(),
                        last_ts.isoformat(),
                    )
                else:
                    last_ts = event_ts
                await clock.wait_until(event_ts)
                await self.bus.publish(event)
                published.count += 1
            await self._report(job, status=JOB_RUNNING, processed=published.count, total=total)

    def _clock(self, first_ts: datetime, speed: float) -> ReplayClock:
        if self._sleep is None:
            return ReplayClock(first_ts, speed=speed)
        return ReplayClock(first_ts, speed=speed, sleep=self._sleep)

    # ------------------------------------------------------------------------ progress

    async def _report(
        self,
        job: IngestJobMessage,
        *,
        status: str,
        processed: int,
        total: int,
        error: str = "",
    ) -> None:
        """Write the live counters to Redis and the durable status to Postgres."""
        await set_progress(
            self.redis, job.job_id, status=status, processed=processed, total=total, error=error
        )
        await self._update_row(
            job, status=status, progress=progress_fraction(processed, total, status), error=error
        )

    async def _update_row(
        self, job: IngestJobMessage, *, status: str, progress: float, error: str
    ) -> None:
        """Update the job row — filtered on `tenant_id`, like every other query."""
        try:
            job_id, tenant_id = uuid.UUID(job.job_id), uuid.UUID(job.tenant_id)
        except ValueError:
            # Not one of ours. Redis already has the status; a failed cast here would
            # leave the message pending and it would be reclaimed forever.
            logger.warning("job %s has non-UUID identifiers; skipping the row update", job.job_id)
            return

        sessions = self._sessions if self._sessions is not None else get_sessionmaker()
        async with sessions() as session:
            await session.execute(
                UPDATE_JOB,
                {
                    "status": status,
                    "progress": progress,
                    "error": error or None,
                    "job_id": str(job_id),
                    "tenant_id": str(tenant_id),
                },
            )
            await session.commit()

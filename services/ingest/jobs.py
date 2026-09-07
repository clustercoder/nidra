"""Where an upload lands, and where its progress is reported.

Shared by the producer (`api/ingest.py`, which writes the file and the job row) and the
consumer (`services/ingest/worker.py`, which replays it), so the two cannot disagree
about a path or a key name.

Progress lives in Redis under `job:{job_id}`, not on the bus: a progress tick is not a
domain event, and putting it on `raw_events` would pollute the log the whole system is
built to reason about. The database row is the durable record — existence, tenant,
filename, terminal status — and Redis is the live counter the console polls.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from redis.asyncio import Redis

from nidra_common.config import REPO_ROOT, get_config
from nidra_common.events import JOB_COMPLETE, JOB_QUEUED

#: `job:{job_id}` — a hash, not a JSON blob, so a progress tick is one HSET of two fields.
JOB_KEY_PREFIX = "job:"

#: Long enough to inspect a finished job after a demo, short enough not to leak forever.
JOB_TTL_S = 24 * 3600

#: Path components come from a JWT claim and a generated UUID, but a traversal here would
#: be a filesystem write outside the upload root, so they are checked rather than trusted.
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def job_key(job_id: str) -> str:
    """Redis key holding one job's live progress."""
    return f"{JOB_KEY_PREFIX}{job_id}"


def upload_root(cfg: dict[str, Any] | None = None) -> Path:
    """Root directory for validated uploads (`NIDRA_UPLOAD_DIR` overrides the YAML)."""
    config = cfg if cfg is not None else get_config()
    configured = config.get("ingest", {}).get("upload_dir")
    if not configured:
        raise ValueError("ingest.upload_dir is missing from the NIDRA config")
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def safe_component(value: str, *, what: str) -> str:
    """Reject anything that would not be a single, ordinary directory name."""
    if not SAFE_COMPONENT.match(value):
        raise ValueError(f"unsafe {what} for a filesystem path: {value!r}")
    return value


def job_dir(tenant_id: str, job_id: str, cfg: dict[str, Any] | None = None) -> Path:
    """`{upload_dir}/{tenant_id}/{job_id}/` — one directory per job, never shared."""
    return (
        upload_root(cfg)
        / safe_component(tenant_id, what="tenant_id")
        / safe_component(job_id, what="job_id")
    )


def upload_path(tenant_id: str, job_id: str, ext: str, cfg: dict[str, Any] | None = None) -> Path:
    """Destination file. Named after the job, never after the client's filename."""
    return job_dir(tenant_id, job_id, cfg) / f"{job_id}{ext}"


async def init_progress(
    redis: Redis, job_id: str, *, kind: str, speed: float, total: int = 0
) -> None:
    """Seed the progress hash at queue time so a poll before the worker starts still works."""
    await redis.hset(  # type: ignore[misc]
        job_key(job_id),
        mapping={
            "status": JOB_QUEUED,
            "kind": kind,
            "speed": str(speed),
            "processed": "0",
            "total": str(total),
            "error": "",
        },
    )
    await redis.expire(job_key(job_id), JOB_TTL_S)


async def set_progress(
    redis: Redis,
    job_id: str,
    *,
    status: str,
    processed: int,
    total: int,
    error: str = "",
) -> None:
    """Overwrite the live counters. Called once per chunk and at every status change."""
    await redis.hset(  # type: ignore[misc]
        job_key(job_id),
        mapping={
            "status": status,
            "processed": str(processed),
            "total": str(total),
            "error": error,
        },
    )
    await redis.expire(job_key(job_id), JOB_TTL_S)


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


async def read_progress(redis: Redis, job_id: str) -> dict[str, str]:
    """The progress hash as plain strings, or `{}` if it has expired or never existed."""
    raw = await redis.hgetall(job_key(job_id))  # type: ignore[misc]
    return {_decode(k): _decode(v) for k, v in (raw or {}).items()}


def progress_fraction(processed: int, total: int, status: str) -> float:
    """0..1 for the API and the `ingest_jobs.progress` column.

    `total` is unknown for a PCAP (tshark does not count packets up front), so an
    in-flight capture reports 0 and a finished one reports 1 — a progress bar that lies
    smoothly would be worse than one that only moves when it knows something.
    """
    if status == JOB_COMPLETE:
        return 1.0
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, processed / total))

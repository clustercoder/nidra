"""Upload endpoints: accept a capture, queue it, report its progress.

Everything a client sends is treated as hostile until proven otherwise. The extension
must be on the allowlist, the first bytes must match what the extension claims, and the
size is checked *while streaming* rather than after — a 2 GB cap enforced by reading 2 GB
first is not a cap. The file is written to a path built from the tenant claim and a
generated job id; the client's filename is stored for display and never used to name
anything on disk.

Progress is read from Redis (`job:{job_id}`, written by the worker) and existence from
Postgres, both filtered on the tenant from the token. There is no request parameter that
names a tenant, which is what makes cross-tenant access a code change rather than a
crafted request.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import CurrentTenant, tenant_uuid
from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.db import get_session
from nidra_common.events import JOB_QUEUED, IngestJobMessage, JobKind
from services.ingest.jobs import (
    init_progress,
    progress_fraction,
    read_progress,
    upload_path,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]

#: libpcap little- and big-endian, and pcapng. tshark reads all three regardless of the
#: extension used, so any of them satisfies `.pcap` or `.pcapng`.
PCAP_MAGICS: tuple[bytes, ...] = (
    bytes.fromhex("d4c3b2a1"),
    bytes.fromhex("a1b2c3d4"),
    bytes.fromhex("0a0d0d0a"),
)

#: Bytes inspected before the file is accepted: enough for any magic, and enough of a
#: first line to tell a CSV header from a binary blob.
SNIFF_BYTES = 4096

UPLOAD_CHUNK_BYTES = 1 << 20

#: Printable ASCII plus tab and the line terminators — what a CSV header can contain.
_PRINTABLE = frozenset(range(0x20, 0x7F)) | {0x09, 0x0D, 0x0A}

MAX_LIST_LIMIT = 200

INSERT_JOB = text(
    "INSERT INTO ingest_jobs (id, tenant_id, filename, kind, status, progress) "
    "VALUES (:id, :tenant_id, :filename, :kind, :status, 0)"
)

SELECT_JOB = text(
    "SELECT id, filename, kind, status, progress, error, created_at FROM ingest_jobs "
    "WHERE id = :id AND tenant_id = :tenant_id"
)

SELECT_JOBS = text(
    "SELECT id, filename, kind, status, progress, error, created_at FROM ingest_jobs "
    "WHERE tenant_id = :tenant_id ORDER BY created_at DESC, id DESC LIMIT :limit"
)


async def get_redis() -> AsyncIterator[Redis]:
    """One Redis client per request.

    Deliberately not a process-wide singleton: a pool binds to the event loop it is first
    used on, and these endpoints are cheap enough that a connection per request costs
    nothing worth the failure mode.
    """
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


RedisDep = Annotated[Redis, Depends(get_redis)]


# --------------------------------------------------------------------------- schemas


class IngestJobCreated(BaseModel):
    """What the upload produced. The client polls the status endpoint from here."""

    job_id: uuid.UUID
    kind: JobKind
    filename: str
    speed: float
    status: str


class IngestJobStatus(BaseModel):
    """Durable job row merged with the worker's live counters from Redis."""

    job_id: uuid.UUID
    kind: str | None
    filename: str | None
    status: str
    processed: int
    total: int
    progress: float = Field(ge=0.0, le=1.0)
    speed: float | None
    error: str | None
    created_at: datetime | None


class IngestJobList(BaseModel):
    jobs: list[IngestJobStatus]


# ------------------------------------------------------------------------ validation


def _ingest_config() -> dict[str, Any]:
    return dict(get_config().get("ingest", {}))


def allowed_extensions() -> list[str]:
    """Extension allowlist from `config/default.yaml`."""
    return [str(ext).lower() for ext in _ingest_config().get("allowed_ext", [])]


def max_upload_bytes() -> int:
    """Upload cap in bytes, from `ingest.max_upload_mb`."""
    return int(float(_ingest_config().get("max_upload_mb", 2048)) * 1024 * 1024)


def default_speed() -> float:
    """Replay multiplier used when the request does not name one."""
    return float(get_config().get("replay", {}).get("default_speed", 60.0))


def _reject(code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=code, detail=detail)


def validated_extension(filename: str | None) -> str:
    """Lowercased extension of the uploaded name, checked against the allowlist."""
    suffix = Path(filename or "").suffix.lower()
    allowed = allowed_extensions()
    if suffix not in allowed:
        raise _reject(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type {suffix or '(none)'}; allowed: {', '.join(allowed)}",
        )
    return suffix


def _looks_like_text(head: bytes) -> bool:
    """True when the first line is printable ASCII — a CSV header, not a capture."""
    first_line = head.split(b"\n", 1)[0]
    return bool(first_line) and all(byte in _PRINTABLE for byte in first_line)


def job_kind_for(head: bytes, ext: str) -> JobKind:
    """Map extension + magic bytes to a parser, or reject the mismatch with 415.

    An extension is a claim; the first four bytes are evidence. A CSV renamed to `.pcap`
    would otherwise reach tshark, and a capture renamed to `.csv` would reach pandas.
    """
    if ext in (".pcap", ".pcapng"):
        if head[:4] not in PCAP_MAGICS:
            raise _reject(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"{ext} content does not start with a pcap or pcapng magic number",
            )
        return "pcap"
    if not _looks_like_text(head):
        raise _reject(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "csv content is not text; its first line contains non-printable bytes",
        )
    return "csv"


def _display_filename(filename: str | None) -> str:
    """The client's name, kept for display only — never used to build a path."""
    return Path(filename or "upload").name[:255] or "upload"


async def _store_upload(file: UploadFile, destination: Path, cap_bytes: int) -> bytes:
    """Stream the upload to `destination`, enforcing the cap as it goes.

    Returns the sniffed head. The partial file is removed on rejection, so a refused
    upload leaves nothing behind.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    head = b""
    written = 0
    try:
        with destination.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > cap_bytes:
                    raise _reject(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"upload exceeds the {cap_bytes // (1024 * 1024)} MB limit",
                    )
                if len(head) < SNIFF_BYTES:
                    head += chunk[: SNIFF_BYTES - len(head)]
                out.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if not written:
        destination.unlink(missing_ok=True)
        raise _reject(status.HTTP_400_BAD_REQUEST, "uploaded file is empty")
    return head


def _cleanup(directory: Path) -> None:
    """Remove a rejected job's directory. Best effort — a leftover file is not an error."""
    try:
        for child in directory.iterdir():
            child.unlink(missing_ok=True)
        directory.rmdir()
    except OSError:  # pragma: no cover — the next upload does not depend on this
        logger.warning("could not clean up %s", directory)


# ---------------------------------------------------------------------------- routes


@router.post("", response_model=IngestJobCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_ingest_job(
    tenant_id: CurrentTenant,
    session: SessionDep,
    redis: RedisDep,
    file: Annotated[UploadFile, File(description="pcap, pcapng, or CICFlowMeter CSV")],
    speed: Annotated[float | None, Form(description="replay multiplier; 0 = unpaced")] = None,
) -> IngestJobCreated:
    """Accept a capture, queue it on `ingest_jobs`, and return the job id."""
    ext = validated_extension(file.filename)
    job_id = uuid.uuid4()
    replay_speed = default_speed() if speed is None else float(speed)
    if replay_speed < 0:
        raise _reject(status.HTTP_400_BAD_REQUEST, "speed must be >= 0")

    tenant = tenant_uuid(tenant_id)
    destination = upload_path(str(tenant), str(job_id), ext)
    try:
        head = await _store_upload(file, destination, max_upload_bytes())
        kind = job_kind_for(head, ext)
    except HTTPException:
        # Too large, or the content does not match the extension: leave nothing behind.
        _cleanup(destination.parent)
        raise

    filename = _display_filename(file.filename)
    await session.execute(
        INSERT_JOB,
        {
            "id": job_id,
            "tenant_id": tenant,
            "filename": filename,
            "kind": kind,
            "status": JOB_QUEUED,
        },
    )
    await session.commit()

    await init_progress(redis, str(job_id), kind=kind, speed=replay_speed)
    bus = Bus(redis, stream_name("ingest_jobs"), group="ingest", consumer="api")
    await bus.publish(
        IngestJobMessage(
            job_id=str(job_id),
            tenant_id=tenant_id,
            kind=kind,
            path=str(destination),
            filename=filename,
            speed=replay_speed,
        )
    )
    logger.info("queued ingest job %s (%s, %s) for tenant %s", job_id, kind, filename, tenant_id)
    return IngestJobCreated(
        job_id=job_id, kind=kind, filename=filename, speed=replay_speed, status=JOB_QUEUED
    )


def _merge(row: Any, live: dict[str, str]) -> IngestJobStatus:
    """Durable row plus live counters. Redis wins on status; it is the fresher writer."""
    status_value = live.get("status") or str(row.status or JOB_QUEUED)
    processed = int(live.get("processed") or 0)
    total = int(live.get("total") or 0)
    progress = (
        progress_fraction(processed, total, status_value) if live else float(row.progress or 0.0)
    )
    speed = float(live["speed"]) if live.get("speed") else None
    return IngestJobStatus(
        job_id=row.id,
        kind=row.kind,
        filename=row.filename,
        status=status_value,
        processed=processed,
        total=total,
        progress=progress,
        speed=speed,
        error=live.get("error") or row.error or None,
        created_at=row.created_at,
    )


@router.get("", response_model=IngestJobList)
async def list_ingest_jobs(
    tenant_id: CurrentTenant,
    session: SessionDep,
    redis: RedisDep,
    limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = 50,
) -> IngestJobList:
    """This tenant's jobs, newest first."""
    rows = (
        await session.execute(SELECT_JOBS, {"tenant_id": tenant_uuid(tenant_id), "limit": limit})
    ).all()
    return IngestJobList(
        jobs=[_merge(row, await read_progress(redis, str(row.id))) for row in rows]
    )


@router.get("/{job_id}", response_model=IngestJobStatus)
async def get_ingest_job(
    job_id: uuid.UUID,
    tenant_id: CurrentTenant,
    session: SessionDep,
    redis: RedisDep,
) -> IngestJobStatus:
    """One job's status. 404 for another tenant's job — the same answer as for no job."""
    row = (
        await session.execute(SELECT_JOB, {"id": job_id, "tenant_id": tenant_uuid(tenant_id)})
    ).first()
    if row is None:
        raise _reject(status.HTTP_404_NOT_FOUND, "job not found")
    return _merge(row, await read_progress(redis, str(job_id)))

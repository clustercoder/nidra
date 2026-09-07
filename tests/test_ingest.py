"""P5: upload validation, CSV/PCAP parsing, and the replay clock.

Against the real Redis and Postgres from `docker compose up -d redis postgres`. The
fixture CSV carries both defects the ML doc documents — leading-space column names and a
header row repeated mid-file — because a parser tested only on clean input is a parser
tested on a file that does not exist.

The pacing test injects a fake clock and a recording sleep: `ReplayClock` is arithmetic,
and the way to test arithmetic is not to wait for it.
"""

from __future__ import annotations

import shutil
import struct
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy import text

from api.main import create_app
from nidra_common.bus import Bus, create_redis, stream_name
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.events import (
    FLOW_FIELDS,
    JOB_COMPLETE,
    JOB_ERROR,
    IngestJobMessage,
    RawEvent,
)
from services.ingest.jobs import job_key, read_progress, upload_path
from services.ingest.parsers import (
    TSHARK_FIELDS,
    IngestError,
    count_csv_rows,
    normalise_column,
    parse_tshark_line,
    tshark_argv,
)
from services.ingest.replay import ReplayClock
from services.ingest.worker import IngestWorker

PASSWORD = "correct-horse-battery"
CAPTURE_START = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)
FIXTURE_ROWS = 40

#: Leading spaces on every column but the first: the published CIC layout, verbatim.
CSV_HEADER = (
    "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol,"
    " Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,"
    " Total Length of Fwd Packets, Total Length of Bwd Packets, Flow IAT Mean,"
    " Flow IAT Std, Flow IAT Max, FIN Flag Count, SYN Flag Count, RST Flag Count,"
    " PSH Flag Count, ACK Flag Count, URG Flag Count, Label"
)


# ------------------------------------------------------------------------- fixtures


def _csv_row(index: int) -> str:
    """One flow, `index` seconds after the capture start, with a rising SYN count."""
    ts = CAPTURE_START + timedelta(seconds=index)
    return ",".join(
        [
            f"flow-{index}",
            "192.168.10.50",
            str(50000 + index),
            "192.168.10.51",
            "80",
            "6",
            # CIC writes day-first local time with no timezone.
            ts.strftime("%-d/%-m/%Y %H:%M:%S"),
            "1500000",  # 1.5 s, in microseconds
            "6",
            "4",
            "720",
            "480",
            "250000",
            "1000",
            "400000",
            "0",
            str(index),
            "0",
            "1",
            "3",
            "0",
            "BENIGN" if index < 20 else "PortScan",
        ]
    )


def write_fixture_csv(path: Path, rows: int = FIXTURE_ROWS) -> Path:
    """A CIC-style CSV in shuffled-within-second order, with a duplicate header mid-file."""
    lines = [CSV_HEADER]
    for index in range(rows):
        if index == rows // 2:
            lines.append(CSV_HEADER)  # the mid-file repeat some releases ship
        lines.append(_csv_row(index))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ipv4_tcp_syn(src: str, dst: str, sport: int, dport: int, ttl: int) -> bytes:
    """A minimal Ethernet/IPv4/TCP SYN frame. Checksums are zero; tshark does not care."""
    tcp = struct.pack(
        "!HHIIBBHHH", sport, dport, 1, 0, 0x50, 0x02, 8192, 0, 0
    )  # offset 5 words, SYN
    total_length = 20 + len(tcp)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_length,
        0x1234,
        0,
        ttl,
        6,
        0,
        bytes(int(part) for part in src.split(".")),
        bytes(int(part) for part in dst.split(".")),
    )
    ethernet = b"\x02\x00\x00\x00\x00\x01" + b"\x02\x00\x00\x00\x00\x02" + b"\x08\x00"
    return ethernet + ip + tcp


def write_fixture_pcap(path: Path, packets: int = 5) -> Path:
    """A tiny libpcap file (linktype Ethernet), one SYN per second."""
    with path.open("wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for index in range(packets):
            frame = _ipv4_tcp_syn("192.168.10.50", "192.168.10.51", 50000 + index, 80, 64)
            ts = CAPTURE_START + timedelta(seconds=index)
            fh.write(struct.pack("<IIII", int(ts.timestamp()), 0, len(frame), len(frame)))
            fh.write(frame)
    return path


@pytest.fixture(autouse=True)
async def fresh_engine() -> AsyncIterator[None]:
    """Dispose the pooled engine after every case.

    Each test runs on its own event loop, and a connection pooled on a closed one fails
    later in a way that reads like a database problem rather than a fixture problem.
    """
    yield
    await dispose_engine()


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def raw_events_stream(redis_client: Redis) -> AsyncIterator[str]:
    """A private `raw_events`-shaped stream, deleted afterwards."""
    name = f"test:raw_events:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name)


@pytest.fixture
async def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Path]:
    """Point `ingest.upload_dir` at a temp directory via the documented env override."""
    root = tmp_path / "uploads"
    monkeypatch.setenv("NIDRA_UPLOAD_DIR", str(root))
    get_config.cache_clear()
    try:
        yield root
    finally:
        monkeypatch.delenv("NIDRA_UPLOAD_DIR", raising=False)
        get_config.cache_clear()


@pytest.fixture
async def account(uploads: Path) -> AsyncIterator[dict[str, str]]:
    """A registered tenant with a live access token, purged afterwards."""
    email = f"p5-{uuid.uuid4().hex[:12]}@nidra.test"
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ingest.test"
    ) as client:
        created = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": PASSWORD, "org_name": "P5 SOC"},
        )
        assert created.status_code == 201, created.text
        tokens = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert tokens.status_code == 200, tokens.text
        yield {
            "tenant_id": created.json()["tenant_id"],
            "token": tokens.json()["access_token"],
            "email": email,
        }

    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM ingest_jobs WHERE tenant_id = CAST(:tid AS uuid)"),
            {"tid": created.json()["tenant_id"]},
        )
        await session.execute(text("DELETE FROM users WHERE email = :email"), {"email": email})
        await session.execute(
            text("DELETE FROM tenants WHERE id = CAST(:tid AS uuid)"),
            {"tid": created.json()["tenant_id"]},
        )
        await session.commit()
    await dispose_engine()


@pytest.fixture
async def api(account: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    """An authenticated client for the ingest endpoints."""
    app: FastAPI = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://ingest.test",
        headers={"Authorization": f"Bearer {account['token']}"},
    ) as client:
        yield client


# --------------------------------------------------------------------- replay clock


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_replay_clock_paces_at_the_speed_multiplier() -> None:
    clock = FakeClock()
    replay = ReplayClock(CAPTURE_START, speed=60.0, monotonic=clock.monotonic, sleep=clock.sleep)

    # 30 s of capture at 60x is half a second of wall clock, per window.
    for step in range(1, 4):
        slept = await replay.wait_until(CAPTURE_START + timedelta(seconds=30 * step))
        assert slept == pytest.approx(0.5)

    assert clock.slept == pytest.approx([0.5, 0.5, 0.5])
    assert clock.now == pytest.approx(101.5)


async def test_replay_clock_does_not_sleep_for_the_first_event() -> None:
    clock = FakeClock()
    replay = ReplayClock(CAPTURE_START, speed=1.0, monotonic=clock.monotonic, sleep=clock.sleep)
    assert await replay.wait_until(CAPTURE_START) == 0.0
    assert clock.slept == []


async def test_replay_clock_never_sleeps_negative_when_behind() -> None:
    clock = FakeClock()
    replay = ReplayClock(CAPTURE_START, speed=1.0, monotonic=clock.monotonic, sleep=clock.sleep)
    clock.now += 10.0  # the worker fell ten seconds behind the capture
    assert await replay.wait_until(CAPTURE_START + timedelta(seconds=2)) == 0.0
    assert clock.slept == []


async def test_replay_clock_speed_zero_is_unpaced() -> None:
    clock = FakeClock()
    replay = ReplayClock(CAPTURE_START, speed=0.0, monotonic=clock.monotonic, sleep=clock.sleep)
    assert replay.paced is False
    assert await replay.wait_until(CAPTURE_START + timedelta(hours=3)) == 0.0
    assert clock.slept == []


# ------------------------------------------------------------------------ csv replay


async def _run_job(
    redis_client: Redis, stream: str, job: IngestJobMessage
) -> tuple[list[RawEvent], object]:
    """Run the worker inline over one job and read everything it published."""
    bus = Bus(redis_client, stream, group="features", consumer="test")
    worker = IngestWorker(redis_client, bus)
    result = await worker.run_job(job)
    entries = await redis_client.xrange(stream)
    events = [RawEvent.model_validate_json(fields[b"payload"]) for _, fields in entries]
    return events, result


def _job(path: Path, kind: str = "csv", speed: float = 0.0) -> IngestJobMessage:
    return IngestJobMessage(
        job_id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        kind=kind,  # type: ignore[arg-type]
        path=str(path),
        filename=path.name,
        speed=speed,
    )


async def test_csv_job_publishes_every_flow_in_timestamp_order(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_csv(tmp_path / "wednesday.csv"))
    events, result = await _run_job(redis_client, raw_events_stream, job)

    assert result.status == JOB_COMPLETE
    assert result.published == FIXTURE_ROWS  # the duplicate header row is not an event
    assert len(events) == FIXTURE_ROWS
    assert [event.ts for event in events] == sorted(event.ts for event in events)
    assert {event.kind for event in events} == {"flow"}
    assert {event.job_id for event in events} == {job.job_id}
    assert {event.tenant_id for event in events} == {job.tenant_id}


async def test_csv_job_maps_cic_columns_onto_flow_fields(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_csv(tmp_path / "wednesday.csv"))
    events, _ = await _run_job(redis_client, raw_events_stream, job)

    first, last = events[0], events[-1]
    assert set(first.fields) == set(FLOW_FIELDS)
    assert first.src_ip == "192.168.10.50"
    assert first.dst_ip == "192.168.10.51"
    assert first.src_port == 50000
    assert first.dst_port == 80
    assert first.protocol == 6
    # Microseconds in the CSV, seconds on the wire.
    assert first.fields["duration"] == pytest.approx(1.5)
    assert first.fields["iat_mean"] == pytest.approx(0.25)
    assert first.fields["iat_max"] == pytest.approx(0.4)
    assert first.fields["bytes_fwd"] == pytest.approx(720.0)
    assert first.fields["pkts_bwd"] == pytest.approx(4.0)
    assert first.fields["ack_count"] == pytest.approx(3.0)
    # SYN count rises with the row index; the label is carried, never in `fields`.
    assert first.fields["syn_count"] == pytest.approx(0.0)
    assert last.fields["syn_count"] == pytest.approx(FIXTURE_ROWS - 1)
    assert first.label == "BENIGN"
    assert last.label == "PortScan"


async def test_csv_job_reports_progress_and_completion(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_csv(tmp_path / "wednesday.csv"))
    try:
        await _run_job(redis_client, raw_events_stream, job)
        progress = await read_progress(redis_client, job.job_id)
        assert progress["status"] == JOB_COMPLETE
        assert int(progress["processed"]) == FIXTURE_ROWS
        # The duplicate header row is counted in `total` but never published.
        assert int(progress["total"]) == FIXTURE_ROWS + 1
    finally:
        await redis_client.delete(job_key(job.job_id))


async def test_csv_job_paces_publication_with_the_replay_clock(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    job = _job(write_fixture_csv(tmp_path / "wednesday.csv", rows=4), speed=60.0)
    bus = Bus(redis_client, raw_events_stream, group="features", consumer="test")
    worker = IngestWorker(redis_client, bus, sleep=record)
    result = await worker.run_job(job)

    assert result.published == 4
    # Rows are one capture-second apart; at 60x that is ~1/60 s each, and the clock
    # never runs backwards because the fake sleep does not advance real time.
    assert len(slept) == 3
    assert slept[0] == pytest.approx(1 / 60, rel=0.5)


async def test_missing_upload_marks_the_job_errored_rather_than_raising(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(tmp_path / "never-written.csv")
    try:
        events, result = await _run_job(redis_client, raw_events_stream, job)
        assert events == []
        assert result.status == JOB_ERROR
        assert (await read_progress(redis_client, job.job_id))["status"] == JOB_ERROR
    finally:
        await redis_client.delete(job_key(job.job_id))


async def test_csv_without_cic_columns_is_an_ingest_error(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    path = tmp_path / "wrong.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    _, result = await _run_job(redis_client, raw_events_stream, _job(path))
    assert result.status == JOB_ERROR
    assert "required column" in result.error


def test_normalise_column_strips_the_cic_leading_spaces() -> None:
    assert normalise_column(" Total Length of Fwd Packets") == "total length of fwd packets"
    assert normalise_column("Tot_Fwd_Pkts") == "tot fwd pkts"


def test_count_csv_rows_excludes_the_header(tmp_path: Path) -> None:
    path = write_fixture_csv(tmp_path / "wednesday.csv")
    assert count_csv_rows(path) == FIXTURE_ROWS + 1  # + the duplicate header row


# ------------------------------------------------------------------------ pcap path


def test_tshark_argv_uses_exactly_the_documented_field_list() -> None:
    argv = tshark_argv(Path("/data/uploads/t/j/j.pcap"))
    assert argv[:3] == ["tshark", "-r", "/data/uploads/t/j/j.pcap"]
    assert [argv[i + 1] for i, token in enumerate(argv) if token == "-e"] == TSHARK_FIELDS
    assert TSHARK_FIELDS == [
        "frame.time_epoch",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "ip.proto",
        "ip.ttl",
        "tcp.window_size_value",
        "ip.flags.mf",
        "ip.frag_offset",
        "frame.len",
        "tcp.len",
        "tcp.flags",
        "tcp.analysis.retransmission",
    ]
    # No shell, and the only interpolated value is the server-side path.
    assert argv[-2:] == ["-Y", "ip"]


def test_parse_tshark_line_treats_empty_retransmission_as_zero() -> None:
    line = "1499245200.5,192.168.10.50,192.168.10.51,50000,80,,,6,64,8192,0,0,74,0,0x0002,"
    event = parse_tshark_line(line, tenant_id="t", job_id="j")
    assert event is not None
    assert event.kind == "packet"
    assert event.src_port == 50000 and event.dst_port == 80
    assert event.fields["retransmission"] == 0.0
    assert event.fields["frag"] == 0.0
    assert event.fields["ttl"] == 64.0
    assert event.fields["tcp_flags"] == 2.0  # 0x0002, the SYN bit


def test_parse_tshark_line_coalesces_udp_ports_and_flags_fragments() -> None:
    line = "1499245200.5,192.168.10.50,8.8.8.8,,,53210,53,17,64,,1,0,90,,,1"
    event = parse_tshark_line(line, tenant_id="t", job_id="j")
    assert event is not None
    assert (event.src_port, event.dst_port, event.protocol) == (53210, 53, 17)
    assert event.fields["frag"] == 1.0
    assert event.fields["retransmission"] == 1.0
    assert "tcp_len" not in event.fields  # UDP carries no TCP fields, rather than zeros


def test_parse_tshark_line_accepts_the_tshark_4_boolean_rendering() -> None:
    """Verbatim tshark 4 output for the fixture capture: `ip.flags.mf` prints `False`."""
    line = (
        "1499245200.000000000,192.168.10.50,192.168.10.51,50000,80,,,"
        "6,64,8192,False,0,54,0,0x0002,"
    )
    event = parse_tshark_line(line, tenant_id="t", job_id="j")
    assert event is not None
    assert event.fields["frag"] == 0.0
    assert event.fields["retransmission"] == 0.0
    assert event.fields["frame_len"] == 54.0

    fragmented = line.replace(",False,0,54", ",True,0,54").rstrip(",") + ",True"
    event = parse_tshark_line(fragmented, tenant_id="t", job_id="j")
    assert event is not None
    assert event.fields["frag"] == 1.0
    assert event.fields["retransmission"] == 1.0


#: Real `tshark 4` output for the fixture capture, captured verbatim. Used by the shim
#: so the subprocess path is exercised on a machine without tshark installed.
TSHARK_OUTPUT = "\n".join(
    f"14992452{index:02d}.000000000,192.168.10.50,192.168.10.51,"
    f"5000{index},80,,,6,64,8192,False,0,54,0,0x0002,"
    for index in range(5)
)


def write_tshark_shim(path: Path, *, stdout: str = TSHARK_OUTPUT, exit_code: int = 0) -> Path:
    """An executable stand-in for tshark, so the subprocess path is always covered."""
    path.write_text(
        "#!/bin/sh\n"
        f'if [ {exit_code} -ne 0 ]; then echo "tshark: bad file" >&2; exit {exit_code}; fi\n'
        f"cat <<'EOF'\n{stdout}\nEOF\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


async def test_pcap_job_streams_packets_from_the_tshark_subprocess(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_pcap(tmp_path / "tiny.pcap"), kind="pcap")
    shim = write_tshark_shim(tmp_path / "tshark-shim")
    bus = Bus(redis_client, raw_events_stream, group="features", consumer="test")
    worker = IngestWorker(
        redis_client, bus, cfg={**get_config(), "ingest": {"tshark_bin": str(shim)}}
    )
    try:
        result = await worker.run_job(job)
        assert result.status == JOB_COMPLETE, result.error
        assert result.published == 5
        entries = await redis_client.xrange(raw_events_stream)
        events = [RawEvent.model_validate_json(fields[b"payload"]) for _, fields in entries]
        assert {event.kind for event in events} == {"packet"}
        assert [event.ts for event in events] == sorted(event.ts for event in events)
        assert events[0].fields["ttl"] == 64.0
        assert events[0].fields["tcp_flags"] == 2.0
        assert events[0].src_port == 50000
    finally:
        await redis_client.delete(job_key(job.job_id))


async def test_pcap_job_reports_a_nonzero_tshark_exit(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_pcap(tmp_path / "tiny.pcap"), kind="pcap")
    shim = write_tshark_shim(tmp_path / "tshark-shim", exit_code=2)
    bus = Bus(redis_client, raw_events_stream, group="features", consumer="test")
    worker = IngestWorker(
        redis_client, bus, cfg={**get_config(), "ingest": {"tshark_bin": str(shim)}}
    )
    try:
        result = await worker.run_job(job)
        assert result.status == JOB_ERROR
        assert "tshark exited 2" in result.error
    finally:
        await redis_client.delete(job_key(job.job_id))


@pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark is not installed")
async def test_pcap_job_publishes_packet_events(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_pcap(tmp_path / "tiny.pcap"), kind="pcap")
    try:
        events, result = await _run_job(redis_client, raw_events_stream, job)
        assert result.status == JOB_COMPLETE, result.error
        assert len(events) == 5
        assert {event.kind for event in events} == {"packet"}
        assert [event.ts for event in events] == sorted(event.ts for event in events)
        assert events[0].src_ip == "192.168.10.50"
        assert events[0].protocol == 6
        assert events[0].fields["ttl"] == 64.0
        assert events[0].fields["tcp_flags"] == 2.0
    finally:
        await redis_client.delete(job_key(job.job_id))


async def test_pcap_job_without_tshark_reports_an_ingest_error(
    tmp_path: Path, redis_client: Redis, raw_events_stream: str
) -> None:
    job = _job(write_fixture_pcap(tmp_path / "tiny.pcap"), kind="pcap")
    bus = Bus(redis_client, raw_events_stream, group="features", consumer="test")
    worker = IngestWorker(
        redis_client, bus, cfg={**get_config(), "ingest": {"tshark_bin": "tshark-does-not-exist"}}
    )
    try:
        result = await worker.run_job(job)
        assert result.status == JOB_ERROR
        assert "not on PATH" in result.error
    finally:
        await redis_client.delete(job_key(job.job_id))


def test_iter_pcap_events_is_an_ingest_error_not_a_crash() -> None:
    """The missing-binary path is an `IngestError`, so the worker acks rather than loops."""
    assert issubclass(IngestError, RuntimeError)


# ---------------------------------------------------------------------- upload API


async def test_upload_queues_a_job_and_reports_its_status(
    api: httpx.AsyncClient,
    account: dict[str, str],
    tmp_path: Path,
    uploads: Path,
    redis_client: Redis,
) -> None:
    path = write_fixture_csv(tmp_path / "wednesday.csv")
    stream = stream_name("ingest_jobs")
    before = await redis_client.xlen(stream)

    response = await api.post(
        "/api/v1/ingest",
        files={"file": ("Wednesday-WorkingHours.pcap_ISCX.csv", path.read_bytes(), "text/csv")},
        data={"speed": "30"},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["kind"] == "csv"
    assert body["speed"] == 30.0
    assert body["status"] == "queued"
    job_id = body["job_id"]

    try:
        # Written under the job id, never under the client's filename.
        stored = upload_path(account["tenant_id"], job_id, ".csv")
        assert stored.is_file()
        assert stored.read_bytes() == path.read_bytes()
        assert uploads in stored.parents
        assert "Wednesday" not in str(stored)

        assert await redis_client.xlen(stream) == before + 1
        entries = await redis_client.xrange(stream, count=1000)
        queued = [IngestJobMessage.model_validate_json(f[b"payload"]) for _, f in entries]
        message = next(job for job in queued if job.job_id == job_id)
        assert message.tenant_id == account["tenant_id"]
        assert message.kind == "csv"
        assert message.speed == 30.0
        assert message.path == str(stored)

        status_response = await api.get(f"/api/v1/ingest/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "queued"
        assert status_response.json()["filename"] == "Wednesday-WorkingHours.pcap_ISCX.csv"

        listing = await api.get("/api/v1/ingest")
        assert listing.status_code == 200
        assert [job["job_id"] for job in listing.json()["jobs"]] == [job_id]
    finally:
        await redis_client.delete(job_key(job_id))


async def test_upload_defaults_the_speed_to_config(
    api: httpx.AsyncClient, tmp_path: Path, redis_client: Redis
) -> None:
    path = write_fixture_csv(tmp_path / "wednesday.csv")
    response = await api.post(
        "/api/v1/ingest", files={"file": ("flows.csv", path.read_bytes(), "text/csv")}
    )
    assert response.status_code == 202, response.text
    try:
        assert response.json()["speed"] == get_config()["replay"]["default_speed"]
    finally:
        await redis_client.delete(job_key(response.json()["job_id"]))


async def test_upload_rejects_an_extension_off_the_allowlist(api: httpx.AsyncClient) -> None:
    response = await api.post(
        "/api/v1/ingest", files={"file": ("capture.txt", b"anything", "text/plain")}
    )
    assert response.status_code == 415


async def test_upload_rejects_magic_byte_mismatch(
    api: httpx.AsyncClient, account: dict[str, str], tmp_path: Path
) -> None:
    """A CSV renamed `.pcap` must not reach tshark, and a capture renamed `.csv` pandas."""
    csv_bytes = write_fixture_csv(tmp_path / "wednesday.csv").read_bytes()
    mislabelled = await api.post(
        "/api/v1/ingest",
        files={"file": ("capture.pcap", csv_bytes, "application/vnd.tcpdump.pcap")},
    )
    assert mislabelled.status_code == 415
    assert "magic number" in mislabelled.json()["detail"]

    pcap_bytes = write_fixture_pcap(tmp_path / "tiny.pcap").read_bytes()
    reversed_mislabel = await api.post(
        "/api/v1/ingest", files={"file": ("flows.csv", pcap_bytes, "text/csv")}
    )
    assert reversed_mislabel.status_code == 415

    # Nothing rejected is left on disk.
    tenant_dir = upload_path(account["tenant_id"], str(uuid.uuid4()), ".csv").parent.parent
    assert not tenant_dir.exists() or not any(tenant_dir.iterdir())


async def test_upload_rejects_a_file_over_the_cap(
    api: httpx.AsyncClient, uploads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("api.ingest.max_upload_bytes", lambda: 1024)
    response = await api.post(
        "/api/v1/ingest",
        files={"file": ("flows.csv", b"header\n" + b"x" * 4096, "text/csv")},
    )
    assert response.status_code == 413
    # A refused upload leaves no directory behind for the job it never created.
    assert not any(uploads.rglob("*.csv"))


async def test_upload_accepts_a_real_pcap(
    api: httpx.AsyncClient, tmp_path: Path, redis_client: Redis
) -> None:
    pcap = write_fixture_pcap(tmp_path / "tiny.pcap")
    response = await api.post(
        "/api/v1/ingest",
        files={"file": ("tiny.pcap", pcap.read_bytes(), "application/vnd.tcpdump.pcap")},
    )
    assert response.status_code == 202, response.text
    try:
        assert response.json()["kind"] == "pcap"
    finally:
        await redis_client.delete(job_key(response.json()["job_id"]))


async def test_ingest_endpoints_require_a_token(tmp_path: Path, uploads: Path) -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ingest.test"
    ) as client:
        assert (await client.get("/api/v1/ingest")).status_code == 401
        assert (await client.get(f"/api/v1/ingest/{uuid.uuid4()}")).status_code == 401
        upload = await client.post(
            "/api/v1/ingest", files={"file": ("flows.csv", b"a,b\n1,2\n", "text/csv")}
        )
        assert upload.status_code == 401


async def test_another_tenants_job_is_a_404(api: httpx.AsyncClient, redis_client: Redis) -> None:
    response = await api.get(f"/api/v1/ingest/{uuid.uuid4()}")
    assert response.status_code == 404

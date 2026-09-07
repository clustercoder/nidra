"""File → `RawEvent` chunks. Two paths, one output shape.

**CSV** is the published CICFlowMeter output, with the two defects the ML doc warns
about: column names carry leading spaces, and some releases repeat the header row
mid-file. Both are handled at parse time — the header row's `Timestamp` value fails to
parse as a date and the row is dropped with everything else that has no usable timestamp.
Reading is chunked; the whole file is never resident.

**PCAP** is `tshark -T fields` with exactly the field list from IMPLEMENTATION-ML.md
§2.2, consumed line by line off the subprocess's stdout. PyShark and Scapy are roughly
two orders of magnitude too slow for bulk extraction, and loading a capture into memory
to sort it defeats the point of streaming it.

`argv` is built as a list containing only the server-side path — no shell, and no
user-supplied string ever reaches the command line.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nidra_common.events import RawEvent

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """A job cannot be parsed: bad file, missing columns, tshark unavailable.

    Distinct from an infrastructure failure on purpose. This one will fail identically
    on every redelivery, so the worker records it on the job and acks; a Redis or
    Postgres error propagates instead and stays pending for reclaim.
    """


# ------------------------------------------------------------------------------- CSV

#: Normalised CIC column name → the name ingest uses. Both the 2017 (`Source IP`) and
#: 2018 (`Src IP`) spellings appear in published releases; accept either.
CIC_COLUMNS: dict[str, str] = {
    "source ip": "src_ip",
    "src ip": "src_ip",
    "destination ip": "dst_ip",
    "dst ip": "dst_ip",
    "source port": "src_port",
    "src port": "src_port",
    "destination port": "dst_port",
    "dst port": "dst_port",
    "protocol": "protocol",
    "timestamp": "ts",
    "flow duration": "duration_us",
    "total fwd packets": "pkts_fwd",
    "tot fwd pkts": "pkts_fwd",
    "total backward packets": "pkts_bwd",
    "tot bwd pkts": "pkts_bwd",
    "total length of fwd packets": "bytes_fwd",
    "totlen fwd pkts": "bytes_fwd",
    "total length of bwd packets": "bytes_bwd",
    "totlen bwd pkts": "bytes_bwd",
    "flow iat mean": "iat_mean_us",
    "flow iat std": "iat_std_us",
    "flow iat max": "iat_max_us",
    "syn flag count": "syn_count",
    "ack flag count": "ack_count",
    "rst flag count": "rst_count",
    "fin flag count": "fin_count",
    "psh flag count": "psh_count",
    "urg flag count": "urg_count",
    "label": "label",
}

#: Without these a row cannot be placed on a host or a window, so the file is unusable.
CSV_REQUIRED = ("src_ip", "dst_ip", "ts")

#: Internal column → `RawEvent.fields` key, for the columns that need no conversion.
_CSV_COUNTS = {
    "pkts_fwd": "pkts_fwd",
    "pkts_bwd": "pkts_bwd",
    "bytes_fwd": "bytes_fwd",
    "bytes_bwd": "bytes_bwd",
    "syn_count": "syn_count",
    "ack_count": "ack_count",
    "rst_count": "rst_count",
    "fin_count": "fin_count",
    "psh_count": "psh_count",
    "urg_count": "urg_count",
}

#: Internal column → field key, for the microsecond durations CIC reports.
_CSV_MICROSECONDS = {
    "duration_us": "duration",
    "iat_mean_us": "iat_mean",
    "iat_std_us": "iat_std",
    "iat_max_us": "iat_max",
}

MICROSECONDS_PER_SECOND = 1_000_000.0


def normalise_column(name: str) -> str:
    """`' Total Length of Fwd Packets'` → `'total length of fwd packets'`.

    Leading spaces are the documented CIC defect; the rest of the normalisation makes
    the 2017 and 2018 spellings meet in the same lookup table.
    """
    return " ".join(str(name).replace("_", " ").split()).lower()


def count_csv_rows(path: Path) -> int:
    """Data rows in a CSV, counted by newlines. One sequential read, no parsing.

    Used only to give the progress bar a denominator, so an off-by-one on a file with no
    trailing newline is immaterial.
    """
    newlines = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            newlines += chunk.count(b"\n")
    return max(newlines - 1, 0)


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        column: CIC_COLUMNS[normalise_column(column)]
        for column in frame.columns
        if normalise_column(column) in CIC_COLUMNS
    }
    return frame[list(mapping)].rename(columns=mapping)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Column as floats, with unparseable values (duplicate header rows) becoming 0."""
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").astype(float).fillna(0.0)


def _optional_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return int(number)


def _csv_chunk_to_events(
    chunk: pd.DataFrame, *, tenant_id: str, job_id: str
) -> tuple[list[RawEvent], int]:
    """Convert one chunk to events sorted by timestamp; returns (events, rows_dropped)."""
    frame = _rename_columns(chunk)
    missing = [name for name in CSV_REQUIRED if name not in frame.columns]
    if missing:
        raise IngestError(
            f"CSV is missing required column(s) {missing}; "
            "expected CICFlowMeter output (see IMPLEMENTATION-ML.md §2.3)"
        )

    rows_in = len(frame)
    # A repeated header row has 'Timestamp' in the timestamp column: it coerces to NaT
    # and is dropped here along with any other row whose time cannot be read.
    frame = frame.assign(
        ts=pd.to_datetime(frame["ts"], dayfirst=True, errors="coerce", format="mixed")
    )
    frame = frame.dropna(subset=["ts", "src_ip", "dst_ip"])

    numeric = {name: _numeric(frame, name) for name in (*_CSV_COUNTS, *_CSV_MICROSECONDS)}
    frame = frame.sort_values("ts", kind="stable")

    events: list[RawEvent] = []
    for index, row in frame.iterrows():
        fields = {key: float(numeric[column].loc[index]) for column, key in _CSV_COUNTS.items()}
        fields.update(
            {
                key: float(numeric[column].loc[index]) / MICROSECONDS_PER_SECOND
                for column, key in _CSV_MICROSECONDS.items()
            }
        )
        label = row.get("label")
        ts = row["ts"].to_pydatetime()
        events.append(
            RawEvent(
                tenant_id=tenant_id,
                job_id=job_id,
                kind="flow",
                ts=ts if ts.tzinfo else ts.replace(tzinfo=UTC),
                src_ip=str(row["src_ip"]).strip(),
                dst_ip=str(row["dst_ip"]).strip(),
                src_port=_optional_int(row.get("src_port")),
                dst_port=_optional_int(row.get("dst_port")),
                protocol=_optional_int(row.get("protocol")),
                fields=fields,
                label=str(label).strip() if isinstance(label, str) and label.strip() else None,
            )
        )
    return events, rows_in - len(frame)


def _read_csv_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    try:
        yield from pd.read_csv(
            path,
            chunksize=chunk_rows,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            skipinitialspace=False,
            on_bad_lines="skip",
            encoding="utf-8",
            encoding_errors="replace",
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestError(f"could not read CSV: {exc}") from exc


async def iter_csv_events(
    path: Path, *, tenant_id: str, job_id: str, chunk_rows: int
) -> AsyncIterator[list[RawEvent]]:
    """Yield chunks of flow events, each chunk sorted by timestamp.

    The pandas work happens on a worker thread: it is CPU-bound, and blocking the event
    loop through a multi-gigabyte file would stall the replay pacing it feeds.
    """
    chunks = _read_csv_chunks(path, chunk_rows)
    dropped = 0
    while True:
        chunk = await asyncio.to_thread(next, chunks, None)
        if chunk is None:
            break
        events, chunk_dropped = await asyncio.to_thread(
            _csv_chunk_to_events, chunk, tenant_id=tenant_id, job_id=job_id
        )
        dropped += chunk_dropped
        if events:
            yield events
    if dropped:
        logger.info("dropped %d unparseable CSV row(s) from job %s", dropped, job_id)


# ----------------------------------------------------------------------------- PCAP

#: Exactly the fields from IMPLEMENTATION-ML.md §2.2, in that order. The parser indexes
#: this list positionally, so the two cannot drift apart.
TSHARK_FIELDS: list[str] = [
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

#: Only IP packets have a host to attribute state to.
TSHARK_DISPLAY_FILTER = "ip"

#: Events per progress update. Pacing is per-event regardless; this only bounds how often
#: the worker touches Redis.
PACKET_BATCH = 500


def tshark_argv(path: Path, binary: str = "tshark") -> list[str]:
    """The extraction command, as a list. No shell, and only the validated path in it.

    `occurrence=f` is the one addition to §2.2's invocation: with tunnelled or duplicated
    layers tshark emits several values for a field, comma-joined, which would silently
    shift every column after it given `separator=,`.
    """
    argv = [
        binary,
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "separator=,",
        "-E",
        "quote=n",
        "-E",
        "occurrence=f",
    ]
    for field in TSHARK_FIELDS:
        argv += ["-e", field]
    argv += ["-Y", TSHARK_DISPLAY_FILTER]
    return argv


def _float_or_none(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


#: tshark 4 renders boolean fields as `True`/`False`; tshark 2/3 rendered them `1`/`0`.
#: Both appear in captures processed on different machines, so both are accepted.
_FALSEY = frozenset({"", "0", "false"})


def _is_set(value: str) -> bool:
    """True when tshark reported a boolean field as set, in either rendering."""
    return value.strip().lower() not in _FALSEY


def _flags_or_none(value: str) -> float | None:
    """`tcp.flags` is hex (`0x0002`); older builds print decimal. Accept both."""
    if not value:
        return None
    try:
        return float(int(value, 16 if value.lower().startswith("0x") else 10))
    except ValueError:
        return None


def parse_tshark_line(line: str, *, tenant_id: str, job_id: str) -> RawEvent | None:
    """One tshark row → a packet event, or None if it is unusable (no IPs, no time)."""
    parts = line.rstrip("\n").split(",")
    if len(parts) != len(TSHARK_FIELDS):
        return None
    (
        epoch,
        src_ip,
        dst_ip,
        tcp_sport,
        tcp_dport,
        udp_sport,
        udp_dport,
        proto,
        ttl,
        window,
        mf,
        frag_offset,
        frame_len,
        tcp_len,
        flags,
        retransmission,
    ) = (part.strip() for part in parts)

    seconds = _float_or_none(epoch)
    if seconds is None or not src_ip or not dst_ip:
        return None

    fields: dict[str, float] = {}
    for key, raw in (
        ("ttl", ttl),
        ("tcp_window", window),
        ("frame_len", frame_len),
        ("tcp_len", tcp_len),
    ):
        value = _float_or_none(raw)
        if value is not None:
            fields[key] = value

    flag_bits = _flags_or_none(flags)
    if flag_bits is not None:
        fields["tcp_flags"] = flag_bits

    # More-fragments set, or a non-zero offset: either means this packet is a fragment.
    offset = _float_or_none(frag_offset) or 0.0
    fields["frag"] = 1.0 if (_is_set(mf) or offset > 0) else 0.0
    # Empty for every normal packet; set only when tshark detected a retransmission.
    fields["retransmission"] = 1.0 if _is_set(retransmission) else 0.0

    return RawEvent(
        tenant_id=tenant_id,
        job_id=job_id,
        kind="packet",
        ts=datetime.fromtimestamp(seconds, tz=UTC),
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=_optional_int(tcp_sport or udp_sport or None),
        dst_port=_optional_int(tcp_dport or udp_dport or None),
        protocol=_optional_int(proto or None),
        fields=fields,
    )


async def iter_pcap_events(
    path: Path, *, tenant_id: str, job_id: str, binary: str = "tshark", batch: int = PACKET_BATCH
) -> AsyncIterator[list[RawEvent]]:
    """Yield batches of packet events, streamed from tshark's stdout.

    Capture order is timestamp order, so no sorting is needed — and could not be done
    without buffering the capture, which is the thing this path exists to avoid.
    """
    resolved = shutil.which(binary)
    if resolved is None:
        raise IngestError(
            f"{binary!r} is not on PATH; packet features require tshark (see CLAUDE.md)"
        )

    process = await asyncio.create_subprocess_exec(
        *tshark_argv(path, resolved),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    batched: list[RawEvent] = []
    try:
        async for raw in process.stdout:
            event = parse_tshark_line(
                raw.decode("utf-8", "replace"), tenant_id=tenant_id, job_id=job_id
            )
            if event is None:
                continue
            batched.append(event)
            if len(batched) >= batch:
                yield batched
                batched = []
        if batched:
            yield batched
    finally:
        if process.returncode is None:
            process.kill()
        stderr = await process.stderr.read() if process.stderr is not None else b""
        await process.wait()

    if process.returncode:
        raise IngestError(
            f"tshark exited {process.returncode}: {stderr.decode('utf-8', 'replace').strip()}"
        )

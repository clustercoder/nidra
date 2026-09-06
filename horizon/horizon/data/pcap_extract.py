"""Packet-level extraction via tshark field output, streamed to parquet.

Uses `tshark -T fields` directly rather than PyShark or Scapy: PyShark shells
out to tshark per packet (~2 orders of magnitude too slow for a full capture
day); Scapy's pure-Python parser is not built for bulk extraction either.

Never loads a whole capture into memory: tshark is invoked as a subprocess
with stdout piped and consumed in chunks, each chunk parsed and appended to a
parquet file immediately. The raw CSV/tshark text is never fully materialized
on disk or in memory.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Exact set of fields required by the ML spec (IMPLEMENTATION-ML.md §2.2).
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

# Column dtypes after parsing tshark's text output. Numeric columns are
# coerced with errors="coerce" -> NaN, then filled per-column below; this
# never silently corrupts a row, it produces an explicit, loggable NaN count.
_NUMERIC_COLS = [
    "frame_time_epoch",
    "tcp_srcport",
    "tcp_dstport",
    "udp_srcport",
    "udp_dstport",
    "ip_proto",
    "ip_ttl",
    "tcp_window_size_value",
    "ip_flags_mf",
    "ip_frag_offset",
    "frame_len",
    "tcp_len",
    "tcp_analysis_retransmission",
]


def _field_to_column(field: str) -> str:
    return field.replace(".", "_")


def build_tshark_argv(
    pcap_path: str | Path,
    time_filter: str | None = None,
) -> list[str]:
    """Build the tshark argv list. Never interpolate untrusted strings into a
    shell command — argv is a list, passed directly to subprocess without a
    shell, and `pcap_path` is the only externally-influenced value."""
    tshark_bin = shutil.which("tshark")
    if tshark_bin is None:
        raise RuntimeError(
            "tshark not found on PATH. The packet-level extraction path "
            "requires tshark (wireshark-common). Run `tshark -v` to verify."
        )
    argv = [tshark_bin, "-r", str(pcap_path), "-T", "fields", "-E", "separator=,", "-E", "quote=n"]
    for f in TSHARK_FIELDS:
        argv += ["-e", f]
    display_filter = "ip"
    if time_filter:
        display_filter = f"({display_filter}) and ({time_filter})"
    argv += ["-Y", display_filter]
    return argv


def stream_tshark_chunks(
    pcap_path: str | Path,
    time_filter: str | None = None,
    chunk_size: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Run tshark as a subprocess and yield parsed DataFrame chunks.

    Never materializes the full capture: reads stdout line-by-line, batches
    `chunk_size` lines, parses each batch, and yields it before reading more.
    """
    argv = build_tshark_argv(pcap_path, time_filter=time_filter)
    columns = [_field_to_column(f) for f in TSHARK_FIELDS]

    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    buffer: list[list[str]] = []
    try:
        for line in proc.stdout:
            row = line.rstrip("\n").split(",")
            if len(row) != len(columns):
                # Malformed row (e.g. an embedded comma in a rare field) —
                # log and drop rather than silently misaligning columns.
                logger.warning("dropping malformed tshark row: %r", line)
                continue
            buffer.append(row)
            if len(buffer) >= chunk_size:
                yield _finalize_chunk(buffer, columns)
                buffer = []
        if buffer:
            yield _finalize_chunk(buffer, columns)
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"tshark exited with code {ret}: {stderr}")


def _finalize_chunk(rows: list[list[str]], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)

    # tcp/udp ports are separate fields; coalesce into one port pair.
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["src_port"] = df["tcp_srcport"].fillna(df["udp_srcport"])
    df["dst_port"] = df["tcp_dstport"].fillna(df["udp_dstport"])

    # tcp.analysis.retransmission is empty for normal packets, "1" for
    # retransmissions — empty must be treated as 0, not NaN or dropped.
    df["is_retransmission"] = df["tcp_analysis_retransmission"].fillna(0).astype(int).clip(0, 1)

    # ip.flags.mf (more-fragments bit) combined with a non-zero frag_offset
    # gives a fragmentation indicator.
    mf = df["ip_flags_mf"].fillna(0).astype(int)
    frag_offset = df["ip_frag_offset"].fillna(0).astype(int)
    df["is_fragment"] = ((mf == 1) | (frag_offset > 0)).astype(int)

    # payload length: total frame length minus header overhead is
    # approximated by tcp.len for TCP; for non-TCP packets fall back to
    # frame_len (payload_size features are computed downstream per-protocol
    # if a more precise figure is needed).
    df["payload_len"] = df["tcp_len"].fillna(df["frame_len"])

    df = df.dropna(subset=["frame_time_epoch", "ip_src", "ip_dst"])
    df["frame_time_epoch"] = df["frame_time_epoch"].astype(float)

    keep = [
        "frame_time_epoch",
        "ip_src",
        "ip_dst",
        "src_port",
        "dst_port",
        "ip_proto",
        "ip_ttl",
        "tcp_window_size_value",
        "is_fragment",
        "frame_len",
        "payload_len",
        "tcp_flags",
        "is_retransmission",
    ]
    return df[keep]


def extract_pcap_to_parquet(
    pcap_path: str | Path,
    out_parquet: str | Path,
    time_filter: str | None = None,
    chunk_size: int = 200_000,
) -> int:
    """Extract one pcap to a parquet file, chunked. Returns row count.

    Writes incrementally via a ParquetWriter so memory stays bounded
    regardless of capture size. The caller is responsible for deleting the
    source pcap/CSV once the parquet is confirmed, per the "delete
    intermediate giant files where safe" guidance — this function does not
    delete the source itself.
    """
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for chunk in stream_tshark_chunks(pcap_path, time_filter=time_filter, chunk_size=chunk_size):
            if chunk.empty:
                continue
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        logger.warning("extract_pcap_to_parquet: no packets extracted from %s", pcap_path)
    return total_rows


def bounded_time_filter(onset_epoch: float, end_epoch: float, pre_minutes: int = 30, post_minutes: int = 10) -> str:
    """Build a tshark display-filter time bound: [onset - pre, end + post]."""
    lo = onset_epoch - pre_minutes * 60
    hi = end_epoch + post_minutes * 60
    return f"frame.time_epoch >= {lo} and frame.time_epoch <= {hi}"


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract packet-level fields from a pcap to parquet via tshark.")
    parser.add_argument("pcap", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--onset-epoch", type=float, default=None)
    parser.add_argument("--end-epoch", type=float, default=None)
    args = parser.parse_args()

    tf = None
    if args.onset_epoch is not None and args.end_epoch is not None:
        tf = bounded_time_filter(args.onset_epoch, args.end_epoch)

    n = extract_pcap_to_parquet(args.pcap, args.out, time_filter=tf)
    logger.info("wrote %d packet rows to %s", n, args.out)

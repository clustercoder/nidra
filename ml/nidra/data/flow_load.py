"""Loader for published CICFlowMeter CSVs (CIC-IDS2017 flow-only releases).

Known issues with these releases, handled explicitly here:
- Column names carry leading spaces (" Flow Duration").
- Some releases have duplicate header rows embedded mid-file.
- Numeric columns occasionally fail to parse; such rows are dropped, not
  silently coerced, and the drop is logged with a reason.
- At least one real CIC-IDS2017 file (Thursday-WorkingHours-Morning-
  WebAttacks) is not valid UTF-8 — its "Web Attack" labels contain a
  Windows-1252 en-dash (byte 0x96), which is not valid UTF-8 and crashes a
  plain `pd.read_csv`. Handled by trying UTF-8 first (the common case) and
  falling back to latin-1 (which never raises — every byte value is a
  valid latin-1 codepoint) only on a genuine decode failure, logged when it
  happens rather than silently switching encodings.

Logs input row count, accepted rows, dropped rows, and reasons for drops —
never a silent corruption of malformed data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical rename map from raw (stripped) CICFlowMeter column names to the
# snake_case names used internally. Only the columns actually consumed by
# windowize.py are mapped; anything else keeps its stripped original name.
_RENAME_MAP = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_len_fwd",
    "Total Length of Bwd Packets": "total_len_bwd",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "SYN Flag Count": "syn_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "RST Flag Count": "rst_flag_count",
    "FIN Flag Count": "fin_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "URG Flag Count": "urg_flag_count",
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Protocol": "protocol",
    "Timestamp": "timestamp",
    "Label": "label",
}

_REQUIRED_NUMERIC = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd",
    "total_len_bwd",
]


@dataclass
class LoadReport:
    path: str
    input_rows: int = 0
    accepted_rows: int = 0
    dropped_rows: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def log(self) -> None:
        logger.info(
            "flow_load %s: input=%d accepted=%d dropped=%d reasons=%s",
            self.path,
            self.input_rows,
            self.accepted_rows,
            self.dropped_rows,
            self.drop_reasons,
        )


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    return df


def load_cicflowmeter_csv(path: str | Path, chunksize: int = 200_000, row_cap: int | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """Load one CICFlowMeter CSV robustly. Returns (dataframe, report).

    Has-IP columns are optional: some CIC-IDS2017 releases omit Source/
    Destination IP from the CSV (5-tuple only available via the PCAP-side
    join in that case). When absent, src_ip/dst_ip are left null and the
    caller must fall back to whatever host-identification path is available.

    `row_cap`, if set, stops reading once at least that many INPUT rows have
    been consumed — the file's remaining chunks are never read. This is for
    MVP-scale runs against a day file too large to fully load, where reading
    it entirely just to subsample it afterward would be wasteful; `None`
    reads the whole file (what both config/default.yaml and
    config/mvp_2017.yaml currently use — CIC-IDS2017 day files are small
    enough that stratified sample capping at the tensor-construction stage,
    not row_cap, is what bounds MVP compute cost).
    """
    path = Path(path)

    def _open_reader(encoding: str):
        return pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
            skip_blank_lines=True,
            on_bad_lines="skip",
            encoding=encoding,
        )

    def _run(encoding: str) -> tuple[LoadReport, list[pd.DataFrame]]:
        """Attempt a full read at the given encoding. Raises UnicodeDecodeError
        (from anywhere in the file, not just the first chunk) if this
        encoding is wrong, letting the caller retry with a different one."""
        report = LoadReport(path=str(path))
        chunks: list[pd.DataFrame] = []
        reader = _open_reader(encoding)
        try:
            for raw_chunk in reader:
                report.input_rows += len(raw_chunk)
                chunk = _strip_columns(raw_chunk)

                # Mid-file duplicate header rows (a data row whose cells
                # literally repeat the header text, e.g. "Destination Port")
                # are caught below: coercing required numeric columns
                # to_numeric turns the header text into NaN, and the row is
                # dropped as unparseable_numeric.
                n_before = len(chunk)
                chunk = chunk.rename(columns=_RENAME_MAP)

                present_required = [c for c in _REQUIRED_NUMERIC if c in chunk.columns]
                for col in present_required:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                bad_numeric = chunk[present_required].isna().any(axis=1) if present_required else pd.Series(False, index=chunk.index)
                n_bad_numeric = int(bad_numeric.sum())
                if n_bad_numeric:
                    report.drop_reasons["unparseable_numeric"] = report.drop_reasons.get("unparseable_numeric", 0) + n_bad_numeric
                chunk = chunk[~bad_numeric]

                n_dropped_this_chunk = n_before - len(chunk)
                report.dropped_rows += n_dropped_this_chunk
                report.accepted_rows += len(chunk)
                chunks.append(chunk)

                if row_cap is not None and report.input_rows >= row_cap:
                    logger.info("load_cicflowmeter_csv: row_cap=%d reached after %d input rows, stopping early (%s)",
                                row_cap, report.input_rows, path)
                    break
        finally:
            reader.close()
        return report, chunks

    try:
        report, chunks = _run("utf-8")
    except UnicodeDecodeError:
        logger.warning("load_cicflowmeter_csv: %s is not valid UTF-8, retrying with latin-1 "
                        "(known issue: some CIC-IDS2017 files use Windows-1252 punctuation "
                        "in Label values, e.g. an en-dash in 'Web Attack – XSS')", path)
        report, chunks = _run("latin-1")

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame()

    # Numeric flag-count columns: coerce, fill missing with 0 (absence of a
    # flag-count column in older CIC-IDS2017 releases means "not measured",
    # treated as 0 flags rather than dropping the row).
    for col in ["syn_flag_count", "ack_flag_count", "rst_flag_count", "fin_flag_count", "psh_flag_count", "urg_flag_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()

    report.log()
    return df, report


def load_multiple(paths: list[str | Path]) -> tuple[pd.DataFrame, list[LoadReport]]:
    dfs, reports = [], []
    for p in paths:
        df, report = load_cicflowmeter_csv(p)
        df["source_file"] = Path(p).name
        dfs.append(df)
        reports.append(report)
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined, reports


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Load and validate CICFlowMeter CSVs.")
    parser.add_argument("csv", type=str, nargs="+")
    parser.add_argument("--out", type=str, default=None, help="optional parquet output path")
    args = parser.parse_args()

    df, reports = load_multiple(args.csv)
    if args.out:
        df.to_parquet(args.out, index=False)
        logger.info("wrote %d rows to %s", len(df), args.out)

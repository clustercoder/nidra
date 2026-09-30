"""Loader for CTU-13 Argus bidirectional NetFlow (`*.binetflow`) files.

CTU-13 (Garcia et al., 2014) is a second real dataset for this project. Its
flow records come from Argus, not CICFlowMeter, so this module's whole job
is to emit exactly the internal column names `nidra.data.join` and
`nidra.data.windowize` already consume — after that every downstream stage
(windowing, graph scalars, dynamics, labelling, splits, training,
benchmark) is shared with CIC-IDS2017 and the two datasets are directly
comparable by construction.

The mapping is not lossless, and the places where it is not are the whole
reason `reports/CTU13_DATASET_ASSESSMENT_2026-09-23.md` exists:

    Argus column        internal name           fidelity
    ----------------    --------------------    --------------------------
    StartTime           timestamp (epoch s)     exact (microsecond clock)
    Dur                 flow_duration           exact, converted s -> us
    SrcAddr/DstAddr     src_ip/dst_ip           exact
    Sport/Dport         src_port/dst_port       exact where numeric
    Proto               protocol                exact
    TotPkts             total_fwd_packets       total only; see below
    SrcBytes            total_len_fwd           exact
    TotBytes - SrcBytes total_len_bwd           exact
    Dur / (TotPkts-1)   flow_iat_mean           exact for the same estimator
    (absent)            flow_iat_max            NOT AVAILABLE -> 0
    State               *_flag_count            PRESENCE, not count
    Label               label                   exact

Two of those need stating plainly.

`total_fwd_packets` / `total_bwd_packets`: Argus's `TotPkts` is the flow's
packet total with no directional split. Every feature in FEATURE_ORDER that
touches packet counts uses only their SUM (`windowize.aggregate_flows`
computes `total_packets = fwd + bwd` and never reads either alone), so all
of `TotPkts` is assigned to the forward column and the reverse column is
zero. That is lossless for every feature actually computed, and wrong for
anything that later reads the two apart — hence this paragraph.

TCP flag counts: CICFlowMeter reports how MANY packets in the flow carried
each flag. Argus reports which flags were SEEN, per direction, in its
`State` word (`SRPA_FSPA` = source sent SYN/RST/PSH/ACK, destination sent
FIN/SYN/PSH/ACK). Presence is therefore counted as one per direction, which
is close to the true count for SYN/FIN/RST/URG (those appear about once per
direction in a well-behaved connection) and badly under-counts ACK and PSH
(a bulk transfer carries thousands of each). `ack_ratio` and `psh_ratio`
are consequently NOT comparable across the two datasets; the cross-dataset
feature mask in `nidra.data.schema` is what keeps that out of the
transfer experiments, and `nidra.data.dataset_shift` measures it rather
than assuming it.

Non-TCP flows have no TCP flags at all. Argus writes a word there instead
(`CON`, `INT`, `URP`, `RED`, `ECO`, `UNK`), and those words contain the very
letters a naive reader scores as flags — `URP` would read as URG+RST+PSH.
Flags are therefore parsed ONLY for `proto == tcp` states carrying the
`_` direction separator.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The CVUT university network the CTU-13 captures were taken inside. Every
#: labelled host — botnet and verified-normal alike — lives here; everything
#: else in the capture is Internet background appearing as a flow source.
#: NIDRA forecasts the risk of hosts you monitor, so the monitored range is
#: the host population, exactly as CIC-IDS2017's victim subnet is there.
CTU_INTERNAL_CIDRS: tuple[str, ...] = ("147.32.0.0/16",)

_COLUMNS = [
    "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr",
    "Dport", "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label",
]

#: Argus TCP state letters -> the CICFlowMeter flag-count column they feed.
_FLAG_LETTERS = {
    "S": "syn_flag_count",
    "A": "ack_flag_count",
    "R": "rst_flag_count",
    "F": "fin_flag_count",
    "P": "psh_flag_count",
    "U": "urg_flag_count",
}
FLAG_COLUMNS = list(dict.fromkeys(_FLAG_LETTERS.values()))


@dataclass
class CTULoadReport:
    path: str
    input_rows: int = 0
    accepted_rows: int = 0
    dropped_rows: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str, n: int) -> None:
        if n:
            self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + int(n)

    def log(self) -> None:
        logger.info(
            "ctu_load %s: input=%d accepted=%d dropped=%d reasons=%s",
            self.path, self.input_rows, self.accepted_rows, self.dropped_rows, self.drop_reasons,
        )


def parse_ctu_timestamp(ts: pd.Series) -> pd.Series:
    """`2011/08/10 09:46:59.607825` -> epoch seconds as float64.

    The printed clock is read literally (the captures are CEST; no offset is
    applied). Nothing in the CTU path joins against another clock — there is
    no packet table to meet — so the only thing an offset would move is which
    UTC calendar day `splits.capture_day` assigns a window to, and reading
    the local clock literally is what keeps a capture's own midnight at
    midnight. Unparseable values become NaN and are dropped by the caller.
    """
    parsed = pd.to_datetime(ts, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    fallback = parsed.isna() & ts.notna()
    if fallback.any():
        parsed.loc[fallback] = pd.to_datetime(ts[fallback], errors="coerce")
    out = pd.Series(np.nan, index=ts.index, dtype="float64")
    valid = parsed.notna()
    out.loc[valid] = parsed[valid].dt.as_unit("ns").astype("int64") / 1e9
    return out


def parse_argus_state_flags(state: pd.Series, proto: pd.Series) -> pd.DataFrame:
    """Argus `State` -> the six CICFlowMeter flag-count columns.

    Counted as PRESENCE per direction (see the module docstring), and only
    for TCP states carrying the `_` direction separator — a UDP `CON` or an
    ICMP `URP` is a state word, not a flag set.
    """
    s = state.astype("string").fillna("")
    is_tcp = proto.astype("string").str.lower().fillna("") == "tcp"
    parsable = is_tcp & s.str.contains("_", regex=False)
    src = s.where(parsable, "").str.split("_").str[0].fillna("")
    dst = s.where(parsable, "").str.split("_").str[1].fillna("")
    out = {}
    for letter, column in _FLAG_LETTERS.items():
        out[column] = (
            src.str.contains(letter, regex=False).fillna(False).astype("int64")
            + dst.str.contains(letter, regex=False).fillna(False).astype("int64")
        )
    return pd.DataFrame(out, index=state.index)


def _numeric_port(port: pd.Series) -> pd.Series:
    """Argus prints ICMP type/code in the port columns as hex (`0x0303`).
    Those are not ports; they are parsed to their integer value rather than
    dropping the row, and plain decimal ports parse unchanged."""
    text = port.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    hexish = numeric.isna() & text.str.lower().str.startswith("0x", na=False)
    if hexish.any():
        numeric.loc[hexish] = text[hexish].apply(lambda v: int(v, 16))
    return numeric.fillna(-1).astype("int64")


def _in_any_cidr(ips: pd.Series, cidrs: tuple[str, ...]) -> pd.Series:
    """Vectorized membership of dotted-quad strings in a set of IPv4 CIDRs.

    Done on the packed 32-bit integer rather than per-row `ipaddress` calls:
    a CTU scenario carries up to 4.7M flows and a Python-level check per row
    is minutes of wall clock per file.
    """
    octets = ips.astype("string").str.split(".", expand=True)
    if octets.shape[1] != 4:
        return pd.Series(False, index=ips.index)
    parts = [pd.to_numeric(octets[i], errors="coerce") for i in range(4)]
    packed = parts[0] * 2**24 + parts[1] * 2**16 + parts[2] * 2**8 + parts[3]
    inside = pd.Series(False, index=ips.index)
    for cidr in cidrs:
        net = ipaddress.ip_network(cidr)
        lo, hi = int(net.network_address), int(net.broadcast_address)
        inside |= packed.between(lo, hi)
    return inside.fillna(False)


def load_binetflow(
    path: str | Path,
    chunksize: int = 500_000,
    internal_cidrs: tuple[str, ...] | None = CTU_INTERNAL_CIDRS,
    row_cap: int | None = None,
) -> tuple[pd.DataFrame, CTULoadReport]:
    """Read one `.binetflow` into the internal flow-record columns.

    Streams the file in chunks (the largest scenario is 640 MB / 4.7M rows)
    and reports every dropped row with a reason — a silently shrinking flow
    table would move every downstream number with nothing to show for it.

    `internal_cidrs=None` disables the monitored-network host filter and
    keeps Internet background sources as hosts in their own right.
    """
    path = Path(path)
    report = CTULoadReport(path=str(path))
    kept: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path, chunksize=chunksize, low_memory=False, skip_blank_lines=True,
        on_bad_lines="skip", dtype="string",
    )
    try:
        for raw in reader:
            report.input_rows += len(raw)
            chunk = raw.rename(columns=lambda c: c.strip())
            missing = [c for c in _COLUMNS if c not in chunk.columns]
            if missing:
                raise ValueError(f"{path}: not a CTU-13 binetflow — missing columns {missing}")

            n_before = len(chunk)
            # Embedded header rows repeat the column names as data; every
            # numeric column coerces to NaN and the row is dropped here
            # rather than becoming a zero-valued flow.
            dur = pd.to_numeric(chunk["Dur"], errors="coerce")
            tot_pkts = pd.to_numeric(chunk["TotPkts"], errors="coerce")
            tot_bytes = pd.to_numeric(chunk["TotBytes"], errors="coerce")
            src_bytes = pd.to_numeric(chunk["SrcBytes"], errors="coerce")
            epoch = parse_ctu_timestamp(chunk["StartTime"])

            bad = dur.isna() | tot_pkts.isna() | tot_bytes.isna() | src_bytes.isna() | epoch.isna()
            bad |= chunk["SrcAddr"].isna() | chunk["DstAddr"].isna()
            report.note("unparseable_row", int(bad.sum()))
            chunk = chunk[~bad]
            dur, tot_pkts, tot_bytes, src_bytes, epoch = (
                x[~bad] for x in (dur, tot_pkts, tot_bytes, src_bytes, epoch)
            )

            if internal_cidrs:
                internal = _in_any_cidr(chunk["SrcAddr"], tuple(internal_cidrs))
                report.note("external_source_host", int((~internal).sum()))
                chunk = chunk[internal]
                dur, tot_pkts, tot_bytes, src_bytes, epoch = (
                    x[internal] for x in (dur, tot_pkts, tot_bytes, src_bytes, epoch)
                )

            report.dropped_rows += n_before - len(chunk)
            if chunk.empty:
                continue

            reverse_bytes = tot_bytes - src_bytes
            report.note("src_bytes_exceeds_total", int((reverse_bytes < 0).sum()))
            reverse_bytes = reverse_bytes.clip(lower=0.0)

            # Mean inter-arrival time of a flow's packets = duration / gaps.
            # This is the same estimator CICFlowMeter's "Flow IAT Mean"
            # reports, in the same microsecond unit.
            gaps = (tot_pkts - 1.0).clip(lower=0.0)
            iat_mean = np.where(gaps > 0, dur * 1e6 / gaps.where(gaps > 0, 1.0), 0.0)

            out = pd.DataFrame({
                "timestamp": epoch.to_numpy(dtype="float64"),
                "src_ip": chunk["SrcAddr"].astype(str).to_numpy(),
                "dst_ip": chunk["DstAddr"].astype(str).to_numpy(),
                "src_port": _numeric_port(chunk["Sport"]).to_numpy(),
                "dst_port": _numeric_port(chunk["Dport"]).to_numpy(),
                "protocol": chunk["Proto"].astype(str).str.lower().to_numpy(),
                "flow_duration": (dur * 1e6).to_numpy(dtype="float64"),
                "total_fwd_packets": tot_pkts.to_numpy(dtype="float64"),
                "total_bwd_packets": np.zeros(len(chunk), dtype="float64"),
                "total_len_fwd": src_bytes.to_numpy(dtype="float64"),
                "total_len_bwd": reverse_bytes.to_numpy(dtype="float64"),
                "flow_iat_mean": np.asarray(iat_mean, dtype="float64"),
                "flow_iat_max": np.zeros(len(chunk), dtype="float64"),
                "label": chunk["Label"].astype(str).str.strip().to_numpy(),
            })
            flags = parse_argus_state_flags(chunk["State"], chunk["Proto"])
            for column in FLAG_COLUMNS:
                out[column] = flags[column].to_numpy(dtype="float64")

            report.accepted_rows += len(out)
            kept.append(out)
            if row_cap is not None and report.input_rows >= row_cap:
                logger.info("load_binetflow: row_cap=%d reached after %d input rows (%s)",
                            row_cap, report.input_rows, path)
                break
    finally:
        reader.close()

    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                 "flow_duration", "total_fwd_packets", "total_bwd_packets", "total_len_fwd",
                 "total_len_bwd", "flow_iat_mean", "flow_iat_max", "label"] + FLAG_COLUMNS
    )
    report.log()
    return df, report

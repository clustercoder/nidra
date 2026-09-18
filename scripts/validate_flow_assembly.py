"""Does flow assembly from packets produce the features the model was trained on?

The model's 15 flow features came from CICFlowMeter CSVs. Uploaded pcaps have
no CSV, so `nidra.data.flow_assemble` rebuilds the flows from packets. If that
reconstruction is systematically different, every flow feature shifts and the
model degrades with nothing raising an error.

CIC-IDS2017 is the one case where both sides exist for the same capture, so
this compares them:

  reference : CICFlowMeter CSV flows  -> build_state_rows -> 45-feature windows
  assembled : packets -> assemble_flows -> build_state_rows -> 45-feature windows

Per-flow equality is not the target and is not achievable from packets alone
(CICFlowMeter's splitting depends on internal activity timers). What has to
hold is that the per-(host, window) feature distributions match, because those
vectors are what the model actually consumes. The CSV's timestamps are local
and the packets' are UTC, so windows are not compared pairwise — the
comparison is between marginal distributions over all windows, which is
exactly where a systematic shift would show.

Run:  .venv/bin/python scripts/validate_flow_assembly.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nidra.data.flow_assemble import assemble_flows  # noqa: E402
from nidra.data.flow_load import load_cicflowmeter_csv  # noqa: E402
from nidra.data.join import prepare_flows_for_windowing, prepare_packets_for_windowing  # noqa: E402
from nidra.data.schema import WINDOW_SECONDS  # noqa: E402
from nidra.data.windowize import align_window, build_state_rows  # noqa: E402

RAW = Path("~/nidra/cicids2017").expanduser()
PACKETS = RAW / "pcap" / "parquet" / "Friday-WorkingHours_packets.parquet"
FLOW_CSV = RAW / "csv" / "extracted" / "TrafficLabelling " / "Friday-WorkingHours-Morning.pcap_ISCX.csv"

FLOW_FEATURES = [
    "syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "psh_ratio", "urg_ratio",
    "bytes_total", "bytes_up_down_ratio", "pkts_per_flow_mean",
    "flow_duration_mean", "flow_duration_var", "iat_mean", "iat_var", "iat_max",
    "active_flow_count",
]


def windows_from_assembled(packets: pd.DataFrame) -> pd.DataFrame:
    flows = assemble_flows(packets)
    flows = flows.assign(window_ts=align_window(flows["timestamp"], WINDOW_SECONDS))
    packets_w = prepare_packets_for_windowing(packets, WINDOW_SECONDS)
    return build_state_rows(flows, packets_w, window_seconds=WINDOW_SECONDS)


#: CIC-IDS2017's CICFlowMeter CSVs carry LOCAL timestamps (UTC-3) while the
#: pcaps carry true UTC epochs, and parse_cic_timestamp reads the CSV as UTC.
#: Aligning the two costs nothing here and is required for the comparison to
#: be between the same minutes of traffic. See CSV_UTC_OFFSET_H in
#: nidra/data/join.py for what this means for the committed training tables.
CSV_UTC_OFFSET_H = 3


def windows_from_csv(csv_path: Path, packets: pd.DataFrame) -> pd.DataFrame:
    flows_raw, _ = load_cicflowmeter_csv(csv_path)
    flows_w = prepare_flows_for_windowing(flows_raw, WINDOW_SECONDS)
    flows_w = flows_w.assign(window_ts=flows_w.window_ts + CSV_UTC_OFFSET_H * 3600)
    packets_w = prepare_packets_for_windowing(packets, WINDOW_SECONDS)
    keep = set(packets_w.window_ts.unique())
    flows_w = flows_w[flows_w.window_ts.isin(keep)]
    return build_state_rows(flows_w, packets_w, window_seconds=WINDOW_SECONDS)


def compare(reference: pd.DataFrame, assembled: pd.DataFrame) -> pd.DataFrame:
    """Marginal distribution per feature. `ratio` is assembled/reference on the
    median; 1.0 is agreement and an order of magnitude is a broken unit."""
    # Inactive windows are structurally zero on both sides and would drown
    # any real difference in a sea of agreeing zeros.
    reference = reference[reference.is_active == 1]
    assembled = assembled[assembled.is_active == 1]
    rows = []
    for feat in FLOW_FEATURES:
        r, a = reference[feat].to_numpy(float), assembled[feat].to_numpy(float)
        r_med, a_med = float(np.median(r)), float(np.median(a))
        denom = abs(r_med) if abs(r_med) > 1e-12 else np.nan
        rows.append({
            "feature": feat,
            "ref_median": r_med,
            "asm_median": a_med,
            "ratio": (a_med / denom) if denom == denom else np.nan,
            "ref_p90": float(np.percentile(r, 90)),
            "asm_p90": float(np.percentile(a, 90)),
            "ref_frac_nonzero": float((r != 0).mean()),
            "asm_frac_nonzero": float((a != 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=60,
                    help="minutes of capture to compare (the full day is ~10M packets)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not PACKETS.exists() or not FLOW_CSV.exists():
        raise SystemExit(f"needs the raw CIC-IDS2017 release:\n  {PACKETS}\n  {FLOW_CSV}")

    packets = pd.read_parquet(PACKETS)
    t0 = float(packets.frame_time_epoch.min())
    packets = packets[packets.frame_time_epoch < t0 + args.minutes * 60].copy()
    print(f"packets in slice: {len(packets):,} ({args.minutes} min from capture start)")

    assembled = windows_from_assembled(packets)
    reference = windows_from_csv(FLOW_CSV, packets)
    print(f"reference windows: {len(reference):,}   assembled windows: {len(assembled):,}")

    table = compare(reference, assembled)
    print(f"active windows compared — reference {int((reference.is_active == 1).sum()):,}, "
          f"assembled {int((assembled.is_active == 1).sum()):,}")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))

    bad = table[(table.ratio.notna()) & ((table.ratio > 3) | (table.ratio < 1 / 3))]
    print("\nfeatures off by more than 3x on the median:",
          ", ".join(bad.feature) if len(bad) else "none")


if __name__ == "__main__":
    main()

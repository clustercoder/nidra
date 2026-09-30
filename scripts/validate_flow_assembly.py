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
vectors are what the model actually consumes.

Both sides are now on the same clock (windowize.parse_cic_timestamp corrects
the CSV's local, 12-hour timestamps), so the comparison is PAIRED: the same
(host, window) computed two ways. That matters, because the two sides do not
see the same set of hosts — CICFlowMeter publishes flows this reconstruction
never sources to the same host — and comparing marginal distributions over
all windows then compares two different populations and reports the
difference between them as if it were reconstruction error. Unpaired
marginals are printed too, with the host populations alongside, so the gap
between the two readings is visible rather than a choice made silently.

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


def windows_from_csv(csv_path: Path, packets: pd.DataFrame) -> pd.DataFrame:
    # The CSV clock corrections (local time, 12-hour with no meridiem) are
    # applied by windowize.parse_cic_timestamp, so both sides of this
    # comparison are already on the pcaps' UTC timeline — nothing to shift
    # here. This harness used to add the 3 hours itself; doing that now would
    # put the reference 3 hours ahead of the packets it is compared against.
    flows_raw, _ = load_cicflowmeter_csv(csv_path)
    flows_w = prepare_flows_for_windowing(flows_raw, WINDOW_SECONDS)
    packets_w = prepare_packets_for_windowing(packets, WINDOW_SECONDS)
    keep = set(packets_w.window_ts.unique())
    flows_w = flows_w[flows_w.window_ts.isin(keep)]
    return build_state_rows(flows_w, packets_w, window_seconds=WINDOW_SECONDS)


ACTIVE_KEY = ["host_id", "window_ts"]


def _active(df: pd.DataFrame) -> pd.DataFrame:
    """Inactive windows are structurally zero on both sides and would drown
    any real difference in a sea of agreeing zeros."""
    return df[df.is_active == 1]


def pair(reference: pd.DataFrame, assembled: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two state tables on (host, window), so every row is one
    moment computed both ways."""
    r, a = _active(reference), _active(assembled)
    return r.merge(a, on=ACTIVE_KEY, how="inner", suffixes=("__ref", "__asm"))


def compare_paired(paired: pd.DataFrame) -> pd.DataFrame:
    """Per feature, over the (host, window) rows both sides produced.

    `median_ratio` is the median of the per-row ratio — not the ratio of the
    medians, which can look fine while every individual row is wrong.
    `agree_frac` is the share of rows within 2x, the loosest standard under
    which a feature is not misleading the model.
    """
    rows = []
    for feat in FLOW_FEATURES:
        r = paired[f"{feat}__ref"].to_numpy(float)
        a = paired[f"{feat}__asm"].to_numpy(float)
        both_zero = (r == 0) & (a == 0)
        nz = ~both_zero & (r != 0)
        ratio = np.divide(a[nz], r[nz]) if nz.any() else np.array([np.nan])
        within = np.abs(np.log2(np.where(ratio > 0, ratio, np.nan))) <= 1.0
        rows.append({
            "feature": feat,
            "n_compared": int(nz.sum()),
            "both_zero": float(both_zero.mean()),
            "median_ratio": float(np.nanmedian(ratio)),
            "p10_ratio": float(np.nanpercentile(ratio, 10)),
            "p90_ratio": float(np.nanpercentile(ratio, 90)),
            "agree_within_2x": float(np.nanmean(within)) if nz.any() else np.nan,
            "spearman": float(pd.Series(r).corr(pd.Series(a), method="spearman")),
        })
    return pd.DataFrame(rows)


def compare_marginal(reference: pd.DataFrame, assembled: pd.DataFrame) -> pd.DataFrame:
    """Unpaired marginals, for comparison with the paired table above. A large
    disagreement here that the paired table does not show is a difference in
    which hosts each side sees, not reconstruction error."""
    reference, assembled = _active(reference), _active(assembled)
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
    pd.set_option("display.width", 220, "display.max_columns", 20)

    ref_hosts = set(_active(reference).host_id)
    asm_hosts = set(_active(assembled).host_id)
    print(f"reference windows: {len(reference):,}   assembled windows: {len(assembled):,}")
    print(f"active windows — reference {int((reference.is_active == 1).sum()):,}, "
          f"assembled {int((assembled.is_active == 1).sum()):,}")
    print(f"hosts with an active window — reference {len(ref_hosts):,}, assembled "
          f"{len(asm_hosts):,}, shared {len(ref_hosts & asm_hosts):,}")

    paired = pair(reference, assembled)
    print(f"\nPAIRED — {len(paired):,} (host, window) moments computed both ways")
    ptable = compare_paired(paired)
    print(ptable.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))

    bad = ptable[ptable.agree_within_2x < 0.5]
    print("\nfeatures agreeing within 2x on fewer than half of paired moments:",
          ", ".join(bad.feature) if len(bad) else "none")

    print("\nUNPAIRED marginals over all active windows (different host "
          "populations — see the module docstring)")
    print(compare_marginal(reference, assembled).to_string(index=False, float_format=lambda v: f"{v:,.4g}"))


if __name__ == "__main__":
    main()

"""Day-2 sanity check (IMPLEMENTATION-ML.md §8): plot a known PortScan
host's windowed features over time to visually confirm the escalation is
present in the pipeline's OUTPUT, not just the raw label column. This is a
data-validation tool, not part of training or serving.

Usage:
    python -m nidra.scripts.portscan_sanity_plot --csv <path to Friday PortScan CSV> --out <png path>
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nidra.train.pipeline import load_and_label_day

logger = logging.getLogger(__name__)

SCAN_SIGNATURE_FEATURES = ["dst_port_entropy", "out_degree", "syn_ratio", "new_peer_count"]


def pick_busiest_portscan_host(state_rows) -> str:
    """Picks the host with the most non-benign (PortScan) windows — the
    clearest case to plot, not necessarily representative of every host."""
    scan_rows = state_rows[state_rows["stage_label"] != "benign"]
    if scan_rows.empty:
        raise ValueError("no non-benign windows found in this slice — cannot pick a PortScan host to plot")
    counts = scan_rows.groupby("host_id").size().sort_values(ascending=False)
    return counts.index[0]


def plot_host_escalation(state_rows, host_id: str, out_path: str) -> None:
    host_rows = state_rows[state_rows["host_id"] == host_id].sort_values("window_ts")
    t = (host_rows["window_ts"] - host_rows["window_ts"].min()).to_numpy()
    is_attack = (host_rows["stage_label"] != "benign").to_numpy()

    fig, axes = plt.subplots(len(SCAN_SIGNATURE_FEATURES), 1, figsize=(10, 8), sharex=True)
    for ax, feat in zip(axes, SCAN_SIGNATURE_FEATURES):
        ax.plot(t, host_rows[feat].to_numpy(), color="tab:blue", linewidth=1.2)
        onset_idx = is_attack.argmax() if is_attack.any() else None
        if onset_idx is not None and is_attack.any():
            ax.axvspan(t[onset_idx], t[-1], color="tab:red", alpha=0.12, label="labelled PortScan window")
        ax.set_ylabel(feat, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(loc="upper left", fontsize=7)
    axes[-1].set_xlabel("seconds since first window")
    fig.suptitle(f"PortScan sanity check — host {host_id}\n(pipeline output, not raw labels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("wrote sanity plot for host=%s to %s", host_id, out_path)


def main():
    parser = argparse.ArgumentParser(description="Sanity-plot a known PortScan host's feature escalation.")
    parser.add_argument("--csv", type=str, required=True, help="Friday PortScan CICFlowMeter CSV")
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--min-windows-per-host", type=int, default=5,
                         help="lower than the production threshold (36) so a single-host sanity check "
                              "isn't starved by a small bounded slice")
    parser.add_argument("--out", type=str, default="portscan_sanity.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    labelled = load_and_label_day(args.csv, window_seconds=args.window_seconds, min_windows_per_host=args.min_windows_per_host)

    host_id = pick_busiest_portscan_host(labelled)
    logger.info("selected host_id=%s for sanity plot", host_id)
    plot_host_escalation(labelled, host_id, args.out)


if __name__ == "__main__":
    main()

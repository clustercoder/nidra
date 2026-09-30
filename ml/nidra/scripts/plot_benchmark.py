"""Figures from benchmark.json records (matplotlib, no seaborn):

    python -m nidra.scripts.plot_benchmark --run experiments/runs/production --splits test,holdout --out reports/run8

Writes: systems_ap_<split>.png (AP with episode-bootstrap CI), horizon_<split>.png
(per-horizon AP: NIDRA / deterministic / oracle, plus stage top-1), state_skill_<split>.png,
reliability_<split>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nidra.scripts.report_tables import SYSTEM_LABEL, SYSTEM_ORDER, _load  # noqa: E402


def systems_ap(m: dict, split: str, out: Path) -> None:
    pub = m["task_published_label"]["systems"]
    names = [n for n in SYSTEM_ORDER if n in pub]
    vals = [pub[n]["auc_pr"] for n in names]
    lo = [pub[n]["auc_pr"] - pub[n]["auc_pr_bootstrap"]["ci_low"] if pub[n].get("auc_pr_bootstrap") else 0 for n in names]
    hi = [pub[n]["auc_pr_bootstrap"]["ci_high"] - pub[n]["auc_pr"] if pub[n].get("auc_pr_bootstrap") else 0 for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.2))
    y = range(len(names))
    ax.barh(list(y), vals, xerr=[[max(v, 0) for v in lo], [max(v, 0) for v in hi]], color="#5b8def", ecolor="#333", capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([SYSTEM_LABEL.get(n, n) for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP at natural prevalence (published label); bars = episode-bootstrap 95% CI")
    ax.set_title(f"{split}: {m['eval_set']['n_rows']} rows, prevalence {m['eval_set']['natural_prevalence_published']:.4f}")
    fig.tight_layout()
    fig.savefig(out / f"systems_ap_{split}.png", dpi=150)
    plt.close(fig)


def horizon(m: dict, split: str, out: Path) -> None:
    per = m["task_C_progression"]["per_horizon"]
    k = [e["minutes_ahead"] for e in per]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(k, [e["auc_pr_natural"] for e in per], "o-", label="NIDRA (calibrated)")
    if "auc_pr_natural_deterministic" in per[0]:
        ax.plot(k, [e["auc_pr_natural_deterministic"] for e in per], "s--", label="deterministic rollout")
    if "auc_pr_natural_oracle" in per[0]:
        ax.plot(k, [e["auc_pr_natural_oracle"] for e in per], "^:", label="oracle (true future)")
    if "stage_top1_accuracy_on_attack_futures" in per[0]:
        ax.plot(k, [e.get("stage_top1_accuracy_on_attack_futures", float("nan")) for e in per], "d-.", label="stage top-1 on attack futures")
    ax.set_ylim(0, 1)
    ax.set_xlabel("minutes ahead")
    ax.set_ylabel("AP (natural prevalence) / accuracy")
    ax.set_title(f"{split}: is t+k an attack window?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"horizon_{split}.png", dpi=150)
    plt.close(fig)


def state_skill(m: dict, split: str, out: Path) -> None:
    sf = m["state_forecast"]["skill_vs_persistence_by_k"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, vals in sf.items():
        ax.plot(range(1, len(vals) + 1), vals, "o-", label=name)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("horizon k (windows)")
    ax.set_ylabel("1 - MSE / MSE_persistence")
    ax.set_title(f"{split}: state-forecast skill vs persistence (kept features)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"state_skill_{split}.png", dpi=150)
    plt.close(fig)


def reliability(m: dict, split: str, out: Path) -> None:
    cal = m["calibration_published_label"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "--", color="#999")
    for key, style in (("raw", "s:"), ("calibrated", "o-")):
        if key not in cal:
            continue
        rel = cal[key]["reliability"]
        xs = [r["bin_center"] for r in rel if r["observed_frequency_natural"] is not None]
        ys = [r["observed_frequency_natural"] for r in rel if r["observed_frequency_natural"] is not None]
        ax.plot(xs, ys, style, label=f"{key} (Brier {cal[key]['brier_natural']:.5f})")
    ax.set_xlabel("forecast score")
    ax.set_ylabel("observed positive rate (natural weights)")
    ax.set_title(f"{split}: reliability", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"reliability_{split}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--splits", default="test,holdout")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    metrics_dir = root / "artifacts" / "metrics" if (root / "artifacts" / "metrics").exists() else root
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        path = metrics_dir / split / "benchmark.json"
        if not path.exists():
            print(f"skip {split}: {path} missing")
            continue
        m = _load(path)
        systems_ap(m, split, out)
        horizon(m, split, out)
        state_skill(m, split, out)
        reliability(m, split, out)
        print(f"wrote figures for {split} to {out}")


if __name__ == "__main__":
    main()

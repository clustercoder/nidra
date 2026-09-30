"""PS 26153's benchmark: the world model against logistic regression on F1,
precision, recall and false-positive rate — each system at its OWN threshold.

The problem statement asks for "benchmark results comparing model performance
(F1 score, precision, recall, false positive rate) against a logistic
regression baseline trained on the same features". `benchmark.py` records all
of those, but `_task_table` reads every system at one threshold: the world
model's, selected on validation for the world model's calibrated score. A
logistic regression trained with class weighting puts most of its scores near
1.0, so at 0.718 it alerts on a great deal that its own validation-chosen
threshold would not — and the recorded comparison reports that mismatch as a
false-alarm rate.

This gives every system the rule the world model got: the weighted-F1-optimal
threshold on validation, frozen, then applied unchanged to test and holdout.
It reads the per-row score dumps the benchmark already writes, so no model is
re-run and no Run 8 artifact is modified. AP is reported alongside because it
needs no threshold at all, with a paired episode-cluster interval on the
world model's AP margin over each baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nidra.eval.metrics_natural import (
    best_f1_threshold,
    bootstrap_ap_difference,
    confusion_at,
    weighted_ap,
)

#: The published arm first; the two logistic regressions are the PS's named
#: baseline (same features) and its fairer temporal variant (30 min of history).
SYSTEMS: tuple[str, ...] = (
    "world_model_calibrated",
    "lr_current_state",
    "lr_flattened_history",
    "gru_classifier",
    "gbdt_current_state",
)
LABEL = "y_published"
REPORT_SPLITS: tuple[str, ...] = ("test", "holdout")


def select_thresholds(y: np.ndarray, scores: dict[str, np.ndarray], w: np.ndarray,
                      systems: Sequence[str]) -> dict[str, tuple[float, float]]:
    """{system: (threshold, validation F1)}. Takes validation arrays and
    nothing else, so no other split can reach the choice."""
    return {s: best_f1_threshold(y, scores[s], w) for s in systems if s in scores}


def score_at_threshold(y: np.ndarray, s: np.ndarray, w: np.ndarray, threshold: float,
                       span_hours: float) -> dict[str, float]:
    if not span_hours > 0:
        raise ValueError(f"span_hours must be positive, got {span_hours}")
    c = confusion_at(y, s, threshold, w)
    return {
        "threshold": float(threshold),
        "ap": weighted_ap(y, s, w),
        "precision": float(c["precision"]),
        "recall": float(c["recall"]),
        "f1": float(c["f1"]),
        "fpr": float(c["fpr"]),
        "false_alarms_per_hour": float(c["fp"]) / span_hours,
    }


def paired_ap_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray,
                         clusters: np.ndarray, n_resamples: int = 300, seed: int = 0) -> dict[str, float]:
    d = bootstrap_ap_difference(y, a, b, w, clusters, n_resamples=n_resamples, seed=seed)
    return {"point": float(d["point"]), "ci_low": float(d["ci_low"]), "ci_high": float(d["ci_high"])}


def _row(split: str, system: str, r: dict[str, Any], d: dict[str, float] | None) -> str:
    margin = "—" if d is None else f"{d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
    return (f"| {split} | {system} | {r['threshold']:.4f} | {r['ap']:.3f} | {margin} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['fpr']:.6f} | "
            f"{r['false_alarms_per_hour']:.2f} |")


def markdown_report(rows: dict[str, dict[str, dict]], paired: dict[str, dict[str, dict]],
                    spans: dict[str, float]) -> str:
    out = [
        "Each system is read at **its own validation-selected threshold** — the "
        "weighted-F1-optimal point on the validation split, frozen before test or "
        "holdout was scored. That is the rule the world model's operating point "
        "already used; `benchmark.json` instead reads every baseline at the world "
        "model's threshold, which was chosen for nothing but the world model.",
        "",
        "Natural prevalence throughout. *AP margin* is the world model's AP minus "
        "the baseline's, paired episode-cluster bootstrap, 95% interval.",
        "",
        "| split | system | threshold | AP | AP margin of the world model | precision | recall | F1 | FPR | false alarms/h |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for split, per in rows.items():
        for system in SYSTEMS:
            if system in per:
                out.append(_row(split, system, per[system], paired.get(split, {}).get(system)))
    out.append("")
    out.append("Span of each split in hours (the denominator of false alarms/h): " +
               ", ".join(f"{k} {v:.2f}" for k, v in spans.items()) + ".")
    return "\n".join(out)


def _load(metrics_dir: Path, split: str) -> tuple[Any, float]:
    z = np.load(metrics_dir / split / "benchmark_scores.npz", allow_pickle=True)
    span = json.loads((metrics_dir / split / "benchmark.json").read_text())["metrics"]["eval_set"]["span_hours"]
    return z, float(span)


def run(metrics_dir: Path, n_resamples: int, seed: int) -> str:
    val, _ = _load(metrics_dir, "val")
    thresholds = select_thresholds(val[LABEL], {s: val[s] for s in SYSTEMS if s in val.files},
                                   val["weight"], SYSTEMS)
    rows: dict[str, dict[str, dict]] = {}
    paired: dict[str, dict[str, dict]] = {}
    spans: dict[str, float] = {}
    for split in REPORT_SPLITS:
        z, span = _load(metrics_dir, split)
        spans[split] = span
        y, w, clusters = z[LABEL], z["weight"], z["cluster"]
        rows[split] = {s: {**score_at_threshold(y, z[s], w, thr, span), "val_f1": f1}
                       for s, (thr, f1) in thresholds.items()}
        paired[split] = {s: paired_ap_difference(y, z[SYSTEMS[0]], z[s], w, clusters, n_resamples, seed)
                         for s in thresholds if s != SYSTEMS[0]}
    return markdown_report(rows, paired, spans)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", default="artifacts/metrics",
                    help="holds val/, test/, holdout/ with benchmark.json + benchmark_scores.npz")
    ap.add_argument("--n-resamples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = run(Path(args.metrics_dir), args.n_resamples, args.seed)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()

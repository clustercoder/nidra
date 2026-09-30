"""Paired confidence intervals on the PUBLISHED arm's margin over its best baseline.

`benchmark.py` writes an `attribution_detection` block, but every pair in it
starts from `world_model` — the uncalibrated, single-readout arm. The number
this project publishes is `world_model_calibrated`: ~200 pooled trajectories
through a per-horizon Platt layer at a validation-frozen threshold. That arm has
a marginal cluster-bootstrap interval and no paired one, so §36 item 21's
question — which improvements are statistically supported rather than noise? —
could not be answered about the system actually reported.

This recomputes the paired episode-cluster bootstrap offline, from the per-row
score dumps the benchmark already writes next to each `benchmark.json`. It
re-runs no model and re-scores no split; the same rows, weights and clusters the
benchmark used are read back and paired.

The baseline is chosen **per cell, as the strongest one in that cell**, which is
the only comparison that can support "the model beats its baselines". Choosing
one fixed baseline everywhere would let a cell win by beating whichever rival
happened to be weak there.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from nidra.eval.metrics_natural import bootstrap_ap_difference, weighted_ap

#: The arm the project reports. Not `world_model`, which is the uncalibrated
#: readout the existing attribution pairs use.
PUBLISHED_SYSTEM = "world_model_calibrated"

#: Everything the published arm has to beat. Deliberately excludes the oracle
#: (it reads the true future) and both other arms of the model itself — a
#: margin over `world_model_deterministic` is an ablation, not a baseline.
BASELINE_SYSTEMS: tuple[str, ...] = (
    "persistence",
    "persistence_rollout",
    "noised_persistence",
    "isotropic_noise_persistence",
    "ridge_two_lag",
    "lr_current_state",
    "gbdt_current_state",
    "lr_flattened_history",
    "gru_classifier",
)


#: An interval endpoint this close to zero has not cleared it in any useful
#: sense: at 300 resamples the percentile endpoint is an order statistic of the
#: resampled differences, and a gap of 1e-05 is below the resolution that
#: produced it. Cells like this are counted by the rule fixed before the data
#: was seen, and flagged so the count is not read as stronger than it is.
KNIFE_EDGE = 1e-3


@dataclass(frozen=True)
class CellMargin:
    """One (cell, split): the published arm against the strongest baseline there."""

    cell: str
    split: str
    published_ap: float
    baseline: str
    baseline_ap: float
    point: float
    ci_low: float
    ci_high: float
    n_clusters: int
    verdict: str


def verdict(ci_low: float, ci_high: float) -> str:
    """`above` / `below` only when the interval excludes zero outright.

    An endpoint sitting exactly on zero is not support. At 300 resamples the
    percentile endpoints are order statistics of the resampled differences, so
    an exact 0.0 is a boundary the data touched, not one it cleared.
    """
    if ci_low > 0:
        return "above"
    if ci_high < 0:
        return "below"
    return "spans"


def clearance(m: "CellMargin") -> float:
    """How far the interval's near endpoint sits from zero. 0.0 when it spans."""
    if m.verdict == "above":
        return m.ci_low
    if m.verdict == "below":
        return -m.ci_high
    return 0.0


def is_knife_edge(m: "CellMargin") -> bool:
    return m.verdict != "spans" and clearance(m) < KNIFE_EDGE


def fmt_endpoint(x: float) -> str:
    """Four decimals, except for a nonzero value that would round to zero —
    printing 1.4e-05 as +0.0000 next to a cell marked significant reads as a
    contradiction, and the reader needs to see which it is."""
    if x != 0.0 and abs(x) < 5e-5:
        return f"{x:+.1e}"
    return f"{x:+.4f}"


def best_baseline(y: np.ndarray, scores: dict[str, np.ndarray], w: np.ndarray) -> str | None:
    """The baseline with the highest weighted AP in this cell, or None."""
    present = {k: weighted_ap(y, scores[k], w) for k in BASELINE_SYSTEMS if k in scores}
    present = {k: v for k, v in present.items() if not np.isnan(v)}
    return max(present, key=present.__getitem__) if present else None


def margin_for_cell(cell: str, split: str, y: np.ndarray, scores: dict[str, np.ndarray],
                    w: np.ndarray, clusters: np.ndarray, n_resamples: int = 300,
                    seed: int = 0) -> CellMargin | None:
    """None — never a nan row — when the cell cannot support the comparison."""
    if PUBLISHED_SYSTEM not in scores or int(np.asarray(y).sum()) == 0:
        return None
    base = best_baseline(y, scores, w)
    if base is None:
        return None
    boot = bootstrap_ap_difference(y, scores[PUBLISHED_SYSTEM], scores[base], w, clusters,
                                   n_resamples=n_resamples, seed=seed)
    lo, hi = float(boot["ci_low"]), float(boot["ci_high"])
    return CellMargin(
        cell=cell, split=split,
        published_ap=weighted_ap(y, scores[PUBLISHED_SYSTEM], w),
        baseline=base, baseline_ap=weighted_ap(y, scores[base], w),
        point=float(boot["point"]), ci_low=lo, ci_high=hi,
        n_clusters=int(boot.get("n_clusters", 0)), verdict=verdict(lo, hi),
    )


def summarise(margins: Sequence[CellMargin]) -> dict[str, int]:
    """Validation is counted separately: the operating point is chosen there,
    so a val cell cannot support a claim about the system's performance."""
    held_out = [m for m in margins if m.split != "val"]
    return {
        "n_cells": len(margins),
        "n_cells_excluding_val": len(held_out),
        "above": sum(1 for m in margins if m.verdict == "above"),
        "below": sum(1 for m in margins if m.verdict == "below"),
        "spans": sum(1 for m in margins if m.verdict == "spans"),
        "above_excluding_val": sum(1 for m in held_out if m.verdict == "above"),
        "below_excluding_val": sum(1 for m in held_out if m.verdict == "below"),
    }


def markdown_table(margins: Sequence[CellMargin]) -> str:
    s = summarise(margins)
    knives = [m for m in margins if is_knife_edge(m)]
    caveat = (
        f" **{len(knives)} of the {s['above'] + s['below']} counted cells clear zero by "
        f"less than {KNIFE_EDGE:g}** and are flagged *knife edge* below: the rule was "
        f"fixed before the data was seen and is applied as written, but an endpoint at "
        f"1e-05 is not a margin.\n" if knives else "\n"
    )
    head = (
        f"**`{PUBLISHED_SYSTEM}` against the strongest baseline in each cell, paired "
        f"episode-cluster bootstrap: the interval excludes zero in {s['above']} of "
        f"{s['n_cells']} cells** ({s['above_excluding_val']} of "
        f"{s['n_cells_excluding_val']} once validation is set aside), and is entirely "
        f"below zero in {s['below']}. The baseline is picked per cell as the highest-AP "
        f"one there, so a cell cannot win by beating a rival that happened to be weak "
        f"in it.\n" + caveat + "\n"
        "| cell | split | published AP | strongest baseline | its AP | margin | 95% CI | clusters | |\n"
        "|---|---|---:|---|---:|---:|---|---:|---|"
    )
    lines = [head]
    for m in margins:
        mark = "**" if m.verdict in ("above", "below") else ""
        note = "knife edge" if is_knife_edge(m) else ""
        lines.append(
            f"| {m.cell} | {m.split} | {m.published_ap:.4f} | {m.baseline} | "
            f"{m.baseline_ap:.4f} | {mark}{m.point:+.4f}{mark} | "
            f"[{fmt_endpoint(m.ci_low)}, {fmt_endpoint(m.ci_high)}] | {m.n_clusters} | {note} |"
        )
    return "\n".join(lines)


def iter_cells(runs: Path, prefix: str, splits: Sequence[str]) -> Iterator[tuple[str, str, Path]]:
    for run in sorted(runs.glob(f"{prefix}*")):
        for split in splits:
            dump = run / "artifacts" / "metrics" / split / "benchmark_scores.npz"
            if dump.exists():
                yield run.name.removeprefix(prefix), split, dump


def load_cell(dump: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    z = np.load(dump, allow_pickle=True)
    scores = {k: z[k] for k in z.files if k in (PUBLISHED_SYSTEM, *BASELINE_SYSTEMS)}
    return z["y_detect"], scores, z["weight"], z["cluster"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="experiments/runs")
    ap.add_argument("--prefix", default="mask_", help="run-label prefix; mask_ is the D145-corrected set")
    ap.add_argument("--splits", default="val,test,holdout")
    ap.add_argument("--n-resamples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    margins: list[CellMargin] = []
    for cell, split, dump in iter_cells(Path(args.runs), args.prefix, args.splits.split(",")):
        y, scores, w, clusters = load_cell(dump)
        m = margin_for_cell(cell, split, y, scores, w, clusters,
                            n_resamples=args.n_resamples, seed=args.seed)
        if m is not None:
            margins.append(m)

    text = markdown_table(margins)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()

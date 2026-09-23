"""§3.41: is the rollout a projection onto the model's own training manifold?

In both cells that beat their oracle, the frozen head ranks at or below chance
on the TRUE future states and far above chance on states the transition model
produced. The hypothesis is that a rolled-out state lies nearer the head's
training distribution than the truth of a corpus the head was not trained on
alone does — so the rollout acts as a projection, and the head is better off on
the projection than on the truth.

The criterion was fixed in §3.41 BEFORE this ran, including the statistic, the
horizon, the standardisation and the threshold. §3.27 is the standing reminder
of what happens on this question otherwise.

Runs on the `mask_*` artifacts, so the measurement is made on the corrected
model rather than the one with phantom features — the first attempt at this
measurement is what found D145.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Below this many cells on either side the comparison is underpowered and says
#: so, rather than reading a pattern off two cells.
MIN_CELLS_PER_GROUP = 3

#: A training feature whose spread is below this has no scale, so a deviation in
#: it has no meaning in sd units. Excluded rather than floored: flooring is what
#: produced distances of 2e6 on the first attempt and led to D145.
DEGENERATE_SD = 1e-6


def non_degenerate(reference: np.ndarray) -> np.ndarray:
    """[F] bool — which features of the reference cloud carry a usable scale."""
    return reference.std(axis=0) > DEGENERATE_SD


def cloud_distance(reference: np.ndarray, probe: np.ndarray) -> float:
    """Mean standardised L2 distance from `reference`'s centre to `probe`'s rows.

    Standardised by the reference cloud's own per-feature spread, over
    non-degenerate features only, so a wide feature does not dominate a narrow
    one and a constant one contributes nothing at all.
    """
    keep = non_degenerate(reference)
    if not keep.any():
        return float("nan")
    mu = reference[:, keep].mean(axis=0)
    sd = reference[:, keep].std(axis=0)
    z = (probe[:, keep] - mu) / sd
    return float(np.sqrt((z ** 2).sum(axis=-1)).mean())


def verdict(cells: list[dict[str, Any]]) -> dict[str, str]:
    """§3.41's criterion applied to the measured cells, and nothing else.

    Supported only when every oracle-beating cell has ratio < 1 AND every
    oracle-winning cell has ratio >= 1. One exception either way refutes it.
    """
    beating = [c for c in cells if c["beats_oracle"]]
    losing = [c for c in cells if not c["beats_oracle"]]
    if len(beating) < MIN_CELLS_PER_GROUP or len(losing) < MIN_CELLS_PER_GROUP:
        return {"verdict": "underpowered",
                "reason": (f"{len(beating)} oracle-beating and {len(losing)} oracle-winning cells; "
                           f"the criterion needs at least {MIN_CELLS_PER_GROUP} of each and does not "
                           "read a pattern off fewer")}

    def named(group: list[dict[str, Any]]) -> str:
        return ", ".join(f"{c.get('cell', '?')} ({c['ratio']:.3f})" for c in group)

    bad_beating = [c for c in beating if c["ratio"] >= 1.0]
    bad_losing = [c for c in losing if c["ratio"] < 1.0]
    if bad_beating:
        return {"verdict": "refuted",
                "reason": ("a cell that beats its oracle has its rollout no closer to the training "
                           f"cloud than the truth: {named(bad_beating)}")}
    if bad_losing:
        return {"verdict": "refuted",
                "reason": ("a cell whose oracle wins has its rollout closer to the training cloud "
                           f"anyway: {named(bad_losing)}")}
    return {"verdict": "supported",
            "reason": (f"all {len(beating)} oracle-beating cells have ratio < 1 and all {len(losing)} "
                       "oracle-winning cells have ratio >= 1")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="experiments/runs")
    ap.add_argument("--prefix", default="mask_")
    ap.add_argument("--splits", default="test,holdout")
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from nidra.data.normalize import FeatureScaler
    from nidra.eval.benchmark import load_models
    from nidra.eval.eval_set import build_eval_set
    from nidra.scripts.run_experiment import apply_overrides
    from nidra.train.pipeline import (build_all_splits, build_windowed_arrays, geometry_from_config,
                                      scale_arrays)
    from nidra.utils.config import load_config, resolve_path

    runs = Path(args.runs)
    cells: list[dict[str, Any]] = []
    for d in sorted(runs.glob(f"{args.prefix}*")):
        for split in args.splits.split(","):
            bp = d / "artifacts" / "metrics" / split / "benchmark.json"
            if not bp.exists():
                continue
            bench = json.loads(bp.read_text())
            systems = bench["metrics"]["task_published_label"]["systems"]
            if "oracle_true_future" not in systems or "world_model" not in systems:
                continue
            beats = systems["world_model"]["auc_pr"] > systems["oracle_true_future"]["auc_pr"]

            cfg = apply_overrides(load_config(bench["config_path"]), list(bench["config_overrides"]))
            _, L, K = geometry_from_config(cfg)
            window_seconds = geometry_from_config(cfg)[0]
            scaler = FeatureScaler.load(*FeatureScaler.default_paths(
                resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
            model = load_models(cfg, [0])[0]
            model.eval()
            splits = build_all_splits(cfg)

            tr = build_windowed_arrays(splits.train, L=L, K=K, max_samples=args.cap, seed=0)
            Xtr, _ = scale_arrays(tr, scaler)
            reference = Xtr.reshape(-1, Xtr.shape[-1])

            ev = build_eval_set(getattr(splits, split), L, K, window_seconds, split, seed=0)
            X, Y = scale_arrays(ev.arrays, scaler)
            n = min(len(X), args.cap)
            X, Y = X[:n], Y[:n]
            with torch.no_grad():
                pred = model.rollout(torch.from_numpy(X).float(), K=K,
                                     n_samples=8, stochastic=True).states.mean(1).numpy()

            d_truth = cloud_distance(reference, Y[:, K - 1])
            d_roll = cloud_distance(reference, pred[:, K - 1])
            cells.append({"cell": f"{d.name.removeprefix(args.prefix)}/{split}",
                          "beats_oracle": bool(beats), "d_truth": d_truth, "d_rollout": d_roll,
                          "ratio": d_roll / d_truth if d_truth else float("nan")})
            print(f"{cells[-1]['cell']:38s} beats_oracle={beats!s:5s} "
                  f"d_truth={d_truth:7.3f} d_rollout={d_roll:7.3f} ratio={cells[-1]['ratio']:.3f}", flush=True)

    v = verdict(cells)
    lines = [f"**§3.41 verdict: {v['verdict'].upper()}** — {v['reason']}", "",
             "| cell | beats its oracle | d(truth) | d(rollout) | ratio |", "|---|:--:|---:|---:|---:|"]
    for c in sorted(cells, key=lambda c: c["ratio"]):
        lines.append(f"| {c['cell']} | {'yes' if c['beats_oracle'] else 'no'} | "
                     f"{c['d_truth']:.3f} | {c['d_rollout']:.3f} | {c['ratio']:.3f} |")
    text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    print("\n" + text)


if __name__ == "__main__":
    main()

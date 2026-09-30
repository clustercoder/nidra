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

#: A training feature whose spread is below this has no usable scale, so a
#: deviation in it has no meaning in sd units.
#:
#: §3.41 pre-registered "features with non-degenerate training variance only".
#: The first implementation read that as sd > 1e-6 and did NOT implement it: a
#: feature with training sd 5.3e-5 passed, contributed a mean z^2 of 1,009,400
#: and produced a distance of 806 against a truth distance of 4.9 — the same
#: failure the pre-registration named. The kept-feature set also moved with the
#: training sample size (31 features at 1,500 samples, 32 at 2,500), so the
#: statistic was unstable in a parameter that should not matter.
#:
#: 1e-2 is set against the scaling convention rather than against this data: the
#: scaler maps features to roughly unit spread, so a training sd two orders
#: below that is a constant. Because any such threshold is a judgement, the
#: verdict is reported across THRESHOLD_SWEEP and is only a verdict if it
#: survives all of them.
DEGENERATE_SD = 1e-2

#: The verdict is computed at each of these and reported at all of them. A
#: conclusion that holds at one threshold and not another is not a conclusion.
THRESHOLD_SWEEP = (3e-3, 1e-2, 3e-2, 1e-1)


def non_degenerate(reference: np.ndarray, threshold: float = DEGENERATE_SD) -> np.ndarray:
    """[F] bool — which features of the reference cloud carry a usable scale."""
    return reference.std(axis=0) > threshold


def cloud_distance(reference: np.ndarray, probe: np.ndarray,
                   threshold: float = DEGENERATE_SD) -> float:
    """Mean standardised L2 distance from `reference`'s centre to `probe`'s rows.

    Standardised by the reference cloud's own per-feature spread, over
    non-degenerate features only, so a wide feature does not dominate a narrow
    one and a constant one contributes nothing at all.
    """
    keep = non_degenerate(reference, threshold)
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


def sweep_verdicts(cells: list[dict[str, Any]]) -> dict[float, dict[str, str]]:
    """§3.41's criterion applied at every threshold in the sweep."""
    out = {}
    for t in THRESHOLD_SWEEP:
        at_t = [{"cell": c.get("cell"), "beats_oracle": c["beats_oracle"], "ratio": c["ratios"][t]}
                for c in cells]
        out[t] = verdict(at_t)
    return out


def sweep_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """One verdict only if every threshold agrees; otherwise, plainly, none.

    The threshold is a judgement call made after the first implementation of it
    failed, so a conclusion that depends on which value was chosen is a
    conclusion about the choice.
    """
    per = sweep_verdicts(cells)
    verdicts = {v["verdict"] for v in per.values()}
    if len(verdicts) == 1:
        only = verdicts.pop()
        return {"stable": True, "verdict": only, "per_threshold": per,
                "summary": f"**{only.upper()}**, and the same at every threshold in "
                           f"{', '.join(str(t) for t in THRESHOLD_SWEEP)}."}
    return {"stable": False, "verdict": "inconclusive", "per_threshold": per,
            "summary": ("**INCONCLUSIVE — the verdict does not survive the threshold sweep.** "
                        + "; ".join(f"{t}: {per[t]['verdict']}" for t in THRESHOLD_SWEEP)
                        + ". The threshold is a judgement call, so a conclusion that depends on it "
                          "is a conclusion about the choice and not about the model.")}


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

            ratios, dt, dr = {}, {}, {}
            for t in THRESHOLD_SWEEP:
                dt[t] = cloud_distance(reference, Y[:, K - 1], t)
                dr[t] = cloud_distance(reference, pred[:, K - 1], t)
                ratios[t] = dr[t] / dt[t] if dt[t] else float("nan")
            cells.append({"cell": f"{d.name.removeprefix(args.prefix)}/{split}",
                          "beats_oracle": bool(beats), "ratios": ratios,
                          "d_truth": dt[DEGENERATE_SD], "d_rollout": dr[DEGENERATE_SD],
                          "ratio": ratios[DEGENERATE_SD]})
            print(f"{cells[-1]['cell']:38s} beats_oracle={beats!s:5s} "
                  f"d_truth={dt[DEGENERATE_SD]:7.3f} d_rollout={dr[DEGENERATE_SD]:7.3f} "
                  f"ratio={ratios[DEGENERATE_SD]:.3f}  "
                  f"(sweep {' '.join(f'{ratios[t]:.2f}' for t in THRESHOLD_SWEEP)})", flush=True)

    summary = sweep_summary(cells)
    lines = [f"## §3.41 verdict: {summary['summary']}", ""]
    for t in THRESHOLD_SWEEP:
        v = summary["per_threshold"][t]
        lines.append(f"- **sd > {t}** — {v['verdict']}: {v['reason']}")
    lines += ["", f"Distances below are at the default threshold {DEGENERATE_SD}.", "",
              "| cell | beats its oracle | d(truth) | d(rollout) | ratio | ratio across the sweep |",
              "|---|:--:|---:|---:|---:|---|"]
    for c in sorted(cells, key=lambda c: c["ratio"]):
        sweep = " / ".join(f"{c['ratios'][t]:.2f}" for t in THRESHOLD_SWEEP)
        lines.append(f"| {c['cell']} | {'yes' if c['beats_oracle'] else 'no'} | "
                     f"{c['d_truth']:.3f} | {c['d_rollout']:.3f} | {c['ratio']:.3f} | {sweep} |")
    text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    print("\n" + text)


if __name__ == "__main__":
    main()

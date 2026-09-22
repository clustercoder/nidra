"""Does a history-aware head actually attribute to history?

    python -m nidra.scripts.explanation_comparison \\
        --runs ctu_heads__state ctu_heads__hidden ctu_heads__state+hidden \\
        --split val --n-origins 40 --out reports/run9/explanations.md

The head ablation says the encoder's hidden state is worth +40% AP. That is a
claim about a number, not about a mechanism. Integrated gradients through the
rollout back to the input history answers the mechanism question directly: if
a `hidden` head's attributions concentrate on the last window exactly as the
per-state head's do, then whatever it gained, it did not gain by using the
host's history, and the story told about it would be wrong.

Two things are reported per run, on the SAME origins:

    temporal profile   share of |attribution| falling on each history window,
                       summarised as the share in the last 5 windows and the
                       attribution-weighted mean age in minutes
    faithfulness       the existing deletion test — does zeroing the top-m
                       attributed cells move the forecast more than zeroing
                       m random ones

A per-state head is included as the control. Nothing here is a selection
signal; it is a description of the model that was selected.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def temporal_profile(attr: np.ndarray, window_seconds: int = 60, recent_windows: int = 5) -> dict:
    """Where in the history the attribution mass sits. attr: [L, F].

    `share_recent` is the fraction of total |attribution| in the last
    `recent_windows` windows; `mean_age_min` is the attribution-weighted mean
    age of a cell, in minutes before the origin. A head that only looks at
    the present has share_recent near 1 and mean_age near 0."""
    a = np.abs(np.asarray(attr, dtype="float64"))
    per_window = a.sum(axis=1)
    total = per_window.sum()
    L = len(per_window)
    if total <= 0:
        return {"share_recent": float("nan"), "mean_age_min": float("nan"),
                "per_window_share": [0.0] * L, "n_windows": L}
    share = per_window / total
    age_min = (np.arange(L)[::-1]) * window_seconds / 60.0    # 0 = the origin window
    return {
        "share_recent": float(share[-recent_windows:].sum()),
        "mean_age_min": float((share * age_min).sum()),
        "per_window_share": [float(s) for s in share],
        "n_windows": int(L),
    }


def summarise(rows: list[dict]) -> dict:
    """Mean over origins of each per-origin statistic, with its spread."""
    def stat(key):
        v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))], dtype="float64")
        return {"mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v))} if len(v) else None
    return {k: stat(k) for k in ("share_recent", "mean_age_min", "drop_top_m", "drop_random_m_mean",
                                 "beats_random_fraction", "forecast_score")}


def comparison_markdown(per_run: dict[str, dict], n_origins: int, split: str) -> str:
    out = [f"### Explanation comparison — {n_origins} highest-risk **{split}** origins", "",
           "Integrated gradients through the rollout back to the input history, on the same origins for "
           "every head. `share recent` is the fraction of absolute attribution in the last 5 windows of "
           "the 30-window history; `mean age` is the attribution-weighted mean age of a cell. The deletion "
           "test zeroes the top-8 attributed cells and compares the drop against 8 random ones.", "",
           "| head | forecast score | share recent (5 of 30) | mean age (min) | drop top-8 | drop random | beats random |",
           "|---|---|---|---|---|---|---|"]
    def f(s, nd=3):
        return "—" if not s else f"{s['mean']:.{nd}f} ± {s['sd']:.{nd}f}"
    for label, s in per_run.items():
        out.append(f"| {label} | {f(s['forecast_score'])} | {f(s['share_recent'])} | {f(s['mean_age_min'], 1)} | "
                   f"{f(s['drop_top_m'])} | {f(s['drop_random_m_mean'])} | {f(s['beats_random_fraction'], 2)} |")
    return "\n".join(out)


def _origins(cfg: dict, split: str, n: int, seed: int = 0):
    """The n highest-risk origins of the split, as scaled [L, F] histories.

    Highest-risk rather than random: an explanation of a silent host is not
    the explanation anyone wants, and a random sample of a 0.3%-prevalence
    split is almost all silent hosts."""
    from nidra.data.normalize import FeatureScaler
    from nidra.train.pipeline import build_all_splits, geometry_from_config, scale_arrays
    from nidra.data.dataset import build_windowed_arrays
    from nidra.utils.config import resolve_path

    _, L, K = geometry_from_config(cfg)
    splits = build_all_splits(cfg)
    arrays = build_windowed_arrays(getattr(splits, split), L=L, K=K, max_samples=20000, seed=seed)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    X, _ = scale_arrays(arrays, scaler)
    pos = np.where(arrays.risk_label == 1)[0]
    pick = pos[:n] if len(pos) >= n else np.concatenate([pos, np.arange(len(X))[:n - len(pos)]])
    return X[pick].astype("float32"), scaler.zero_state_scaled().astype("float32"), int(cfg["windowing"]["window_seconds"])


def main() -> None:
    from nidra.eval.benchmark import load_models
    from nidra.explain.forecast_attribution import explain_forecast, forecast_score_fn, integrated_gradients
    from nidra.explain.forecast_attribution import faithfulness_check
    from nidra.scripts.run_experiment import RUNS_DIR
    from nidra.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-origins", type=int, default=40)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--out", default=None)
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    seeds = [int(s) for s in args.seeds.split(",")]
    per_run, raw = {}, {}
    X = baseline = None
    window_seconds = 60
    for label in args.runs:
        cfg = load_config(runs_dir / label / "config.yaml")
        if X is None:
            # The SAME origins for every head, from the first run's scaler.
            X, baseline, window_seconds = _origins(cfg, args.split, args.n_origins)
            logger.info("explaining %d origins of %s", len(X), args.split)
        models = load_models(cfg, seeds)
        K = int(cfg["windowing"]["horizon_length"])
        f = forecast_score_fn(models, K)
        rows = []
        for i, x in enumerate(X):
            b = np.tile(baseline.reshape(1, -1), (x.shape[0], 1))
            attr, fx, _ = integrated_gradients(f, x, b, steps=args.steps)
            prof = temporal_profile(attr, window_seconds)
            faith = faithfulness_check(f, x, b, attr, m=8, n_random=20, seed=i)
            rows.append({**prof, **faith, "forecast_score": fx})
        per_run[label] = summarise(rows)
        raw[label] = {"per_window_share_mean":
                      np.mean([r["per_window_share"] for r in rows], axis=0).tolist()}
        logger.info("%s: %s", label, {k: (v and round(v["mean"], 3)) for k, v in per_run[label].items()})

    md = comparison_markdown(per_run, len(X), args.split)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps({"summary": per_run, "profiles": raw}, indent=2))


if __name__ == "__main__":
    main()

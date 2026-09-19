"""Operating point selected on VALIDATION only, then frozen.

The Δ=30 pooling quantile (q=0.85) was chosen by looking at test and holdout
F1 (it said so in the config). Everything decision-like now comes from
here, from the validation split, before test is touched:

  pooling statistic     which reduction of the sampled trajectories is the
                        forecast score (mean / median / quantile / max /
                        P(trajectory > 0.5)), and how horizons are collapsed
  calibration           per-horizon Platt scaling of the pooled score, fit
                        with natural-prevalence weights so the number that
                        is called a probability is one
  threshold             the score threshold that maximizes weighted F1 on
                        validation; the mandated 0.75 is reported alongside
                        (on the calibrated score), never tuned for

The selection is written as JSON with every candidate's validation score,
so the choice is auditable, and test/holdout runs only ever LOAD it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from nidra.eval.calibrate import apply_platt_by_horizon, fit_platt_by_horizon
from nidra.eval.eval_set import EvalSet
from nidra.eval.metrics_natural import best_f1_threshold, weighted_ap
from nidra.eval.systems import ScoreBundle

OPERATING_POINT_FILENAME = "operating_point.json"

POOLING_CANDIDATES: list[dict[str, Any]] = [
    {"method": "mean", "quantile": None, "horizon_reduction": "max"},
    {"method": "mean", "quantile": None, "horizon_reduction": "integrated"},
    {"method": "median", "quantile": None, "horizon_reduction": "max"},
    {"method": "quantile", "quantile": 0.75, "horizon_reduction": "max"},
    {"method": "quantile", "quantile": 0.85, "horizon_reduction": "max"},
    {"method": "quantile", "quantile": 0.90, "horizon_reduction": "max"},
    {"method": "quantile", "quantile": 0.95, "horizon_reduction": "max"},
    {"method": "max", "quantile": None, "horizon_reduction": "max"},
    {"method": "p_above_half", "quantile": None, "horizon_reduction": "max"},
]


def pooling_key(p: dict[str, Any]) -> str:
    q = f"{p['quantile']:.2f}" if p.get("quantile") is not None else "-"
    return f"{p['method']}|q={q}|{p.get('horizon_reduction', 'max')}"


def _composite(bundle: ScoreBundle, pooling: dict[str, Any], calibration: list[dict] | None) -> np.ndarray:
    risk_k = bundle.pooled(pooling["method"], pooling.get("quantile") or 0.85)
    if calibration is not None:
        risk_k = apply_platt_by_horizon(risk_k, calibration)
    return ScoreBundle.over_horizon(risk_k, pooling.get("horizon_reduction", "max"))


def select_operating_point(bundle: ScoreBundle, val: EvalSet, mandated_threshold: float = 0.75,
                           candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pick pooling by validation AP (natural prevalence, published label),
    fit per-horizon Platt scaling on validation for that pooling, choose the
    weighted-F1-optimal threshold on the calibrated composite score."""
    candidates = candidates or POOLING_CANDIDATES
    y = val.y_published
    w = val.weight
    scored = {}
    for p in candidates:
        s = _composite(bundle, p, None)
        scored[pooling_key(p)] = {"pooling": p, "val_auc_pr": weighted_ap(y, s, w)}
    best_key = max(scored, key=lambda k: (np.nan_to_num(scored[k]["val_auc_pr"], nan=-1.0)))
    best = scored[best_key]["pooling"]

    risk_k_raw = bundle.pooled(best["method"], best.get("quantile") or 0.85)
    calibration = fit_platt_by_horizon(risk_k_raw, val.arrays.future_is_attack, sample_weight=w)
    s_cal = _composite(bundle, best, calibration)
    thr, f1 = best_f1_threshold(y, s_cal, w)
    s_raw = _composite(bundle, best, None)
    thr_raw, f1_raw = best_f1_threshold(y, s_raw, w)
    return {
        "selected_on": "val",
        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pooling": best,
        "pooling_key": best_key,
        "candidates": scored,
        "calibration": {"method": "platt_per_horizon_weighted", "params_by_k": calibration,
                        "label": "future_is_attack[k]", "weighted_by": "natural_prevalence"},
        "threshold": {"f1_optimal_calibrated": thr, "val_f1_calibrated": f1,
                      "f1_optimal_raw": thr_raw, "val_f1_raw": f1_raw,
                      "mandated": mandated_threshold},
        "val_auc_pr_selected": scored[best_key]["val_auc_pr"],
        "val_summary": val.summary(),
        "n_members": bundle.n_members,
        "n_samples_per_member": bundle.n_samples_per_member,
    }


def apply_operating_point(bundle: ScoreBundle, op: dict[str, Any]) -> dict[str, np.ndarray]:
    """Raw and calibrated composite scores under a frozen operating point."""
    pooling = op["pooling"]
    cal = op["calibration"]["params_by_k"]
    return {
        "raw": _composite(bundle, pooling, None),
        "calibrated": _composite(bundle, pooling, cal),
        "risk_k_raw": bundle.pooled(pooling["method"], pooling.get("quantile") or 0.85),
        "risk_k_calibrated": apply_platt_by_horizon(bundle.pooled(pooling["method"], pooling.get("quantile") or 0.85), cal),
    }


def save_operating_point(op: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(op, f, indent=2, default=float)
    return path


def load_operating_point(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)

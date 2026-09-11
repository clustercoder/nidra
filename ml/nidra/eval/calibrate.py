"""Post-hoc probability recalibration for the frozen risk head's forecast
output — Platt scaling (a 2-parameter logistic remap), fit ONLY on the
validation split, never on test/holdout.

Why this exists (see the calibration-gap investigation in
REAL_DATA_RESULTS.md / MODEL_CARD.md): `risk_mean_k` is the Monte-Carlo mean
of the frozen risk head's sigmoid output across ~100-500 stochastic rollout
trajectories. Measured directly against the trained model: individual
trajectory scores are near-binary (std across trajectories ~0.45, close to
the 0.5 theoretical max for a bounded probability), so `risk_mean_k` behaves
like "fraction of plausible sampled futures the head calls risky" — for
genuine future-attack windows that fraction averages only ~45%, well under
the mandated 0.75 decision threshold, even though ~98-100% of those windows
have at least one individual trajectory that DOES cross 0.75. AUC-PR
(rank-based) is unaffected by this — the ordering is fine — but F1/recall/
lead-time (all threshold-based) are not, because the absolute magnitude of
the aggregate score is compressed.

Platt scaling fixes the MAGNITUDE while providing a mathematical guarantee
about what it does NOT change, WITHIN ONE HORIZON: it is a strictly
monotonic transform of the score (`sigmoid(a * logit(p) + b)` with any real
a>0), so for a FIXED k it can only ever relabel which raw scores map to
which calibrated scores — it cannot invert any pair's relative order.
`calibration_by_horizon`'s per-k Brier/reliability numbers are therefore
comparing like for like, and per-k AUC-PR (e.g. `ablations.horizon_curve`)
is unaffected.

**This guarantee does NOT extend to `risk_over_horizon` (`risk_mean_k.max(axis=1)`,
used by `baselines.json`'s `world_model`/`world_model_calibrated` rows and
by the lead-time risk curve).** Each horizon k gets its OWN (a_k, b_k), so
two samples whose raw scores peak at different horizons can have their
*relative* order after the max-reduction changed by calibration even
though each individual horizon's ranking was preserved — measured directly
in this project: `world_model` vs `world_model_calibrated` AUC-PR differed
by about 0.012 on the test split, not exactly zero. Small here, but real,
and it must not be described as an exact invariance in that composite
context — only the per-horizon claim is exact.

This does NOT retrain the risk head (Rule 1: heads are trained only on
observed states, then frozen — enforced in train/train_heads.py and
unaffected by anything in this module). Platt scaling here operates
entirely outside that boundary, on the head's already-frozen OUTPUT
probability, using labeled validation data the same way a decision
threshold itself would be chosen — it is closer to "choosing a threshold"
than to "training a classifier."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p_clipped / (1.0 - p_clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Fits `sigmoid(a * logit(probs) + b)` to `labels` via a 1-D logistic
    regression on `logit(probs)` (standard Platt scaling). Falls back to the
    identity transform (a=1, b=0) if `labels` has only one class present —
    a real possibility on a small or heavily-capped validation slice — since
    a 2-class fit is undefined there; this is reported in the returned dict
    (`"degenerate": True`) rather than silently returned as if it were a
    real fit.
    """
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    if n == 0 or len(np.unique(labels)) < 2:
        return {"a": 1.0, "b": 0.0, "n": int(n), "degenerate": True}

    from sklearn.linear_model import LogisticRegression

    x = _logit(np.asarray(probs)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x, labels)
    a = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    if a <= 0:
        # A non-positive slope would invert the ranking, which defeats the
        # entire point of a magnitude-only recalibration — fall back to
        # identity rather than silently degrading AUC-PR.
        logger.warning("fit_platt: fitted slope a=%.4f <= 0, falling back to identity (n=%d)", a, n)
        return {"a": 1.0, "b": 0.0, "n": int(n), "degenerate": True}
    return {"a": a, "b": b, "n": int(n), "degenerate": False}


def apply_platt(probs: np.ndarray, params: dict) -> np.ndarray:
    """Applies a fitted (or identity/degenerate) Platt transform. Safe to
    call with `params={"a": 1.0, "b": 0.0}` as a no-op."""
    probs = np.asarray(probs)
    return _sigmoid(params["a"] * _logit(probs) + params["b"])


def fit_platt_by_horizon(risk_mean_k: np.ndarray, future_is_attack: np.ndarray) -> list[dict]:
    """risk_mean_k, future_is_attack: [N, K]. Returns one fitted-Platt dict
    per horizon k — calibration is fit separately per k because the
    per-horizon Brier/reliability numbers already show materially different
    behavior across k (the horizon-parity oscillation documented in
    REAL_DATA_RESULTS.md); a single global (a, b) would paper over that
    rather than correct for it.
    """
    K = risk_mean_k.shape[1]
    return [fit_platt(risk_mean_k[:, k], future_is_attack[:, k]) for k in range(K)]


def apply_platt_by_horizon(risk_mean_k: np.ndarray, params_by_k: list[dict]) -> np.ndarray:
    """risk_mean_k: [..., K] (works on [N,K] or [K]). params_by_k: length-K
    list from fit_platt_by_horizon (or loaded from disk)."""
    risk_mean_k = np.asarray(risk_mean_k)
    K = risk_mean_k.shape[-1]
    if len(params_by_k) != K:
        raise ValueError(f"expected {K} per-horizon calibration params, got {len(params_by_k)}")
    out = np.empty_like(risk_mean_k, dtype="float64")
    for k in range(K):
        out[..., k] = apply_platt(risk_mean_k[..., k], params_by_k[k])
    return out


def save_calibration(path: str | Path, params_by_k: list[dict], metadata: dict) -> None:
    Path(path).write_text(json.dumps({"params_by_k": params_by_k, "metadata": metadata}, indent=2))


def load_calibration(path: str | Path) -> tuple[list[dict], dict] | None:
    """Returns (params_by_k, metadata), or None if no calibration file has
    been fit yet — callers must treat that as "run uncalibrated", not as an
    error, since calibration is an optional refinement on top of a model
    that works without it."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data["params_by_k"], data["metadata"]

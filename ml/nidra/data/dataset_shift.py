"""How far apart two datasets' feature distributions actually are.

A cross-dataset result is only interpretable if you know which features
moved. `schema.FEATURE_REGIMES` decides what a transfer experiment may read
on an argument about instrumentation — CTU-13's Argus records cannot produce
a TTL — but "can be computed in both" is not the same as "means the same in
both", and the TCP flag ratios are the case in point: CICFlowMeter counts
packets carrying a flag, Argus records that the flag was seen. This module
measures the gap instead of asserting it.

Two statistics per feature, on the ACTIVE rows of each dataset's TRAINING
population only (never validation, test or holdout — a mask chosen with a
glance at the target split is a mask tuned on test):

  * `wasserstein`  — earth-mover distance between the two empirical
    distributions after each is standardised by the POOLED training spread,
    so the number is in shared standard deviations and comparable across
    features. 0 is identical; above ~1 the two datasets barely overlap.
  * `auc`          — how well a single threshold on that one feature tells
    the two datasets apart, as the Mann-Whitney statistic. 0.5 means the
    feature carries no dataset signature at all; 1.0 means it is a perfect
    dataset label, which is the failure mode that makes a "transfer" result
    really a domain-detection result.

Neither statistic decides anything on its own. They go in the assessment
next to the regime definition so a reader can see which of the kept features
are doing the transferring and which are along for the ride.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER

logger = logging.getLogger(__name__)

#: Rows sampled per dataset per feature. The statistics are rank-based and
#: converge quickly; a 2.3M x 2.3M exact Mann-Whitney is minutes of wall
#: clock for a number that is stable at 1e5.
DEFAULT_MAX_ROWS = 100_000


@dataclass(frozen=True)
class FeatureShift:
    feature: str
    wasserstein: float
    auc: float
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    zero_fraction_a: float
    zero_fraction_b: float

    def as_dict(self) -> dict:
        return {
            "feature": self.feature, "wasserstein": self.wasserstein, "auc": self.auc,
            "mean_a": self.mean_a, "mean_b": self.mean_b, "std_a": self.std_a, "std_b": self.std_b,
            "zero_fraction_a": self.zero_fraction_a, "zero_fraction_b": self.zero_fraction_b,
        }


def _rank_auc(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney AUC of `b` over `a`, folded to [0.5, 1] — direction is not
    the question, separability is. Ties contribute 0.5, so a feature that is
    constant in both datasets scores exactly 0.5 rather than 0 or 1."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(n_a + n_b, dtype="float64")
    values = np.concatenate([a, b])[order]
    ranks[order] = np.arange(1, n_a + n_b + 1, dtype="float64")
    # average ranks within tied groups
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    rank_sum_b = ranks[n_a:].sum()
    auc = (rank_sum_b - n_b * (n_b + 1) / 2) / (n_a * n_b)
    return float(max(auc, 1.0 - auc))


def feature_shift(states_a: np.ndarray, states_b: np.ndarray, active_a: np.ndarray | None = None,
                  active_b: np.ndarray | None = None, max_rows: int = DEFAULT_MAX_ROWS,
                  seed: int = 0) -> list[FeatureShift]:
    """Per-feature distance between two RAW (unscaled) state populations.

    `states_*` are [N, 45]. Rows are restricted to active windows by default
    — the silent rows are identically zero in both datasets and would drag
    every distance towards zero for a reason that has nothing to do with the
    instrumentation.
    """
    rng = np.random.default_rng(seed)

    def prepare(states: np.ndarray, active: np.ndarray | None) -> np.ndarray:
        arr = np.asarray(states, dtype="float64")
        if arr.shape[-1] != len(FEATURE_ORDER):
            raise ValueError(f"expected {len(FEATURE_ORDER)} features, got {arr.shape[-1]}")
        mask = arr[:, FEATURE_INDEX["is_active"]] > 0 if active is None else np.asarray(active, dtype=bool)
        arr = arr[mask]
        if len(arr) > max_rows:
            arr = arr[rng.choice(len(arr), max_rows, replace=False)]
        return arr

    a, b = prepare(states_a, active_a), prepare(states_b, active_b)
    logger.info("feature_shift: %d active rows vs %d active rows", len(a), len(b))
    out: list[FeatureShift] = []
    for i, name in enumerate(FEATURE_ORDER):
        xa, xb = a[:, i], b[:, i]
        pooled = float(np.std(np.concatenate([xa, xb])))
        scale = pooled if pooled > 1e-12 else 1.0
        out.append(FeatureShift(
            feature=name,
            wasserstein=float(wasserstein_distance(xa / scale, xb / scale)),
            auc=_rank_auc(xa, xb),
            mean_a=float(xa.mean()), mean_b=float(xb.mean()),
            std_a=float(xa.std()), std_b=float(xb.std()),
            zero_fraction_a=float((xa == 0).mean()), zero_fraction_b=float((xb == 0).mean()),
        ))
    return out


def shift_report(shifts: list[FeatureShift], regime_dropped: list[str] | None = None) -> dict:
    """Summary for the assessment: the worst offenders, and whether the
    features a regime KEEPS are the comparable ones."""
    dropped = set(regime_dropped or [])
    kept = [s for s in shifts if s.feature not in dropped]
    ranked = sorted(kept, key=lambda s: -s.wasserstein)
    return {
        "n_features": len(shifts),
        "n_kept_by_regime": len(kept),
        "kept_median_wasserstein": float(np.median([s.wasserstein for s in kept])) if kept else float("nan"),
        "kept_max_wasserstein": float(max((s.wasserstein for s in kept), default=float("nan"))),
        "kept_median_auc": float(np.median([s.auc for s in kept])) if kept else float("nan"),
        "worst_kept": [s.as_dict() for s in ranked[:10]],
        "per_feature": [s.as_dict() for s in shifts],
    }

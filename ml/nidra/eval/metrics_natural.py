"""Natural-prevalence metrics for a weighted (stratified) evaluation set,
with a cluster bootstrap over attack episodes and benign hosts.

Everything here takes `weight` — the stratum weights from eval_set.py — so
a metric is an unbiased estimate for the whole split, not for the sample
that was rolled out. `cluster_bootstrap` resamples clusters (episodes for
attack-related rows, hosts for the rest) because windows inside one episode
are not independent; a per-row bootstrap would give intervals several
times too narrow (the reevaluation's §3 intervals used exactly this scheme).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _clean(y: np.ndarray, s: np.ndarray, w: np.ndarray | None):
    y = np.asarray(y).astype(int)
    s = np.nan_to_num(np.asarray(s, dtype="float64"), nan=0.0)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
    return y, s, w


def weighted_ap(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    y, s, w = _clean(y, s, w)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s, sample_weight=w))


def weighted_roc_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    y, s, w = _clean(y, s, w)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def confusion_at(y: np.ndarray, s: np.ndarray, threshold: float, w: np.ndarray | None = None) -> dict[str, float]:
    """Weighted confusion counts at a threshold (weights make them estimates
    of the full split's counts)."""
    y, s, w = _clean(y, s, w)
    pred = s >= threshold
    tp = float(w[pred & (y == 1)].sum())
    fp = float(w[pred & (y == 0)].sum())
    fn = float(w[~pred & (y == 1)].sum())
    tn = float(w[~pred & (y == 0)].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "n_alerts": tp + fp}


def best_f1_threshold(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """(threshold, F1) maximizing weighted F1 — for choosing an operating
    point on VALIDATION only."""
    y, s, w = _clean(y, s, w)
    if y.sum() == 0:
        return 0.5, 0.0
    p, r, t = precision_recall_curve(y, s, sample_weight=w)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[min(i, len(t) - 1)]), float(f1[i])


def precision_at_recall(y: np.ndarray, s: np.ndarray, target_recall: float, w: np.ndarray | None = None) -> float:
    y, s, w = _clean(y, s, w)
    if y.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y, s, sample_weight=w)
    ok = r >= target_recall
    return float(p[ok].max()) if ok.any() else 0.0


def recall_at_precision(y: np.ndarray, s: np.ndarray, target_precision: float, w: np.ndarray | None = None) -> float:
    y, s, w = _clean(y, s, w)
    if y.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y, s, sample_weight=w)
    ok = p >= target_precision
    return float(r[ok].max()) if ok.any() else 0.0


def pr_curve_points(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None, n_points: int = 50) -> list[dict]:
    """A thinned weighted PR curve for plotting / the report."""
    y, s, w = _clean(y, s, w)
    if y.sum() == 0:
        return []
    p, r, t = precision_recall_curve(y, s, sample_weight=w)
    idx = np.unique(np.linspace(0, len(p) - 1, num=min(n_points, len(p))).astype(int))
    return [{"recall": float(r[i]), "precision": float(p[i]),
             "threshold": float(t[i]) if i < len(t) else None} for i in idx]


def brier(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    y, s, w = _clean(y, s, w)
    return float(np.average((np.clip(s, 0, 1) - y) ** 2, weights=w))


def summarize_scores(y: np.ndarray, s: np.ndarray, w: np.ndarray, threshold: float,
                     active_benign_mask: np.ndarray | None = None, span_hours: float | None = None) -> dict:
    """The standard block reported for one system on one label."""
    out = {
        "auc_pr": weighted_ap(y, s, w),
        "auc_pr_unweighted": weighted_ap(y, s, None),
        "roc_auc": weighted_roc_auc(y, s, w),
        "n_pos": int(np.asarray(y).sum()),
        "n_rows": int(len(y)),
        "prevalence": float(np.average(np.asarray(y).astype(float), weights=w)) if len(y) else float("nan"),
        "at_threshold": confusion_at(y, s, threshold, w),
        "precision_at_recall_0.5": precision_at_recall(y, s, 0.5, w),
        "precision_at_recall_0.8": precision_at_recall(y, s, 0.8, w),
        "recall_at_precision_0.9": recall_at_precision(y, s, 0.9, w),
        "recall_at_precision_0.5": recall_at_precision(y, s, 0.5, w),
        "brier": brier(y, s, w),
        "mean_score_pos": float(np.mean(np.asarray(s)[np.asarray(y) == 1])) if np.asarray(y).sum() else float("nan"),
        "mean_score_neg": float(np.average(np.asarray(s)[np.asarray(y) == 0], weights=np.asarray(w)[np.asarray(y) == 0])) if (np.asarray(y) == 0).any() else float("nan"),
    }
    if active_benign_mask is not None and active_benign_mask.any():
        sa = np.asarray(s)[active_benign_mask]
        wa = np.asarray(w)[active_benign_mask]
        out["active_benign_false_alarm_rate"] = float(np.average(sa >= threshold, weights=wa))
    if span_hours:
        out["false_alarms_per_hour"] = out["at_threshold"]["fp"] / span_hours
        out["alerts_per_hour"] = out["at_threshold"]["n_alerts"] / span_hours
    return out


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_resamples: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a statistic over row-index arrays by resampling CLUSTERS
    with replacement. `stat(idx)` computes the statistic on the rows `idx`
    (with repeats). Returns point estimate, 2.5/97.5 percentiles, std."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = float(stat(np.arange(len(clusters))))
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[c] for c in pick]) if len(pick) else np.zeros(0, dtype=int)
        val = stat(idx)
        if np.isfinite(val):
            draws.append(float(val))
    if not draws:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan"), "n_resamples": 0}
    d = np.asarray(draws)
    return {"point": point, "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "std": float(d.std()), "n_resamples": int(len(d))}


def bootstrap_ap(y: np.ndarray, s: np.ndarray, w: np.ndarray, clusters: np.ndarray,
                 n_resamples: int = 300, seed: int = 0) -> dict[str, float]:
    y, s, w = _clean(y, s, w)
    return cluster_bootstrap(lambda idx: weighted_ap(y[idx], s[idx], w[idx]), clusters, n_resamples, seed)


def bootstrap_ap_difference(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, w: np.ndarray, clusters: np.ndarray,
                            n_resamples: int = 300, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of AP(a) - AP(b) on the same resampled clusters —
    the attribution statistic (world model minus persistence, etc.)."""
    y, a, w = _clean(y, s_a, w)
    _, b, _ = _clean(y, s_b, w)
    return cluster_bootstrap(lambda idx: weighted_ap(y[idx], a[idx], w[idx]) - weighted_ap(y[idx], b[idx], w[idx]),
                             clusters, n_resamples, seed)

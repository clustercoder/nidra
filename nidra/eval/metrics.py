"""Standard and forecast-specific metrics.

The forecast-specific metrics (lead time, state nRMSE, calibration) are what
distinguish a world model from a classifier — a classifier never emits a
state, so it cannot report state_nrmse at all, and it has no notion of
"lead time" beyond a single fixed operating point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------

def standard_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.75) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "fpr": fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def fpr_alerts_per_hour_per_host(fp_count: int, n_hosts: int, n_hours: float) -> float:
    """The number a SOC actually cares about, not a bare fraction."""
    if n_hosts <= 0 or n_hours <= 0:
        return 0.0
    return fp_count / (n_hosts * n_hours)


# ---------------------------------------------------------------------------
# Forecast lead time — the headline metric, defined exactly once.
# ---------------------------------------------------------------------------

def lead_time_for_episode(
    risk_curve: list[tuple[int, float]],
    onset_ts: int,
    threshold: float = 0.75,
    m: int = 2,
) -> float | None:
    """risk_curve: chronological [(window_ts, p_compromise), ...] for one
    host, covering time strictly before `onset_ts` (the true attack-stage
    onset). Returns lead time in SECONDS, or None if no sustained warning
    fired. `m` consecutive windows above threshold prevents a single noisy
    spike from counting as a prediction. Never report only the maximum lead
    time across episodes — always the full distribution (see
    `lead_time_distribution`)."""
    run = 0
    for i, (ts, p) in enumerate(risk_curve):
        if ts >= onset_ts:
            break
        run = run + 1 if p >= threshold else 0
        if run >= m:
            first_crossing_ts = risk_curve[i - m + 1][0]
            return float(onset_ts - first_crossing_ts)
    return None


@dataclass
class LeadTimeReport:
    lead_times: list[float] = field(default_factory=list)   # only non-None
    n_episodes: int = 0
    n_no_warning: int = 0

    @property
    def median(self) -> float | None:
        return float(np.median(self.lead_times)) if self.lead_times else None

    @property
    def fraction_no_warning(self) -> float:
        return self.n_no_warning / self.n_episodes if self.n_episodes else 0.0

    def to_dict(self) -> dict:
        return {
            "median_lead_time_s": self.median,
            "lead_time_distribution_s": self.lead_times,
            "n_episodes": self.n_episodes,
            "n_no_warning": self.n_no_warning,
            "fraction_no_warning": self.fraction_no_warning,
        }


def lead_time_distribution(
    episodes: list[tuple[list[tuple[int, float]], int]],
    threshold: float = 0.75,
    m: int = 2,
) -> LeadTimeReport:
    """episodes: list of (risk_curve, onset_ts) pairs, one per attack
    episode. Reports median + full distribution + no-warning fraction —
    NEVER only the maximum, which a single lucky episode could produce."""
    report = LeadTimeReport(n_episodes=len(episodes))
    for risk_curve, onset_ts in episodes:
        lt = lead_time_for_episode(risk_curve, onset_ts, threshold, m)
        if lt is None:
            report.n_no_warning += 1
        else:
            report.lead_times.append(lt)
    return report


# ---------------------------------------------------------------------------
# Forecast-specific: early-warning precision, detection-before-completion,
# stage accuracy
# ---------------------------------------------------------------------------

def early_warning_precision_at_k(alerts_k: np.ndarray, followed_by_attack_k: np.ndarray) -> float:
    """Of alarms raised at horizon k, the fraction followed by a real
    attack within k windows."""
    n_alerts = int(alerts_k.sum())
    if n_alerts == 0:
        return float("nan")
    return float((alerts_k & followed_by_attack_k).sum()) / n_alerts


def detection_before_stage_completion(warned: np.ndarray) -> float:
    """Fraction of attack episodes where an alarm fired before the labelled
    stage ended. `warned`: bool array, one entry per episode."""
    if len(warned) == 0:
        return float("nan")
    return float(warned.sum()) / len(warned)


def stage_accuracy_at_k(true_stage: np.ndarray, pred_stage_probs: np.ndarray, top_k: int = 1) -> float:
    """true_stage: [N] int class indices. pred_stage_probs: [N, n_stages].
    Computed only over episodes where an attack occurred (caller filters)."""
    if len(true_stage) == 0:
        return float("nan")
    top_k_preds = np.argsort(-pred_stage_probs, axis=1)[:, :top_k]
    hit = (top_k_preds == true_stage[:, None]).any(axis=1)
    return float(hit.mean())


# ---------------------------------------------------------------------------
# State forecast nRMSE — the metric that proves a world model was built.
# ---------------------------------------------------------------------------

# Floor for the nRMSE normalizer, in scaled (RobustScaler) units where 1.0
# is one training-IQR. Many of the 45 features are structurally near-constant
# for large slices of this dataset (packet aggregates are all-zero on a
# flow-only day; urg_ratio/frag_flag_rate are ~0 for most benign hosts), so
# their true variance is genuinely tiny. Dividing by that tiny number (the
# textbook nRMSE = RMSE/std definition) inflates the ratio by orders of
# magnitude for reasons that have nothing to do with forecast quality — this
# is what produced the 10^5-10^6 nRMSE values in earlier runs. The floor
# bounds the worst case (state_clamp=10 => rmse<=20) to a legible ~20/0.05=400
# rather than ~20/1e-8=2e9, without touching the numerator.
_NRMSE_SCALE_FLOOR = 0.05


def state_nrmse(y_true: np.ndarray, y_pred: np.ndarray, feature_scale: np.ndarray | None = None) -> np.ndarray:
    """y_true, y_pred: [N, F] (already selected for one horizon k). Returns
    per-feature nRMSE.

    `feature_scale`, when given, should be the per-feature std of the TRAIN
    population in scaled units (`FeatureScaler.reference_std_`) — a stable
    statistic computed once over millions of rows. This is strongly
    preferred over the alternative of recomputing std(y_true) on whatever
    (often small, stratified) eval batch is passed in: a batch of a few
    thousand rows can have near-zero variance on a near-constant feature
    purely by chance, which does not mean that feature's forecast is
    actually 10^5x worse than a well-behaved one. Falls back to the
    eval-batch std, floored, if no reference is supplied (e.g. an older
    scaler artifact saved before `reference_std_` existed)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    scale = feature_scale if feature_scale is not None else np.std(y_true, axis=0)
    return rmse / np.clip(scale, _NRMSE_SCALE_FLOOR, None)


def state_nrmse_by_horizon(
    y_true_k: np.ndarray, y_pred_k: np.ndarray, feature_scale: np.ndarray | None = None
) -> np.ndarray:
    """y_true_k, y_pred_k: [N, K, F]. Returns [K, F] nRMSE grid — per
    feature, per horizon. This is the metric a classifier cannot report at
    all, since it never emits a state. See `state_nrmse` re: `feature_scale`."""
    K = y_true_k.shape[1]
    return np.stack([state_nrmse(y_true_k[:, k, :], y_pred_k[:, k, :], feature_scale) for k in range(K)])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Returns bin centers, observed frequency per bin, and bin counts —
    the raw material for a reliability plot. A forecast saying 78% should
    be right about 78% of the time; this is where that claim is checked."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    centers, observed, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        counts.append(n)
        centers.append(float((bins[b] + bins[b + 1]) / 2))
        observed.append(float(y_true[mask].mean()) if n > 0 else float("nan"))
    return {"bin_centers": centers, "observed_frequency": observed, "bin_counts": counts}

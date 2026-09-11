"""Builds per-host risk curves from windowed samples and computes the
canonical lead-time distribution across attack episodes.

The per-window "risk at time t" used for the curve is `risk_over_horizon`
(max over k=1..K of the ensemble-mean rollout risk) — the model's estimate
of "does the current trajectory reach compromise within the horizon,"
which is exactly what risk_label supervises during Stage 2 training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nidra.data.dataset import WindowedArrays
from nidra.data.normalize import FeatureScaler
from nidra.eval.baselines import ensemble_world_model_forecast, world_model_forecast
from nidra.eval.metrics import LeadTimeReport, lead_time_distribution
from nidra.models.world_model import WorldModel


def _forecast(batch: np.ndarray, model: WorldModel | list[WorldModel], K: int, n_samples: int,
              calibration: list[dict] | None) -> dict:
    """Dispatches to the single-model or pooled-ensemble forecast statistic
    depending on whether `model` is one WorldModel or a list of them — the
    same distinction as `world_model_forecast` vs `ensemble_world_model_forecast`
    in baselines.py, kept in one place so callers don't need to branch."""
    if isinstance(model, list):
        return ensemble_world_model_forecast(batch, model, K=K, n_samples_per_member=n_samples,
                                              calibration=calibration)
    return world_model_forecast(batch, model, K=K, n_samples=n_samples, calibration=calibration)


def _host_onset_ts(labelled_state_table: pd.DataFrame) -> dict[str, int]:
    attacks = labelled_state_table[labelled_state_table["stage_label"] != "benign"]
    return attacks.groupby("host_id")["window_ts"].min().to_dict()


def build_episode_risk_curves(
    windowed: WindowedArrays,
    labelled_state_table: pd.DataFrame,
    model: WorldModel | list[WorldModel],
    scaler: FeatureScaler,
    n_samples: int = 50,
    batch_size: int = 64,
    calibration: list[dict] | None = None,
) -> list[tuple[list[tuple[int, float]], int]]:
    """Returns [(risk_curve, onset_ts), ...] — one entry per host that has
    at least one attack window in `labelled_state_table`, restricted to
    samples strictly before that host's onset.

    `model` may be a single WorldModel (single-seed statistic, historically
    the only supported mode) or a list of WorldModels (pooled-ensemble
    statistic, matching exactly what `NidraPredictor` serves — see
    `_forecast` above). Passing the real ensemble here is what lets a
    lead-time claim be verified against production serving behavior instead
    of a single arbitrary seed."""
    onset_by_host = _host_onset_ts(labelled_state_table)
    episodes = []

    for host, onset_ts in onset_by_host.items():
        mask = windowed.host_id == host
        if not mask.any():
            continue
        host_origin_ts = windowed.origin_ts[mask]
        host_X = windowed.X[mask]
        order = np.argsort(host_origin_ts)
        host_origin_ts = host_origin_ts[order]
        host_X = host_X[order]

        pre_onset = host_origin_ts < onset_ts
        if not pre_onset.any():
            episodes.append(([], onset_ts))
            continue

        X_pre = scaler.transform(host_X[pre_onset])
        ts_pre = host_origin_ts[pre_onset]

        risks = []
        for i in range(0, len(X_pre), batch_size):
            batch = X_pre[i : i + batch_size]
            out = _forecast(batch, model, K=windowed.Y.shape[1], n_samples=n_samples, calibration=calibration)
            risks.append(out["risk_over_horizon"])
        risks = np.concatenate(risks) if risks else np.array([])

        risk_curve = list(zip((int(t) for t in ts_pre), (float(p) for p in risks)))
        episodes.append((risk_curve, int(onset_ts)))

    return episodes


def compute_lead_time_report(
    windowed: WindowedArrays,
    labelled_state_table: pd.DataFrame,
    model: WorldModel | list[WorldModel],
    scaler: FeatureScaler,
    threshold: float = 0.75,
    m: int = 2,
    n_samples: int = 50,
    calibration: list[dict] | None = None,
) -> LeadTimeReport:
    """`n_samples` means rollout trajectories total when `model` is a single
    WorldModel, or trajectories PER ENSEMBLE MEMBER when `model` is a list
    (so total trajectories = n_samples * len(model)) — same convention as
    `world_model_forecast`'s `n_samples` vs. `ensemble_world_model_forecast`'s
    `n_samples_per_member`."""
    episodes = build_episode_risk_curves(
        windowed, labelled_state_table, model, scaler, n_samples=n_samples, calibration=calibration
    )
    return lead_time_distribution(episodes, threshold=threshold, m=m)

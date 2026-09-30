"""State-forecast metrics: the numbers that prove a world model was built.

MSE / RMSE per horizon in the model's scaled units over the KEPT features
(dropped features are constant 0 for every predictor and would dilute the
ratio), skill = 1 - MSE / MSE_persistence, per-feature skill, the same on
active origins only, and 90% band coverage of the sampled trajectories.
"""

from __future__ import annotations

import numpy as np

from nidra.data.schema import FEATURE_ORDER


def mse_by_k(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """pred, true: [N, K, F]; mask: bool [F]. Returns [K]."""
    if rows is not None:
        pred, true = pred[rows], true[rows]
    if len(pred) == 0:
        return np.full(true.shape[1], np.nan)
    err = ((pred.astype("float64") - true.astype("float64")) ** 2)[..., mask]
    return err.mean(axis=(0, 2))


def skill(pred: np.ndarray, true: np.ndarray, pers: np.ndarray, mask: np.ndarray, rows: np.ndarray | None = None) -> float:
    m = mse_by_k(pred, true, mask, rows)
    p = mse_by_k(pers, true, mask, rows)
    return float(1.0 - m.sum() / p.sum()) if p.sum() > 0 else float("nan")


def per_feature_skill(pred: np.ndarray, true: np.ndarray, pers: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    e_m = ((pred.astype("float64") - true) ** 2).mean(axis=(0, 1))
    e_p = ((pers.astype("float64") - true) ** 2).mean(axis=(0, 1))
    out = {}
    for i, name in enumerate(FEATURE_ORDER):
        if not mask[i]:
            continue
        out[name] = float(1.0 - e_m[i] / e_p[i]) if e_p[i] > 0 else float("nan")
    return out


def band_coverage(pred_mean: np.ndarray, pred_std: np.ndarray, true: np.ndarray, mask: np.ndarray, z: float = 1.645) -> np.ndarray:
    """Fraction of kept feature-elements whose true value lies inside
    mean ± z·std, per horizon. [K]."""
    inside = (np.abs(true - pred_mean) <= z * pred_std)[..., mask]
    return inside.mean(axis=(0, 2))


def state_forecast_report(states_det: np.ndarray, states_ridge: np.ndarray | None, X_scaled: np.ndarray,
                          Y_scaled: np.ndarray, mask: np.ndarray, active_origin: np.ndarray,
                          states_std: np.ndarray | None = None, traj_mean: np.ndarray | None = None) -> dict:
    K = Y_scaled.shape[1]
    pers = np.repeat(X_scaled[:, -1:, :], K, axis=1)
    # period-2 persistence: alternate the last two observed states
    p2 = np.zeros_like(Y_scaled)
    hist = [X_scaled[:, -2, :], X_scaled[:, -1, :]]
    for k in range(K):
        p2[:, k] = hist[-2]
        hist.append(p2[:, k])
    predictors = {"world_model_deterministic": states_det, "persistence": pers, "period2_persistence": p2}
    if states_ridge is not None:
        predictors["ridge_two_lag"] = states_ridge
    if traj_mean is not None:
        predictors["world_model_trajectory_mean"] = traj_mean
    act = np.where(active_origin)[0]
    out = {
        "n_rows": int(len(Y_scaled)), "n_active_origins": int(len(act)), "features_scored": int(mask.sum()),
        "rmse_by_k": {n: np.sqrt(mse_by_k(p, Y_scaled, mask)).round(4).tolist() for n, p in predictors.items()},
        "skill_vs_persistence": {n: skill(p, Y_scaled, pers, mask) for n, p in predictors.items() if n != "persistence"},
        "skill_vs_persistence_by_k": {
            n: (1.0 - mse_by_k(p, Y_scaled, mask) / mse_by_k(pers, Y_scaled, mask)).round(4).tolist()
            for n, p in predictors.items() if n != "persistence"},
        "active_origins": {
            "rmse_by_k": {n: np.sqrt(mse_by_k(p, Y_scaled, mask, act)).round(4).tolist() for n, p in predictors.items()},
            "skill_vs_persistence": {n: skill(p, Y_scaled, pers, mask, act) for n, p in predictors.items() if n != "persistence"},
        },
        "per_feature_skill_world_model": per_feature_skill(states_det, Y_scaled, pers, mask),
    }
    if states_ridge is not None:
        out["skill_vs_ridge"] = float(1.0 - mse_by_k(states_det, Y_scaled, mask).sum() / mse_by_k(states_ridge, Y_scaled, mask).sum())
        out["active_origins"]["skill_vs_ridge"] = float(
            1.0 - mse_by_k(states_det, Y_scaled, mask, act).sum() / mse_by_k(states_ridge, Y_scaled, mask, act).sum())
    if states_std is not None and traj_mean is not None:
        out["band90_coverage_by_k"] = band_coverage(traj_mean, states_std, Y_scaled, mask).round(4).tolist()
        out["trajectory_std_by_k"] = states_std[..., mask].mean(axis=(0, 2)).round(4).tolist()
    return out

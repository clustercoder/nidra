"""The four mandated baselines (IMPLEMENTATION-ML.md §5.1).

    1. LR on S_t                      — mandated by the problem statement.
    2. LR on flattened S[t-L..t]      — THE ONE THAT MATTERS. History, no
                                         dynamics. Beating this is the real
                                         test of whether the transition model
                                         earns its place.
    3. Persistence (S_hat[t+k] = S_t) — reuses the SAME frozen risk head as
                                         the world model, just skips the
                                         transition step. This isolates the
                                         transition model's contribution
                                         rather than comparing two different
                                         classifiers.
    4. Oracle                         — frozen risk head applied to the
                                         TRUE future state. Upper bound.

If the world model cannot beat baseline #2, that is reported, not hidden —
see eval/ablations.py and the ablation-suite honesty rule in CLAUDE.md.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from nidra.models.world_model import WorldModel


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 2000) -> LogisticRegression:
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    clf.fit(X, y)
    return clf


def baseline_lr_current_state(X_train_last: np.ndarray, y_train: np.ndarray,
                               X_eval_last: np.ndarray) -> tuple[LogisticRegression, np.ndarray]:
    """Baseline 1: LogisticRegression on S_t (raw 45-dim, already scaled).
    Returns (fitted classifier, predicted P(risk) on the eval set)."""
    clf = fit_logistic_regression(X_train_last, y_train)
    probs = clf.predict_proba(X_eval_last)[:, 1]
    return clf, probs


def baseline_lr_flattened_history(X_train: np.ndarray, y_train: np.ndarray,
                                   X_eval: np.ndarray) -> tuple[LogisticRegression, np.ndarray]:
    """Baseline 2 (the sharp one): LogisticRegression on flattened
    S[t-L..t] (45*L features — 1350 for L=30). X_* are [N, L, F]."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_eval_flat = X_eval.reshape(X_eval.shape[0], -1)
    clf = fit_logistic_regression(X_train_flat, y_train)
    probs = clf.predict_proba(X_eval_flat)[:, 1]
    return clf, probs


@torch.no_grad()
def baseline_persistence(X_last: np.ndarray, model: WorldModel) -> np.ndarray:
    """Baseline 3: S_hat[t+k] = S_t for every k (the mu=0 case). Because
    that predicted state never changes with k, the composite "risk in
    (t, t+K]" score is just the frozen risk head applied once to S_t —
    reusing the SAME head the world model uses, so any gap between this and
    the world-model score is attributable to the transition model alone."""
    model.eval()
    risk, _ = model.score_states(torch.from_numpy(X_last).float())
    return risk.numpy()


@torch.no_grad()
def baseline_oracle(Y_true: np.ndarray, model: WorldModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline 4: frozen heads applied to the TRUE future states (upper
    bound — no forecasting error at all, only head error). Y_true: [N,K,F].
    Returns (risk_over_horizon [N] = max_k, per_k_risk [N,K], per_k_stage_probs [N,K,n_stages])."""
    model.eval()
    risk_k, stage_k = model.score_states(torch.from_numpy(Y_true).float())
    risk_over_horizon = risk_k.max(dim=1).values.numpy()
    return risk_over_horizon, risk_k.numpy(), stage_k.numpy()


@torch.no_grad()
def ensemble_baseline_persistence(X_last: np.ndarray, models: list[WorldModel]) -> np.ndarray:
    """Ensemble-pooled version of `baseline_persistence`: averages each
    member's own frozen-head score on the same S_t, mirroring how
    `ensemble_world_model_forecast` averages head outputs across members —
    so an "ensemble world model vs. ensemble persistence" comparison isolates
    the transition model's contribution the same way the single-seed
    comparison does, just at the statistic `NidraPredictor` actually serves."""
    x = torch.from_numpy(X_last).float()
    risks = []
    for model in models:
        model.eval()
        r, _ = model.score_states(x)
        risks.append(r)
    return torch.stack(risks, dim=0).mean(dim=0).numpy()


@torch.no_grad()
def ensemble_baseline_oracle(Y_true: np.ndarray, models: list[WorldModel]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble-pooled version of `baseline_oracle`: averages each member's
    own frozen-head score on the true future state. Same contract as
    `baseline_oracle` otherwise."""
    y = torch.from_numpy(Y_true).float()
    risks, stages = [], []
    for model in models:
        model.eval()
        r, s = model.score_states(y)
        risks.append(r)
        stages.append(s)
    risk_k = torch.stack(risks, dim=0).mean(dim=0)
    stage_k = torch.stack(stages, dim=0).mean(dim=0)
    risk_over_horizon = risk_k.max(dim=1).values.numpy()
    return risk_over_horizon, risk_k.numpy(), stage_k.numpy()


@torch.no_grad()
def world_model_forecast(X: np.ndarray, model: WorldModel, K: int, n_samples: int = 200,
                          stochastic: bool = True, calibration: list[dict] | None = None) -> dict:
    """The system under test: recursive rollout -> frozen heads -> quantiles
    across sampled trajectories. Returns per-k mean risk, per-k confidence
    band, the composite risk-over-horizon score (max_k mean risk), and
    per-k mean stage distribution.

    `calibration`, if given (a length-K list from
    `nidra.eval.calibrate.fit_platt_by_horizon`/`load_calibration`), is
    applied to `risk_mean_k`/`risk_ci_low_k`/`risk_ci_high_k` BEFORE
    `risk_over_horizon` is derived from them — see `eval/calibrate.py` for
    why (raw ensemble-mean probabilities are compressed well below the
    0.75 decision threshold even for true positives). Leaving it `None`
    reproduces the original, uncalibrated behavior exactly.
    """
    model.eval()
    x_t = torch.from_numpy(X).float()
    out = model.rollout(x_t, K=K, n_samples=n_samples, stochastic=stochastic)  # states: [B,S,K,F]
    B, S, K_, F = out.states.shape
    flat_states = out.states.reshape(B * S, K_, F)
    risk, stage = model.score_states(flat_states)          # [B*S, K], [B*S, K, n_stages]
    risk = risk.reshape(B, S, K_)
    stage = stage.reshape(B, S, K_, -1)

    risk_mean_k = risk.mean(dim=1).numpy()                  # [B, K]
    risk_ci_low_k = risk.quantile(0.05, dim=1).numpy()
    risk_ci_high_k = risk.quantile(0.95, dim=1).numpy()
    stage_mean_k = stage.mean(dim=1).numpy()                # [B, K, n_stages]

    if calibration is not None:
        from nidra.eval.calibrate import apply_platt_by_horizon
        risk_mean_k = apply_platt_by_horizon(risk_mean_k, calibration)
        risk_ci_low_k = apply_platt_by_horizon(risk_ci_low_k, calibration)
        risk_ci_high_k = apply_platt_by_horizon(risk_ci_high_k, calibration)

    risk_over_horizon = risk_mean_k.max(axis=1)

    return {
        "risk_mean_k": risk_mean_k,
        "risk_ci_low_k": risk_ci_low_k,
        "risk_ci_high_k": risk_ci_high_k,
        "stage_mean_k": stage_mean_k,
        "risk_over_horizon": risk_over_horizon,
        "predicted_states_mean": out.states.mean(dim=1).numpy(),  # [B, K, F] for state_nrmse
    }


@torch.no_grad()
def ensemble_world_model_forecast(X: np.ndarray, models: list[WorldModel], K: int,
                                   n_samples_per_member: int = 100, stochastic: bool = True,
                                   calibration: list[dict] | None = None) -> dict:
    """Same contract as `world_model_forecast`, but pools rollout
    trajectories AND head scores across every ensemble member first —
    mirrors `NidraPredictor._ensemble_rollout` exactly, so calibration fit
    against this function's output matches what serving actually produces
    (a per-seed fit against a single model's own `world_model_forecast`
    would not, since the pooled ensemble mean is a different statistic than
    any one seed's own mean).
    """
    x_t = torch.from_numpy(X).float()
    all_states = []
    for model in models:
        model.eval()
        out = model.rollout(x_t, K=K, n_samples=n_samples_per_member, stochastic=stochastic)
        all_states.append(out.states)  # [B, S, K, F]
    pooled_states = torch.cat(all_states, dim=1)  # [B, S_total, K, F]
    B, S, K_, F = pooled_states.shape
    flat_states = pooled_states.reshape(B * S, K_, F)

    risk_list, stage_list = [], []
    for model in models:
        r, s = model.score_states(flat_states)
        risk_list.append(r.reshape(B, S, K_))
        stage_list.append(s.reshape(B, S, K_, -1))
    risk = torch.stack(risk_list, dim=0).mean(dim=0)     # average head outputs across members too
    stage = torch.stack(stage_list, dim=0).mean(dim=0)

    risk_mean_k = risk.mean(dim=1).numpy()
    risk_ci_low_k = risk.quantile(0.05, dim=1).numpy()
    risk_ci_high_k = risk.quantile(0.95, dim=1).numpy()
    stage_mean_k = stage.mean(dim=1).numpy()

    if calibration is not None:
        from nidra.eval.calibrate import apply_platt_by_horizon
        risk_mean_k = apply_platt_by_horizon(risk_mean_k, calibration)
        risk_ci_low_k = apply_platt_by_horizon(risk_ci_low_k, calibration)
        risk_ci_high_k = apply_platt_by_horizon(risk_ci_high_k, calibration)

    return {
        "risk_mean_k": risk_mean_k,
        "risk_ci_low_k": risk_ci_low_k,
        "risk_ci_high_k": risk_ci_high_k,
        "stage_mean_k": stage_mean_k,
        "risk_over_horizon": risk_mean_k.max(axis=1),
        "predicted_states_mean": pooled_states.mean(dim=1).numpy(),
    }

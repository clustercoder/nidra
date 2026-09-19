"""Score every system under comparison on one evaluation set.

One function per system family, one `ScoreBundle` holding everything the
benchmark and the attribution need. The NIDRA rollout is cached per
trajectory (head-mean risk [N, S, K]) so every pooling statistic is a
reduction of the same sampled futures, and the ablations that isolate the
transition model are computed from the same models:

  persistence        frozen heads on S_t — the mu=0, no-noise case
  noised persistence mu forced to 0, the model's own predicted variance kept
  isotropic noise    mu forced to 0, a single scalar variance in every feature
  deterministic      mu kept, no sampling
  stochastic         the full world model

Baselines that are NOT world models (LR on S_t, LR on flattened history,
GBDT on S_t, a GRU sequence classifier) are scored here too, on the same
rows, so the table is one table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from nidra.data.dataset import WindowedArrays
from nidra.data.normalize import FeatureScaler
from nidra.eval.eval_set import EvalSet
from nidra.models.risk_pooling import pool_trajectories_np
from nidra.models.world_model import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class ScoreBundle:
    K: int
    risk_traj: np.ndarray | None = None          # [N, S, K] fp16, head-mean per sampled trajectory (stochastic WM)
    risk_np_traj: np.ndarray | None = None       # [N, S, K] noised persistence (mu=0, learned variance)
    risk_iso_traj: np.ndarray | None = None      # [N, S, K] isotropic-noise persistence
    risk_det_k: np.ndarray | None = None         # [N, K] deterministic rollout (ensemble mean of heads)
    risk_obs: np.ndarray | None = None           # [N] heads on S_t
    risk_true_k: np.ndarray | None = None        # [N, K] heads on the true future
    stage_traj_mean_k: np.ndarray | None = None  # [N, K, n_stages] stage distribution from the stochastic rollout
    stage_obs: np.ndarray | None = None          # [N, n_stages] stage head on S_t
    states_det: np.ndarray | None = None         # [N, K, F] deterministic rollout states (ensemble mean)
    states_std: np.ndarray | None = None         # [N, K, F] trajectory std of the stochastic rollout
    logvar_mean: np.ndarray | None = None        # [N, K, F] mean predicted logvar along the deterministic rollout
    states_ridge: np.ndarray | None = None       # [N, K, F] ridge two-lag forecast
    risk_ridge_k: np.ndarray | None = None       # [N, K] heads on the ridge forecast
    baselines: dict[str, np.ndarray] = field(default_factory=dict)   # name -> [N]
    iso_sigma: float | None = None
    n_members: int = 0
    n_samples_per_member: int = 0

    # ---------------------------------------------------------------- pooling
    def pooled(self, method: str, quantile: float = 0.85, traj: np.ndarray | None = None) -> np.ndarray:
        """[N, K] pooled risk per horizon from the cached trajectories."""
        r = self.risk_traj if traj is None else traj
        return pool_trajectories_np(r, method, quantile, axis=1)

    @staticmethod
    def over_horizon(risk_k: np.ndarray, reduction: str = "max", k_eval: int | None = None) -> np.ndarray:
        """Collapse [N, K] to [N]: max over the first k_eval horizons, or the
        'integrated' 1 - prod(1 - p_k)."""
        r = risk_k[:, :k_eval] if k_eval else risk_k
        if reduction == "max":
            return r.max(axis=1)
        if reduction == "integrated":
            return 1.0 - np.prod(1.0 - np.clip(r, 0, 1), axis=1)
        raise ValueError(f"unknown horizon reduction {reduction!r}")


@torch.no_grad()
def _rollout_risk(models: list[WorldModel], X: torch.Tensor, K: int, n_samples: int, chunk: int,
                  force_zero_mu: bool = False, iso_sigma: float | None = None, seed: int = 0):
    """Stochastic rollouts for every member, scored by every member's heads
    (soft-vote over heads per trajectory). Returns head-mean risk [N, M*S, K]
    fp16, stage mean [N, K, n_stages], trajectory std [N, K, F]."""
    torch.manual_seed(seed)
    N = X.shape[0]
    risk_out, stage_out, std_out = [], [], []
    for lo in range(0, N, chunk):
        xb = X[lo:lo + chunk]
        B = xb.shape[0]
        all_states = []
        for m in models:
            m.eval()
            if force_zero_mu or iso_sigma is not None:
                orig = m.transition.forward

                def patched(h, *lags, _orig=orig):
                    mu, logvar = _orig(h, *lags)
                    mu = torch.zeros_like(mu)
                    if iso_sigma is not None:
                        logvar = torch.full_like(logvar, float(np.log(iso_sigma ** 2)))
                    return mu, logvar
                m.transition.forward = patched
                try:
                    out = m.rollout(xb, K=K, n_samples=n_samples, stochastic=True)
                finally:
                    m.transition.forward = orig
            else:
                out = m.rollout(xb, K=K, n_samples=n_samples, stochastic=True)
            all_states.append(out.states)                        # [B, S, K, F]
        states = torch.cat(all_states, dim=1)                    # [B, M*S, K, F]
        S = states.shape[1]
        flat = states.reshape(B * S, K, -1)
        risks, stages = [], []
        for m in models:
            r, s = m.score_states(flat)
            risks.append(r.reshape(B, S, K))
            stages.append(s.reshape(B, S, K, -1))
        risk_out.append(torch.stack(risks).mean(0).to(torch.float16).numpy())
        stage_out.append(torch.stack(stages).mean(0).mean(1).numpy())
        std_out.append(states.std(dim=1).numpy())
    return np.concatenate(risk_out), np.concatenate(stage_out), np.concatenate(std_out)


@torch.no_grad()
def _deterministic(models: list[WorldModel], X: torch.Tensor, K: int, chunk: int):
    """Deterministic rollout per member → ensemble-mean states, ensemble-mean
    head risk per k on each member's OWN states, mean predicted logvar."""
    N = X.shape[0]
    states_out, risk_out, logvar_out = [], [], []
    for lo in range(0, N, chunk):
        xb = X[lo:lo + chunk]
        st, rk, lv = [], [], []
        for m in models:
            m.eval()
            out = m.rollout(xb, K=K, n_samples=1, stochastic=False)
            s = out.states[:, 0]                                 # [B, K, F]
            st.append(s)
            lv.append(out.logvars[:, 0])
            r, _ = m.score_states(s)
            rk.append(r)
        states_out.append(torch.stack(st).mean(0).numpy())
        risk_out.append(torch.stack(rk).mean(0).numpy())
        logvar_out.append(torch.stack(lv).mean(0).numpy())
    return np.concatenate(states_out), np.concatenate(risk_out), np.concatenate(logvar_out)


@torch.no_grad()
def _heads_on(models: list[WorldModel], states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    risks, stages = [], []
    for m in models:
        m.eval()
        r, s = m.score_states(states)
        risks.append(r)
        stages.append(s)
    return torch.stack(risks).mean(0).numpy(), torch.stack(stages).mean(0).numpy()


def score_world_model(bundle: ScoreBundle, models: list[WorldModel], X_scaled: np.ndarray, Y_scaled: np.ndarray,
                      n_samples_per_member: int, chunk: int = 250, with_ablations: bool = True, seed: int = 0) -> ScoreBundle:
    K = bundle.K
    X = torch.from_numpy(X_scaled).float()
    Y = torch.from_numpy(Y_scaled).float()
    logger.info("scoring world model: N=%d members=%d samples/member=%d K=%d", X.shape[0], len(models), n_samples_per_member, K)
    bundle.risk_obs, bundle.stage_obs = _heads_on(models, X[:, -1, :])
    bundle.risk_true_k, _ = _heads_on(models, Y)
    bundle.states_det, bundle.risk_det_k, bundle.logvar_mean = _deterministic(models, X, K, chunk)
    bundle.risk_traj, bundle.stage_traj_mean_k, bundle.states_std = _rollout_risk(models, X, K, n_samples_per_member, chunk, seed=seed)
    bundle.n_members = len(models)
    bundle.n_samples_per_member = n_samples_per_member
    if with_ablations:
        bundle.risk_np_traj, _, _ = _rollout_risk(models, X, K, n_samples_per_member, chunk, force_zero_mu=True, seed=seed)
        # isotropic: one scalar std matched to the model's RMS predicted std over the set
        sigma = float(np.sqrt(np.exp(bundle.logvar_mean).mean()))
        bundle.iso_sigma = sigma
        bundle.risk_iso_traj, _, _ = _rollout_risk(models, X, K, n_samples_per_member, chunk, iso_sigma=sigma, seed=seed)
    return bundle


def fit_ridge_two_lag(X_train_scaled: np.ndarray, Y_train_scaled: np.ndarray, K: int, alpha: float = 1.0) -> list[Ridge]:
    F_tr = np.concatenate([X_train_scaled[:, -1, :], X_train_scaled[:, -2, :]], axis=1)
    return [Ridge(alpha=alpha).fit(F_tr, Y_train_scaled[:, k, :]) for k in range(K)]


def score_ridge(bundle: ScoreBundle, models: list[WorldModel], ridges: list[Ridge], X_scaled: np.ndarray,
                clip: float = 10.0) -> ScoreBundle:
    F_ev = np.concatenate([X_scaled[:, -1, :], X_scaled[:, -2, :]], axis=1)
    preds = np.stack([np.clip(r.predict(F_ev), -clip, clip) for r in ridges], axis=1).astype("float32")  # [N, K, F]
    bundle.states_ridge = preds
    bundle.risk_ridge_k, _ = _heads_on(models, torch.from_numpy(preds).float())
    return bundle


def fit_classifier_baselines(train: WindowedArrays, X_train_scaled: np.ndarray, seed: int = 0,
                             flat_negatives: int = 60000) -> dict[str, Any]:
    """LR on S_t, GBDT on S_t (regularised), LR on flattened history (all
    positives + a capped negative sample for the 1350-column fit)."""
    y = train.risk_label.astype(int)
    S = X_train_scaled[:, -1, :]
    out: dict[str, Any] = {}
    out["lr_current_state"] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(S, y)
    out["gbdt_current_state"] = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200, l2_regularization=1.0,
        class_weight="balanced", random_state=seed).fit(S, y)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    sub = np.sort(np.concatenate([pos, rng.choice(neg, size=min(flat_negatives, len(neg)), replace=False)]))
    Xf = X_train_scaled[sub].reshape(len(sub), -1)
    out["lr_flattened_history"] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xf, y[sub])
    out["_flat_n"] = int(len(sub))
    return out


def score_classifier_baselines(bundle: ScoreBundle, fitted: dict[str, Any], X_scaled: np.ndarray) -> ScoreBundle:
    S = X_scaled[:, -1, :]
    bundle.baselines["lr_current_state"] = fitted["lr_current_state"].predict_proba(S)[:, 1]
    bundle.baselines["gbdt_current_state"] = fitted["gbdt_current_state"].predict_proba(S)[:, 1]
    bundle.baselines["lr_flattened_history"] = fitted["lr_flattened_history"].predict_proba(X_scaled.reshape(len(X_scaled), -1))[:, 1]
    return bundle


def system_scores(bundle: ScoreBundle, pooling: dict, k_eval: int | None = None) -> dict[str, np.ndarray]:
    """The [N] composite score of every system, under one pooling rule.
    `pooling` = {"method": ..., "quantile": ..., "horizon_reduction": ...}."""
    method = pooling.get("method", "mean")
    q = float(pooling.get("quantile") or 0.85)
    red = pooling.get("horizon_reduction", "max")
    oh = lambda rk: ScoreBundle.over_horizon(rk, red, k_eval)
    out: dict[str, np.ndarray] = {}
    if bundle.risk_obs is not None:
        out["persistence"] = bundle.risk_obs
    if bundle.risk_true_k is not None:
        out["oracle_true_future"] = oh(bundle.risk_true_k)
    if bundle.risk_ridge_k is not None:
        out["ridge_two_lag"] = oh(bundle.risk_ridge_k)
    if bundle.risk_det_k is not None:
        out["world_model_deterministic"] = oh(bundle.risk_det_k)
    if bundle.risk_np_traj is not None:
        out["noised_persistence"] = oh(bundle.pooled(method, q, bundle.risk_np_traj))
    if bundle.risk_iso_traj is not None:
        out["isotropic_noise_persistence"] = oh(bundle.pooled(method, q, bundle.risk_iso_traj))
    if bundle.risk_traj is not None:
        out["world_model"] = oh(bundle.pooled(method, q))
    for name, s in bundle.baselines.items():
        out[name] = s
    return out

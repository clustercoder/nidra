"""KernelSHAP over the frozen heads.

Two distinct uses, both structurally the same mechanism (KernelSHAP) but
answering different questions — see IMPLEMENTATION-ML.md §6:

  (a) "Why is the CURRENT state risky?"  — SHAP on the risk head, over the
      OBSERVED state S_t.
  (c) "Why THIS STAGE?"                  — SHAP on the stage head, over a
      PREDICTED state (only expressible because the transition model
      decodes to named features rather than an opaque latent).

Background set: 100 k-means centroids of BENIGN TRAINING states, never a
random sample — random background makes KernelSHAP slow and noisy (a
documented failure mode in IMPLEMENTATION-ML.md §9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shap
import torch
from sklearn.cluster import KMeans

from nidra.data.schema import FEATURE_ORDER
from nidra.models.world_model import WorldModel


def build_shap_background(benign_train_states: np.ndarray, n_centroids: int = 100, seed: int = 0) -> np.ndarray:
    """benign_train_states: [N, 45] raw/scaled states labelled benign, drawn
    from the TRAINING split only. Returns [min(n_centroids, N), 45]
    k-means centroids."""
    n = min(n_centroids, len(benign_train_states))
    if n < 2:
        return benign_train_states
    km = KMeans(n_clusters=n, n_init=10, random_state=seed)
    km.fit(benign_train_states)
    return km.cluster_centers_


def save_background(background: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, background)


def load_background(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    return np.load(path) if path.exists() else None


def _risk_predict_fn(model: WorldModel):
    def f(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            risk, _ = model.score_states(torch.from_numpy(X).float())
        return risk.numpy()
    return f


def _stage_predict_fn(model: WorldModel, stage_idx: int):
    def f(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            _, stage_probs = model.score_states(torch.from_numpy(X).float())
        return stage_probs[:, stage_idx].numpy()
    return f


def _format_attributions(shap_values: np.ndarray, state: np.ndarray, feature_names: list[str]) -> list[dict]:
    results = []
    for name, val, feat_val in zip(feature_names, shap_values, state):
        results.append({
            "feature": name,
            "shap_value": float(val),
            "direction": "up" if val > 0 else "down",
            "feature_value": float(feat_val),
        })
    results.sort(key=lambda r: -abs(r["shap_value"]))
    return results


def explain_current_risk(
    state: np.ndarray,
    background: np.ndarray,
    model: WorldModel,
    nsamples: int | str = 100,
) -> list[dict]:
    """(a) Why is the current observed state S_t risky? state: [45]."""
    explainer = shap.KernelExplainer(_risk_predict_fn(model), background)
    shap_values = explainer.shap_values(state.reshape(1, -1), nsamples=nsamples, silent=True)
    shap_values = np.asarray(shap_values).reshape(-1)
    return _format_attributions(shap_values, state, FEATURE_ORDER)


def explain_predicted_stage(
    predicted_state: np.ndarray,
    background: np.ndarray,
    model: WorldModel,
    stage_idx: int,
    nsamples: int | str = 100,
) -> list[dict]:
    """(c) Why THIS stage, for a predicted (rolled-out) state. Only
    expressible because the transition model decodes to named features."""
    explainer = shap.KernelExplainer(_stage_predict_fn(model, stage_idx), background)
    shap_values = explainer.shap_values(predicted_state.reshape(1, -1), nsamples=nsamples, silent=True)
    shap_values = np.asarray(shap_values).reshape(-1)
    return _format_attributions(shap_values, predicted_state, FEATURE_ORDER)


def top_signals(attributions: list[dict], n: int = 5) -> list[dict]:
    """Top-N signed signals for the API/UI's `top_signals` field."""
    return attributions[:n]

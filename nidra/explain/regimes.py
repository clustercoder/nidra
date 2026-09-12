"""Behavioral regimes: unsupervised clustering of the encoder's latent
representation, plus empirical regime-to-regime transition counts.

This is descriptive, not predictive: regimes are discovered from the
encoder's own hidden-state geometry (k-means), never from attack-stage
labels — `discover_regimes` never sees `stage_label`/`risk_label`. Labels
are used only afterward, to *describe* what a regime tends to correspond to
(e.g. "regime 3 is 91% pre-attack windows"), which is presentation, not
training signal. See CLAUDE.md claims-discipline: this is a clustering of
model-internal representations, not a claim about real attacker behavior.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from nidra.models.world_model import WorldModel


@torch.no_grad()
def extract_latent_representations(model: WorldModel, X: np.ndarray) -> np.ndarray:
    """X: [N, L, F] scaled observed histories. Returns [N, H] — the
    encoder's last-step hidden summary h_t for each sample, never a
    rollout-predicted state."""
    model.eval()
    h_t, _ = model.encoder(torch.from_numpy(X).float())
    return h_t.numpy()


def discover_regimes(latents: np.ndarray, n_clusters: int = 5, seed: int = 0) -> tuple[KMeans, np.ndarray]:
    """Unsupervised k-means over latent representations. Returns (fitted
    KMeans, cluster label per row). Deliberately takes only `latents` — no
    label array can be passed in, structurally preventing the clustering
    target from ever being an attack stage."""
    n = min(n_clusters, len(latents))
    if n < 1:
        raise ValueError("discover_regimes: no latents to cluster")
    km = KMeans(n_clusters=n, n_init=10, random_state=seed)
    labels = km.fit_predict(latents)
    return km, labels


def regime_transition_counts(regime_labels_per_host: dict[str, np.ndarray], n_regimes: int) -> np.ndarray:
    """Empirical regime[t] -> regime[t+1] transition counts, computed only
    within each host's own chronological (gap-filled) sequence — never
    across host boundaries. `regime_labels_per_host`: host_id -> chronologically
    ordered regime-label array for that host's OBSERVED windows (never a
    rollout-predicted sequence). Returns an [n_regimes, n_regimes] count
    matrix, counts[i, j] = number of observed i -> j transitions."""
    counts = np.zeros((n_regimes, n_regimes), dtype="int64")
    for labels in regime_labels_per_host.values():
        if len(labels) < 2:
            continue
        for a, b in zip(labels[:-1], labels[1:]):
            counts[a, b] += 1
    return counts


def regime_risk_profile(regime_labels: np.ndarray, risk_label: np.ndarray, n_regimes: int) -> list[dict]:
    """Descriptive-only summary: for each discovered regime, the observed
    fraction of samples with risk_label=1 and the sample count. This
    interprets regimes after the fact — it plays no role in how they were
    formed (see module docstring)."""
    profile = []
    for r in range(n_regimes):
        mask = regime_labels == r
        n = int(mask.sum())
        risk_rate = float(risk_label[mask].mean()) if n > 0 else float("nan")
        profile.append({"regime": r, "n_samples": n, "risk_rate": risk_rate})
    return profile

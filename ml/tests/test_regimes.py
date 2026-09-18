import numpy as np

from nidra.explain.regimes import (
    discover_regimes,
    extract_latent_representations,
    regime_risk_profile,
    regime_transition_counts,
)
from nidra.models.world_model import WorldModel


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def test_extract_latent_representations_shape():
    model = _tiny_model()
    X = np.random.randn(20, 30, 45).astype("float32")
    latents = extract_latent_representations(model, X)
    assert latents.shape == (20, 16)


def test_discover_regimes_never_sees_labels():
    """discover_regimes's signature structurally cannot accept a label
    array — clustering must be unsupervised over latents only."""
    import inspect
    params = inspect.signature(discover_regimes).parameters
    assert "labels" not in params and "stage_label" not in params and "risk_label" not in params


def test_discover_regimes_returns_labels_for_every_row():
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((50, 16))
    km, labels = discover_regimes(latents, n_clusters=4, seed=0)
    assert labels.shape == (50,)
    assert set(labels.tolist()) <= set(range(4))


def test_regime_transition_counts_stays_within_host_sequences():
    labels_per_host = {
        "h1": np.array([0, 0, 1, 1, 2]),
        "h2": np.array([2, 1]),
    }
    counts = regime_transition_counts(labels_per_host, n_regimes=3)
    assert counts.shape == (3, 3)
    # h1: 0->0, 0->1, 1->1, 1->2 ; h2: 2->1
    assert counts[0, 0] == 1
    assert counts[0, 1] == 1
    assert counts[1, 1] == 1
    assert counts[1, 2] == 1
    assert counts[2, 1] == 1
    assert counts.sum() == 5


def test_regime_risk_profile_reports_rate_per_regime():
    regime_labels = np.array([0, 0, 1, 1, 1])
    risk_label = np.array([1, 0, 1, 1, 1])
    profile = regime_risk_profile(regime_labels, risk_label, n_regimes=2)
    assert profile[0]["n_samples"] == 2
    assert profile[0]["risk_rate"] == 0.5
    assert profile[1]["n_samples"] == 3
    assert profile[1]["risk_rate"] == 1.0

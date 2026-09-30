import numpy as np

from nidra.eval.ablations import horizon_curve, persistence_ablation, surprise_signal, time_shuffle_ablation
from nidra.eval.calibration import calibration_by_horizon
from nidra.models.world_model import WorldModel


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def _synthetic_batch(n=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 30, 45)).astype("float32")
    Y = rng.standard_normal((n, 6, 45)).astype("float32")
    risk_label = rng.integers(0, 2, n)
    future_is_attack = rng.integers(0, 2, (n, 6))
    return X, Y, risk_label, future_is_attack


def test_persistence_ablation_runs_and_reports_keys():
    model = _tiny_model()
    X, Y, risk_label, _ = _synthetic_batch()
    result = persistence_ablation(X, Y, risk_label, model)
    for key in ["auc_pr_persistence", "auc_pr_world_model", "auc_collapse", "interpretation"]:
        assert key in result


def test_time_shuffle_ablation_runs_and_reports_keys():
    model = _tiny_model()
    X, _, risk_label, _ = _synthetic_batch()
    result = time_shuffle_ablation(X, risk_label, model, K=6)
    for key in ["auc_pr_normal_order", "auc_pr_shuffled_order", "collapse", "interpretation"]:
        assert key in result


def test_horizon_curve_shapes():
    model = _tiny_model()
    X, Y, _, future_is_attack = _synthetic_batch()
    result = horizon_curve(future_is_attack, Y, model, X, n_samples=10)
    assert len(result["auc_pr_by_k"]) == 6
    assert len(result["nrmse_by_k"]) == 6
    assert isinstance(result["flat_curve_leakage_warning"], bool)


def test_surprise_signal_runs_and_reports_keys():
    model = _tiny_model()
    X, Y, _, future_is_attack = _synthetic_batch()
    result = surprise_signal(X, Y, future_is_attack, model)
    for key in ["mean_error_benign", "mean_error_pre_attack", "n_benign", "n_pre_attack", "error_rises_before_onset"]:
        assert key in result
    assert result["n_benign"] + result["n_pre_attack"] == X.shape[0]


def test_calibration_by_horizon_shapes():
    model = _tiny_model()
    X, _, _, future_is_attack = _synthetic_batch()
    result = calibration_by_horizon(X, future_is_attack, model, n_samples=10, n_bins=5)
    assert len(result["per_horizon"]) == 6
    assert "mean_brier" in result
    for entry in result["per_horizon"]:
        assert len(entry["reliability"]["bin_centers"]) == 5

import torch

from nidra.models.world_model import WorldModel
from nidra.train.losses import (
    dynamics_loss,
    gaussian_nll,
    risk_head_loss,
    stage_head_loss,
    teacher_forcing_schedule,
)


def test_gaussian_nll_is_finite_and_nonnegative_ish():
    pred = torch.zeros(4, 45)
    target = torch.zeros(4, 45)
    logvar = torch.zeros(4, 45)
    loss = gaussian_nll(pred, logvar, target)
    assert torch.isfinite(loss)


def test_gaussian_nll_increases_with_prediction_error():
    logvar = torch.zeros(4, 45)
    target = torch.zeros(4, 45)
    close_pred = torch.full((4, 45), 0.1)
    far_pred = torch.full((4, 45), 5.0)
    assert gaussian_nll(close_pred, logvar, target) < gaussian_nll(far_pred, logvar, target)


def test_dynamics_loss_runs_and_backprops():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    x = torch.randn(4, 30, 45)
    y_future = torch.randn(4, 6, 45)
    loss = dynamics_loss(model, x, y_future, K=6, horizon_discount=0.85, teacher_forcing_p=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_dynamics_loss_with_zero_teacher_forcing_uses_own_predictions():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    x = torch.randn(4, 30, 45)
    y_future = torch.randn(4, 6, 45)
    loss = dynamics_loss(model, x, y_future, K=6, horizon_discount=0.85, teacher_forcing_p=0.0)
    assert torch.isfinite(loss)


def test_teacher_forcing_schedule_anneals_start_to_end():
    assert teacher_forcing_schedule(0, 60, 1.0, 0.3, 0.6) == 1.0
    assert teacher_forcing_schedule(36, 60, 1.0, 0.3, 0.6) == 0.3  # at/after anneal boundary
    mid = teacher_forcing_schedule(18, 60, 1.0, 0.3, 0.6)
    assert 0.3 < mid < 1.0


def test_risk_head_loss_runs():
    logits = torch.randn(10, 1)
    labels = torch.randint(0, 2, (10,))
    loss = risk_head_loss(logits, labels)
    assert torch.isfinite(loss)


def test_stage_head_loss_runs():
    logits = torch.randn(10, 6)
    labels = torch.randint(0, 6, (10,))
    loss = stage_head_loss(logits, labels)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Δ=60 rebuild: masked features, β-NLL, MSE auxiliary, sample weights,
# free-running validation metrics
# ---------------------------------------------------------------------------

def _fixed_model():
    torch.manual_seed(0)
    return WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64).eval()


def test_feature_mask_excludes_dropped_dims_from_the_loss():
    model = _fixed_model()
    torch.manual_seed(1)
    x = torch.randn(4, 30, 45)
    y = torch.randn(4, 6, 45)
    mask = torch.ones(45, dtype=torch.bool)
    mask[:5] = False
    # corrupt the masked dims of the target wildly: a masked loss must not move
    # (teacher_forcing_p=0 so the target never re-enters the encoder; in the
    # real pipeline the scaler zeroes dropped dims in both x and y anyway)
    y_bad = y.clone()
    y_bad[:, :, :5] = 1e3
    a = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=0.0, feature_mask=mask)
    b = dynamics_loss(model, x, y_bad, K=6, horizon_discount=0.85, teacher_forcing_p=0.0, feature_mask=mask)
    assert torch.isclose(a, b)
    c = dynamics_loss(model, x, y_bad, K=6, horizon_discount=0.85, teacher_forcing_p=0.0)
    assert c > a


def test_beta_nll_zero_matches_plain_nll_and_positive_beta_changes_it():
    model = _fixed_model()
    torch.manual_seed(3)
    x = torch.randn(4, 30, 45)
    y = torch.randn(4, 6, 45)
    torch.manual_seed(4)
    plain = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0)
    torch.manual_seed(4)
    beta0 = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0, beta_nll=0.0)
    torch.manual_seed(4)
    beta5 = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0, beta_nll=0.5)
    assert torch.isclose(plain, beta0)
    assert not torch.isclose(plain, beta5)


def test_mse_auxiliary_adds_a_nonnegative_term():
    model = _fixed_model()
    torch.manual_seed(5)
    x = torch.randn(4, 30, 45)
    y = torch.randn(4, 6, 45)
    torch.manual_seed(6)
    base = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0)
    torch.manual_seed(6)
    aux = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0, mse_aux_weight=1.0)
    assert aux > base


def test_sample_weights_reweight_the_batch():
    model = _fixed_model()
    torch.manual_seed(7)
    x = torch.randn(4, 30, 45)
    y = torch.randn(4, 6, 45)
    y[0] += 50.0   # one sample with a huge error
    w_focus = torch.tensor([10.0, 1.0, 1.0, 1.0])
    w_ignore = torch.tensor([0.0, 1.0, 1.0, 1.0])
    torch.manual_seed(8)
    focus = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0, sample_weight=w_focus)
    torch.manual_seed(8)
    ignore = dynamics_loss(model, x, y, K=6, horizon_discount=0.85, teacher_forcing_p=1.0, sample_weight=w_ignore)
    assert focus > ignore


def test_free_running_metrics_have_per_horizon_shape_and_persistence_reference():
    from nidra.train.losses import free_running_metrics
    model = _fixed_model()
    torch.manual_seed(9)
    x = torch.randn(8, 30, 45)
    # a future that IS persistence: the persistence MSE must be exactly zero
    y = x[:, -1:, :].repeat(1, 6, 1)
    out = free_running_metrics(model, x, y, K=6)
    for key in ("nll", "mse", "mse_persistence", "coverage90"):
        assert out[key].shape == (6,)
        assert torch.isfinite(out[key]).all()
    assert torch.allclose(out["mse_persistence"], torch.zeros(6))
    assert (out["coverage90"] >= 0).all() and (out["coverage90"] <= 1).all()


def test_free_running_metrics_respect_the_feature_mask():
    from nidra.train.losses import free_running_metrics
    model = _fixed_model()
    torch.manual_seed(10)
    x = torch.randn(8, 30, 45)
    y = torch.randn(8, 6, 45)
    mask = torch.ones(45, dtype=torch.bool)
    mask[:10] = False
    y_bad = y.clone()
    y_bad[:, :, :10] = 1e3
    a = free_running_metrics(model, x, y, K=6, feature_mask=mask)
    b = free_running_metrics(model, x, y_bad, K=6, feature_mask=mask)
    assert torch.allclose(a["mse"], b["mse"]) and torch.allclose(a["nll"], b["nll"])

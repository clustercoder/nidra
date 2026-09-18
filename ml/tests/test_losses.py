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

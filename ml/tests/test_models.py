import torch

from nidra.models.encoder import Encoder
from nidra.models.heads import RiskHead, StageHead
from nidra.models.transition import Transition
from nidra.models.world_model import WorldModel


def test_encoder_output_shapes():
    enc = Encoder(n_features=45, hidden_size=128, num_layers=2)
    x = torch.randn(4, 30, 45)
    h_t, h = enc(x)
    assert h_t.shape == (4, 128)
    assert h.shape == (2, 4, 128)


def test_encoder_single_step_reentry_shape():
    enc = Encoder(n_features=45, hidden_size=128, num_layers=2)
    x = torch.randn(4, 30, 45)
    h_t, h = enc(x)
    nxt = torch.randn(4, 45)
    h_t2, h2 = enc(nxt.unsqueeze(1), h)
    assert h_t2.shape == (4, 128)
    assert h2.shape == (2, 4, 128)


def test_transition_logvar_is_clamped():
    trans = Transition(hidden_size=128, n_features=45, logvar_min=-6.0, logvar_max=3.0)
    h = torch.randn(8, 128) * 1000  # extreme input to try to blow past the clamp
    mu, logvar = trans(h)
    assert mu.shape == (8, 45)
    assert logvar.shape == (8, 45)
    assert (logvar >= -6.0).all()
    assert (logvar <= 3.0).all()


def test_risk_and_stage_head_shapes():
    risk = RiskHead(n_features=45, hidden=64)
    stage = StageHead(n_features=45, hidden=64, n_stages=6)
    s = torch.randn(10, 45)
    assert risk(s).shape == (10, 1)
    assert stage(s).shape == (10, 6)


def test_world_model_rollout_output_shape_and_no_nan():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    x = torch.randn(3, 30, 45)
    out = model.rollout(x, K=6, n_samples=5, stochastic=True)
    assert out.states.shape == (3, 5, 6, 45)
    assert not torch.isnan(out.states).any()
    assert not torch.isinf(out.states).any()


def test_world_model_rollout_deterministic_reproducible_with_stochastic_false():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    model.eval()
    x = torch.randn(2, 30, 45)
    out1 = model.rollout(x, K=6, n_samples=1, stochastic=False)
    out2 = model.rollout(x, K=6, n_samples=1, stochastic=False)
    torch.testing.assert_close(out1.states, out2.states)


def test_world_model_rollout_length_equals_K():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    x = torch.randn(2, 30, 45)
    out = model.rollout(x, K=6, n_samples=1)
    assert out.states.shape[2] == 6


def test_world_model_rollout_predicts_delta_not_absolute():
    """Zeroing the transition's mu weights should make the rollout converge
    to persistence (S_hat[t+1] ~= S[t] + noise), proving the head predicts
    a delta added to the current state rather than an absolute value."""
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    with torch.no_grad():
        model.transition.mu_head.weight.zero_()
        model.transition.mu_head.bias.zero_()
        model.transition.logvar_head.weight.zero_()
        model.transition.logvar_head.bias.fill_(-6.0)  # near-zero variance
    x = torch.randn(2, 30, 45)
    out = model.rollout(x, K=3, n_samples=1, stochastic=False)
    last_observed = x[:, -1, :]
    for k in range(3):
        torch.testing.assert_close(out.states[:, 0, k, :], last_observed, atol=1e-5, rtol=1e-5)


def test_freeze_dynamics_stops_encoder_and_transition_grad():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    model.freeze_dynamics()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.transition.parameters())
    assert all(p.requires_grad for p in model.risk_head.parameters())


def test_freeze_all_stops_every_parameter():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    model.freeze_all()
    assert all(not p.requires_grad for p in model.parameters())


def test_score_states_output_ranges():
    model = WorldModel(n_features=45, hidden_size=32, transition_mlp_hidden=64)
    states = torch.randn(4, 6, 45)
    risk, stage = model.score_states(states)
    assert risk.shape == (4, 6)
    assert stage.shape == (4, 6, 6)
    assert (risk >= 0).all() and (risk <= 1).all()
    torch.testing.assert_close(stage.sum(dim=-1), torch.ones(4, 6))

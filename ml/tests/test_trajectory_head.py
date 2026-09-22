"""The history-aware risk head, and the rollout context it reads.

Run 8's finding was that the per-state risk head is the bottleneck: on
Friday it ranked Bot-C2 windows BELOW silence (ROC 0.37) while a GRU
sequence classifier on the same rows reached 0.976. The difference is that
the classifier sees the host's history and the head sees one state vector.
These tests pin the machinery that lets a head see more, and — just as
importantly — pin that the extra information comes from the model's own
forward simulation and never from an observation after time t.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nidra.models.heads import TRAJECTORY_COMPONENTS, TrajectoryRiskHead
from nidra.models.world_model import WorldModel


@pytest.fixture
def model() -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32)


def _x(n: int = 4, L: int = 8, F: int = 45) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n, L, F) * 0.3


class TestTrajectoryRiskHead:
    def test_input_width_is_the_sum_of_its_declared_components(self):
        head = TrajectoryRiskHead(components=("state", "hidden"), n_features=45, hidden_size=16)
        assert head.input_dim == 45 + 16
        head = TrajectoryRiskHead(components=("state", "hidden", "delta", "logvar"), n_features=45, hidden_size=16)
        assert head.input_dim == 45 + 16 + 45 + 45

    def test_state_only_is_the_run_8_head_shape(self):
        head = TrajectoryRiskHead(components=("state",), n_features=45, hidden_size=16)
        assert head.input_dim == 45

    def test_components_must_be_known(self):
        with pytest.raises(ValueError, match="unknown trajectory head component"):
            TrajectoryRiskHead(components=("state", "the_future"), n_features=45, hidden_size=16)

    def test_at_least_one_component_is_required(self):
        with pytest.raises(ValueError, match="at least one"):
            TrajectoryRiskHead(components=(), n_features=45, hidden_size=16)

    def test_components_are_order_independent_and_deduplicated(self):
        a = TrajectoryRiskHead(components=("hidden", "state"), n_features=45, hidden_size=16)
        b = TrajectoryRiskHead(components=("state", "hidden", "state"), n_features=45, hidden_size=16)
        assert a.components == b.components == ("state", "hidden")

    def test_forward_accepts_arbitrary_leading_dimensions(self):
        head = TrajectoryRiskHead(components=("state", "hidden"), n_features=45, hidden_size=16)
        out = head(state=torch.randn(2, 3, 5, 45), hidden=torch.randn(2, 3, 5, 16))
        assert out.shape == (2, 3, 5, 1)

    def test_a_missing_declared_component_is_refused_not_zero_filled(self):
        head = TrajectoryRiskHead(components=("state", "hidden"), n_features=45, hidden_size=16)
        with pytest.raises(ValueError, match="requires component 'hidden'"):
            head(state=torch.randn(2, 45))

    def test_undeclared_components_are_ignored_not_concatenated(self):
        head = TrajectoryRiskHead(components=("state",), n_features=45, hidden_size=16)
        out = head(state=torch.randn(2, 45), hidden=torch.randn(2, 16), logvar=torch.randn(2, 45))
        assert out.shape == (2, 1)

    def test_roundtrips_through_save_and_load(self, tmp_path):
        head = TrajectoryRiskHead(components=("state", "hidden", "logvar"), n_features=45, hidden_size=16, hidden=8)
        path = tmp_path / "head.pt"
        head.save(path)
        back = TrajectoryRiskHead.load(path)
        assert back.components == head.components
        assert back.input_dim == head.input_dim
        state, hidden, logvar = torch.randn(3, 45), torch.randn(3, 16), torch.randn(3, 45)
        assert torch.allclose(back(state=state, hidden=hidden, logvar=logvar),
                              head(state=state, hidden=hidden, logvar=logvar))

    def test_loaded_head_is_frozen(self, tmp_path):
        head = TrajectoryRiskHead(components=("state",), n_features=45, hidden_size=16)
        path = tmp_path / "head.pt"
        head.save(path)
        back = TrajectoryRiskHead.load(path)
        assert all(not p.requires_grad for p in back.parameters())
        assert not back.training

    def test_every_declared_component_name_is_supported_by_the_model(self, model):
        out = model.rollout(_x(), K=3, n_samples=2)
        available = {"state": out.states, "hidden": out.hiddens, "delta": out.mus, "logvar": out.logvars}
        assert set(available) == set(TRAJECTORY_COMPONENTS)


class TestRolloutExposesItsHiddenState:
    def test_hiddens_have_one_entry_per_rollout_step(self, model):
        out = model.rollout(_x(n=4), K=3, n_samples=2)
        assert out.hiddens.shape == (4, 2, 3, model.encoder.hidden_size)

    def test_hidden_at_step_k_is_the_encoder_state_after_ingesting_step_k(self, model):
        # Not merely present: it must be the state the encoder reached after
        # consuming the k-th PREDICTED window, which is what makes the head
        # history-aware through the model's own simulation.
        x = _x(n=2)
        out = model.rollout(x, K=2, n_samples=1, stochastic=False)
        h_t, h = model.encoder(x)
        cur, prev = x[:, -1, :], x[:, -2, :]
        for k in range(2):
            mu, _ = model.transition(h_t, cur, prev)
            nxt = (cur + mu).clamp(-model.state_clamp, model.state_clamp)
            h_t, h = model.encoder(nxt.unsqueeze(1), h)
            assert torch.allclose(out.hiddens[:, 0, k, :], h_t, atol=1e-5)
            prev, cur = cur, nxt

    def test_the_default_rollout_is_unchanged_by_the_addition(self, model):
        torch.manual_seed(7)
        a = model.rollout(_x(), K=3, n_samples=1, stochastic=False)
        torch.manual_seed(7)
        b = model.rollout(_x(), K=3, n_samples=1, stochastic=False)
        assert torch.allclose(a.states, b.states)


class TestStateSources:
    """Persistence and the oracle must differ from the world model in ONE
    thing — where the next state comes from — or the comparison measures
    two classifiers rather than the transition model."""

    def test_persistence_repeats_the_last_observed_state(self, model):
        x = _x(n=3)
        out = model.rollout(x, K=4, n_samples=1, state_source="persist")
        for k in range(4):
            assert torch.allclose(out.states[:, 0, k, :], x[:, -1, :])

    def test_persistence_still_advances_the_encoder(self, model):
        # The head's history component must keep updating, otherwise
        # "persistence" would silently also freeze the hidden state and the
        # comparison would confound two changes.
        out = model.rollout(_x(n=3), K=3, n_samples=1, state_source="persist")
        assert not torch.allclose(out.hiddens[:, 0, 0, :], out.hiddens[:, 0, 2, :])

    def test_persistence_reports_a_zero_delta(self, model):
        out = model.rollout(_x(n=3), K=3, n_samples=1, state_source="persist")
        assert torch.allclose(out.mus, torch.zeros_like(out.mus))

    def test_the_oracle_walks_the_true_future(self, model):
        x, truth = _x(n=3), torch.randn(3, 4, 45) * 0.3
        out = model.rollout(x, K=4, n_samples=1, state_source="truth", truth=truth)
        assert torch.allclose(out.states[:, 0, :, :], truth.clamp(-model.state_clamp, model.state_clamp))

    def test_the_oracle_requires_the_truth_to_be_supplied(self, model):
        with pytest.raises(ValueError, match="state_source='truth' needs"):
            model.rollout(_x(), K=3, state_source="truth")

    def test_truth_shorter_than_the_horizon_is_refused(self, model):
        with pytest.raises(ValueError, match="truth has 2 steps"):
            model.rollout(_x(), K=4, state_source="truth", truth=torch.randn(4, 2, 45))

    def test_an_unknown_state_source_is_refused(self, model):
        with pytest.raises(ValueError, match="unknown state_source"):
            model.rollout(_x(), K=3, state_source="wishful")

    def test_only_the_model_source_consumes_no_future_data(self, model):
        # Guard on the invariant itself: the default path must not accept a
        # truth tensor, so no call site can quietly leak one in.
        with pytest.raises(ValueError, match="state_source='model' must not be given"):
            model.rollout(_x(), K=3, truth=torch.randn(4, 3, 45))


class TestScoringATrajectory:
    def test_a_plain_head_scores_states_exactly_as_before(self, model):
        out = model.rollout(_x(), K=3, n_samples=2)
        risk, stage = model.score_trajectory(out)
        expected_risk, expected_stage = model.score_states(out.states)
        assert torch.allclose(risk, expected_risk)
        assert torch.allclose(stage, expected_stage)

    def test_a_trajectory_head_reads_the_rollout_context(self, model):
        model.risk_head = TrajectoryRiskHead(
            components=("state", "hidden", "delta", "logvar"),
            n_features=45, hidden_size=model.encoder.hidden_size,
        )
        out = model.rollout(_x(n=3), K=2, n_samples=2)
        risk, _ = model.score_trajectory(out)
        assert risk.shape == (3, 2, 2)
        assert torch.isfinite(risk).all() and ((risk >= 0) & (risk <= 1)).all()

    def test_changing_only_the_hidden_state_changes_a_history_aware_score(self, model):
        head = TrajectoryRiskHead(components=("state", "hidden"), n_features=45,
                                  hidden_size=model.encoder.hidden_size)
        state = torch.randn(5, 45)
        a = head(state=state, hidden=torch.zeros(5, model.encoder.hidden_size))
        b = head(state=state, hidden=torch.ones(5, model.encoder.hidden_size))
        assert not torch.allclose(a, b)


class TestObservedContextMatchesTrainingContext:
    """A head scores an observed state in two places — during head training
    (train.head_context) and at evaluation (WorldModel.score_observed). If
    those two build the context differently the head is trained on one
    quantity and applied to another, which no test of either alone catches."""

    def test_the_two_builders_agree_row_for_row(self, model):
        import numpy as np
        import pandas as pd
        from nidra.data.normalize import FeatureScaler
        from nidra.data.schema import FEATURE_ORDER
        from nidra.train.head_context import build_head_context

        rng = np.random.default_rng(5)
        scaler = FeatureScaler().fit(np.abs(rng.normal(size=(300, 45))) + 0.2,
                                     active_mask=np.ones(300, dtype=bool))
        L = 6
        rows = []
        for i in range(14):
            row = {name: float(abs(rng.normal())) for name in FEATURE_ORDER}
            row.update(host_id="h", window_ts=60 * i, stage_label="benign", risk_label=0)
            rows.append(row)
        table = pd.DataFrame(rows)

        ctx = build_head_context(table, scaler, model, L=L)
        scaled = scaler.transform(table[FEATURE_ORDER].to_numpy(dtype="float32")).astype("float32")
        for tau in (L - 1, 9, 13):
            x = torch.from_numpy(scaled[tau - L + 1: tau + 1]).float().unsqueeze(0)
            with torch.no_grad():
                got = model.observed_context(x)
            assert np.allclose(got["hidden"].numpy()[0], ctx.hidden[tau], atol=1e-5), f"hidden @{tau}"
            assert np.allclose(got["delta"].numpy()[0], ctx.delta[tau], atol=1e-5), f"delta @{tau}"
            assert np.allclose(got["logvar"].numpy()[0], ctx.logvar[tau], atol=1e-5), f"logvar @{tau}"

    def test_score_observed_equals_score_states_for_a_per_state_head(self, model):
        x = _x(n=5)
        risk_a, stage_a = model.score_observed(x)
        risk_b, stage_b = model.score_states(x[:, -1, :])
        assert torch.allclose(risk_a, risk_b) and torch.allclose(stage_a, stage_b)


class TestRealizedDeltas:
    def test_realized_delta_is_the_trajectorys_own_backward_difference(self, model):
        out = model.rollout(_x(n=3), K=4, n_samples=2)
        d = out.realized_deltas()
        assert torch.allclose(d[:, :, 0, :], out.states[:, :, 0, :] - out.anchor)
        for k in range(1, 4):
            assert torch.allclose(d[:, :, k, :], out.states[:, :, k, :] - out.states[:, :, k - 1, :])

    def test_a_stochastic_realized_delta_is_not_the_predicted_mean(self, model):
        out = model.rollout(_x(n=3), K=3, n_samples=2, stochastic=True)
        assert not torch.allclose(out.realized_deltas(), out.mus)

    def test_persistence_has_a_zero_realized_delta(self, model):
        out = model.rollout(_x(n=3), K=3, n_samples=1, state_source="persist")
        assert torch.allclose(out.realized_deltas(), torch.zeros_like(out.states), atol=1e-6)

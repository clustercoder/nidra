"""The onset head reads the same context the risk head can.

Advance warning is the claim the project has least evidence for: on CTU-13's
validation captures the state-only onset head scored within noise of zero at
every horizon while the oracle reached 0.022, and 0 of 8 episodes were warned
before onset. The risk-head ablation on the same data says what is missing —
the encoder's hidden state raised validation AP from 0.353 to 0.496 — so the
onset head is given the same option, under the same frozen-head discipline.

`components=("state",)` must stay byte-compatible with the heads already
trained, or the comparison is against a moving baseline.
"""

from __future__ import annotations

import pytest
import torch

from nidra.models.heads import OnsetHead, component_widths, concat_components


class TestSharedComponentConcatenation:
    def test_widths_follow_the_declared_order_not_the_call_order(self):
        assert component_widths(("hidden", "state"), n_features=45, hidden_size=128) == [45, 128]

    def test_an_unknown_component_is_refused(self):
        with pytest.raises(ValueError, match="frobnicate"):
            component_widths(("state", "frobnicate"), 45, 128)

    def test_concatenation_is_in_declared_order(self):
        parts = {"state": torch.zeros(2, 45), "hidden": torch.ones(2, 128)}
        out = concat_components(("hidden", "state"), parts)
        assert out.shape == (2, 173)
        assert out[0, 0].item() == 0.0 and out[0, 44].item() == 0.0    # state first
        assert out[0, 45].item() == 1.0

    def test_a_missing_component_is_an_error_never_a_zero_fill(self):
        with pytest.raises(ValueError, match="hidden"):
            concat_components(("state", "hidden"), {"state": torch.zeros(2, 45)})


class TestOnsetHeadComponents:
    def test_the_default_is_the_state_only_head(self):
        head = OnsetHead(45, 8, (1, 5))
        assert head.components == ("state",)
        assert head.net[0].in_features == 45

    def test_a_state_only_head_still_accepts_a_bare_tensor(self):
        """Every existing call site passes a tensor positionally."""
        head = OnsetHead(45, 8, (1, 5))
        assert head(torch.zeros(3, 45)).shape == (3, 2)

    def test_a_history_aware_head_widens_the_input(self):
        head = OnsetHead(45, 8, (1, 5), components=("state", "hidden"), hidden_size=64)
        assert head.net[0].in_features == 109
        assert head(state=torch.zeros(3, 45), hidden=torch.zeros(3, 64)).shape == (3, 2)

    def test_it_refuses_a_bare_tensor_when_it_needs_context(self):
        head = OnsetHead(45, 8, (1, 5), components=("state", "hidden"), hidden_size=64)
        with pytest.raises(ValueError, match="hidden"):
            head(torch.zeros(3, 45))

    def test_hidden_only_needs_no_state(self):
        head = OnsetHead(45, 8, (1, 5), components=("hidden",), hidden_size=64)
        assert head(hidden=torch.zeros(3, 64)).shape == (3, 2)


class TestRoundTrip:
    def test_a_history_aware_head_reloads_into_the_same_architecture(self, tmp_path):
        head = OnsetHead(45, 8, (1, 5, 30), components=("state", "hidden", "logvar"), hidden_size=64)
        path = tmp_path / "onset.pt"
        head.save(path, parameterisation="hazard")
        back = OnsetHead.load(path)
        assert back.components == ("state", "hidden", "logvar")
        assert back.hidden_size == 64
        assert back.parameterisation == "hazard"
        assert back.horizons_min == (1, 5, 30)
        a = head(state=torch.zeros(2, 45), hidden=torch.zeros(2, 64), logvar=torch.zeros(2, 45))
        b = back(state=torch.zeros(2, 45), hidden=torch.zeros(2, 64), logvar=torch.zeros(2, 45))
        assert torch.allclose(a, b)

    def test_it_is_frozen_on_load(self, tmp_path):
        head = OnsetHead(45, 8, (1, 5), components=("state", "hidden"), hidden_size=64)
        path = tmp_path / "onset.pt"
        head.save(path)
        assert not any(p.requires_grad for p in OnsetHead.load(path).parameters())

    def test_a_checkpoint_without_components_loads_as_state_only(self, tmp_path):
        """Heads saved before this change carry no `components` key."""
        path = tmp_path / "old.pt"
        old = OnsetHead(45, 8, (1, 5))
        torch.save({"state_dict": old.state_dict(), "n_features": 45, "hidden": 8,
                    "horizons_min": (1, 5), "parameterisation": "independent"}, path)
        back = OnsetHead.load(path)
        assert back.components == ("state",)
        assert torch.allclose(back(torch.zeros(2, 45)), old(torch.zeros(2, 45)))


class TestTheRiskHeadUsesTheSameMachinery:
    def test_trajectory_risk_head_input_dim_matches_the_shared_width_table(self):
        from nidra.models.heads import TrajectoryRiskHead
        head = TrajectoryRiskHead(("state", "hidden", "delta", "logvar"), 45, 128, 8)
        assert head.input_dim == sum(component_widths(head.components, 45, 128))

"""The rollout must not manufacture mass in features the scaler dropped.

`train/losses.py` masks dropped (constant/duplicate) features out of the
transition loss, so the transition network receives NO gradient on those
output dimensions. Whatever it emits there is untrained. The rollout feeds its
own output back in, so an untrained non-zero drift compounds with every step
and lands in the state the frozen risk head reads — slots where that head has
itself never seen a non-zero input, and so has untrained weights too.

Measured before the fix on `comb2cic_state+hidden` (13 dropped of 45): the
dropped slots grow from rms 0.58 at k=1 to 2.03 at k=6 (abs max 6.21 against a
state clamp of 10) while the input and the true future hold them at exactly
zero and the real features stay flat at rms ~0.69.

The contract these tests pin: a dropped slot is zero in the input, zero in the
truth, and must stay zero everywhere the rollout produces.
"""

import pytest
import torch

from nidra.models.world_model import WorldModel


def _model(n_features: int = 8) -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(n_features=n_features, hidden_size=16, encoder_layers=1,
                      transition_mlp_hidden=16, risk_hidden=8, stage_hidden=8)


def _history(n_rows: int, L: int, n_features: int, dropped: list[int]) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.randn(n_rows, L, n_features)
    x[:, :, dropped] = 0.0          # a dropped feature is zeroed, never removed
    return x


class TestDroppedSlotsStayZero:
    def test_rollout_keeps_dropped_slots_at_zero(self):
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(_history(4, 3, 8, dropped), K=6, n_samples=1, stochastic=False)
        assert out.states[..., dropped].abs().max().item() == 0.0

    def test_holds_under_sampling(self):
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(_history(4, 3, 8, dropped), K=6, n_samples=8, stochastic=True)
        assert out.states[..., dropped].abs().max().item() == 0.0

    def test_predicted_delta_is_masked_too(self):
        """`mu` reaches a `state+delta` head, so it carries the same hazard."""
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(_history(4, 3, 8, dropped), K=6, n_samples=1, stochastic=False)
        assert out.mus[..., dropped].abs().max().item() == 0.0

    def test_kept_slots_are_untouched_by_the_mask(self):
        """The fix must not quietly change the forecast it is meant to leave alone."""
        dropped = [2, 5]
        kept = [i for i in range(8) if i not in dropped]
        x = _history(4, 3, 8, dropped)
        unmasked = _model().rollout(x, K=6, n_samples=1, stochastic=False)
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        masked = m.rollout(x, K=6, n_samples=1, stochastic=False)
        # Step 1 is produced from an identical input state, so the kept slots
        # must agree exactly; later steps legitimately diverge because the fed
        # back state differs.
        torch.testing.assert_close(masked.states[:, :, 0, kept], unmasked.states[:, :, 0, kept])

    def test_without_a_mask_behaviour_is_unchanged(self):
        """Recorded runs must stay reproducible: no mask set == the old path."""
        x = _history(4, 3, 8, [])
        a = _model().rollout(x, K=6, n_samples=1, stochastic=False)
        b = _model().rollout(x, K=6, n_samples=1, stochastic=False)
        torch.testing.assert_close(a.states, b.states)
        assert _model().feature_mask is None

    def test_mask_is_not_a_checkpoint_entry(self):
        """The mask belongs to the scaler, not the weights: adding it must not
        break loading a checkpoint trained before it existed."""
        m = _model()
        before = set(m.state_dict())
        m.set_feature_mask([i not in (2, 5) for i in range(8)])
        assert set(m.state_dict()) == before

    def test_mask_length_is_validated(self):
        m = _model()
        try:
            m.set_feature_mask([True] * 7)
        except ValueError as e:
            assert "7" in str(e) and "8" in str(e)
        else:
            raise AssertionError("a wrong-length mask must be refused")


class TestOracleAndPersistenceAgree:
    def test_truth_path_is_already_zero_there(self):
        """The oracle reads observed states, which hold dropped slots at zero;
        the mask must be a no-op for it, not a correction."""
        dropped = [2, 5]
        x = _history(4, 3, 8, dropped)
        truth = torch.randn(4, 6, 8)
        truth[:, :, dropped] = 0.0
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(x, K=6, n_samples=1, stochastic=False, state_source="truth", truth=truth)
        torch.testing.assert_close(out.states[:, 0], truth)

    def test_persist_path_stays_at_the_anchor(self):
        dropped = [2, 5]
        x = _history(4, 3, 8, dropped)
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(x, K=6, n_samples=1, stochastic=False, state_source="persist")
        for k in range(6):
            torch.testing.assert_close(out.states[:, 0, k], x[:, -1, :])


class TestLogvarInDroppedSlots:
    """`logvar` reaches a `state+logvar` head, which is the most interesting
    head in §3.22 precisely because it is decomposable into the 45 named
    features. Leaving 13 of those 45 carrying an untrained value would put the
    contamination straight into the conclusion that head supports.

    It is NOT zeroed: a log-variance of 0 asserts unit variance, which is a
    claim, and a loud one at the scale the head reads. It is set to
    `logvar_min`, the floor the transition already clamps to — "this slot does
    not vary", which is what a dropped feature means.
    """

    def test_logvar_is_pinned_to_the_floor_not_to_zero(self):
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(_history(4, 3, 8, dropped), K=6, n_samples=1, stochastic=False)
        lv = out.logvars[..., dropped]
        assert lv.min().item() == lv.max().item() == m.transition.logvar_min

    def test_kept_logvars_are_untouched(self):
        dropped = [2, 5]
        kept = [i for i in range(8) if i not in dropped]
        x = _history(4, 3, 8, dropped)
        unmasked = _model().rollout(x, K=6, n_samples=1, stochastic=False)
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        masked = m.rollout(x, K=6, n_samples=1, stochastic=False)
        torch.testing.assert_close(masked.logvars[:, :, 0, kept], unmasked.logvars[:, :, 0, kept])

    def test_sampling_in_a_dropped_slot_cannot_reach_the_state(self):
        """Pinning logvar changes the noise drawn there; the state must not
        notice, because the mask is applied after the noise is added."""
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        out = m.rollout(_history(4, 3, 8, dropped), K=6, n_samples=16, stochastic=True)
        assert out.states[..., dropped].abs().max().item() == 0.0


class TestHeadContextOutsideTheRollout:
    """`rollout()` is not the only path that hands a head a transition output.

    `observed_context` (serving, `score_observed`) and
    `train/head_context.py` (head TRAINING) both call `transition()` directly
    and take its `logvar`. A `state+logvar` head therefore reads 13 untrained
    log-variances in the `cross_core` regime at training time *and* at serving
    time, which is the number §3.38 turns on. Masking only inside the rollout
    would leave that conclusion contaminated.

    `state`, `hidden` and `delta` come from observed data, whose dropped slots
    are already exactly zero, so only `logvar` needs the treatment here.
    """

    def test_observed_context_pins_logvar_in_dropped_slots(self):
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        ctx = m.observed_context(_history(4, 3, 8, dropped))
        lv = ctx["logvar"][:, dropped]
        assert lv.min().item() == lv.max().item() == m.transition.logvar_min

    def test_observed_context_leaves_kept_logvars_alone(self):
        dropped = [2, 5]
        kept = [i for i in range(8) if i not in dropped]
        x = _history(4, 3, 8, dropped)
        plain = _model().observed_context(x)
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        torch.testing.assert_close(m.observed_context(x)["logvar"][:, kept], plain["logvar"][:, kept])

    def test_observed_context_state_delta_hidden_are_untouched(self):
        dropped = [2, 5]
        x = _history(4, 3, 8, dropped)
        plain = _model().observed_context(x)
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        got = m.observed_context(x)
        for key in ("state", "hidden", "delta"):
            torch.testing.assert_close(got[key], plain[key])

    def test_observed_delta_was_already_clean(self):
        """Stated so the scope of the defect stays on the record: `delta` is a
        backward difference of two observed states, so it never carried the
        contamination that `logvar` did."""
        dropped = [2, 5]
        m = _model()
        m.set_feature_mask([i not in dropped for i in range(8)])
        ctx = m.observed_context(_history(4, 3, 8, dropped))
        assert ctx["delta"][:, dropped].abs().max().item() == 0.0

    def test_mask_logvar_is_a_no_op_without_a_mask(self):
        m = _model()
        lv = torch.randn(4, 8)
        torch.testing.assert_close(m.mask_logvar(lv), lv)


class TestServedPathCarriesTheMask:
    """The benchmark sets the mask in `load_models`; the served path loads its
    own ensemble and would otherwise reproduce the phantom drift in production,
    which is the one place it would go unmeasured.
    """

    def test_every_served_model_carries_the_scalers_mask(self, trained_predictor):
        from nidra.data.schema import FEATURE_ORDER

        trained_predictor, _ = trained_predictor
        dropped = set(trained_predictor.scaler.dropped_features or [])
        expected = [f not in dropped for f in FEATURE_ORDER]
        for m in trained_predictor.models:
            if not dropped:
                assert m.feature_mask is None, "no dropped features means no mask, and the old path exactly"
            else:
                assert m.feature_mask is not None, "the served rollout would produce D145's phantom drift"
                assert [bool(v) for v in m.feature_mask.tolist()] == expected

    def test_a_served_rollout_holds_dropped_slots_at_zero(self, trained_predictor):
        """The end-to-end statement, independent of how the mask got there."""
        from nidra.data.schema import FEATURE_ORDER

        trained_predictor, _ = trained_predictor
        dropped = set(trained_predictor.scaler.dropped_features or [])
        if not dropped:
            pytest.skip("this fixture's scaler drops nothing; the regime cannot be exercised here")
        idx = [i for i, f in enumerate(FEATURE_ORDER) if f in dropped]
        m = trained_predictor.models[0]
        x = torch.zeros(2, trained_predictor.L, len(FEATURE_ORDER))
        out = m.rollout(x, K=trained_predictor.K, n_samples=4, stochastic=True)
        assert out.states[..., idx].abs().max().item() == 0.0

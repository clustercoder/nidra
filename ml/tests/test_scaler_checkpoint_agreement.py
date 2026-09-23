"""A model must be evaluated with the scaler it was trained with.

Found by a near-miss, not by design. A probe loaded `config/combined_eval_ctu.yaml`
without the queue's `--set artifacts.scaler_dir=...` override, which resolves to
the shared `ml/artifacts/scaler`. That directory had since been rewritten by a
later run with a DIFFERENT feature regime: 0 dropped features against the
combined model's 13. Nothing errored — shapes match, because a dropped feature is
zeroed rather than removed — and the cell's AP came out 2.9x different from its
recorded value.

The recorded results were all sound (the queue does override it). But the failure
mode is the one this project says is the dangerous one: it does not raise, and it
produces a plausible number.
"""

from __future__ import annotations

import json

import pytest

from nidra.eval.benchmark import assert_scaler_matches_checkpoint


def meta(dropped, **kw):
    return {"dropped_features": dropped, "features_kept": 45 - len(dropped), **kw}


class TestAgreement:
    def test_identical_drop_sets_pass(self):
        assert_scaler_matches_checkpoint(["ttl_mean", "iat_max"], meta(["iat_max", "ttl_mean"]), "seed_0")

    def test_order_does_not_matter(self):
        assert_scaler_matches_checkpoint(["b", "a"], meta(["a", "b"]), "seed_0")

    def test_a_scaler_dropping_nothing_against_a_model_trained_on_32_raises(self):
        """The exact near-miss: the shared scaler's full-45 regime against the
        combined model's cross_core."""
        with pytest.raises(ValueError, match="13"):
            assert_scaler_matches_checkpoint([], meta([f"f{i}" for i in range(13)]), "seed_0")

    def test_the_message_names_the_features_that_differ_not_just_the_count(self):
        with pytest.raises(ValueError) as e:
            assert_scaler_matches_checkpoint(["a"], meta(["a", "ttl_mean"]), "seed_0")
        assert "ttl_mean" in str(e.value)

    def test_the_message_names_the_checkpoint(self):
        with pytest.raises(ValueError, match="seed_3"):
            assert_scaler_matches_checkpoint([], meta(["a"]), "seed_3")

    def test_a_checkpoint_without_the_field_is_not_an_error(self):
        """Older checkpoints predate `dropped_features`. Refusing to evaluate
        them would break replaying Run 8 artifacts, which §29 forbids."""
        assert_scaler_matches_checkpoint(["a"], {"seed": "0"}, "seed_0")

    def test_no_metadata_at_all_is_not_an_error(self):
        assert_scaler_matches_checkpoint(["a"], None, "seed_0")

    def test_both_empty_passes(self):
        assert_scaler_matches_checkpoint([], meta([]), "seed_0")


# --- the same class of failure, on the head rather than the scaler ---

from nidra.eval.benchmark import assert_head_matches_checkpoint


class TestHeadAgreement:
    """Three `state+hidden` transfer arms produced no metrics at all because
    `config/cic2ctu.yaml` and friends do not declare
    `model.risk_head.components`, so the model was built with the 45-wide
    per-state head and the 173-wide checkpoint would not load. `load_state_dict`
    did raise — but the queue recorded `rc=0` (it was reading `tail`'s status),
    so six cells went missing from the cross-dataset matrix and the only symptom
    was gaps in the scorecard."""

    def test_agreeing_components_pass(self):
        assert_head_matches_checkpoint(("state", "hidden"),
                                       {"risk_head_components": "['state', 'hidden']"}, "seed_0")

    def test_a_per_state_config_against_a_history_aware_checkpoint_raises(self):
        with pytest.raises(ValueError, match="hidden"):
            assert_head_matches_checkpoint(("state",), {"risk_head_components": "['state', 'hidden']"}, "seed_0")

    def test_the_message_says_how_to_fix_it(self):
        with pytest.raises(ValueError, match="risk_head.components"):
            assert_head_matches_checkpoint(("state",), {"risk_head_components": "['state', 'hidden']"}, "seed_0")

    def test_a_real_list_is_accepted_as_well_as_its_repr(self):
        """run_experiment writes metadata values through str(), so the field
        arrives as "['state']" — but a hand-written or future record may hold a
        real list, and a guard that only understood one spelling would pass
        everything in the other."""
        assert_head_matches_checkpoint(("state",), {"risk_head_components": ["state"]}, "seed_0")
        with pytest.raises(ValueError):
            assert_head_matches_checkpoint(("state",), {"risk_head_components": ["state", "hidden"]}, "seed_0")

    def test_a_checkpoint_without_the_field_is_not_an_error(self):
        assert_head_matches_checkpoint(("state", "hidden"), {"seed": "0"}, "seed_0")
        assert_head_matches_checkpoint(("state",), None, "seed_0")

    def test_an_unparseable_field_is_not_an_error(self):
        """A guard that crashes on a malformed record is worse than one that
        declines to judge it — load_state_dict still raises on a real mismatch."""
        assert_head_matches_checkpoint(("state",), {"risk_head_components": "not a list"}, "seed_0")

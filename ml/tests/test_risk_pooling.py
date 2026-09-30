import pytest
import torch

from nidra.models.risk_pooling import pool_risk_over_samples


def test_mean_pooling_matches_torch_mean():
    risk = torch.rand(4, 20, 6)
    pooled = pool_risk_over_samples(risk, dim=1, method="mean")
    torch.testing.assert_close(pooled, risk.mean(dim=1))


def test_quantile_pooling_matches_torch_quantile():
    risk = torch.rand(4, 20, 6)
    pooled = pool_risk_over_samples(risk, dim=1, method="quantile", quantile=0.9)
    torch.testing.assert_close(pooled, risk.quantile(0.9, dim=1))


def test_quantile_pooling_is_at_least_mean_for_right_skewed_samples():
    # A handful of high-risk trajectories among many low-risk ones is exactly
    # the shape a stochastic rollout produces for a rare, correctly-detected
    # attack — upper-quantile pooling should not sit below the mean here.
    risk = torch.cat([torch.full((2, 15, 6), 0.05), torch.full((2, 5, 6), 0.95)], dim=1)
    mean_pooled = pool_risk_over_samples(risk, dim=1, method="mean")
    quantile_pooled = pool_risk_over_samples(risk, dim=1, method="quantile", quantile=0.9)
    assert (quantile_pooled >= mean_pooled - 1e-6).all()
    assert (quantile_pooled > mean_pooled + 0.1).all()


def test_unknown_method_raises():
    risk = torch.rand(2, 5, 6)
    with pytest.raises(ValueError):
        pool_risk_over_samples(risk, dim=1, method="max")


# --- Ensemble head-vs-trajectory reduction order -------------------------
# The pooled ensemble averages five heads' opinions on each trajectory BEFORE
# pooling trajectories, which partially re-smooths the tail quantile pooling
# exists to select. `head_reduction` makes that order explicit and testable.

from nidra.models.risk_pooling import pool_ensemble_risk  # noqa: E402


def _per_head_risk() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(5, 3, 20, 6)  # [M, B, S, K]


def test_before_pooling_reproduces_the_original_head_mean_then_pool_order():
    risk = _per_head_risk()
    pooled = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9)
    legacy = risk.mean(dim=0).quantile(0.9, dim=1)
    torch.testing.assert_close(pooled, legacy)


def test_the_two_orders_are_identical_under_mean_pooling():
    # A mean of means over fixed-size axes commutes, so head_reduction must be
    # a no-op here — this is what lets "mean" configs ignore the setting.
    risk = _per_head_risk()
    before = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="mean")
    after = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="mean",
                               head_reduction="after_pooling")
    torch.testing.assert_close(before, after)


def test_after_pooling_matches_pooling_each_head_then_soft_voting():
    risk = _per_head_risk()
    pooled = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9,
                                head_reduction="after_pooling")
    expected = risk.quantile(0.9, dim=2).mean(dim=0)
    torch.testing.assert_close(pooled, expected)


def test_the_orders_agree_when_every_head_ranks_the_same_trajectories_riskiest():
    # A quantile is a fixed linear combination of ORDER statistics, so as long
    # as averaging heads leaves the per-trajectory ranking intact, the two
    # orders coincide exactly — worth pinning down, because it says the choice
    # only matters for a specific, nameable kind of ensemble disagreement.
    risk = torch.full((3, 1, 10, 1), 0.02)
    risk[0, :, 8:, :] = 0.98                      # heads 1 and 2 stay flat
    before = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9)
    after = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9,
                               head_reduction="after_pooling")
    torch.testing.assert_close(before, after)


def test_after_pooling_preserves_the_tail_when_heads_disagree_about_which_futures_are_risky():
    # The case that actually separates the two orders: each head flags a
    # DIFFERENT pair of trajectories. Averaging heads first turns three
    # confident minority warnings into one diluted score on six trajectories
    # (0.98 -> 0.34) before the quantile ever runs; pooling each head's own
    # distribution first keeps each head's warning intact and then soft-votes.
    risk = torch.full((3, 1, 10, 1), 0.02)
    for head in range(3):
        risk[head, :, 2 * head : 2 * head + 2, :] = 0.98
    before = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9)
    after = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9,
                               head_reduction="after_pooling")
    assert (after > before + 0.5).all()
    torch.testing.assert_close(after, torch.full_like(after, 0.98))


def test_both_orders_return_the_same_shape():
    risk = _per_head_risk()
    before = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9)
    after = pool_ensemble_risk(risk, sample_dim=2, head_dim=0, method="quantile", quantile=0.9,
                               head_reduction="after_pooling")
    assert before.shape == after.shape == (3, 6)


def test_unknown_head_reduction_raises():
    with pytest.raises(ValueError):
        pool_ensemble_risk(_per_head_risk(), sample_dim=2, head_dim=0, head_reduction="whenever")

"""Risk-score pooling across sampled rollout trajectories.

Single place this aggregation decision is made, so eval (`eval/baselines.py`,
`eval/lead_time_runner.py`) and serving (`serve/predictor.py`) pool
identically — see REAL_DATA_RESULTS.md's calibration section for why this
matters: mean-pooling across trajectories was the largest single cause of
the mandated-0.75-threshold recall problem, independently of calibration.

Mean pooling across sampled futures answers "how risky is the average
imagined future" — if only a minority of sampled trajectories reach
compromise (very plausible for a stochastic rollout of a rare event), the
mean sits far below any individual risky trajectory's own score, which is
exactly why raw mean-pooled p_compromise for true positives measured well
under the mandated 0.75 alert threshold even where ranking (AUC-PR) was
good. Upper-quantile pooling answers "how risky is the riskier tail of
plausible futures" instead — still a well-defined statistic of the same
sampled distribution, not a different model or a lowered bar, and, being
monotonic in the underlying per-trajectory risk, it does not invent
separability that the ranking didn't already have.
"""

from __future__ import annotations

import torch

VALID_METHODS = ("mean", "quantile")


def pool_risk_over_samples(
    risk: torch.Tensor,
    dim: int,
    method: str = "mean",
    quantile: float = 0.9,
) -> torch.Tensor:
    """Reduces `risk`'s sample dimension `dim` away. `risk` is any shape
    with a rollout-trajectory dimension at `dim` (single-model sampling) or
    a pooled-trajectory dimension (ensemble sampling — see
    `eval.baselines.ensemble_world_model_forecast`, which concatenates every
    member's trajectories along this same axis before calling this).

    `method="mean"` reproduces the original behavior exactly (the default,
    for any call site not deliberately opting into quantile pooling).
    `method="quantile"` uses `quantile` (e.g. 0.9 = 90th percentile) instead.
    """
    if method == "mean":
        return risk.mean(dim=dim)
    if method == "quantile":
        return risk.quantile(quantile, dim=dim)
    raise ValueError(f"unknown risk pooling method {method!r}, expected one of {VALID_METHODS}")


VALID_HEAD_REDUCTIONS = ("before_pooling", "after_pooling")


def pool_ensemble_risk(
    per_head_risk: torch.Tensor,
    sample_dim: int,
    head_dim: int,
    method: str = "mean",
    quantile: float = 0.9,
    head_reduction: str = "before_pooling",
) -> torch.Tensor:
    """Reduces both the ensemble-head dimension (`head_dim`) and the
    rollout-trajectory dimension (`sample_dim`) of `per_head_risk` away.

    The head dimension is always reduced by a plain mean (standard soft
    voting); `head_reduction` only controls the ORDER of the two reductions,
    which matters whenever `method` is not itself a mean:

    - `"before_pooling"` (default, the original behavior): average the heads'
      opinions on each individual trajectory first, then pool trajectories.
      Head disagreement on a risky trajectory drags that trajectory's score
      toward the middle BEFORE the quantile can select it, so member
      averaging partially re-smooths the very tail quantile pooling exists
      to preserve.
    - `"after_pooling"`: pool each head's own trajectory distribution first,
      then average those per-head tail estimates. Each head keeps its own
      view of its riskier futures, and soft voting happens over those
      estimates instead of over individual trajectories.

    Under `method="mean"` the two orders are mathematically identical (a mean
    of means over a fixed-size axis), so this argument is a no-op there — see
    `tests/test_risk_pooling.py`.
    """
    if head_reduction not in VALID_HEAD_REDUCTIONS:
        raise ValueError(
            f"unknown head reduction {head_reduction!r}, expected one of {VALID_HEAD_REDUCTIONS}"
        )
    if head_reduction == "before_pooling":
        head_mean = per_head_risk.mean(dim=head_dim)
        pool_dim = sample_dim - 1 if sample_dim > head_dim else sample_dim
        return pool_risk_over_samples(head_mean, dim=pool_dim, method=method, quantile=quantile)
    pooled = pool_risk_over_samples(
        per_head_risk, dim=sample_dim, method=method, quantile=quantile
    )
    collapsed_head_dim = head_dim - 1 if head_dim > sample_dim else head_dim
    return pooled.mean(dim=collapsed_head_dim)

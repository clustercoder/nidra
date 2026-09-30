"""Stage-1 dynamics loss: multi-step unrolled Gaussian NLL with scheduled
sampling, and the horizon-discounted weighting that keeps distant, more
uncertain steps from dominating the gradient.

This loop OWNS gradients through the transition and encoder across K steps
— it is written directly (not via WorldModel.rollout, which is the
`torch.no_grad()` inference-time rollout used for sampling/serving/eval).

Options beyond the original objective (all off by default, all in config
under `train_dynamics`, each one a measured experiment — see
experiments/runs/):

  feature_mask     — bool [F]: dropped (constant/duplicate) features are
                     excluded from the loss instead of contributing a large
                     negative constant (logvar at its floor) that hides
                     changes in the real terms.
  beta_nll         — β-NLL (Seitzer et al. 2022): each element's NLL is
                     weighted by stop_grad(var)^β. β=0 is plain NLL; β>0
                     keeps high-variance features from being written off,
                     which is what the variance head otherwise learns to do
                     with the rarest, most informative transitions.
  mse_aux_weight   — adds λ · MSE(pred, target) so the mean is pulled toward
                     the target even where the variance head is confident.
  sample_weight    — [B] per-sample weight (e.g. up-weighting origins whose
                     horizon contains a state transition).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from nidra.models.world_model import WorldModel


def gaussian_nll(pred: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element diagonal-Gaussian negative log-likelihood, averaged over
    the batch and feature dims. logvar is assumed already clamped upstream
    (Transition.forward does this)."""
    var = logvar.exp()
    nll = 0.5 * (logvar + (pred - target) ** 2 / var)
    return nll.mean()


def gaussian_nll_elements(pred: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element NLL [B, F] — no reduction."""
    var = logvar.exp()
    return 0.5 * (logvar + (pred - target) ** 2 / var)


def _reduce(elements: torch.Tensor, feature_mask: torch.Tensor | None, sample_weight: torch.Tensor | None) -> torch.Tensor:
    """Mean over kept features, then (weighted) mean over the batch."""
    if feature_mask is not None:
        per_sample = (elements * feature_mask).sum(dim=-1) / feature_mask.sum().clamp_min(1.0)
    else:
        per_sample = elements.mean(dim=-1)
    if sample_weight is not None:
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    return per_sample.mean()


def dynamics_loss(
    model: WorldModel,
    x: torch.Tensor,
    y_future: torch.Tensor,
    K: int,
    horizon_discount: float,
    teacher_forcing_p: float,
    feature_mask: torch.Tensor | None = None,
    beta_nll: float = 0.0,
    mse_aux_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """x: [B, L, F] observed history (scaled). y_future: [B, K, F] ground
    truth next K states (scaled). Returns the scalar training loss.

    Scheduled sampling: at each step, the true future state is fed back
    with probability `teacher_forcing_p`; otherwise the model's own
    (detached) prediction is fed back. Early in training this keeps the
    unroll stable; annealing it toward the model's own predictions is what
    prevents a model that looks fine at k=1 but diverges by k=6.
    """
    h_t, h = model.encoder(x)
    cur = x[:, -1, :]
    prev = x[:, -2, :] if x.shape[1] > 1 else cur
    total = x.new_zeros(())
    mask = feature_mask.to(x.dtype) if feature_mask is not None else None

    for k in range(K):
        mu, logvar = model.transition(h_t, cur, prev)
        pred = cur + mu
        target = y_future[:, k, :]
        elements = gaussian_nll_elements(pred, logvar, target)
        if beta_nll > 0.0:
            elements = elements * logvar.exp().detach().pow(beta_nll)
        step = _reduce(elements, mask, sample_weight)
        if mse_aux_weight > 0.0:
            step = step + mse_aux_weight * _reduce((pred - target) ** 2, mask, sample_weight)
        total = total + step * (horizon_discount ** k)

        if random.random() < teacher_forcing_p:
            nxt = target
        else:
            nxt = pred.detach()

        h_t, h = model.encoder(nxt.unsqueeze(1), h)
        prev, cur = cur, nxt

    return total / K


@torch.no_grad()
def free_running_metrics(
    model: WorldModel,
    x: torch.Tensor,
    y_future: torch.Tensor,
    K: int,
    feature_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """What the model does at deployment: a deterministic free-running
    rollout (its own mean fed back, no teacher forcing, no sampling) scored
    against the true future. Returns per-horizon tensors [K]:
    `nll` (Gaussian NLL of the rollout mean/variance), `mse` (rollout mean),
    `mse_persistence` (S_t repeated) — so skill = 1 - mse/mse_persistence is
    computable per k — and `coverage90` (fraction of kept feature-elements
    inside the ±1.645σ band). Everything is over kept features only."""
    h_t, h = model.encoder(x)
    cur = x[:, -1, :]
    prev = x[:, -2, :] if x.shape[1] > 1 else cur
    last = x[:, -1, :]
    mask = feature_mask.to(x.dtype) if feature_mask is not None else torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    denom = mask.sum().clamp_min(1.0)
    nll_k, mse_k, pers_k, cov_k = [], [], [], []
    for k in range(K):
        mu, logvar = model.transition(h_t, cur, prev)
        pred = (cur + mu).clamp(-model.state_clamp, model.state_clamp)
        target = y_future[:, k, :]
        elements = gaussian_nll_elements(pred, logvar, target)
        nll_k.append(((elements * mask).sum(-1) / denom).mean())
        mse_k.append((((pred - target) ** 2) * mask).sum(-1).div(denom).mean())
        pers_k.append((((last - target) ** 2) * mask).sum(-1).div(denom).mean())
        inside = ((pred - target).abs() <= 1.645 * (0.5 * logvar).exp()).to(x.dtype)
        cov_k.append(((inside * mask).sum(-1) / denom).mean())
        h_t, h = model.encoder(pred.unsqueeze(1), h)
        prev, cur = cur, pred
    return {
        "nll": torch.stack(nll_k),
        "mse": torch.stack(mse_k),
        "mse_persistence": torch.stack(pers_k),
        "coverage90": torch.stack(cov_k),
    }


def teacher_forcing_schedule(epoch: int, total_epochs: int, start_p: float, end_p: float, anneal_fraction: float) -> float:
    """Linear anneal from start_p to end_p over the first `anneal_fraction`
    of training, then held at end_p."""
    anneal_epochs = max(1, int(total_epochs * anneal_fraction))
    if epoch >= anneal_epochs:
        return end_p
    frac = epoch / anneal_epochs
    return start_p + (end_p - start_p) * frac


def risk_head_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None,
                   sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE with the usual positive upweighting, optionally reweighted per row.

    `sample_weight` exists for the stage-balanced objective. The pooled
    `risk_label` makes every positive equal, so whichever attack stage supplies
    most of the positives supplies most of the gradient: on CTU that is exfil
    at 172 of 213 attack windows, and the resulting head ranks recon and c2
    BELOW chance while the stage head — trained on the same states with class
    weights — ranks them at 0.638 and 0.854. Per-row weights let the positive
    class be rebalanced across stages without touching the positive/negative
    balance, which is what makes the two runs a controlled comparison.
    """
    per_row = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float(),
                                                 pos_weight=pos_weight, reduction="none")
    if sample_weight is None:
        return per_row.mean()
    w = sample_weight.to(per_row.dtype)
    total = w.sum()
    if total <= 0:
        return per_row.mean()
    # Normalised by the weight total, so the loss keeps the scale an unweighted
    # mean would have and the learning rate does not have to move with it.
    return (per_row * w).sum() / total


def stage_head_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long(), weight=class_weights)

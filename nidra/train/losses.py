"""Stage-1 dynamics loss: multi-step unrolled Gaussian NLL with scheduled
sampling, and the horizon-discounted weighting that keeps distant, more
uncertain steps from dominating the gradient.

This loop OWNS gradients through the transition and encoder across K steps
— it is written directly (not via WorldModel.rollout, which is the
`torch.no_grad()` inference-time rollout used for sampling/serving/eval).
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


def dynamics_loss(
    model: WorldModel,
    x: torch.Tensor,
    y_future: torch.Tensor,
    K: int,
    horizon_discount: float,
    teacher_forcing_p: float,
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
    total = x.new_zeros(())

    for k in range(K):
        mu, logvar = model.transition(h_t)
        pred = cur + mu
        target = y_future[:, k, :]
        total = total + gaussian_nll(pred, logvar, target) * (horizon_discount ** k)

        if random.random() < teacher_forcing_p:
            nxt = target
        else:
            nxt = pred.detach()

        h_t, h = model.encoder(nxt.unsqueeze(1), h)
        cur = nxt

    return total / K


def teacher_forcing_schedule(epoch: int, total_epochs: int, start_p: float, end_p: float, anneal_fraction: float) -> float:
    """Linear anneal from start_p to end_p over the first `anneal_fraction`
    of training, then held at end_p."""
    anneal_epochs = max(1, int(total_epochs * anneal_fraction))
    if epoch >= anneal_epochs:
        return end_p
    frac = epoch / anneal_epochs
    return start_p + (end_p - start_p) * frac


def risk_head_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float(), pos_weight=pos_weight)


def stage_head_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long(), weight=class_weights)

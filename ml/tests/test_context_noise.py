"""Training-time noise on the head's CONTEXT components, not just its state.

Measured on CTU-13 validation with the `hidden` head: the head applied to the
observed origin scores AP 0.489, the same head applied to the rollout scores
0.459, and the rollout with the transition disabled also scores 0.459. The
loss is not caused by the transition's predictions — it appears as soon as the
encoder ingests ANY synthetic window. The head was fit on hidden states the
encoder reached over real observations and is asked at inference for a
judgement on hidden states it reached over six of its own.

The state component already carries training noise for exactly this reason.
This extends the same regularizer to the context components, which is the
version of the fix that stays inside the frozen-head discipline: the inputs
are still observed, still from at or before t, and the head still never sees
a predicted state during training.
"""

from __future__ import annotations

import numpy as np
import torch

from nidra.models.heads import TrajectoryRiskHead
from nidra.train.train_heads import _risk_logits


def _parts(n=4, H=8, F=45) -> dict:
    g = torch.Generator().manual_seed(0)
    return {"hidden": torch.randn(n, H, generator=g), "delta": torch.randn(n, F, generator=g),
            "logvar": torch.randn(n, F, generator=g)}


class TestContextNoise:
    def test_zero_noise_leaves_the_context_untouched(self):
        head = TrajectoryRiskHead(("state", "hidden"), 45, 8, 4)
        parts, s = _parts(), torch.zeros(4, 45)
        torch.manual_seed(0)
        a = _risk_logits(head, parts, None, "cpu", 0.0, s, context_noise=0.0)
        torch.manual_seed(0)
        b = _risk_logits(head, parts, None, "cpu", 0.0, s, context_noise=0.0)
        assert torch.allclose(a, b)

    def test_noise_changes_the_logits(self):
        head = TrajectoryRiskHead(("state", "hidden"), 45, 8, 4)
        parts, s = _parts(), torch.zeros(4, 45)
        clean = _risk_logits(head, parts, None, "cpu", 0.0, s, context_noise=0.0)
        torch.manual_seed(1)
        noisy = _risk_logits(head, parts, None, "cpu", 0.0, s, context_noise=0.5)
        assert not torch.allclose(clean, noisy)

    def test_the_default_is_no_context_noise(self):
        """Absent from a config, behaviour is exactly what it was."""
        head = TrajectoryRiskHead(("state", "hidden"), 45, 8, 4)
        parts, s = _parts(), torch.zeros(4, 45)
        assert torch.allclose(_risk_logits(head, parts, None, "cpu", 0.0, s),
                              _risk_logits(head, parts, None, "cpu", 0.0, s, context_noise=0.0))

    def test_the_noise_is_scaled_per_component_by_its_own_spread(self):
        """`hidden` and `logvar` live on different scales; one absolute sigma
        would be a rounding error on one and destroy the other."""
        from nidra.train.train_heads import _noise_like
        torch.manual_seed(0)
        wide = torch.randn(4096, 8) * 10.0
        narrow = torch.randn(4096, 45) * 0.0001
        assert _noise_like(wide, 1.0).std().item() > 100 * _noise_like(narrow, 1.0).std().item()

    def test_a_per_state_head_is_unaffected(self):
        from nidra.models.heads import RiskHead
        head = RiskHead(45, 4)
        s = torch.randn(4, 45)
        assert torch.allclose(_risk_logits(head, {}, None, "cpu", 0.0, s, context_noise=0.9), head(s))


class TestNoiseScale:
    def test_zero_sigma_is_exactly_zero(self):
        from nidra.train.train_heads import _noise_like
        assert torch.equal(_noise_like(torch.randn(8, 3), 0.0), torch.zeros(8, 3))

    def test_a_constant_component_gets_no_noise(self):
        """sd 0 means the component carries no information to perturb."""
        from nidra.train.train_heads import _noise_like
        assert torch.equal(_noise_like(torch.full((8, 3), 2.0), 1.0), torch.zeros(8, 3))

    def test_sigma_is_a_multiple_of_the_batch_standard_deviation(self):
        from nidra.train.train_heads import _noise_like
        x = torch.randn(4096, 4) * 3.0
        torch.manual_seed(0)
        n = _noise_like(x, 0.5)
        assert np.isclose(n.std().item(), 0.5 * 3.0, rtol=0.1)

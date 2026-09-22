"""The stage-balanced risk objective.

§3.9's diagnosis: `risk_label` pools every attack stage into one positive
class, so the stage with the most positives supplies most of the gradient. On
CTU that is exfil (172 of 213 attack windows) and the head ends up ranking
recon and c2 below chance, while the stage head — same states, class-weighted
— ranks them at 0.638 and 0.854. Score fusion cannot recover it (§3.9), so the
fix has to be in the objective.

These tests pin the two properties that make the experiment controlled: the
weights change only the COMPOSITION of the positive class, not its total mass
against the negatives, and the default path is bit-identical to the old one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nidra.train.losses import risk_head_loss


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(64, 1), (torch.rand(64) > 0.7).float()


def test_default_is_unchanged(batch):
    """Anything else silently invalidates every earlier run."""
    logits, y = batch
    import torch.nn.functional as F
    assert risk_head_loss(logits, y) == pytest.approx(
        float(F.binary_cross_entropy_with_logits(logits.squeeze(-1), y)), abs=1e-7)


def test_pos_weight_still_applies(batch):
    logits, y = batch
    pw = torch.tensor(10.0)
    import torch.nn.functional as F
    assert risk_head_loss(logits, y, pos_weight=pw) == pytest.approx(
        float(F.binary_cross_entropy_with_logits(logits.squeeze(-1), y, pos_weight=pw)), abs=1e-7)


def test_uniform_sample_weight_is_the_unweighted_loss(batch):
    logits, y = batch
    assert risk_head_loss(logits, y, sample_weight=torch.full((64,), 3.0)) == pytest.approx(
        float(risk_head_loss(logits, y)), abs=1e-7)


def test_weights_are_normalised_so_the_scale_does_not_move(batch):
    """A loss whose magnitude tracks the weight total would make the learning
    rate depend on the stage mix, and the ablation would not be controlled."""
    logits, y = batch
    w = torch.rand(64) + 0.1
    a = float(risk_head_loss(logits, y, sample_weight=w))
    b = float(risk_head_loss(logits, y, sample_weight=w * 1000.0))
    assert a == pytest.approx(b, rel=1e-6)


def test_a_zero_weight_row_is_excluded():
    logits = torch.tensor([[10.0], [-10.0]])
    y = torch.tensor([0.0, 1.0])                       # both wrong, both large losses
    only_first = float(risk_head_loss(logits, y, sample_weight=torch.tensor([1.0, 0.0])))
    alone = float(risk_head_loss(logits[:1], y[:1]))
    assert only_first == pytest.approx(alone, abs=1e-6)


def test_all_zero_weights_fall_back_rather_than_dividing_by_zero(batch):
    logits, y = batch
    out = risk_head_loss(logits, y, sample_weight=torch.zeros(64))
    assert torch.isfinite(out)
    assert float(out) == pytest.approx(float(risk_head_loss(logits, y)), abs=1e-7)


def test_upweighting_a_stage_moves_the_gradient_toward_it():
    """The mechanism the experiment depends on: a rare stage's rows must be
    able to dominate the update."""
    logits = torch.tensor([[5.0], [5.0], [5.0], [-5.0]], requires_grad=True)
    y = torch.tensor([1.0, 1.0, 1.0, 1.0])             # row 3 is the rare, badly-scored stage
    plain = torch.autograd.grad(risk_head_loss(logits, y), logits, retain_graph=True)[0]
    rare = torch.autograd.grad(
        risk_head_loss(logits, y, sample_weight=torch.tensor([1.0, 1.0, 1.0, 100.0])), logits)[0]
    assert abs(float(rare[3])) > abs(float(plain[3]))
    assert abs(float(rare[0])) < abs(float(plain[0]))


def _w(risk, stages):
    from nidra.train.train_heads import stage_balanced_sample_weights
    return stage_balanced_sample_weights(np.array(risk), np.array(stages, dtype=object))


def test_positive_weight_total_is_preserved():
    """The experiment must change the stage MIX and nothing else. If the total
    positive mass moved, the run would confound two variables."""
    risk = [1] * 10 + [0] * 90
    stages = ["exfil"] * 8 + ["c2", "recon"] + ["benign"] * 90
    w = _w(risk, stages)
    assert w[:10].sum() == pytest.approx(10.0)
    assert np.all(w[10:] == 1.0)


def test_a_rare_stage_outweighs_a_common_one_per_row():
    risk = [1] * 10
    stages = ["exfil"] * 8 + ["c2", "recon"]
    w = _w(risk, stages)
    assert w[8] == pytest.approx(w[9])                  # c2 and recon, one each
    assert w[8] > w[0] * 7                              # eight exfil rows share what one c2 row gets
    # Each STAGE ends up with the same total mass: 8 exfil rows together weigh
    # what the single c2 row does, and what the single recon row does.
    assert w[:8].sum() == pytest.approx(w[8])
    assert w[8] == pytest.approx(w[9])
    assert w[:8].sum() + w[8] + w[9] == pytest.approx(10.0)


def test_equal_stages_give_uniform_weights():
    w = _w([1, 1, 1, 1], ["a", "a", "b", "b"])
    assert np.allclose(w, 1.0)


def test_pre_onset_positives_are_their_own_group():
    """stage_label is 'benign' on a pre-onset positive — the window before an
    attack. They are a distinct kind of positive, not exfil."""
    risk = [1, 1, 1, 0]
    stages = ["exfil", "exfil", "benign", "benign"]
    w = _w(risk, stages)
    assert w[2] > w[0]                                  # one pre-onset against two exfil
    assert w[3] == 1.0                                  # the negative is untouched


def test_no_positives_is_not_an_error():
    assert np.allclose(_w([0, 0, 0], ["benign"] * 3), 1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="against"):
        _w([1, 0], ["exfil"])


def test_integer_stage_indices_group_the_same_way_as_labels():
    """The trainer carries integer stage indices, the tests above use strings.
    Both must group by distinct value or the production path differs from the
    tested one."""
    risk = [1, 1, 1, 0]
    by_label = _w(risk, ["exfil", "exfil", "c2", "benign"])
    by_index = _w(risk, [4, 4, 2, 0])
    assert np.allclose(by_label, by_index)


def test_weights_line_up_with_a_shuffled_epoch_index():
    """The trainer indexes weights with the same batch index it uses for the
    labels; a mismatch would pair each row with another row's weight."""
    from nidra.train.train_heads import stage_balanced_sample_weights
    rng = np.random.default_rng(0)
    risk = np.array([1] * 6 + [0] * 20)
    stages = np.array(["exfil"] * 5 + ["c2"] + ["benign"] * 20, dtype=object)
    w = stage_balanced_sample_weights(risk, stages)
    perm = rng.permutation(len(risk))
    assert np.allclose(w[perm], stage_balanced_sample_weights(risk[perm], stages[perm]))

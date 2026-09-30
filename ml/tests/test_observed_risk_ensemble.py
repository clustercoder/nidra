"""`observed_risk` must be the whole ensemble's opinion, not seed 0's.

Everything else NidraPredictor returns is pooled across all five members: the
forecast risk curve soft-votes the five heads and then pools sampled
trajectories. `observed_risk` — the number a console renders as "risk right
now", and the anchor the forecast cone is drawn from — was computed from
`self.models[0]` alone.

That is a different and noisier statistic than the one every published
number refers to, presented next to them as though it were comparable. On a
real CIC-IDS2017 host it scored 0 true positives against 4 false ones over
48 windows while the pooled forecast on the same windows got 17 against 12.
Seed 0 is not the model; it is one fifth of it.

Averaging the five heads' probabilities is the same soft vote the forecast
already performs under the shipped `head_reduction: before_pooling`, so this
makes the anchor consistent with the cone rather than inventing a statistic.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from datetime import datetime, timezone


@contextmanager
def disagreeing_ensemble(predictor):
    """The shared fixture trains a single seed, where the mean of the members
    IS member zero and this whole distinction is invisible. Stand up a second,
    deliberately different member so the two candidate behaviours give
    different answers — otherwise the test passes without testing anything."""
    original = predictor.models
    twin = copy.deepcopy(original[0])
    with torch.no_grad():
        twin.risk_head.net[-1].bias += 3.0      # a member that disagrees loudly
        twin.stage_head.net[-1].bias[2] += 8.0  # ...and votes for a different stage
    try:
        predictor.models = [original[0], twin]
        yield predictor
    finally:
        predictor.models = original


def _member_risks(predictor, scaled):
    out = []
    with torch.no_grad():
        for m in predictor.models:
            r, _ = m.score_states(torch.from_numpy(scaled[-1]).float().unsqueeze(0))
            out.append(float(r.item()))
    return out


def test_observed_risk_is_the_mean_of_every_member_not_seed_zero(trained_predictor):
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    scaled = predictor._validate_and_scale(states)

    with disagreeing_ensemble(predictor) as p:
        per_member = _member_risks(p, scaled)
        assert abs(per_member[0] - np.mean(per_member)) > 1e-6, (
            "fixture members agree; this test would pass under either behaviour"
        )
        result = p.forecast(states, host_id="h1",
                            origin_ts=datetime(2017, 7, 4, 9, 0, tzinfo=timezone.utc))

    assert result["observed_risk"] == pytest.approx(float(np.mean(per_member))), (
        "observed_risk must soft-vote every ensemble member, matching how the "
        "forecast reduces the heads"
    )
    assert result["observed_risk"] != pytest.approx(per_member[0]), (
        "observed_risk is still seed 0's opinion alone"
    )


def test_observed_stage_is_also_decided_by_the_whole_ensemble(trained_predictor):
    """The stage shown beside the risk comes from the same observed state, so
    it must be voted the same way — otherwise the pair can disagree about
    which member they describe."""
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    scaled = predictor._validate_and_scale(states)

    from nidra.data.schema import STAGE_LABELS

    with disagreeing_ensemble(predictor) as p:
        probs = []
        with torch.no_grad():
            for m in p.models:
                _, s = m.score_states(torch.from_numpy(scaled[-1]).float().unsqueeze(0))
                probs.append(s.numpy())
        stacked = np.stack(probs, axis=0)
        expected_idx = int(stacked.mean(axis=0).argmax())
        assert expected_idx != int(stacked[0].argmax()), (
            "members agree on the stage; this test would pass under either behaviour"
        )
        result = p.forecast(states, host_id="h1",
                            origin_ts=datetime(2017, 7, 4, 9, 0, tzinfo=timezone.utc))

    assert result["observed_stage"] == STAGE_LABELS[expected_idx]


def test_single_member_ensemble_is_unchanged(trained_predictor):
    """With one member the mean is that member — the fix must not perturb a
    single-seed deployment (config/mvp_2017.yaml)."""
    predictor, windowed = trained_predictor
    states = windowed["train"].X[0]
    scaled = predictor._validate_and_scale(states)
    kept = predictor.models
    try:
        predictor.models = kept[:1]
        with torch.no_grad():
            r, _ = kept[0].score_states(torch.from_numpy(scaled[-1]).float().unsqueeze(0))
        result = predictor.forecast(states, host_id="h1",
                                    origin_ts=datetime(2017, 7, 4, 9, 0, tzinfo=timezone.utc))
        assert result["observed_risk"] == float(r.item())
    finally:
        predictor.models = kept

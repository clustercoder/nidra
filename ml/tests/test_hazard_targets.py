"""Discrete-time survival targets for onset forecasting."""

from __future__ import annotations

import numpy as np
import pytest

from nidra.data.onset import discrete_hazard_targets, onset_targets, survival_to_cumulative

H = (1, 3, 5, 10, 15, 30)


class TestHazardTargets:
    def test_the_event_lands_in_exactly_one_bucket(self):
        minutes = np.array([0.5, 2.0, 4.0, 7.0, 12.0, 20.0])
        event, _ = discrete_hazard_targets(np.zeros(6, dtype=bool), minutes, H)
        assert list(event.sum(axis=1)) == [1] * 6
        assert list(event.argmax(axis=1)) == [0, 1, 2, 3, 4, 5]

    def test_a_row_is_at_risk_up_to_and_including_its_event_bucket(self):
        event, at_risk = discrete_hazard_targets(np.zeros(1, dtype=bool), np.array([7.0]), H)
        assert list(at_risk[0]) == [True, True, True, True, False, False]
        assert event[0, 3] == 1

    def test_an_onset_beyond_the_last_horizon_is_censored_not_negative_forever(self):
        event, at_risk = discrete_hazard_targets(np.zeros(2, dtype=bool), np.array([45.0, np.inf]), H)
        assert event.sum() == 0
        assert at_risk.all(), "a censored row is at risk in every bucket and carries no event"

    def test_rows_inside_an_episode_are_at_risk_nowhere(self):
        event, at_risk = discrete_hazard_targets(np.ones(3, dtype=bool), np.array([2.0, 7.0, np.inf]), H)
        assert not at_risk.any() and event.sum() == 0

    def test_an_onset_exactly_on_a_boundary_belongs_to_the_lower_bucket(self):
        event, _ = discrete_hazard_targets(np.zeros(1, dtype=bool), np.array([3.0]), H)
        assert event[0, 1] == 1


class TestCumulativeIsCoherent:
    def test_cumulative_probability_is_monotone(self):
        rng = np.random.default_rng(0)
        cum = survival_to_cumulative(rng.random((200, len(H))))
        assert (np.diff(cum, axis=1) >= -1e-12).all()

    def test_zero_hazard_is_zero_probability_and_unit_hazard_saturates(self):
        assert survival_to_cumulative(np.zeros((1, 3))).max() == 0.0
        assert survival_to_cumulative(np.array([[0.0, 1.0, 0.0]]))[0, 2] == pytest.approx(1.0)

    def test_it_agrees_with_the_independent_target_on_the_ground_truth(self):
        # Given the true bucket, the cumulative of a perfect hazard equals the
        # independent "within h" target — the two parameterisations describe
        # the same event, which is what makes the comparison controlled.
        minutes = np.array([0.5, 4.0, 20.0, np.inf])
        inside = np.zeros(4, dtype=bool)
        event, _ = discrete_hazard_targets(inside, minutes, H)
        assert np.array_equal(survival_to_cumulative(event.astype(float)).astype(int),
                              onset_targets(inside, minutes, H))

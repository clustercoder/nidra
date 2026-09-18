import numpy as np

from nidra.eval.metrics import (
    brier_score,
    lead_time_distribution,
    lead_time_for_episode,
    reliability_diagram,
    standard_metrics,
    state_nrmse,
    state_nrmse_by_horizon,
)


def test_lead_time_exact_worked_example():
    # Mirrors the PRD §7.5 worked example: sustained crossing at k=3 with
    # m=2, threshold=0.75, 30s windows -> lead time 90s.
    onset_ts = 1000
    risk_curve = [
        (700, 0.30),
        (730, 0.55),
        (760, 0.61),
        (790, 0.69),
        (820, 0.78),  # first sustained crossing starts here (with next window)
        (850, 0.83),
        (880, 0.85),
    ]
    lt = lead_time_for_episode(risk_curve, onset_ts, threshold=0.75, m=2)
    assert lt == 1000 - 820


def test_lead_time_none_when_never_crosses():
    onset_ts = 1000
    risk_curve = [(t, 0.1) for t in range(700, 1000, 30)]
    assert lead_time_for_episode(risk_curve, onset_ts, threshold=0.75, m=2) is None


def test_lead_time_single_spike_does_not_count():
    onset_ts = 1000
    risk_curve = [(700, 0.1), (730, 0.9), (760, 0.1), (790, 0.1)]
    assert lead_time_for_episode(risk_curve, onset_ts, threshold=0.75, m=2) is None


def test_lead_time_distribution_reports_median_not_max():
    episodes = [
        ([(0, 0.9), (30, 0.9)], 100),   # sustained crossing starts at ts=0 -> lead time 100
        ([(0, 0.1), (30, 0.1)], 100),   # no warning
        ([(60, 0.9), (90, 0.9)], 100),  # sustained crossing starts at ts=60 -> lead time 40
    ]
    report = lead_time_distribution(episodes, threshold=0.75, m=2)
    assert report.n_episodes == 3
    assert report.n_no_warning == 1
    assert report.median == 70.0  # median of [100, 40]
    assert report.fraction_no_warning == 1 / 3


def test_state_nrmse_zero_for_perfect_prediction():
    y = np.random.randn(50, 45)
    nrmse = state_nrmse(y, y.copy())
    np.testing.assert_allclose(nrmse, 0.0, atol=1e-10)


def test_state_nrmse_by_horizon_shape():
    y_true = np.random.randn(20, 6, 45)
    y_pred = y_true + np.random.randn(20, 6, 45) * 0.1
    grid = state_nrmse_by_horizon(y_true, y_pred)
    assert grid.shape == (6, 45)
    assert (grid >= 0).all()


def test_state_nrmse_does_not_blow_up_on_degenerate_eval_batch():
    """Regression test for the real, observed failure mode: a small eval
    batch where one feature is exactly constant (e.g. a packet aggregate
    zero-filled on a flow-only day) has std=0 for that feature. The old
    behavior (divide by std(y_true), floored at 1e-8) inflated nRMSE for
    that single feature into the 10^5-10^9 range from a small prediction
    error alone — orders of magnitude beyond every other feature, and
    dominating the mean nRMSE reported in REAL_DATA_RESULTS.md. The floor
    must keep any single feature's contribution bounded to a legible range."""
    n = 200
    y_true = np.random.randn(n, 45) * 0.5
    y_pred = y_true.copy()
    y_true[:, 3] = 0.0
    y_pred[:, 3] = 0.02  # small, plausible model noise on an exactly-constant feature
    nrmse = state_nrmse(y_true, y_pred)
    assert nrmse[3] < 10.0, f"degenerate feature should not dominate the metric, got {nrmse[3]}"


def test_state_nrmse_prefers_supplied_reference_scale_over_batch_std():
    n = 50
    y_true = np.zeros((n, 45))
    y_pred = np.full((n, 45), 0.1)
    # Batch std is exactly 0 everywhere; without a reference scale this
    # would divide by the floor directly. With a realistic reference scale
    # (as if computed from millions of training rows) the ratio should be
    # governed by that reference, not the degenerate batch.
    reference_scale = np.full(45, 2.0)
    nrmse = state_nrmse(y_true, y_pred, feature_scale=reference_scale)
    np.testing.assert_allclose(nrmse, 0.1 / 2.0)


def test_brier_score_perfect_is_zero():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == 0.0


def test_brier_score_worst_case():
    y_true = np.array([1, 0])
    y_prob = np.array([0.0, 1.0])
    assert brier_score(y_true, y_prob) == 1.0


def test_reliability_diagram_shape():
    y_true = np.random.randint(0, 2, 200)
    y_prob = np.random.rand(200)
    diag = reliability_diagram(y_true, y_prob, n_bins=10)
    assert len(diag["bin_centers"]) == 10
    assert len(diag["observed_frequency"]) == 10
    assert sum(diag["bin_counts"]) == 200


def test_standard_metrics_perfect_classifier():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.1, 0.1])
    m = standard_metrics(y_true, y_prob, threshold=0.5)
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0

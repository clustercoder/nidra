import pytest
import torch

from nidra.data.dataset import build_windowed_arrays
from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.normalize import FeatureScaler
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.windowize import build_state_rows
from nidra.eval.lead_time_runner import build_episode_risk_curves, compute_lead_time_report
from nidra.models.world_model import WorldModel
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def _tiny_model():
    return WorldModel(n_features=45, hidden_size=16, encoder_layers=1, transition_mlp_hidden=32,
                       risk_hidden=16, stage_hidden=16, n_stages=6)


def _labelled_day():
    flows = make_synthetic_flows(n_hosts=2, n_windows=80, portscan_start_window=40, portscan_len=10)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    return attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=WINDOW_SECONDS)


def _fitted_scaler(arrays):
    scaler = FeatureScaler()
    scaler.fit(arrays.X.reshape(-1, arrays.X.shape[-1]))
    return scaler


def test_build_episode_risk_curves_single_model():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    scaler = _fitted_scaler(arrays)
    model = _tiny_model()
    model.eval()

    episodes = build_episode_risk_curves(arrays, day, model, scaler, n_samples=5)
    assert len(episodes) > 0
    for risk_curve, onset_ts in episodes:
        assert isinstance(onset_ts, int)
        for ts, p in risk_curve:
            assert isinstance(ts, int)
            assert 0.0 <= p <= 1.0


def test_build_episode_risk_curves_ensemble_matches_single_model_with_one_member():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    scaler = _fitted_scaler(arrays)
    model = _tiny_model()
    model.eval()

    torch.manual_seed(0)
    single = build_episode_risk_curves(arrays, day, model, scaler, n_samples=5)
    torch.manual_seed(0)
    ensemble = build_episode_risk_curves(arrays, day, [model], scaler, n_samples=5)

    assert len(single) == len(ensemble)
    for (curve_a, onset_a), (curve_b, onset_b) in zip(single, ensemble):
        assert onset_a == onset_b
        assert len(curve_a) == len(curve_b)
        for (ts_a, p_a), (ts_b, p_b) in zip(curve_a, curve_b):
            assert ts_a == ts_b
            assert p_a == pytest.approx(p_b, abs=1e-5)


def test_compute_lead_time_report_accepts_model_list():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    scaler = _fitted_scaler(arrays)
    models = [_tiny_model() for _ in range(2)]
    for m in models:
        m.eval()

    report = compute_lead_time_report(arrays, day, models, scaler, threshold=0.75, m=2, n_samples=5)
    assert report.n_episodes >= 0
    assert 0.0 <= report.fraction_no_warning <= 1.0

from nidra.data.dataset import WorldModelDataset, build_windowed_arrays
from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import CONTEXT_LENGTH, FEATURE_ORDER, HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.windowize import build_state_rows
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def _labelled_day():
    flows = make_synthetic_flows(n_hosts=2, n_windows=60, portscan_start_window=30, portscan_len=8)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    return attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=WINDOW_SECONDS)


def test_windowed_array_shapes():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    assert arrays.X.shape[1:] == (CONTEXT_LENGTH, len(FEATURE_ORDER))
    assert arrays.Y.shape[1:] == (HORIZON_LENGTH, len(FEATURE_ORDER))
    assert arrays.X.shape[0] == arrays.Y.shape[0] == len(arrays.host_id) == len(arrays.origin_ts)


def test_every_sample_traceable_to_host_and_origin():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    assert (arrays.host_id != "").all()
    assert (arrays.origin_ts > 0).all()


def test_samples_inside_attack_episode_get_positive_episode_id():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    attack_mask = arrays.stage_label != "benign"
    if attack_mask.any():
        assert (arrays.episode_id[attack_mask] >= 0).all()
    benign_far_from_attack = arrays.episode_id == -1
    assert benign_far_from_attack.any()


def test_dataset_wrapper_getitem_shapes():
    day = _labelled_day()
    arrays = build_windowed_arrays(day, L=CONTEXT_LENGTH, K=HORIZON_LENGTH)
    ds = WorldModelDataset(arrays)
    sample = ds[0]
    assert sample["x"].shape == (CONTEXT_LENGTH, len(FEATURE_ORDER))
    assert sample["y"].shape == (HORIZON_LENGTH, len(FEATURE_ORDER))
    assert len(ds) == arrays.X.shape[0]

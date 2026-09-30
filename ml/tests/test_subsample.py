import numpy as np

from nidra.data.dataset import WindowedArrays
from nidra.data.dataset import subsample_stratified_by_risk as _subsample


def _fake_arrays(n, n_pos):
    risk = np.zeros(n, dtype="int64")
    risk[:n_pos] = 1
    return WindowedArrays(
        X=np.random.randn(n, 30, 45).astype("float32"),
        Y=np.random.randn(n, 6, 45).astype("float32"),
        host_id=np.array([f"h{i}" for i in range(n)]),
        origin_ts=np.arange(n, dtype="int64"),
        episode_id=np.full(n, -1, dtype="int64"),
        stage_label=np.array(["benign"] * n),
        risk_label=risk,
        future_stage_idx=np.zeros((n, 6), dtype="int64"),
        future_is_attack=np.zeros((n, 6), dtype="int64"),
    )


def test_subsample_preserves_all_rare_positives():
    arrays = _fake_arrays(n=10000, n_pos=5)
    sub = _subsample(arrays, max_n=100, seed=0)
    assert sub.risk_label.sum() == 5  # all 5 positives survive a 100-sample subsample of 10000
    assert len(sub.X) == 100


def test_subsample_propagates_future_arrays():
    arrays = _fake_arrays(n=1000, n_pos=10)
    sub = _subsample(arrays, max_n=50, seed=0)
    assert sub.future_stage_idx is not None
    assert sub.future_is_attack is not None
    assert sub.future_stage_idx.shape == (50, 6)


def test_subsample_no_op_when_under_max():
    arrays = _fake_arrays(n=20, n_pos=1)
    sub = _subsample(arrays, max_n=100, seed=0)
    assert sub is arrays


def test_subsample_caps_positives_when_exceeding_max_n():
    arrays = _fake_arrays(n=1000, n_pos=200)
    sub = _subsample(arrays, max_n=50, seed=0)
    assert len(sub.X) == 50
    assert sub.risk_label.sum() <= 50

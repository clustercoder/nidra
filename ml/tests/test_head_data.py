"""Head arrays cover every row of a split and agree with the windowed path
on the rows both contain."""

from __future__ import annotations

import numpy as np

from nidra.train.head_data import build_head_arrays
from nidra.train.pipeline import scale_arrays
from tests.test_head_selection import head_ready_artifacts  # noqa: F401


def test_head_arrays_cover_every_row_and_match_windowed_states(head_ready_artifacts):
    cfg, tmp_path, windowed, scaler = head_ready_artifacts
    # rebuild the val table from the windowed arrays' source: conftest's fixture keeps no table,
    # so check invariants on a table assembled from the arrays themselves
    import pandas as pd
    from nidra.data.schema import FEATURE_ORDER
    arrays = windowed["val"]
    rows = []
    for i in range(len(arrays.X)):
        rows.append({"host_id": arrays.host_id[i], "window_ts": int(arrays.origin_ts[i]), "stage_label": arrays.stage_label[i],
                     "risk_label": int(arrays.risk_label[i]), **{f: float(v) for f, v in zip(FEATURE_ORDER, arrays.X[i, -1, :])}})
    table = pd.DataFrame(rows)
    ha = build_head_arrays(table, scaler)
    assert len(ha) == len(table)
    assert ha.states.shape == (len(table), len(FEATURE_ORDER)) and ha.states.dtype == np.float32
    Xs, _ = scale_arrays(arrays, scaler)
    # same host/ts ordering after the sort -> compare by key
    key_w = {(h, int(t)): Xs[i, -1, :] for i, (h, t) in enumerate(zip(arrays.host_id, arrays.origin_ts))}
    for i in range(len(ha)):
        np.testing.assert_allclose(ha.states[i], key_w[(ha.host_id[i], int(ha.window_ts[i]))], rtol=1e-5, atol=1e-5)
    assert ha.summary()["n_pos"] == int(arrays.risk_label.sum())
    assert ha.onset_targets((1, 5)).shape == (len(ha), 2)


def test_train_heads_accepts_full_split_head_data(head_ready_artifacts):
    from nidra.train.train_heads import train_heads_for_seed
    cfg, tmp_path, windowed, scaler = head_ready_artifacts
    import pandas as pd
    from nidra.data.schema import FEATURE_ORDER

    def _table(arrays):
        return pd.DataFrame([{"host_id": arrays.host_id[i], "window_ts": int(arrays.origin_ts[i]), "stage_label": arrays.stage_label[i],
                              "risk_label": int(arrays.risk_label[i]), **{f: float(v) for f, v in zip(FEATURE_ORDER, arrays.X[i, -1, :])}}
                             for i in range(len(arrays.X))])
    head_data = {"train": build_head_arrays(_table(windowed["train"]), scaler), "val": build_head_arrays(_table(windowed["val"]), scaler)}
    meta = train_heads_for_seed(cfg, 0, windowed, scaler, "cpu", head_data=head_data)
    assert meta["heads_data"] == "all_split_rows"
    assert meta["heads_n_train"] == len(head_data["train"])

"""The demo entry point is the first thing a reviewer runs after cloning, so
its window-selection has to be correct rather than approximately right: an
[L, F] context window must be L rows of ONE host, contiguous in time, ending
on the row whose risk_label is being demonstrated. A silently non-contiguous
window would feed the model a fabricated history and make the demo's risk
curve meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.data.schema import FEATURE_ORDER
from nidra.scripts.demo_forecast import pick_window


def _table(rows):
    """rows: list of (host_id, window_ts, risk_label, stage_label)."""
    base = {name: 0.0 for name in FEATURE_ORDER}
    records = []
    for host_id, ts, risk, stage in rows:
        rec = dict(base)
        rec.update(host_id=host_id, window_ts=ts, risk_label=risk, stage_label=stage)
        # make each row distinguishable so ordering is checkable
        rec[FEATURE_ORDER[0]] = float(ts)
        records.append(rec)
    return pd.DataFrame(records)


def test_picks_contiguous_window_ending_on_requested_label():
    rows = [("h1", 30 * i, 0, "benign") for i in range(5)]
    rows.append(("h1", 30 * 5, 1, "exfil"))
    win = pick_window(_table(rows), L=3, window_seconds=30, want_risk=1)

    assert win.host_id == "h1"
    assert win.states.shape == (3, len(FEATURE_ORDER))
    assert win.risk_label == 1
    # oldest-first, ending on the labelled row (ts 90, 120, 150)
    assert list(win.states[:, 0]) == [90.0, 120.0, 150.0]


def test_skips_time_gaps_rather_than_fabricating_history():
    """h1 has a labelled positive but a missing window behind it; h2 has a
    clean contiguous run. The clean one must win."""
    rows = [("h1", 0, 0, "benign"), ("h1", 30, 0, "benign"), ("h1", 300, 1, "exfil")]
    rows += [("h2", 30 * i, 0, "benign") for i in range(2)]
    rows.append(("h2", 60, 1, "exfil"))
    win = pick_window(_table(rows), L=3, window_seconds=30, want_risk=1)
    assert win.host_id == "h2"


def test_raises_when_no_host_has_enough_contiguous_history():
    rows = [("h1", 0, 1, "exfil"), ("h1", 30, 1, "exfil")]
    with pytest.raises(ValueError, match="no host"):
        pick_window(_table(rows), L=5, window_seconds=30, want_risk=1)


def test_can_request_a_benign_window():
    rows = [("h1", 30 * i, 0, "benign") for i in range(4)]
    win = pick_window(_table(rows), L=4, window_seconds=30, want_risk=0)
    assert win.risk_label == 0
    assert win.stage_label == "benign"


def test_states_are_float_and_feature_ordered():
    rows = [("h1", 30 * i, 0, "benign") for i in range(3)]
    win = pick_window(_table(rows), L=3, window_seconds=30, want_risk=0)
    assert win.states.dtype == np.float64 or win.states.dtype == np.float32
    assert win.states.shape[1] == len(FEATURE_ORDER)

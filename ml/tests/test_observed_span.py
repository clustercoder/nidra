"""How long the evaluation split actually watched the network.

`false_alarms_per_hour` divides by this. Taking it as
`max(window_ts) - min(window_ts)` is only right when the split is one
continuous capture. CIC-IDS2017's splits are single working days, so the
error there is a rounding difference — but CTU-13's splits are captures made
on different days of August 2011, and the idle nights between them are not
time the system was watching. Measured on the real tables, the naive range
overstates the CTU holdout's observation time by 5.0x, which would divide
the false-alarm rate by five.

The span is therefore the number of DISTINCT window timestamps in the split
times the window length: exactly the wall clock the model was fed, with no
assumption that the split is contiguous, no double counting when two
captures overlap, and — measured on the real tables — within 0.25% of the
old number on every CIC split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.eval.eval_set import observed_span_hours


def _table(ts: list[int], hosts: int = 2) -> pd.DataFrame:
    return pd.DataFrame([{"host_id": f"h{h}", "window_ts": t} for t in ts for h in range(hosts)])


class TestObservedSpan:
    def test_a_contiguous_capture_is_its_own_length(self):
        # 60 windows of 60 s, one continuous hour.
        assert observed_span_hours(_table([60 * i for i in range(60)]), 60) == pytest.approx(1.0)

    def test_two_captures_with_a_gap_count_only_the_captures(self):
        # one hour, then a day of nothing, then one hour.
        day = 24 * 3600
        ts = [60 * i for i in range(60)] + [day + 60 * i for i in range(60)]
        assert observed_span_hours(_table(ts), 60) == pytest.approx(2.0)

    def test_the_naive_range_would_be_much_larger_here(self):
        day = 24 * 3600
        ts = [60 * i for i in range(60)] + [day + 60 * i for i in range(60)]
        naive = (max(ts) - min(ts)) / 3600.0
        assert naive > 24.0
        assert observed_span_hours(_table(ts), 60) == pytest.approx(2.0)

    def test_many_hosts_in_one_window_are_one_window_of_wall_clock(self):
        one = observed_span_hours(_table([0, 60, 120], hosts=1), 60)
        many = observed_span_hours(_table([0, 60, 120], hosts=50), 60)
        assert one == many == pytest.approx(3 * 60 / 3600)

    def test_overlapping_captures_are_not_double_counted(self):
        # two captures covering the same ten minutes.
        a = [60 * i for i in range(10)]
        assert observed_span_hours(_table(a + a), 60) == pytest.approx(10 * 60 / 3600)

    def test_the_window_length_scales_it(self):
        assert observed_span_hours(_table([0, 30, 60]), 30) == pytest.approx(3 * 30 / 3600)

    def test_an_empty_table_is_zero_not_an_error(self):
        assert observed_span_hours(pd.DataFrame({"host_id": [], "window_ts": []}), 60) == 0.0


class TestItIsWhatTheEvalSetCarries:
    def test_build_eval_set_uses_the_observed_span(self):
        from nidra.data.schema import FEATURE_ORDER
        from nidra.eval.eval_set import build_eval_set

        day = 24 * 3600
        rng = np.random.default_rng(0)
        rows = []
        for block in (0, day):
            for i in range(40):
                for h in range(2):
                    row = {f: float(abs(rng.normal())) for f in FEATURE_ORDER}
                    row.update(host_id=f"h{h}", window_ts=block + 60 * i, risk_label=0,
                               stage_label="benign", is_active=1.0)
                    rows.append(row)
        table = pd.DataFrame(rows)
        ev = build_eval_set(table, L=5, K=2, window_seconds=60, split="test")
        # 80 distinct minutes of observation, not the 24 h + 40 min range.
        assert ev.span_hours == pytest.approx(80 * 60 / 3600)

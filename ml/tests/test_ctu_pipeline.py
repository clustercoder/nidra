"""CTU-13 days flowing through the shared per-day pipeline.

The point of the adapter is that nothing downstream has to know which
dataset a day came from. These tests pin that: a CTU scenario produces the
same canonical labelled state table a CIC day does, its cache key cannot
collide with a CIC day's, and the validation block is carved per scenario
rather than per calendar day (three CTU scenarios share 2011-08-15).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.data.schema import FEATURE_ORDER
from nidra.data.splits import temporal_train_val_split
from nidra.train.pipeline import day_cache_path, day_format, load_and_label_day

HEADER = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"


def _binetflow(tmp_path, n_windows: int = 40) -> str:
    """One internal benign host plus one internal host that turns botnet
    halfway through, one flow per minute each."""
    rows = []
    base = pd.Timestamp("2011-08-10 09:00:00")
    for i in range(n_windows):
        ts = (base + pd.Timedelta(minutes=i)).strftime("%Y/%m/%d %H:%M:%S.%f")
        rows.append(f"{ts},1.0,tcp,147.32.84.10,1000,   ->,8.8.8.8,53,SRPA_FSPA,0,0,6,600,300,flow=Background-TCP-Established")
        label = "flow=From-Botnet-V42-TCP-CC1-HTTP-Not-Encrypted" if i >= n_windows // 2 else "flow=Background-TCP-Established"
        rows.append(f"{ts},2.0,tcp,147.32.84.165,1001,   ->,9.9.9.9,80,SRPA_FSPA,0,0,8,800,400,{label}")
    path = tmp_path / "capture.binetflow"
    path.write_text("\n".join([HEADER] + rows) + "\n")
    return str(path)


class TestLoadAndLabelCTUDay:
    def test_produces_the_canonical_state_table(self, tmp_path):
        table = load_and_label_day(_binetflow(tmp_path), window_seconds=60, min_windows_per_host=1,
                                   flow_format="ctu_binetflow", label_dialect="ctu")
        assert list(table.columns[:2]) == ["host_id", "window_ts"]
        for name in FEATURE_ORDER:
            assert name in table.columns
        assert {"stage_label", "risk_label"} <= set(table.columns)
        assert set(table["host_id"]) == {"147.32.84.10", "147.32.84.165"}

    def test_packet_features_are_zero_because_ctu_has_no_usable_packet_source(self, tmp_path):
        table = load_and_label_day(_binetflow(tmp_path), window_seconds=60, min_windows_per_host=1,
                                   flow_format="ctu_binetflow", label_dialect="ctu")
        packet_features = FEATURE_ORDER[15:26]
        assert (table[packet_features].to_numpy() == 0.0).all()

    def test_only_the_botnet_host_gets_attack_windows(self, tmp_path):
        table = load_and_label_day(_binetflow(tmp_path), window_seconds=60, min_windows_per_host=1,
                                   flow_format="ctu_binetflow", label_dialect="ctu")
        attack = table[table["stage_label"] != "benign"]
        assert set(attack["host_id"]) == {"147.32.84.165"}
        assert (attack["stage_label"] == "c2").all()
        assert table.loc[table["host_id"] == "147.32.84.10", "risk_label"].sum() == 0

    def test_risk_label_leads_the_episode_by_the_horizon(self, tmp_path):
        table = load_and_label_day(_binetflow(tmp_path), window_seconds=60, min_windows_per_host=1,
                                   flow_format="ctu_binetflow", label_dialect="ctu", horizon_k=6)
        bot = table[table["host_id"] == "147.32.84.165"].sort_values("window_ts").reset_index(drop=True)
        onset = int(bot.index[bot["stage_label"] != "benign"][0])
        assert bot.loc[onset - 1, "risk_label"] == 1
        assert bot.loc[onset - 6, "risk_label"] == 1
        assert bot.loc[onset - 7, "risk_label"] == 0

    def test_an_unknown_format_is_refused_rather_than_guessed(self, tmp_path):
        with pytest.raises(ValueError, match="unknown flow format"):
            load_and_label_day(_binetflow(tmp_path), window_seconds=60, min_windows_per_host=1,
                               flow_format="netflow9")


class TestDayFormatAndCacheKey:
    def test_format_defaults_to_cicflowmeter(self):
        assert day_format({"file": "Monday.csv", "role": "train"}) == "cicflowmeter"

    def test_format_is_read_from_the_day_entry(self):
        assert day_format({"file": "1/capture.binetflow", "format": "ctu_binetflow"}) == "ctu_binetflow"

    def test_cache_key_separates_formats(self, tmp_path):
        windowing = {"window_seconds": 60}
        cic = day_cache_path(tmp_path, "d", windowing, None, "flowonly", flow_format="cicflowmeter")
        ctu = day_cache_path(tmp_path, "d", windowing, None, "flowonly", flow_format="ctu_binetflow")
        assert cic != ctu
        assert "ctu_binetflow" in ctu.name

    def test_cache_key_separates_host_filters(self, tmp_path):
        windowing = {"window_seconds": 60}
        a = day_cache_path(tmp_path, "d", windowing, None, "flowonly", flow_format="ctu_binetflow",
                           host_scope="147.32.0.0-16")
        b = day_cache_path(tmp_path, "d", windowing, None, "flowonly", flow_format="ctu_binetflow",
                           host_scope="all")
        assert a != b


class TestPerScenarioValidationBlock:
    def test_scenarios_sharing_a_calendar_day_are_blocked_separately(self):
        # Two CTU scenarios on 2011-08-15, one in the morning and one in the
        # evening. Blocking by calendar day would put the whole evening
        # scenario in validation and none of the morning one.
        rows = []
        for group, start in (("s4", pd.Timestamp("2011-08-15 11:00")), ("s5", pd.Timestamp("2011-08-15 16:43"))):
            for i in range(10):
                rows.append({"host_id": "h", "window_ts": int((start + pd.Timedelta(minutes=i)).timestamp()),
                             "stage_label": "benign", "risk_label": 0, "split_group": group})
        df = pd.DataFrame(rows)
        train, val = temporal_train_val_split(df, val_fraction=0.3, per_day=True)
        assert set(val["split_group"]) == {"s4", "s5"}
        assert len(val) == 6 and len(train) == 14

    def test_without_split_group_the_calendar_day_is_still_used(self):
        base = pd.Timestamp("2017-07-04 09:00")
        df = pd.DataFrame([{"host_id": "h", "window_ts": int((base + pd.Timedelta(minutes=i)).timestamp()),
                            "stage_label": "benign", "risk_label": 0} for i in range(10)])
        train, val = temporal_train_val_split(df, val_fraction=0.3, per_day=True)
        assert len(val) == 3 and len(train) == 7

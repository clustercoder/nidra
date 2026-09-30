"""CTU-13 binetflow adapter: mapping fidelity, not just "it parses".

Every assertion here pins one column of the Argus -> CICFlowMeter-internal
mapping, because a silent unit error (Dur is seconds, CICFlowMeter's
Flow Duration is microseconds) would not raise anywhere downstream — it
would just shift one feature's distribution and make every cross-dataset
number wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.data.ctu_load import (
    CTU_INTERNAL_CIDRS,
    load_binetflow,
    parse_argus_state_flags,
    parse_ctu_timestamp,
)

HEADER = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"


def _write(tmp_path, rows: list[str], name: str = "capture.binetflow"):
    path = tmp_path / name
    path.write_text("\n".join([HEADER] + rows) + "\n")
    return path


def test_timestamp_parses_to_epoch_seconds_with_microseconds():
    ts = pd.Series(["2011/08/10 09:46:59.607825", "2011/08/10 09:47:00.634364"])
    epoch = parse_ctu_timestamp(ts)
    assert epoch.iloc[1] - epoch.iloc[0] == pytest.approx(1.026539, abs=1e-5)
    # printed wall clock read literally (documented choice: no tz shift)
    assert epoch.iloc[0] == pytest.approx(
        pd.Timestamp("2011-08-10 09:46:59.607825").value / 1e9, abs=1e-6
    )


def test_unparseable_timestamp_becomes_nan_not_a_wrong_instant():
    epoch = parse_ctu_timestamp(pd.Series(["not a time", "2011/08/10 09:46:59.607825"]))
    assert np.isnan(epoch.iloc[0])
    assert not np.isnan(epoch.iloc[1])


class TestArgusStateFlags:
    def test_tcp_state_counts_each_flag_once_per_direction(self):
        counts = parse_argus_state_flags(pd.Series(["SRPA_FSPA"]), pd.Series(["tcp"]))
        assert counts["syn_flag_count"].iloc[0] == 2   # S in both directions
        assert counts["fin_flag_count"].iloc[0] == 1   # F only on the dst side
        assert counts["rst_flag_count"].iloc[0] == 1
        assert counts["psh_flag_count"].iloc[0] == 2
        assert counts["ack_flag_count"].iloc[0] == 2
        assert counts["urg_flag_count"].iloc[0] == 0

    def test_one_sided_state_counts_only_that_side(self):
        counts = parse_argus_state_flags(pd.Series(["S_"]), pd.Series(["tcp"]))
        assert counts["syn_flag_count"].iloc[0] == 1
        assert counts["ack_flag_count"].iloc[0] == 0

    def test_non_tcp_state_words_are_never_read_as_flags(self):
        # "URP", "CON", "RED", "ECR" contain R/P/C/E/O letters that a naive
        # reader would score as RST/PSH flags. UDP and ICMP carry no TCP flags.
        protos = pd.Series(["udp", "icmp", "icmp", "icmp", "igmp"])
        states = pd.Series(["CON", "URP", "RED", "ECR", "INT"])
        counts = parse_argus_state_flags(states, protos)
        assert counts.to_numpy().sum() == 0

    def test_tcp_state_without_a_direction_separator_is_not_parsed(self):
        counts = parse_argus_state_flags(pd.Series(["CON"]), pd.Series(["tcp"]))
        assert counts.to_numpy().sum() == 0

    def test_missing_state_is_zero_not_an_exception(self):
        counts = parse_argus_state_flags(pd.Series([None, np.nan]), pd.Series(["tcp", "tcp"]))
        assert counts.to_numpy().sum() == 0


class TestLoadBinetflow:
    def test_maps_bytes_packets_and_duration_to_cicflowmeter_units(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.026539,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,SRPA_FSPA,0,0,4,276,156,flow=From-Botnet-V42-UDP-DNS",
        ])
        df, report = load_binetflow(path)
        row = df.iloc[0]
        # Dur is SECONDS in Argus; CICFlowMeter Flow Duration is MICROseconds.
        assert row["flow_duration"] == pytest.approx(1.026539e6)
        # TotPkts is not split by direction; the sum is what every consumer uses.
        assert row["total_fwd_packets"] + row["total_bwd_packets"] == 4
        assert row["total_len_fwd"] == 156
        assert row["total_len_bwd"] == 276 - 156
        assert row["src_ip"] == "147.32.84.165"
        assert row["dst_ip"] == "8.8.8.8"
        assert row["dst_port"] == 53
        assert report.accepted_rows == 1

    def test_mean_iat_is_duration_over_gaps_and_iat_max_is_absent(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,3.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,SRPA_FSPA,0,0,4,276,156,flow=Background",
            "2011/08/10 09:47:00.000000,5.0,tcp,147.32.84.165,1578,   ->,8.8.8.8,53,S_,0,0,1,60,60,flow=Background",
        ])
        df, _ = load_binetflow(path)
        assert df["flow_iat_mean"].iloc[0] == pytest.approx(3.0e6 / 3)
        # A single-packet flow has no inter-arrival gap at all.
        assert df["flow_iat_mean"].iloc[1] == 0.0
        # Argus reports no per-packet timing, so there is no flow IAT maximum.
        assert (df["flow_iat_max"] == 0.0).all()

    def test_negative_reverse_bytes_are_clipped_and_counted(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,S_,0,0,1,60,90,flow=Background",
        ])
        df, report = load_binetflow(path)
        assert df["total_len_bwd"].iloc[0] == 0.0
        assert report.drop_reasons.get("src_bytes_exceeds_total", 0) == 1

    def test_embedded_header_rows_are_dropped_not_coerced(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,S_,0,0,1,60,60,flow=Background",
            HEADER,
            "2011/08/10 09:47:59.607825,1.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,S_,0,0,1,60,60,flow=Background",
        ])
        df, report = load_binetflow(path)
        assert len(df) == 2
        assert report.dropped_rows == 1

    def test_host_filter_keeps_only_flows_whose_source_is_internal(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,S_,0,0,1,60,60,flow=From-Botnet-V42-TCP",
            "2011/08/10 09:47:59.607825,1.0,tcp,94.44.127.113,1577,   ->,147.32.84.59,6881,S_,0,0,1,60,60,flow=Background",
        ])
        df, report = load_binetflow(path, internal_cidrs=CTU_INTERNAL_CIDRS)
        assert list(df["src_ip"]) == ["147.32.84.165"]
        assert report.drop_reasons["external_source_host"] == 1

    def test_no_host_filter_keeps_everything(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,S_,0,0,1,60,60,flow=From-Botnet-V42-TCP",
            "2011/08/10 09:47:59.607825,1.0,tcp,94.44.127.113,1577,   ->,147.32.84.59,6881,S_,0,0,1,60,60,flow=Background",
        ])
        df, _ = load_binetflow(path, internal_cidrs=None)
        assert len(df) == 2

    def test_timestamp_column_is_epoch_seconds_ready_for_join(self, tmp_path):
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.0,tcp,147.32.84.165,1577,   ->,8.8.8.8,53,S_,0,0,1,60,60,flow=Background",
        ])
        df, _ = load_binetflow(path)
        assert df["timestamp"].dtype.kind == "f"
        assert df["timestamp"].iloc[0] > 1_300_000_000

    def test_non_numeric_ports_do_not_drop_the_row(self, tmp_path):
        # Argus prints ICMP "ports" as hex type/code strings like 0x0303.
        path = _write(tmp_path, [
            "2011/08/10 09:46:59.607825,1.0,icmp,147.32.84.165,0x0303,   ->,8.8.8.8,0x0303,URP,0,0,1,60,60,flow=Background",
        ])
        df, _ = load_binetflow(path)
        assert len(df) == 1
        assert df["dst_port"].iloc[0] >= 0

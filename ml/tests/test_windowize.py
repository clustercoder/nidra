import numpy as np
import pandas as pd

from nidra.data.graph_features import safe_div, shannon_entropy
from nidra.data.schema import FEATURE_ORDER, WINDOW_SECONDS
from nidra.data.windowize import align_window, build_state_rows, parse_cic_timestamp
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def test_align_window_is_absolute_epoch_aligned_not_relative_to_first_event():
    # Two series starting at different epochs but sharing an absolute
    # window boundary must align to the SAME window_ts for overlapping time
    # — alignment must not depend on which series' first timestamp anchors it.
    epochs_a = pd.Series([1000.0, 1015.0, 1029.9])
    epochs_b = pd.Series([1000.5, 1010.0])
    wa = align_window(epochs_a, WINDOW_SECONDS)
    wb = align_window(epochs_b, WINDOW_SECONDS)
    # floor(1000/30)*30=990, floor(1015/30)*30=990, floor(1029.9/30)*30=1020
    assert list(wa) == [990, 990, 1020]
    assert list(wb) == [990, 990]


def test_parse_cic_timestamp_returns_correct_epoch_seconds_regardless_of_pandas_datetime_resolution():
    # Regression test: pd.to_datetime's default resolution (ns vs. us) has
    # varied across pandas versions, and a prior implementation of
    # parse_cic_timestamp assumed nanoseconds when converting to an epoch
    # int, silently producing an epoch 1000x too small (and therefore
    # corrupting every window_ts) whenever pd.to_datetime returned
    # microsecond resolution instead — with no exception raised anywhere.
    #
    # 1497513601 is 15/06/2017 08:00:01 read as a naive wall clock; the
    # dataset corrections then move it to the UTC instant the PCAPs use.
    ts = pd.Series(["15/06/2017 08:00:01", "15/06/2017 08:00:02", "not a date"])
    epoch = parse_cic_timestamp(ts, twelve_hour=False, utc_offset_hours=0.0)
    assert epoch.iloc[0] == 1497513601
    assert epoch.iloc[1] == 1497513602
    assert pd.isna(epoch.iloc[2])


def test_parse_cic_timestamp_lifts_local_wall_clock_to_the_utc_the_pcaps_use():
    # CIC-IDS2017's CSVs record the capture site's local clock (UNB, Atlantic
    # Daylight Time = UTC-3 in July 2017); the PCAPs carry true UTC epochs.
    # Reading the CSV as UTC put every flow 3 hours before its own packets,
    # so the (host_id, window_ts) join in join.py matched almost nothing and
    # all 11 packet-derived features were ~always zero. Friday's capture
    # opens at 08:59 local == 11:59 UTC, which is where the Friday PCAP's
    # first packet (11:59:50 UTC) actually is.
    epoch = parse_cic_timestamp(pd.Series(["7/7/2017 8:59"]))
    assert pd.Timestamp(epoch.iloc[0], unit="s") == pd.Timestamp("2017-07-07 11:59:00")


def test_parse_cic_timestamp_reads_the_hour_as_12_hour_with_no_meridiem():
    # The published CSVs print a 12-hour clock and drop the AM/PM marker
    # entirely, so the afternoon day-files say "1:00" for 13:00. Read
    # literally, every afternoon capture lands in the small hours and can
    # never meet its own packets. The captures run 09:00-17:00 local, so the
    # hours present are exactly {8..12} (morning) and {1..5} (afternoon) —
    # hours 6 and 7 never occur and the disambiguation is total.
    epoch = parse_cic_timestamp(pd.Series(["7/7/2017 1:00", "7/7/2017 11:00"]))
    # 13:00 local -> 16:00 UTC, and 11:00 local -> 14:00 UTC.
    assert pd.Timestamp(epoch.iloc[0], unit="s") == pd.Timestamp("2017-07-07 16:00:00")
    assert pd.Timestamp(epoch.iloc[1], unit="s") == pd.Timestamp("2017-07-07 14:00:00")


def test_parse_cic_timestamp_keeps_noon_at_noon():
    # The one hour a naive `hour < 8 -> hour + 12` rule gets wrong in the
    # other direction: 12:xx is noon, not midnight, and Monday/Wednesday both
    # carry tens of thousands of 12:xx rows.
    epoch = parse_cic_timestamp(pd.Series(["03/07/2017 12:30:00"]))
    assert pd.Timestamp(epoch.iloc[0], unit="s") == pd.Timestamp("2017-07-03 15:30:00")


def test_align_window_exact_multiples():
    epochs = pd.Series([0.0, 29.9, 30.0, 59.9, 60.0])
    w = align_window(epochs, 30)
    assert list(w) == [0, 0, 30, 30, 60]


def test_safe_div_zero_denominator():
    assert safe_div(5, 0) == 0.0
    assert safe_div(0, 0) == 0.0
    assert safe_div(10, 2) == 5.0


def test_shannon_entropy_single_bucket_is_zero():
    assert shannon_entropy([10]) == 0.0
    assert shannon_entropy([]) == 0.0


def test_shannon_entropy_uniform_is_positive():
    assert shannon_entropy([5, 5, 5, 5]) > 0.0


def test_build_state_rows_output_schema():
    flows = make_synthetic_flows(n_hosts=3, n_windows=50)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    assert list(states.columns) == ["host_id", "window_ts"] + FEATURE_ORDER
    assert not states[FEATURE_ORDER].isna().any().any()


def test_empty_windows_preserved_with_is_active_zero():
    # A host with a silent gap in the middle of its activity range must
    # still get zero-valued rows with is_active=0 for the gap — never a
    # dropped window, which would corrupt the time axis.
    flows = make_synthetic_flows(n_hosts=1, n_windows=20)
    gap_start, gap_end = flows["window_ts"].sort_values().unique()[8], flows["window_ts"].sort_values().unique()[12]
    flows = flows[~flows["window_ts"].between(gap_start, gap_end)].reset_index(drop=True)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)

    assert (states["is_active"].isin([0.0, 1.0])).all()
    gap_rows = states[states["window_ts"].between(gap_start, gap_end)]
    assert len(gap_rows) > 0, "silent windows were dropped instead of zero-filled"
    assert (gap_rows["is_active"] == 0.0).all()
    # Base (non-derived) features must be the literal zero fill. The
    # dynamics features (d_*, slope3_*) are computed AFTER filling and
    # legitimately go non-zero exactly at the active->silent transition
    # edges (a real drop-to-zero delta is informative, not a bug) — they
    # are excluded from this check.
    dynamic_features = {
        "d_syn_ratio", "d_dst_port_entropy", "d_out_degree", "d_new_peer_count",
        "d_iat_var", "d_retrans_rate", "slope3_syn_ratio", "slope3_dst_port_entropy",
        "slope3_out_degree", "slope3_iat_var",
    }
    base_features = [f for f in FEATURE_ORDER if f != "is_active" and f not in dynamic_features]
    assert (gap_rows[base_features] == 0.0).all().all()


def test_portscan_escalation_shows_rising_dst_port_entropy_and_new_peers():
    flows = make_synthetic_flows(
        n_hosts=2, n_windows=80, portscan_host_idx=0,
        portscan_start_window=40, portscan_len=15,
    )
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    scanner = states[states["host_id"] == "10.0.0.1"].sort_values("window_ts").reset_index(drop=True)

    pre_scan = scanner.iloc[35:40]
    during_scan = scanner.iloc[45:50]

    assert during_scan["dst_port_entropy"].mean() > pre_scan["dst_port_entropy"].mean()
    assert during_scan["new_peer_count"].mean() > pre_scan["new_peer_count"].mean()
    assert during_scan["syn_ratio"].mean() > pre_scan["syn_ratio"].mean()


def test_deltas_and_slopes_pad_zero_at_sequence_start():
    flows = make_synthetic_flows(n_hosts=1, n_windows=10)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    first_two = states.sort_values("window_ts").iloc[:2]
    assert (first_two["slope3_syn_ratio"] == 0.0).all()

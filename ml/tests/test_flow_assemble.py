"""Assembling CICFlowMeter-shaped flows from raw packets.

The model's 15 flow features were trained on CICFlowMeter's output. An
uploaded pcap has no such CSV, so these flows have to be reconstructed from
packets — and if the reconstruction differs systematically from
CICFlowMeter, every flow feature shifts and the model degrades silently.
That is the failure this module is written against, so the semantics that
were calibrated against the real CIC-IDS2017 capture are pinned here:

  * durations and inter-arrival times in MICROSECONDS (a CSV flow spanning
    112.74 seconds records 112,740,690)
  * flows are bidirectional, keyed on the unordered address/port pair plus
    protocol, with "forward" fixed by whichever side sent first
  * one 5-tuple recurs many times in a capture (port reuse), so packets must
    be split into separate flows on teardown and timeout rather than
    collapsed into one long-lived flow

The splitting rules below are not guesses about CICFlowMeter's source; each
one is measured off the published CSVs for the same captures, and the
measurement is cited on the test that pins it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.data.flow_assemble import (
    FIN, RST, SYN, ACK, PSH, URG,
    assemble_flows,
    decode_flag_bits,
)


_US_PER_S = 1_000_000.0


def _pkts(rows):
    """rows: (t, src, sport, dst, dport, proto, payload, frame, flags_hex)."""
    cols = ["frame_time_epoch", "ip_src", "src_port", "ip_dst", "dst_port",
            "ip_proto", "payload_len", "frame_len", "tcp_flags"]
    return pd.DataFrame(rows, columns=cols)


def test_decode_flag_bits_reads_the_hex_field():
    s = pd.Series(["0x0002", "0x0011", "0x0018", "", None])
    bits = decode_flag_bits(s)
    assert list(bits[SYN]) == [True, False, False, False, False]
    assert list(bits[FIN]) == [False, True, False, False, False]
    assert list(bits[ACK]) == [False, True, True, False, False]
    assert list(bits[PSH]) == [False, False, True, False, False]
    assert not bits[RST].any() and not bits[URG].any()


def test_flag_columns_record_presence_not_a_packet_count():
    """CICFlowMeter's `... Flag Count` columns are 0/1 — "did any packet in
    this flow carry this flag" — despite the name. Measured on Friday
    morning's 191,033 flows: every one of SYN/ACK/PSH/FIN/RST/URG takes only
    the values 0 and 1, while a real ACK count over the same flows reaches
    into the hundreds. Emitting true counts made ack_ratio (a per-window sum
    of this column over packet count) land at 1.0 against a reference of 0.5.
    """
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0012"),
        (2.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0018"),
        (3.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0018"),
        (4.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0010"),
    ])
    f = assemble_flows(p).iloc[0]
    assert f.ack_flag_count == 1       # four packets carry ACK
    assert f.syn_flag_count == 1       # two carry SYN
    assert f.psh_flag_count == 1       # two carry PSH
    assert f.rst_flag_count == 0 and f.urg_flag_count == 0


def test_a_timeout_split_keeps_the_original_direction():
    """CICFlowMeter carries the original src/dst into the flow it opens after
    a timeout, so a long download stays one-directional across all its
    segments. Taking the direction from each segment's own first packet
    instead flipped every continuation whose first packet came from the
    server — which showed up as forward byte totals ~90x the reference on the
    mean, with the reference's client-sends-little/receives-a-lot asymmetry
    washed out."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),     # client opens
        (1.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 1000, 1054, "0x0018"),  # server sends
        # ... past the timeout, and the next packet is the server's:
        (200.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 1000, 1054, "0x0018"),
        (201.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0010"),
    ])
    out = assemble_flows(p, flow_timeout_s=120.0)
    assert len(out) == 2
    # Both segments keep the client as the forward direction.
    assert list(out.src_ip) == ["10.0.0.1", "10.0.0.1"]
    assert list(out.total_len_fwd) == [10, 10]
    assert list(out.total_len_bwd) == [1000, 1000]


def test_a_teardown_split_re_reads_the_direction():
    """Unlike a timeout, a close removes the connection outright — what comes
    next is a new connection and takes its direction from its own first
    packet, even when that reverses the roles."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0014"),   # RST closes
        (2.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 99, 153, "0x0002"),  # other side opens
        (3.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0012"),
    ])
    out = assemble_flows(p)
    assert list(out.src_ip) == ["10.0.0.1", "10.0.0.2"]
    assert list(out.total_len_fwd) == [10, 99]


def test_one_exchange_becomes_one_bidirectional_flow():
    p = _pkts([
        (1000.0, "10.0.0.1", 1234, "10.0.0.2", 80, 6, 100, 154, "0x0002"),
        (1000.5, "10.0.0.2", 80, "10.0.0.1", 1234, 6, 200, 254, "0x0012"),
        (1001.0, "10.0.0.1", 1234, "10.0.0.2", 80, 6, 300, 354, "0x0010"),
    ])
    out = assemble_flows(p)
    assert len(out) == 1
    f = out.iloc[0]
    # direction is fixed by who spoke first
    assert f.src_ip == "10.0.0.1" and f.dst_ip == "10.0.0.2" and f.dst_port == 80
    assert f.total_fwd_packets == 2 and f.total_bwd_packets == 1
    assert f.total_len_fwd == 400 and f.total_len_bwd == 200
    assert f.flow_duration == pytest.approx(1_000_000.0)   # 1.0s in microseconds
    assert f.syn_flag_count == 1 and f.ack_flag_count == 1


def test_inter_arrival_times_are_microseconds_over_both_directions():
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0002"),
        (1.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 0, 54, "0x0012"),
        (4.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0010"),
    ])
    f = assemble_flows(p).iloc[0]
    assert f.flow_iat_mean == pytest.approx(2_000_000.0)   # (1s + 3s) / 2
    assert f.flow_iat_max == pytest.approx(3_000_000.0)


def test_the_timeout_caps_a_flow_s_total_duration_not_its_idle_gap():
    """CICFlowMeter's 120s timeout is measured from the flow's FIRST packet,
    not from the previous one. Measured on the published CSVs: of 191,033
    Friday-morning flows the longest is 119.999993s and none exceed 120s,
    which an idle-gap rule cannot produce. Splitting on idle gap instead let
    a chatty connection run for the whole capture, which is what put
    assembled flow_duration_mean 32x under the reference."""
    p = _pkts([
        # One packet a minute for five minutes: never idle for 120s, but well
        # past a 120s cap on total duration.
        (float(60 * i), "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0010")
        for i in range(6)
    ])
    out = assemble_flows(p, flow_timeout_s=120.0)
    # Each flow runs 120s from its own first packet: t=0,60,120 then
    # t=180,240,300. An idle-gap rule would see no gap over 60s and emit one.
    assert len(out) == 2
    assert list(out.flow_duration) == [120 * _US_PER_S, 120 * _US_PER_S]


def test_a_reused_five_tuple_is_split_on_the_flow_timeout():
    """The same 5-tuple recurs throughout a real capture. Collapsing those
    into one flow would inflate flow_duration_mean and deflate
    active_flow_count for every window the host appears in."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0010"),
        (500.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (501.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0010"),
    ])
    out = assemble_flows(p, flow_timeout_s=120.0)
    assert len(out) == 2
    assert list(out.total_fwd_packets) == [2, 2]


def test_rst_closes_the_flow_immediately():
    """A reset ends the connection outright; the next packet on the tuple is
    a new one."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0014"),   # RST
        (2.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),   # new SYN
        (3.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0012"),
    ])
    out = assemble_flows(p)
    assert len(out) == 2
    assert list(out.rst_flag_count) == [1, 0]


def test_one_fin_does_not_close_the_flow_but_the_second_does():
    """A graceful TCP close is two FINs, one per direction, and the ACKs that
    follow the first FIN belong to the SAME flow. Splitting on the first FIN
    cut nearly every real connection in half: it drove fin_ratio to a nonzero
    value in 74% of assembled active windows against 3.7% of reference ones,
    and halved pkts_per_flow_mean."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0011"),   # FIN (fwd)
        (2.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0010"),   # ACK of it
        (3.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0011"),   # FIN (bwd)
        (4.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),   # new connection
        (5.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 10, 64, "0x0012"),
    ])
    out = assemble_flows(p)
    assert len(out) == 2
    assert list(out.fin_flag_count) == [1, 0]
    # The four packets of the close, including the ACK between the two FINs.
    assert int(out.iloc[0].total_fwd_packets + out.iloc[0].total_bwd_packets) == 4


def test_lone_packets_are_not_emitted_as_flows():
    """CICFlowMeter never publishes a one-packet flow: across Friday morning
    (191,033 flows), Friday afternoon's port scan (286,467) and Monday
    (529,918) there is not a single one. Emitting them adds a population of
    zero-duration, zero-byte flows that the model never saw in training and
    that drags every per-window median toward zero."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),   # unanswered
        (0.0, "10.0.0.3", 2, "10.0.0.4", 80, 6, 10, 64, "0x0002"),
        (0.5, "10.0.0.4", 80, "10.0.0.3", 2, 6, 10, 64, "0x0014"),
    ])
    out = assemble_flows(p)
    assert len(out) == 1
    assert out.iloc[0].src_ip == "10.0.0.3"


def test_udp_and_icmp_survive_without_tcp_flags():
    """Only TCP carries flags; a UDP or ICMP flow must still produce a row
    with zeroed flag counts rather than being dropped."""
    p = _pkts([
        (0.0, "10.0.0.1", 53, "10.0.0.2", 53, 17, 40, 68, ""),
        (0.2, "10.0.0.2", 53, "10.0.0.1", 53, 17, 60, 88, ""),
        (1.0, "10.0.0.1", 0, "10.0.0.2", 0, 1, 0, 84, None),
        (1.1, "10.0.0.2", 0, "10.0.0.1", 0, 1, 0, 84, None),
    ])
    out = assemble_flows(p)
    assert len(out) == 2
    assert (out.syn_flag_count == 0).all()
    assert set(out.protocol) == {17, 1}


def test_empty_input_returns_the_right_columns_not_an_exception():
    out = assemble_flows(_pkts([]))
    assert len(out) == 0
    for col in ("src_ip", "dst_ip", "dst_port", "flow_duration", "total_fwd_packets",
                "total_len_fwd", "syn_flag_count", "flow_iat_mean", "flow_iat_max"):
        assert col in out.columns


def test_a_two_packet_one_sided_flow_keeps_zero_backward_totals():
    """A retransmitted probe with no answer is two packets in one direction.
    The backward totals must be 0, never NaN, which would poison every
    window aggregate the flow lands in."""
    p = _pkts([
        (5.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0002"),
        (6.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0002"),
    ])
    f = assemble_flows(p).iloc[0]
    assert f.flow_duration == pytest.approx(_US_PER_S)
    assert f.flow_iat_mean == pytest.approx(_US_PER_S) and f.flow_iat_max == pytest.approx(_US_PER_S)
    assert f.total_bwd_packets == 0 and f.total_len_bwd == 0

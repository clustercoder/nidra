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
    be split into separate flows on teardown and idle timeout rather than
    collapsed into one long-lived flow
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
    assert f.syn_flag_count == 2 and f.ack_flag_count == 2


def test_inter_arrival_times_are_microseconds_over_both_directions():
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0002"),
        (1.0, "10.0.0.2", 80, "10.0.0.1", 1, 6, 0, 54, "0x0012"),
        (4.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0010"),
    ])
    f = assemble_flows(p).iloc[0]
    assert f.flow_iat_mean == pytest.approx(2_000_000.0)   # (1s + 3s) / 2
    assert f.flow_iat_max == pytest.approx(3_000_000.0)


def test_a_reused_five_tuple_is_split_on_idle_timeout():
    """The same 5-tuple recurs throughout a real capture. Collapsing those
    into one flow would inflate flow_duration_mean and deflate
    active_flow_count for every window the host appears in."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0010"),
        (500.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
    ])
    out = assemble_flows(p, idle_timeout_s=120.0)
    assert len(out) == 2
    assert list(out.total_fwd_packets) == [2, 1]


def test_teardown_starts_a_new_flow():
    """A FIN or RST closes the flow; the next packet on the same tuple is a
    new connection, which is how CICFlowMeter splits a busy port pair."""
    p = _pkts([
        (0.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),
        (1.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0011"),   # FIN
        (2.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 10, 64, "0x0002"),   # new SYN
    ])
    out = assemble_flows(p)
    assert len(out) == 2


def test_udp_and_icmp_survive_without_tcp_flags():
    """Only TCP carries flags; a UDP or ICMP flow must still produce a row
    with zeroed flag counts rather than being dropped."""
    p = _pkts([
        (0.0, "10.0.0.1", 53, "10.0.0.2", 53, 17, 40, 68, ""),
        (0.2, "10.0.0.2", 53, "10.0.0.1", 53, 17, 60, 88, ""),
        (1.0, "10.0.0.1", 0, "10.0.0.2", 0, 1, 0, 84, None),
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


def test_single_packet_flow_has_zero_duration_and_no_iat():
    """A lone packet is a real flow — a scan produces thousands of them —
    and must not become NaN, which would poison the window aggregate."""
    p = _pkts([(5.0, "10.0.0.1", 1, "10.0.0.2", 80, 6, 0, 54, "0x0002")])
    f = assemble_flows(p).iloc[0]
    assert f.flow_duration == 0.0
    assert f.flow_iat_mean == 0.0 and f.flow_iat_max == 0.0
    assert f.total_bwd_packets == 0 and f.total_len_bwd == 0

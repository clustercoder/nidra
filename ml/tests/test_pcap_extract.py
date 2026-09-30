from nidra.data.pcap_extract import TSHARK_FIELDS, _field_to_column, _finalize_chunk, build_tshark_argv


def test_build_tshark_argv_uses_occurrence_first_to_avoid_multivalue_rows():
    """Real bug found against Monday-WorkingHours.pcap: without -E
    occurrence=f, an ICMP error packet's embedded original-packet headers
    make tshark emit ip.src/ip.dst twice, joined by the same comma used as
    the field separator — silently corrupting ~0.25% of rows. occurrence=f
    takes only the first value, keeping row width fixed."""
    argv = build_tshark_argv("/tmp/fake.pcap")
    assert "-E" in argv
    assert "occurrence=f" in argv
    assert "separator=," in argv
    assert "quote=n" in argv


def test_build_tshark_argv_applies_time_filter():
    argv = build_tshark_argv("/tmp/fake.pcap", time_filter="frame.time_epoch >= 100")
    joined = " ".join(argv)
    assert "frame.time_epoch >= 100" in joined
    assert "(ip) and (frame.time_epoch >= 100)" in joined


def _row(**overrides) -> list[str]:
    columns = [_field_to_column(f) for f in TSHARK_FIELDS]
    defaults = {
        "frame_time_epoch": "1499082958.598308000",
        "ip_src": "192.168.10.1",
        "ip_dst": "192.168.10.2",
        "tcp_srcport": "443",
        "tcp_dstport": "51000",
        "udp_srcport": "",
        "udp_dstport": "",
        "ip_proto": "6",
        "ip_ttl": "64",
        "tcp_window_size_value": "8192",
        "ip_flags_mf": "0",
        "ip_frag_offset": "0",
        "frame_len": "100",
        "tcp_len": "40",
        "tcp_flags": "0x0018",
        "tcp_analysis_retransmission": "",
    }
    defaults.update(overrides)
    return [defaults[c] for c in columns]


def test_finalize_chunk_coalesces_tcp_and_udp_ports():
    df = _finalize_chunk([_row()], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert df.iloc[0]["src_port"] == 443
    assert df.iloc[0]["dst_port"] == 51000


def test_finalize_chunk_udp_row_falls_back_to_udp_ports():
    row = _row(tcp_srcport="", tcp_dstport="", udp_srcport="53", udp_dstport="51000", ip_proto="17")
    df = _finalize_chunk([row], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert df.iloc[0]["src_port"] == 53
    assert df.iloc[0]["dst_port"] == 51000


def test_finalize_chunk_empty_retransmission_field_means_zero_not_dropped():
    df = _finalize_chunk([_row(tcp_analysis_retransmission="")], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert df.iloc[0]["is_retransmission"] == 0


def test_finalize_chunk_retransmission_flag_set():
    df = _finalize_chunk([_row(tcp_analysis_retransmission="1")], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert df.iloc[0]["is_retransmission"] == 1


def test_finalize_chunk_fragment_flag_from_more_fragments_bit():
    df = _finalize_chunk([_row(ip_flags_mf="1")], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert df.iloc[0]["is_fragment"] == 1


def test_finalize_chunk_fragment_flag_from_nonzero_offset():
    df = _finalize_chunk([_row(ip_frag_offset="185")], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert df.iloc[0]["is_fragment"] == 1


def test_finalize_chunk_drops_rows_missing_core_identity_fields():
    good = _row()
    bad = _row(ip_src="")
    df = _finalize_chunk([good, bad], [_field_to_column(f) for f in TSHARK_FIELDS])
    assert len(df) == 1

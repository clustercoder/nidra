import io

import pandas as pd

from nidra.data.flow_load import load_cicflowmeter_csv


def _write_csv(tmp_path, text: str):
    p = tmp_path / "flows.csv"
    p.write_text(text)
    return p


def test_strips_leading_spaces_from_columns(tmp_path):
    text = (
        " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
        "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
        "80,100,2,2,120,240,BENIGN\n"
    )
    p = _write_csv(tmp_path, text)
    df, report = load_cicflowmeter_csv(p)
    assert "dst_port" in df.columns
    assert "flow_duration" in df.columns
    assert report.accepted_rows == 1


def test_drops_duplicate_header_row_mid_file(tmp_path):
    text = (
        " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
        "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
        "80,100,2,2,120,240,BENIGN\n"
        " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
        "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
        "443,200,3,3,150,300,BENIGN\n"
    )
    p = _write_csv(tmp_path, text)
    df, report = load_cicflowmeter_csv(p)
    assert report.accepted_rows == 2
    assert report.dropped_rows == 1
    assert report.input_rows == 3


def test_missing_flag_columns_default_to_zero_not_dropped(tmp_path):
    text = (
        " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
        "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
        "80,100,2,2,120,240,BENIGN\n"
    )
    p = _write_csv(tmp_path, text)
    df, _ = load_cicflowmeter_csv(p)
    assert (df["syn_flag_count"] == 0).all()


def test_falls_back_to_latin1_on_non_utf8_bytes(tmp_path):
    """Real CIC-IDS2017 file (Thursday-WorkingHours-Morning-WebAttacks) has
    a Windows-1252 en-dash (byte 0x96) in its 'Web Attack' labels, which is
    not valid UTF-8 and crashes a plain pd.read_csv without this fallback."""
    header = (
        " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
        "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
    ).encode("utf-8")
    row = "80,100,2,2,120,240,Web Attack ".encode("utf-8") + b"\x96" + " XSS\n".encode("utf-8")
    p = tmp_path / "flows.csv"
    p.write_bytes(header + row)

    df, report = load_cicflowmeter_csv(p)
    assert report.accepted_rows == 1
    assert "Web Attack" in df["label"].iloc[0]


def test_row_cap_stops_reading_early_without_consuming_whole_file(tmp_path):
    header = (
        " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
        "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
    )
    rows = "".join(f"80,100,2,2,120,240,BENIGN\n" for _ in range(10))
    p = _write_csv(tmp_path, header + rows)

    df, report = load_cicflowmeter_csv(p, chunksize=3, row_cap=5)
    # row_cap stops after the chunk that crosses the cap, not mid-chunk —
    # with chunksize=3 that's the second chunk (rows 4-6), i.e. 6 input rows,
    # not exactly 5. The guarantee is "at least row_cap, never the whole
    # file", not an exact count.
    assert report.input_rows < 10
    assert report.input_rows >= 5
    assert len(df) == report.input_rows

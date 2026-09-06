import io

import pandas as pd

from horizon.data.flow_load import load_cicflowmeter_csv


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

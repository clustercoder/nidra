import pandas as pd
import pytest

from nidra.scripts.portscan_sanity_plot import pick_busiest_portscan_host


def test_pick_busiest_portscan_host_picks_host_with_most_attack_windows():
    df = pd.DataFrame({
        "host_id": ["a"] * 5 + ["b"] * 2 + ["c"] * 10,
        "stage_label": (["portscan"] * 3 + ["benign"] * 2) + ["benign"] * 2 + (["portscan"] * 8 + ["benign"] * 2),
    })
    assert pick_busiest_portscan_host(df) == "c"


def test_pick_busiest_portscan_host_raises_when_all_benign():
    df = pd.DataFrame({"host_id": ["a", "b"], "stage_label": ["benign", "benign"]})
    with pytest.raises(ValueError):
        pick_busiest_portscan_host(df)

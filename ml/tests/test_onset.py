"""Onset targets: strictly forward-looking, undefined inside an episode,
identical between the training arrays and the evaluation set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nidra.data.dataset import build_windowed_arrays
from nidra.data.labels import reattach_risk_label
from nidra.data.onset import episode_geometry, infer_window_seconds, onset_targets
from nidra.data.schema import FEATURE_ORDER


def _table(window_seconds: int = 60, n: int = 40, attack_windows=(20, 21, 22, 30)) -> pd.DataFrame:
    rows = []
    for i in range(n):
        feats = {f: 0.0 for f in FEATURE_ORDER}
        feats["is_active"] = 1.0
        stage = "recon" if i in attack_windows else "benign"
        rows.append({"host_id": "h1", "window_ts": 1000 + i * window_seconds, "stage_label": stage,
                     "episode_id": -1, **feats})
    df = pd.DataFrame(rows)
    df["risk_label"] = 0
    return df


def test_infer_window_seconds_from_contiguous_table():
    assert infer_window_seconds(_table(60)) == 60
    assert infer_window_seconds(_table(30)) == 30


def test_geometry_minutes_to_onset_and_inside():
    df = _table(60, attack_windows=(20, 21, 22, 30))
    origin_ts = df["window_ts"].to_numpy()
    hosts = df["host_id"].to_numpy()
    inside, to_onset, key = episode_geometry(df, hosts, origin_ts, 60, merge_gap_windows=5)
    # windows 20-22 and 30 merge into one episode (gap of 7 benign windows > 5? no: 23..29 is 7 windows)
    # gap 7 > merge gap 5 -> two episodes
    assert inside[20] and inside[22] and inside[30]
    assert not inside[23]
    assert to_onset[19] == pytest.approx(1.0)
    assert to_onset[15] == pytest.approx(5.0)
    assert to_onset[25] == pytest.approx(5.0)  # next onset is window 30
    assert np.isinf(to_onset[31])
    assert key[19] == "h1@" + str(1000 + 20 * 60)


def test_onset_targets_zero_inside_episode_and_monotone_in_h():
    inside = np.array([False, False, True, False])
    to_onset = np.array([2.0, 12.0, 0.0, np.inf])
    t = onset_targets(inside, to_onset, (1, 3, 5, 10, 15, 30))
    assert t[0].tolist() == [0, 1, 1, 1, 1, 1]
    assert t[1].tolist() == [0, 0, 0, 0, 1, 1]
    assert t[2].tolist() == [0, 0, 0, 0, 0, 0]
    assert t[3].tolist() == [0, 0, 0, 0, 0, 0]


def test_windowed_arrays_carry_onset_geometry():
    df = reattach_risk_label(_table(60, n=60, attack_windows=(40, 41, 42)), horizon_k=3)
    arrays = build_windowed_arrays(df, L=5, K=3)
    assert arrays.inside_episode is not None and arrays.minutes_to_onset is not None
    targets = arrays.onset_targets((1, 3, 5))
    # origin at window 39 (one minute before onset) -> 1 at every horizon
    i39 = np.where(arrays.origin_ts == 1000 + 39 * 60)[0][0]
    assert targets[i39].tolist() == [1, 1, 1]
    i35 = np.where(arrays.origin_ts == 1000 + 35 * 60)[0][0]
    assert targets[i35].tolist() == [0, 0, 1]
    i41 = np.where(arrays.origin_ts == 1000 + 41 * 60)[0][0]
    assert arrays.inside_episode[i41] and targets[i41].tolist() == [0, 0, 0]
    # the published risk label at 39 is also 1 (attack within K=3) — the two agree there,
    # but differ at 41 where risk_label is 1 (ongoing) and onset is masked
    assert arrays.risk_label[i39] == 1 and arrays.risk_label[i41] == 1


def test_two_episodes_on_one_host_keep_separate_keys():
    """A host that attacks twice (SSH-Patator on Tuesday, Heartbleed on
    Wednesday from the same attacker IP) has two episodes; rows inside the
    first must not be re-keyed to the second."""
    df = _table(60, n=80, attack_windows=(10, 11, 12, 60, 61))
    origin_ts = df["window_ts"].to_numpy()
    hosts = df["host_id"].to_numpy()
    inside, to_onset, key = episode_geometry(df, hosts, origin_ts, 60, merge_gap_windows=5)
    k1 = "h1@" + str(1000 + 10 * 60)
    k2 = "h1@" + str(1000 + 60 * 60)
    assert key[11] == k1 and key[61] == k2
    assert inside[11] and inside[61]
    assert key[9] == k1 and key[59] == k2
    assert key[30] == k2 and to_onset[30] == pytest.approx(30.0)
    assert to_onset[11] == pytest.approx(49.0)  # minutes from inside episode 1 to the onset of episode 2

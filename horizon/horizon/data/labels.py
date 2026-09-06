"""Stage and risk label construction.

Future information is allowed here (labels are supervision targets) but
never leaks into the model INPUT — that boundary is enforced by keeping this
module entirely separate from windowize.py's feature construction.

The dataset-label -> tactic mapping is a curated PRESENTATION mapping, not
ATT&CK technique-level ground truth. CIC-IDS2017 does not support
technique-level resolution. Two simplifications are called out explicitly:

  - "Infiltration" is mapped wholesale to `lateral` (the PRD's own table
    describes it as "Initial Access -> Lateral Movement"; the dataset gives
    no per-flow signal to split the two sub-phases, so the later, more
    severe tactic is used as a single label for the whole labelled episode).
  - The dataset table in the spec maps DoS/DDoS to "Impact", which is not
    one of the six stage categories used here (benign, recon,
    initial_access, lateral, c2, exfil — chosen to match the frozen
    stage-head output dimension). DoS/DDoS/Heartbleed are mapped to `exfil`
    as the closest available terminal/impact-like bucket. This is a
    deliberate scoping choice, documented rather than silently made.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from horizon.data.schema import STAGE_INDEX, STAGE_LABELS

# Ordered (pattern, stage) rules, checked in order — first match wins so
# more specific labels can be listed before broad fallbacks.
_LABEL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"portscan", re.I), "recon"),
    (re.compile(r"ftp[\s\-]?patator", re.I), "initial_access"),
    (re.compile(r"ssh[\s\-]?patator", re.I), "initial_access"),
    (re.compile(r"web attack", re.I), "initial_access"),
    (re.compile(r"heartbleed", re.I), "initial_access"),
    (re.compile(r"infiltrat", re.I), "lateral"),
    (re.compile(r"\bbot\b", re.I), "c2"),
    (re.compile(r"ddos", re.I), "exfil"),
    (re.compile(r"\bdos\b", re.I), "exfil"),
    (re.compile(r"benign", re.I), "benign"),
]


def map_label_to_stage(raw_label: str) -> str:
    """Map one raw CIC-IDS2017 `Label` value to a curated tactic bucket.
    Unrecognized labels fall back to 'benign' rather than raising, but this
    is logged upstream via the unmapped-label report in `label_stage_table`.
    """
    label = str(raw_label).strip()
    for pattern, stage in _LABEL_RULES:
        if pattern.search(label):
            return stage
    return "benign"


def label_stage_table(flows_windowed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Given windowed flow rows (host_id via src_ip, window_ts, label),
    return one row per (host_id, window_ts) with the most severe stage
    present in that window, plus a report of unmapped raw label strings.
    """
    df = flows_windowed.copy()
    df["stage"] = df["label"].map(map_label_to_stage)

    matched = df["label"].apply(lambda l: any(p.search(str(l)) for p, _ in _LABEL_RULES))
    unmapped = sorted(df.loc[~matched, "label"].unique().tolist())

    severity = {name: i for i, name in enumerate(STAGE_LABELS)}  # benign=0 least severe
    df["severity"] = df["stage"].map(severity)

    idx = df.groupby(["src_ip", "window_ts"])["severity"].idxmax()
    winners = df.loc[idx, ["src_ip", "window_ts", "stage"]].rename(columns={"src_ip": "host_id"})
    return winners.reset_index(drop=True), {"unmapped_labels": unmapped}


def attach_risk_label(state_table: pd.DataFrame, stage_table: pd.DataFrame, horizon_k: int, window_seconds: int) -> pd.DataFrame:
    """Attach stage_label (per host/window, benign if no attack flow present)
    and risk_label = 1 if any attack-stage window occurs in (t, t+K], else 0.

    Uses future windows ONLY to construct these two target columns — this
    function must never be called anywhere near model-input construction.
    """
    df = state_table.merge(stage_table, on=["host_id", "window_ts"], how="left")
    df["stage"] = df["stage"].fillna("benign")
    df = df.sort_values(["host_id", "window_ts"]).reset_index(drop=True)

    is_attack = (df["stage"] != "benign").astype(int)
    df["_is_attack"] = is_attack

    risk_labels = np.zeros(len(df), dtype=int)
    for host, g in df.groupby("host_id", sort=False):
        g_idx = g.index.to_numpy()
        attack = g["_is_attack"].to_numpy()
        n = len(attack)
        # future window in (t, t+K] means the next K entries in this host's
        # own chronological (gap-filled) sequence — window spacing is
        # uniform (WINDOW_SECONDS) after _fill_empty_windows, so index
        # offset == time offset.
        for i in range(n):
            hi = min(n, i + 1 + horizon_k)
            risk_labels[g_idx[i]] = 1 if attack[i + 1 : hi].any() else 0

    df["risk_label"] = risk_labels
    df["stage_label"] = df["stage"]
    return df.drop(columns=["_is_attack", "stage"])

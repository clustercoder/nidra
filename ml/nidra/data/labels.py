"""Stage and risk label construction.

Future information is allowed here (labels are supervision targets) but
never leaks into the model INPUT — that boundary is enforced by keeping this
module entirely separate from windowize.py's feature construction.

The dataset-label -> tactic mapping is a curated PRESENTATION mapping, not
ATT&CK technique-level ground truth. CIC-IDS2017 (the actual dataset this
project trains and evaluates on) does not support technique-level
resolution. Two simplifications are called out explicitly:

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

Dormant CSE-CIC-IDS2018 label support: `_LABEL_RULES` below also covers
CSE-CIC-IDS2018's raw `Label` column strings (e.g. "FTP-BruteForce",
"SSH-Bruteforce", "Infilteration" — note the dataset's own misspelling with
an extra "e" — "Brute Force -Web", "Brute Force -XSS", "SQL Injection").
This project does NOT use CSE-CIC-IDS2018 (an earlier plan to switch to it
was abandoned; CIC-IDS2017 is the sole real dataset — see
../../REAL_DATA_RESULTS.md) — these extra rules are simply harmless,
unused compatibility left in place in case a CSE-CIC-IDS2018 slice is ever
added later, cross-referenced only against the CIC website's published
attack/day table, never against a real downloaded file. Any label that
fails to match a rule falls back to "benign" but is also surfaced in
`label_stage_table`'s `unmapped_labels` report.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from nidra.data.schema import STAGE_INDEX, STAGE_LABELS

# Ordered (pattern, stage) rules, checked in order — first match wins so
# more specific labels can be listed before broad fallbacks.
_LABEL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"portscan", re.I), "recon"),
    # CIC-IDS2017 phrasing:
    (re.compile(r"ftp[\s\-]?patator", re.I), "initial_access"),
    (re.compile(r"ssh[\s\-]?patator", re.I), "initial_access"),
    (re.compile(r"web attack", re.I), "initial_access"),
    # CSE-CIC-IDS2018 phrasing (per the CIC's own published attack/day
    # table — dormant, unused rules; this project does not use that dataset):
    (re.compile(r"ftp[\s\-]?bruteforce", re.I), "initial_access"),
    (re.compile(r"ssh[\s\-]?bruteforce", re.I), "initial_access"),
    (re.compile(r"brute\s*force\s*-?\s*(web|xss)", re.I), "initial_access"),
    (re.compile(r"sql injection", re.I), "initial_access"),
    (re.compile(r"heartbleed", re.I), "initial_access"),
    # "infilt" (not "infiltrat") deliberately also matches CSE-CIC-IDS2018's
    # own misspelling "Infilteration" (extra "e"), as well as 2017's
    # "Infiltration".
    (re.compile(r"infilt", re.I), "lateral"),
    (re.compile(r"\bbot\b", re.I), "c2"),
    (re.compile(r"ddos", re.I), "exfil"),
    (re.compile(r"\bdos\b", re.I), "exfil"),
    (re.compile(r"benign", re.I), "benign"),
]

# CTU-13's own taxonomy. Its `Label` column is a sentence, not a class name
# ("flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted"), and two of its
# families of strings read like attacks under the CIC rules above:
# `flow=Background-TCP-Attempt` carries the word that marks botnet scanning
# and `flow=To-Background-CVUT-Proxy` carries a hostname. Only flows whose
# label states the botnet as the SOURCE are attacks — CTU-13 labels every
# such flow `From-Botnet` and there are no `To-Botnet` flows in the release
# (verified across all 13 scenarios) — so the dialect is gated on that
# prefix before any behaviour rule is consulted.
#
# The behaviour -> tactic mapping is curated PRESENTATION, the same standing
# as the CIC one, and follows the same two conventions: flood/impact
# behaviour lands in `exfil` (the closest terminal bucket among the six
# frozen stages), and scanning lands in `recon`.
_CTU_BOTNET_MARKER = re.compile(r"from-botnet", re.I)
_CTU_RULES: list[tuple[re.Pattern, str]] = [
    # command and control first: a CC channel is the defining botnet behaviour
    (re.compile(r"\bcc\d*\b", re.I), "c2"),
    (re.compile(r"\birc\b", re.I), "c2"),
    # payload retrieval
    (re.compile(r"binary-download", re.I), "initial_access"),
    # monetisation / impact
    (re.compile(r"spam", re.I), "exfil"),
    (re.compile(r"http-ad", re.I), "exfil"),
    (re.compile(r"click", re.I), "exfil"),
    (re.compile(r"ddos", re.I), "exfil"),
    (re.compile(r"icmp", re.I), "exfil"),
    # scanning / failed connection sweeps
    (re.compile(r"attempt", re.I), "recon"),
    (re.compile(r"\bps\b", re.I), "recon"),
]
#: Anything labelled Botnet that matches no behaviour rule above is still an
#: attack. A new sub-label must never be able to become a negative.
_CTU_DEFAULT_ATTACK_STAGE = "c2"


def map_label_to_stage(raw_label: str, dialect: str = "cic") -> str:
    """Map one raw `Label` value to a curated tactic bucket.

    `dialect` selects the label taxonomy: "cic" for CIC-IDS2017's class
    names (the default; CSE-CIC-IDS2018 strings are also matched but
    dormant — see module docstring), "ctu" for CTU-13's `flow=...`
    sentences. The two are not interchangeable and there is no
    auto-detection: a CTU label read under the CIC rules silently scores
    every attack as benign, which is exactly the kind of quiet corruption
    that shows up only as a suspiciously low prevalence.

    Unrecognized labels fall back to 'benign' rather than raising, and are
    surfaced by the unmapped-label report in `label_stage_table`.
    """
    label = str(raw_label).strip()
    if dialect == "ctu":
        if not _CTU_BOTNET_MARKER.search(label):
            return "benign"
        for pattern, stage in _CTU_RULES:
            if pattern.search(label):
                return stage
        return _CTU_DEFAULT_ATTACK_STAGE
    if dialect != "cic":
        raise ValueError(f"unknown label dialect {dialect!r}; expected 'cic' or 'ctu'")
    for pattern, stage in _LABEL_RULES:
        if pattern.search(label):
            return stage
    return "benign"


def _label_is_recognised(label: str, dialect: str) -> bool:
    """Whether any rule of this dialect actually matched, as opposed to the
    value falling through to the benign default."""
    text = str(label).strip()
    if dialect == "ctu":
        # A Background/Normal flow is a recognised negative, not an unmapped
        # label; only a string that is neither botnet nor background is one.
        return bool(
            _CTU_BOTNET_MARKER.search(text)
            or re.search(r"background|normal", text, re.I)
        )
    return any(p.search(text) for p, _ in _LABEL_RULES)


def label_stage_table(flows_windowed: pd.DataFrame, dialect: str = "cic") -> tuple[pd.DataFrame, dict]:
    """Given windowed flow rows (host_id via src_ip, window_ts, label),
    return one row per (host_id, window_ts) with the most severe stage
    present in that window, plus a report of unmapped raw label strings.

    `dialect` picks the label taxonomy — see `map_label_to_stage`.
    """
    df = flows_windowed.copy()
    # One map() per DISTINCT label rather than per row: a CTU scenario
    # carries millions of flows drawn from a few hundred label strings.
    vocabulary = pd.unique(df["label"].astype(str))
    stage_of = {v: map_label_to_stage(v, dialect=dialect) for v in vocabulary}
    recognised = {v: _label_is_recognised(v, dialect) for v in vocabulary}
    df["stage"] = df["label"].astype(str).map(stage_of)

    matched = df["label"].astype(str).map(recognised)
    unmapped = sorted(df.loc[~matched, "label"].unique().tolist())

    severity = {name: i for i, name in enumerate(STAGE_LABELS)}  # benign=0 least severe
    df["severity"] = df["stage"].map(severity)

    idx = df.groupby(["src_ip", "window_ts"])["severity"].idxmax()
    winners = df.loc[idx, ["src_ip", "window_ts", "stage"]].rename(columns={"src_ip": "host_id"})
    return winners.reset_index(drop=True), {"unmapped_labels": unmapped}


def risk_label_from_attack_flags(is_attack: np.ndarray, horizon_k: int) -> np.ndarray:
    """risk_label[i] = 1 iff any of is_attack[i+1 .. i+K] is 1, for ONE
    host's chronological, gap-filled sequence. Vectorized forward-looking
    window max (the per-row Python loop this replaces took minutes on a
    real day of ~500k windows). Rows near the end of the sequence look at
    the windows that exist, exactly as before."""
    attack = np.asarray(is_attack, dtype="int64")
    n = len(attack)
    if n == 0:
        return np.zeros(0, dtype="int64")
    # cumulative count of attack windows: cum[j] = attack[0..j-1].sum()
    cum = np.concatenate([[0], np.cumsum(attack)])
    idx = np.arange(n)
    hi = np.minimum(n, idx + 1 + horizon_k)
    ahead = cum[hi] - cum[idx + 1]
    return (ahead > 0).astype("int64")


def attach_risk_label(state_table: pd.DataFrame, stage_table: pd.DataFrame, horizon_k: int, window_seconds: int) -> pd.DataFrame:
    """Attach stage_label (per host/window, benign if no attack flow present)
    and risk_label = 1 if any attack-stage window occurs in (t, t+K], else 0.

    Uses future windows ONLY to construct these two target columns — this
    function must never be called anywhere near model-input construction.
    """
    df = state_table.merge(stage_table, on=["host_id", "window_ts"], how="left")
    df["stage"] = df["stage"].fillna("benign")
    df = df.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    df["stage_label"] = df["stage"]
    df = df.drop(columns=["stage"])
    return reattach_risk_label(df, horizon_k)


def reattach_risk_label(labelled: pd.DataFrame, horizon_k: int) -> pd.DataFrame:
    """(Re)compute `risk_label` from `stage_label` for the given horizon K.
    The cached windowed tables carry stage_label, which does not depend on
    K; risk_label does, so it is derived here for whatever K the current
    config uses. Rows must be per-host chronological and gap-filled (uniform
    window spacing), which is what windowize guarantees, so an index offset
    equals a time offset."""
    df = labelled.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    is_attack = (df["stage_label"] != "benign").to_numpy(dtype="int64")
    risk = np.zeros(len(df), dtype="int64")
    host_codes, host_starts = np.unique(df["host_id"].to_numpy(), return_index=True)
    bounds = np.append(np.sort(host_starts), len(df))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        risk[lo:hi] = risk_label_from_attack_flags(is_attack[lo:hi], horizon_k)
    df["risk_label"] = risk
    return df

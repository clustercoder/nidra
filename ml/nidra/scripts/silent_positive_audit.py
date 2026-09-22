"""How many forecast positives carry no information in the state at all?

    python -m nidra.scripts.silent_positive_audit --config <run>/config.yaml \\
        --splits train,val --out reports/tables/ctu_silent_positives.md

`risk_label[t]` is 1 when an attack starts within the horizon, so a positive
row need not contain any attack traffic — it is the window BEFORE. If the host
happens to be silent in that window, its 45 features sit on the scaler's silent
state, which is the same vector every other silent window has. Those rows are
bit-identical to thousands of negatives, so no function of the state can order
them above those negatives. History can; the state cannot.

That makes this a ceiling, not a complaint. Within the floor stratum the best
achievable AP from the state alone is the stratum's own prevalence, and any
state-only head that appears to beat it on a split is reading something that
is not in the state — worth checking for, which is why the number is written
down rather than asserted.

The audit is per split and reports both the strict floor (all features within
tolerance of the silent state) and the weaker `is_active == 0`, because a
backward-looking delta or slope can be non-zero in a silent window when the
PREVIOUS window was not: that trace of history inside the state vector is the
difference between the two counts.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Scaled-space tolerance for "this row is the silent state". Tight: the point
#: is exact indistinguishability, not approximate quietness.
FLOOR_ATOL = 1e-5


@dataclass(frozen=True)
class SilentAudit:
    """One split's accounting of positives that carry no state information."""

    split: str
    n_rows: int
    n_positive: int
    n_preonset: int            # risk_label == 1 and stage_label == benign
    n_preonset_inactive: int   # ... and is_active == 0
    n_floor_positive: int      # ... and every feature at the silent state
    n_floor_rows: int          # all rows at the silent state, both classes

    @property
    def floor_prevalence(self) -> float:
        """The best AP any function of the STATE can reach inside the floor
        stratum: every row there is the same vector, so the only ranking
        available is a constant one."""
        return self.n_floor_positive / self.n_floor_rows if self.n_floor_rows else float("nan")

    @property
    def share_of_positives(self) -> float:
        return self.n_floor_positive / self.n_positive if self.n_positive else float("nan")


def audit_split(scaled: np.ndarray, is_active: np.ndarray, risk_label: np.ndarray,
                stage_label: np.ndarray, zero_state: np.ndarray, split: str) -> SilentAudit:
    scaled = np.asarray(scaled, dtype="float32")
    if scaled.shape[1] != len(zero_state):
        raise ValueError(f"{split}: {scaled.shape[1]} features against a {len(zero_state)}-feature silent state")
    y = np.asarray(risk_label).astype(int)
    pre = (y == 1) & (np.asarray(stage_label) == "benign")
    at_floor = np.all(np.isclose(scaled, np.asarray(zero_state, dtype="float32"), atol=FLOOR_ATOL), axis=1)
    return SilentAudit(split=split, n_rows=int(len(y)), n_positive=int(y.sum()), n_preonset=int(pre.sum()),
                       n_preonset_inactive=int((pre & (np.asarray(is_active) == 0)).sum()),
                       n_floor_positive=int((pre & at_floor).sum()), n_floor_rows=int(at_floor.sum()))


def audit_markdown(audits: list[SilentAudit]) -> str:
    out = ["### Positives that carry no information in the state", "",
           "`risk_label` marks the windows BEFORE an attack starts, and a host that is silent in such a "
           "window produces the scaler's silent state — the same 45 numbers every other silent window "
           "has. Those rows cannot be ordered above the negatives they are identical to by any function "
           "of the state; only the host's history distinguishes them.", "",
           "| split | rows | positives | pre-onset | of those, silent | at the exact floor | floor stratum | "
           "state-only AP ceiling there | share of all positives |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in audits:
        out.append(f"| {a.split} | {a.n_rows} | {a.n_positive} | {a.n_preonset} | {a.n_preonset_inactive} | "
                   f"{a.n_floor_positive} | {a.n_floor_rows} | {a.floor_prevalence:.6f} | "
                   f"{100 * a.share_of_positives:.1f}% |")
    out += ["", "The ceiling is the stratum's own prevalence. A state-only head that beats it on a split is "
                "reading something the state does not contain, so the number is also a leak check."]
    return "\n".join(out)


def main() -> None:
    from nidra.data.normalize import FeatureScaler
    from nidra.data.schema import FEATURE_ORDER
    from nidra.train.pipeline import build_all_splits
    from nidra.utils.config import load_config, resolve_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    zero = scaler.zero_state_scaled().astype("float32")
    splits = build_all_splits(cfg)

    audits = []
    for name in args.splits.split(","):
        t = getattr(splits, name.strip())
        raw = t[FEATURE_ORDER].to_numpy(dtype="float32")
        a = audit_split(scaler.transform(raw).astype("float32"), raw[:, FEATURE_ORDER.index("is_active")],
                        t["risk_label"].to_numpy(), t["stage_label"].to_numpy(), zero, name.strip())
        logger.info("%s: %d/%d positives at the silence floor (%.1f%%), ceiling AP %.6f",
                    a.split, a.n_floor_positive, a.n_positive, 100 * a.share_of_positives, a.floor_prevalence)
        audits.append(a)

    md = audit_markdown(audits)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        logger.info("wrote %s", args.out)
    print(md)


if __name__ == "__main__":
    main()

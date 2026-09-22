"""What the model was actually trained on, counted rather than asserted.

    python -m nidra.scripts.data_accounting --config config/ctu13.yaml \\
        --epochs 20 --out reports/tables/ctu13_data_accounting.md

"Trained on the whole dataset" has several meanings and they differ by orders
of magnitude, so the roadmap asks for each separately:

    raw flows          rows in the source capture / CSV
    canonical states   [host, window] rows after windowing and gap filling
    active states      of those, the ones with any traffic (is_active = 1)
    eligible origins   states with L windows of history and K of future
                       behind/ahead of them — the actual training population
    sampled per epoch  the cap the sampler draws to (train_dynamics)
    state exposures    sampled x epochs x L: how many state vectors the
                       encoder read in total

The counts come from the cached tables the training run itself used, so this
cannot drift from what was trained.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitAccounting:
    split: str
    captures: int
    raw_flows: int
    canonical_states: int
    active_states: int
    eligible_origins: int
    positive_origins: int
    sampled_per_epoch: int
    epochs: int
    context_length: int

    #: Splits the sampler and the epoch loop actually touch. test and holdout
    #: have neither, and printing a coverage for them reads as a claim that
    #: they were trained on.
    TRAINED_ON = ("train", "val")

    @property
    def trained_on(self) -> bool:
        return self.split in self.TRAINED_ON

    @property
    def origin_coverage(self) -> float:
        """Fraction of the eligible population drawn in ONE epoch. Capped at
        1.0: a sampler asked for more than exists returns everything."""
        if self.eligible_origins <= 0:
            return 0.0
        return min(1.0, self.sampled_per_epoch / self.eligible_origins)

    @property
    def expected_origin_touches(self) -> float:
        """Average number of times an eligible origin is drawn across the
        whole run. Below 1 means most of the split was never seen."""
        if self.eligible_origins <= 0:
            return 0.0
        return self.epochs * min(self.sampled_per_epoch, self.eligible_origins) / self.eligible_origins

    @property
    def origin_prevalence(self) -> float:
        if self.eligible_origins <= 0:
            return 0.0
        return self.positive_origins / self.eligible_origins

    @property
    def total_state_exposures(self) -> int:
        if not self.trained_on:
            return 0
        return int(self.sampled_per_epoch) * int(self.epochs) * int(self.context_length)


def coverage_table(rows: list[SplitAccounting]) -> dict:
    return {
        "splits": len(rows),
        "captures": sum(r.captures for r in rows),
        "raw_flows": sum(r.raw_flows for r in rows),
        "canonical_states": sum(r.canonical_states for r in rows),
        "active_states": sum(r.active_states for r in rows),
        "eligible_origins": sum(r.eligible_origins for r in rows),
        "positive_origins": sum(r.positive_origins for r in rows),
        "total_state_exposures": sum(r.total_state_exposures for r in rows),
    }


def _n(x) -> str:
    return f"{int(x):,}"


def accounting_markdown(rows: list[SplitAccounting], title: str | None = None) -> str:
    out = [f"### {title}" if title else "### Training data accounting", "",
           "| split | captures | raw flows | canonical states | active | eligible origins | positives | "
           "sampled / epoch | coverage / epoch | mean touches | state exposures |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sampling = (f"{_n(r.sampled_per_epoch)} | {r.origin_coverage:.1%} | {r.expected_origin_touches:.1f} | "
                    f"{_n(r.total_state_exposures)}") if r.trained_on else "— | — | — | —"
        out.append(
            f"| {r.split} | {r.captures} | {_n(r.raw_flows)} | {_n(r.canonical_states)} | {_n(r.active_states)} | "
            f"{_n(r.eligible_origins)} | {_n(r.positive_origins)} | {sampling} |")
    t = coverage_table(rows)
    out += ["", f"Totals: {t['captures']} captures, {_n(t['raw_flows'])} raw flows, "
               f"{_n(t['canonical_states'])} canonical states ({_n(t['active_states'])} active), "
               f"{_n(t['eligible_origins'])} eligible origins of which {_n(t['positive_origins'])} positive, "
               f"{_n(t['total_state_exposures'])} state exposures over training. "
               f"Sampling columns apply to the splits the model is fit on; test and holdout are read once."]
    return "\n".join(out)


def _split_accounting(cfg: dict, split: str, epochs: int, audit: dict | None) -> SplitAccounting:
    """Counts for one split, from the same cached tables training reads."""
    import numpy as np

    from nidra.data.dataset import enumerate_candidates
    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.train.train_dynamics import resolve_sample_caps

    _, L, K = geometry_from_config(cfg)
    splits = build_all_splits(cfg)
    df = getattr(splits, split)
    day_keys = list(cfg["splits"].get(f"{split}_days", []) or [])
    raw = 0
    for key in day_keys:
        raw += int(((audit or {}).get(key) or {}).get("raw_rows", 0))

    if df.empty:
        return SplitAccounting(split, len(day_keys), raw, 0, 0, 0, 0, 0, epochs, L)
    cands = enumerate_candidates(df, L, K)
    active = int((df["is_active"].to_numpy() > 0).sum()) if "is_active" in df.columns else 0
    max_train, max_val = resolve_sample_caps(cfg, None, None)
    cap = {"train": max_train, "val": max_val}.get(split)
    sampled = min(cap, len(cands)) if cap else len(cands)
    return SplitAccounting(
        split=split, captures=len(day_keys), raw_flows=raw, canonical_states=int(len(df)),
        active_states=active, eligible_origins=int(len(cands)),
        positive_origins=int(np.sum(cands.risk_label == 1)),
        sampled_per_epoch=int(sampled), epochs=int(epochs), context_length=int(L))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits", default="train,val,test,holdout")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--audit", default=None, help="a dataset audit JSON keyed by day, for raw flow counts")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from nidra.utils.config import load_config
    cfg = load_config(args.config)
    audit = None
    if args.audit and Path(args.audit).exists():
        raw = json.loads(Path(args.audit).read_text())
        entries = raw.get("scenarios", raw) if isinstance(raw, dict) else raw
        if isinstance(entries, list):
            audit = {e["day_key"]: {"raw_rows": e.get("raw", {}).get("input_rows", 0)} for e in entries}
        else:
            audit = {k: {"raw_rows": v.get("raw", {}).get("input_rows", 0)} for k, v in entries.items()}

    rows = [_split_accounting(cfg, s.strip(), args.epochs, audit) for s in args.splits.split(",") if s.strip()]
    md = accounting_markdown(rows, args.title)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()

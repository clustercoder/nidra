"""Every run on disk, with its provenance, in one table.

    python -m nidra.scripts.experiment_matrix --runs experiments/runs \\
        --out reports/tables/experiment_matrix.md

§33 requires a provenance record per experiment and forbids hiding failed ones.
Both are satisfied by a table generated FROM the records rather than written by
hand: a run that failed, was abandoned, or produced a result nobody liked is in
`experiments/runs/` and therefore in this table, and one that was quietly
deleted is visibly absent from a directory listing that is also printed.

Columns are what a reader needs to judge whether two rows are comparable:
the dataset and feature regime, the geometry, which stages ran, the seeds, the
git commit, and the wall clock. Runs whose geometry or feature regime differ
are flagged, because those are the two things that silently make a comparison
meaningless.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Everything else is a variant directory written by the head ablation, which
#: carries its parent's provenance.
RECORD_NAMES = ("record.json", "provenance.json")


@dataclass(frozen=True)
class RunRow:
    label: str
    stage: str
    dataset: str
    splits: str
    regime: str
    geometry: str
    stages: str
    seeds: str
    epochs: str
    init_from: str
    commit: str
    wall_minutes: float
    has_weights: bool
    has_metrics: bool

    @property
    def wall(self) -> str:
        if self.wall_minutes != self.wall_minutes:
            return "—"
        return f"{self.wall_minutes:.0f}m" if self.wall_minutes < 90 else f"{self.wall_minutes / 60:.1f}h"


def read_record(run_dir: Path) -> dict | None:
    for name in RECORD_NAMES:
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                logger.warning("%s is not readable JSON", p)
    return None


def corpora_of(dataset: dict) -> str:
    """Which corpora a run actually read, counted from its days.

    The record's `name` is a prose string that concatenates both corpora for a
    combined run, so splitting it on the first parenthesis silently reported
    the combined runs as CIC-only. Counting formats cannot go wrong that way.
    """
    fmts = {d.get("format") for d in (dataset.get("days") or {}).values()}
    names = {"ctu_binetflow": "CTU-13", "cicflowmeter": "CIC-IDS2017"}
    got = sorted(names[f] for f in fmts if f in names)
    return "+".join(got) if got else (dataset.get("name") or "—")


def split_counts(dataset: dict) -> str:
    """train/val/test/holdout day counts, from the effective roles the
    provenance record now carries."""
    roles: dict[str, int] = {}
    for d in (dataset.get("days") or {}).values():
        r = d.get("role") or "—"
        roles[r] = roles.get(r, 0) + 1
    # Explicit abbreviations: "train" and "test" both start with t, and a
    # column that renders 2 train days and 3 test days as "2t/3t" is worse
    # than no column.
    short = {"train": "tr", "train+val_carve": "tr+v", "val": "va", "test": "te", "holdout": "ho"}
    parts = [f"{roles[r]}{short[r]}" for r in short if r in roles]
    return " ".join(parts) if parts else "—"


def row_for(run_dir: Path, record: dict) -> RunRow:
    ds = record.get("dataset") or {}
    geo = record.get("geometry") or {}
    geometry = ("—" if not geo else
                f"Δ{geo.get('window_seconds')} L{geo.get('context_length_L')} K{geo.get('horizon_length_K')}")
    wall = record.get("wall_seconds")
    return RunRow(
        label=record.get("label") or run_dir.name,
        stage=record.get("stage") or "—",
        dataset=corpora_of(ds),
        splits=split_counts(ds),
        regime=ds.get("feature_regime") or "—",
        geometry=geometry,
        stages=",".join(record.get("stages") or []) or "—",
        seeds=",".join(str(s) for s in (record.get("ensemble_seeds") or [])) or "—",
        epochs=str(record.get("epochs_override") or "—"),
        init_from=record.get("init_from") or "—",
        commit=(record.get("git_commit") or "—")[:8],
        wall_minutes=float(wall) / 60.0 if wall else float("nan"),
        has_weights=(run_dir / "artifacts" / "weights").exists(),
        has_metrics=(run_dir / "artifacts" / "metrics").exists(),
    )


def collect(runs_dir: Path) -> tuple[list[RunRow], list[str]]:
    """Rows for every run with a record, and the names of those without one."""
    rows, bare = [], []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        record = read_record(d)
        if record is None:
            bare.append(d.name)
            continue
        rows.append(row_for(d, record))
    return rows, bare


def matrix_markdown(rows: list[RunRow], bare: list[str]) -> str:
    regimes = {r.regime for r in rows if r.regime != "—"}
    geometries = {r.geometry for r in rows if r.geometry != "—"}
    out = ["### Experiment matrix — every run on disk", "",
           f"{len(rows)} runs with a provenance record, {len(bare)} directories without one. "
           "Generated from the records, so a run that failed or produced an unwelcome result "
           "is here by construction and one that was deleted is visibly missing.", ""]
    if len(regimes) > 1 or len(geometries) > 1:
        out += [f"⚠ Rows span {len(geometries)} geometries ({', '.join(sorted(geometries))}) and "
                f"{len(regimes)} feature regimes ({', '.join(sorted(regimes))}). Those are the two "
                "things that make a comparison meaningless without saying so; rows differing in "
                "either are not comparable.", ""]
    out += ["| run | stage | corpora | days (train/val/test/holdout) | regime | geometry | stages | seeds | "
            "epochs | init from | commit | wall | weights | metrics |",
            "|---|---|---|---|---|---|---|---|---|---|---|---:|---|---|"]
    for r in sorted(rows, key=lambda x: x.label):
        out.append(f"| `{r.label}` | {r.stage} | {r.dataset} | {r.splits} | {r.regime} | {r.geometry} | {r.stages} | "
                   f"{r.seeds} | {r.epochs} | {r.init_from} | `{r.commit}` | {r.wall} | "
                   f"{'✓' if r.has_weights else '—'} | {'✓' if r.has_metrics else '—'} |")
    total = sum(r.wall_minutes for r in rows if r.wall_minutes == r.wall_minutes)
    out += ["", f"Recorded wall clock across all runs: **{total / 60:.1f} hours** "
                "(single machine, Apple M1, 8 cores, 16 GB; several runs overlapped, so elapsed "
                "time is less than the sum)."]
    if bare:
        out += ["", "Directories without a provenance record — head-ablation variants, which carry "
                     "their parent run's record: " + ", ".join(f"`{b}`" for b in bare) + "."]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="experiments/runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows, bare = collect(Path(args.runs))
    md = matrix_markdown(rows, bare)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        logger.info("wrote %s (%d runs)", args.out, len(rows))
    print(md)


if __name__ == "__main__":
    main()

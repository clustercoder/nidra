"""One ranked table from several runs' benchmark records.

    python -m nidra.scripts.compare_runs --split val \\
        --runs ctu_heads__state ctu_heads__state+hidden ctu_heads__state+hidden+delta+logvar

Every ablation in this phase has the same shape — several runs, one split,
one system — and the comparison must be assembled from the recorded
artifacts rather than retyped into a report.

Two things the table is deliberately opinionated about:

* It names the split in its header, and says plainly when that split is not
  the selection split. Selection happens on validation; reading a ranked
  table built from test is how a project talks itself into choosing an
  architecture on test without noticing.
* It never prints a bare point estimate. Each AP carries its episode
  bootstrap interval, each margin over the reference carries the paired
  interval, and an interval resting on fewer than five positive episodes is
  labelled uninformative rather than quietly reported.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

#: The only split an architecture may be chosen on.
SELECTION_SPLIT = "val"

#: Fewer positive episodes than this and the bootstrap interval is a
#: statement about two or three attacks, not about the model.
MIN_INFORMATIVE_CLUSTERS = 5

DEFAULT_SYSTEM = "world_model_calibrated"
DEFAULT_REFERENCE = "persistence"
DEFAULT_TASK = "task_published_label"


@dataclass(frozen=True)
class RunRow:
    """One run's line in the comparison, with everything needed to read it
    honestly: the estimate, its interval, the margin over the reference and
    that margin's interval, the oracle ceiling, and the two flags that say
    when a number should not be leaned on."""

    label: str
    ap: float
    ci_low: float | None
    ci_high: float | None
    roc_auc: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_alarms_per_hour: float | None
    reference_ap: float | None
    margin: float | None
    margin_ci_low: float | None
    margin_ci_high: float | None
    oracle_ap: float | None
    n_positive: int | None
    n_positive_clusters: int | None

    @property
    def oracle_gap(self) -> float | None:
        """How much of the achievable ranking the forecast gives up. The
        oracle runs the same head on the TRUE future, so this is the head's
        headroom, not the transition model's."""
        if self.oracle_ap is None:
            return None
        return self.oracle_ap - self.ap

    @property
    def few_episodes(self) -> bool:
        return (self.n_positive_clusters or 0) < MIN_INFORMATIVE_CLUSTERS

    @property
    def margin_excludes_zero(self) -> bool:
        """True only when the paired interval lies wholly on one side of
        zero. Everything else — including an interval that merely looks
        mostly positive — is 'not distinguishable from the reference'."""
        if self.margin_ci_low is None or self.margin_ci_high is None:
            return False
        return self.margin_ci_low > 0.0 or self.margin_ci_high < 0.0


def run_metrics_path(run_dir: str | Path, split: str) -> Path:
    return Path(run_dir) / "artifacts" / "metrics" / split / "benchmark.json"


def load_run_metrics(run_dir: str | Path, split: str) -> dict:
    """The `metrics` block of a run's benchmark record. A missing record
    names the run — a comparison silently missing a variant is worse than
    one that stops."""
    path = run_metrics_path(run_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"no benchmark record for run {Path(run_dir).name!r} on split {split!r}: {path}")
    rec = json.loads(path.read_text())
    return rec["metrics"] if isinstance(rec.get("metrics"), dict) else rec


def _get(d: dict, key: str):
    return d.get(key) if isinstance(d, dict) else None


def variant_row(label: str, metrics: dict, system: str = DEFAULT_SYSTEM,
                reference: str = DEFAULT_REFERENCE, task: str = DEFAULT_TASK) -> RunRow:
    systems = metrics[task]["systems"]
    if system not in systems:
        raise KeyError(f"system {system!r} is not in this record (has {sorted(systems)})")
    s = systems[system]
    boot = _get(s, "auc_pr_bootstrap") or {}
    at = _get(s, "at_threshold") or {}
    ref = systems.get(reference) or {}
    attribution = metrics.get("attribution_published_label", {}) if task == DEFAULT_TASK \
        else metrics.get("attribution_detection", {})
    # The recorded attribution pairs are keyed on the uncalibrated system —
    # calibration is a monotone per-horizon map fit on validation, so it
    # cannot change a ranking margin, and there is one recorded pair, not two.
    pair = attribution.get(f"{system} - {reference}") or attribution.get(f"world_model - {reference}") or {}
    return RunRow(
        label=label,
        ap=float(s["auc_pr"]),
        ci_low=_get(boot, "ci_low"),
        ci_high=_get(boot, "ci_high"),
        roc_auc=_get(s, "roc_auc"),
        precision=_get(at, "precision"),
        recall=_get(at, "recall"),
        f1=_get(at, "f1"),
        false_alarms_per_hour=_get(s, "false_alarms_per_hour"),
        reference_ap=_get(ref, "auc_pr"),
        margin=_get(pair, "point"),
        margin_ci_low=_get(pair, "ci_low"),
        margin_ci_high=_get(pair, "ci_high"),
        oracle_ap=_get(systems.get("oracle_true_future") or {}, "auc_pr"),
        n_positive=_get(s, "n_pos"),
        n_positive_clusters=_get(boot, "n_positive_clusters"),
    )


def rank_rows(rows: list[RunRow]) -> list[RunRow]:
    """Best point estimate first. A new list — the caller's order is theirs."""
    return sorted(rows, key=lambda r: r.ap, reverse=True)


def _f(x, nd: int = 3) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def comparison_markdown(rows: list[RunRow], split: str, system: str, reference: str,
                        title: str | None = None) -> str:
    ranked = rank_rows(rows)
    head = [f"### {title}" if title else f"### Variant comparison — `{system}` on **{split}**", ""]
    if split != SELECTION_SPLIT:
        head.append(f"> **{split} is not a selection split.** Ranked here for reporting only; the variant "
                    f"was chosen on `{SELECTION_SPLIT}`.")
        head.append("")
    head += [f"Reference system: `{reference}`. AP is at natural prevalence; intervals are episode-cluster "
             f"bootstrap. ΔAP is the paired difference, and is only evidence of an improvement when its "
             f"interval excludes zero.", "",
             "| variant | AP | 95% CI | ΔAP vs reference | 95% CI | sig. | ROC-AUC | P / R / F1 | FA/h | oracle AP | gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ranked:
        ci = f"[{_f(r.ci_low)}, {_f(r.ci_high)}]" if r.ci_low is not None else "—"
        if r.few_episodes:
            ci += f" ({r.n_positive_clusters} positive episodes: not informative)"
        mci = f"[{_f(r.margin_ci_low)}, {_f(r.margin_ci_high)}]" if r.margin_ci_low is not None else "—"
        sig = "yes" if r.margin_excludes_zero else "no"
        head.append(
            f"| {r.label} | {_f(r.ap)} | {ci} | {_f(r.margin)} | {mci} | {sig} | {_f(r.roc_auc)} | "
            f"{_f(r.precision, 2)} / {_f(r.recall, 2)} / {_f(r.f1, 2)} | {_f(r.false_alarms_per_hour, 2)} | "
            f"{_f(r.oracle_ap)} | {_f(r.oracle_gap)} |")
    return "\n".join(head)


def main() -> None:
    from nidra.scripts.run_experiment import RUNS_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run labels under experiments/runs/, optionally LABEL=display name")
    parser.add_argument("--split", default=SELECTION_SPLIT, choices=["val", "test", "holdout"])
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=["task_published_label", "task_A_detection"])
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None, help="write the Markdown here as well as to stdout")
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip runs with no record for this split instead of stopping")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    rows: list[RunRow] = []
    for spec in args.runs:
        label, _, display = spec.partition("=")
        try:
            metrics = load_run_metrics(runs_dir / label, args.split)
        except FileNotFoundError:
            if not args.allow_missing:
                raise
            print(f"(skipped {label}: no {args.split} record)")
            continue
        rows.append(variant_row(display or label, metrics, args.system, args.reference, args.task))

    md = comparison_markdown(rows, args.split, args.system, args.reference, args.title)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()

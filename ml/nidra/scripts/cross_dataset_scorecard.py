"""The cross-dataset scorecard: one table, every training/evaluation regime.

    python -m nidra.scripts.cross_dataset_scorecard \
        --runs experiments/runs --out reports/run9/scorecard.md

Reads whatever `benchmark.json` files exist under the given run directories
and renders the row for each. Nothing is computed here — a cell is either a
measured number with its interval or an explicit gap. A regime that was not
run says so; it never silently borrows another regime's number.

The rows are the six the phase is designed around (training set x evaluation
set), plus whatever else is on disk:

    CIC     -> CIC      the Run 8 protocol, at the cross-dataset feature mask
    CTU     -> CTU      within-dataset, scenario/family holdout
    CIC     -> CTU      transfer, operating point selected on CIC validation
    CTU     -> CIC      transfer the other way
    CIC+CTU -> CIC      does adding CTU help on CIC?
    CIC+CTU -> CTU      does adding CIC help on CTU?
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# Shared with the per-run table rather than reimplemented: an ECE weighted by
# sample count instead of population weight is a different number, and two
# copies of that decision would eventually disagree.
from nidra.scripts.report_tables import _base_rate, _ece

logger = logging.getLogger(__name__)


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if v != v else f"{v:.{nd}f}"


def _ci(entry: dict | None) -> str:
    b = (entry or {}).get("auc_pr_bootstrap")
    if not b:
        return ""
    stars = "*" if b.get("n_positive_clusters", 99) < 5 else ""
    return f" [{b['ci_low']:.3f}, {b['ci_high']:.3f}]{stars}"


def read_benchmark(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("%s is not readable JSON", path)
        return None


def _calibration_fields(m: dict) -> dict:
    """Brier and ECE for both arms, plus the constant a lazy predictor scores.

    §36 item 19 and §32 criterion 8. The constant column is not decoration: at
    these prevalences (0.0003 to 0.007) Brier is dominated by the negatives, so
    a score near zero says almost nothing on its own and several arms here do
    NOT beat predicting the prevalence and never moving. Reusing report_tables'
    helpers so the scorecard and the per-run table cannot drift apart.
    """
    cal = m.get("calibration_published_label")
    if not cal:
        return {"brier_raw": None, "ece_raw": None, "brier_calibrated": None,
                "ece_calibrated": None, "base_rate_from_bins": None, "brier_constant": None}
    raw, cald = cal.get("raw") or {}, cal.get("calibrated") or {}
    rel = cald.get("reliability") or raw.get("reliability") or []
    p_bar = _base_rate(rel)
    return {
        "brier_raw": raw.get("brier_natural"),
        "ece_raw": _ece(raw.get("reliability") or []),
        "brier_calibrated": cald.get("brier_natural"),
        "ece_calibrated": _ece(cald.get("reliability") or []),
        "base_rate_from_bins": p_bar,
        "brier_constant": p_bar * (1 - p_bar),
    }


def row_for(record: dict, system: str = "world_model_calibrated") -> dict:
    m = record["metrics"]
    pub = m["task_published_label"]
    sysrow = pub["systems"].get(system, {})
    at = sysrow.get("at_threshold", {})
    best_baseline, best_ap = None, float("-inf")
    for name, entry in pub["systems"].items():
        if name in ("world_model", "world_model_calibrated", "oracle_true_future"):
            continue
        ap = entry.get("auc_pr")
        if ap is not None and ap == ap and ap > best_ap:
            best_baseline, best_ap = name, ap
    onset = m.get("task_B_onset_forecast", {})
    per_ep = m.get("per_episode", {}).get("world_model_calibrated", {})
    return {
        "split": record.get("split"),
        "n_rows": pub.get("n_rows"),
        "prevalence": pub.get("prevalence_natural"),
        "ap": sysrow.get("auc_pr"),
        "ap_ci": _ci(sysrow),
        "roc": sysrow.get("roc_auc"),
        "precision": at.get("precision"),
        "recall": at.get("recall"),
        "f1": at.get("f1"),
        # These are two different quantities and the scorecard was reporting the
        # second under the first's name: `active_benign_false_alarm_rate` is a
        # fraction of active-benign ROWS, which rounds to 0.00 at two decimals
        # and made every regime look silent. §33 asks for false alarms per hour.
        "false_alarms_per_hour": sysrow.get("false_alarms_per_hour"),
        "alerts_per_hour": sysrow.get("alerts_per_hour"),
        "active_benign_fa_rate": sysrow.get("active_benign_false_alarm_rate"),
        "oracle_ap": pub["systems"].get("oracle_true_future", {}).get("auc_pr"),
        # §25: the oracle is the frozen head on the TRUE future under a SINGLE
        # trajectory, with no Platt layer. `world_model_calibrated` pools ~200
        # stochastic trajectories and then calibrates, so the ratio of the two
        # is not a fraction of an upper bound and exceeds 1 in half the cells
        # measured. The matched comparison is this one, and it holds in 12 of
        # those 14 — so the column has to travel beside the oracle's.
        "deterministic_ap": pub["systems"].get("world_model_deterministic", {}).get("auc_pr"),
        "persistence_ap": pub["systems"].get("persistence", {}).get("auc_pr"),
        "best_baseline": best_baseline,
        "best_baseline_ap": best_ap if best_baseline else None,
        "state_skill_vs_persistence": (m.get("state_forecast", {}).get("skill_vs_persistence", {}) or {}).get("mean"),
        "state_skill_vs_ridge": m.get("state_forecast", {}).get("skill_vs_ridge"),
        "onset_ap_5": onset.get("5", {}).get("systems", {}).get(system, {}).get("auc_pr"),
        "onset_ap_15": onset.get("15", {}).get("systems", {}).get(system, {}).get("auc_pr"),
        "n_episodes": per_ep.get("n_episodes"),
        "warned_pre_onset": per_ep.get("n_warned_pre_onset"),
        "attribution_vs_persistence": (m.get("attribution_published_label", {}) or {}).get("world_model - persistence"),
        # D130: every attack group in both corpora has its positives on one
        # host, so AP alone cannot tell "learned the family" from "learned the
        # machine". Within-host ROC can, and it is prevalence-independent, so
        # it is the one column that compares honestly across these regimes.
        **_host_identity_fields(m, system),
        **_calibration_fields(m),
    }


def _host_identity_fields(m: dict, system: str) -> dict:
    """The host/timing decomposition for this row's system, if the benchmark
    that produced it carried one. Benchmarks predating the block simply have
    no column rather than a fabricated one."""
    h = (m.get("host_identity_published_label") or {}).get(system)
    if not h:
        return {"n_positive_hosts": None, "within_host_roc": None,
                "within_host_prevalence": None, "host_mean_roc": None}
    return {"n_positive_hosts": h.get("n_positive_hosts"), "within_host_roc": h.get("within_host_roc"),
            "within_host_prevalence": h.get("within_host_prevalence"), "host_mean_roc": h.get("host_mean_roc")}


DEFAULT_REGIMES = [
    ("CIC", "CIC", "cic_core"),
    ("CTU", "CTU", "ctu"),
    ("CIC", "CTU", "cic2ctu"),
    ("CTU", "CIC", "ctu2cic"),
    ("CIC+CTU", "CIC", "comb_cic"),
    ("CIC+CTU", "CTU", "comb_ctu"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="experiments/runs")
    parser.add_argument("--map", action="append", default=[],
                        help="TRAIN:EVAL:run_label, e.g. CIC:CTU:cic2ctu_heads__state+hidden")
    parser.add_argument("--splits", default="test,holdout")
    parser.add_argument("--system", default="world_model_calibrated")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs_dir = Path(args.runs)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    regimes = []
    for spec in args.map:
        train, evalset, label = spec.split(":", 2)
        regimes.append((train, evalset, label))
    if not regimes:
        regimes = DEFAULT_REGIMES

    lines: list[str] = []
    lines.append("| Training | Evaluated on | split | rows | prevalence | AP [95% CI] | ROC | P | R | F1 | FA/h | "
                 "alerts/h | FA rate on active benign | best baseline (AP) | oracle AP / deterministic AP | "
                 "state skill vs persistence / ridge | onset AP 5/15 | "
                 "episodes warned | positive hosts | within-host ROC |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---:|---:|")
    raw: dict[str, Any] = {}
    for train, evalset, label in regimes:
        for split in splits:
            path = runs_dir / label / "artifacts" / "metrics" / split / "benchmark.json"
            record = read_benchmark(path)
            if record is None:
                lines.append(f"| {train} | {evalset} | {split} | " + " | ".join(["—"] * 17) + " |")
                continue
            r = row_for(record, args.system)
            raw[f"{label}/{split}"] = r
            warned = "—" if r["n_episodes"] is None else f"{r['warned_pre_onset']} / {r['n_episodes']}"
            lines.append(
                f"| {train} | {evalset} | {split} | {r['n_rows']:,} | {_f(r['prevalence'], 5)} | "
                f"{_f(r['ap'])}{r['ap_ci']} | {_f(r['roc'])} | {_f(r['precision'])} | {_f(r['recall'])} | "
                f"{_f(r['f1'])} | {_f(r['false_alarms_per_hour'], 2)} | {_f(r['alerts_per_hour'], 2)} | "
                f"{_f(r['active_benign_fa_rate'], 5)} | "
                f"{r['best_baseline'] or '—'} ({_f(r['best_baseline_ap'])}) | {_f(r['oracle_ap'])}"
                f"{' / ' + _f(r['deterministic_ap']) if r['deterministic_ap'] is not None else ''} | "
                f"{_f(r['state_skill_vs_persistence'])} / {_f(r['state_skill_vs_ridge'])} | "
                f"{_f(r['onset_ap_5'])} / {_f(r['onset_ap_15'])} | {warned} | "
                f"{r['n_positive_hosts'] if r['n_positive_hosts'] is not None else '—'}"
                f"{'¹' if r['n_positive_hosts'] == 1 else ''} | {_f(r['within_host_roc'], 4)} |")

    if any(r.get("deterministic_ap") is not None for r in raw.values()):
        lines += ["", "**oracle AP / deterministic AP.** The oracle is the frozen head on the TRUE future "
                      "state under a single-trajectory readout and no Platt layer. The published system pools "
                      "~200 stochastic trajectories and then calibrates, so the ratio of the two is NOT a "
                      "fraction of an upper bound — the published score exceeds the oracle in half the cells "
                      "measured, because its readout is a better estimator and the oracle is given neither "
                      "half of it. The matched comparison is the deterministic column, which the oracle does "
                      "bound in 12 of 14 (§3.24)."]

    if any(r.get("n_positive_hosts") == 1 for r in raw.values()):
        lines += ["", "¹ Every positive in that evaluation sits on a single host, so its AP does not "
                      "separate the attack's behaviour from that host's identity. **Within-host ROC** is "
                      "the column that does: on the infected host alone, does the system order the attack "
                      "windows above that host's own benign ones? It is prevalence-independent and is the "
                      "only column here that compares fairly across datasets."]

    # Calibration gets its own table rather than three more columns on a table
    # that is already twenty wide — and because it answers a different question.
    # AP asks whether the ordering is right; these ask whether the number means
    # what it says. §36 item 19, §32 criterion 8.
    if any(r.get("brier_calibrated") is not None for r in raw.values()):
        lines += ["", "### Calibration", "",
                  "| Training | Evaluated on | split | Brier raw | ECE raw | Brier calibrated | "
                  "ECE calibrated | base rate | Brier of that constant | beats it? |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|:--:|"]
        for train, evalset, label in regimes:
            for split in splits:
                r = raw.get(f"{label}/{split}")
                if r is None or r.get("brier_calibrated") is None:
                    continue
                cb, const = r["brier_calibrated"], r["brier_constant"]
                beats = "—" if const is None or const != const else ("yes" if cb < const else "**no**")
                lines.append(
                    f"| {train} | {evalset} | {split} | {_f(r['brier_raw'], 5)} | {_f(r['ece_raw'], 4)} | "
                    f"{_f(cb, 5)} | {_f(r['ece_calibrated'], 4)} | {_f(r['base_rate_from_bins'], 5)} | "
                    f"{_f(const, 5)} | {beats} |")
        lines += ["", "ECE is weighted by each bin's **population** weight, not its sampled row count — the "
                      "evaluation subsamples negatives, so those are different numbers. At these prevalences "
                      "Brier is dominated by the negatives, so the last two columns are the ones that make it "
                      "readable: predicting the base rate for every row and never moving scores p(1−p), and a "
                      "**no** in the final column means the calibrated score does not beat that. It is not a "
                      "verdict on the ranking, which is what AP measures — but a probability that loses to a "
                      "constant should not be displayed to an operator as a probability."]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.parent / (out.stem + ".json")).write_text(json.dumps(raw, indent=2, default=float))
    print("\n".join(lines))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()

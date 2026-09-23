"""Does a recorded cross-evaluation cell still reproduce under the current code?

Twenty cells were scored before `world_model_calibrated` was bootstrapped
(§3.31) and before the host-identity block, the calibration reader and D142/D143's
guards landed. `q_reprod` re-runs each one into `repro_<label>/` rather than over
the original, because overwriting would leave the log's published figures
unmatched by any artifact precisely when the interesting answer is that they
drifted.

This compares the two and says which cells moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: A re-run is seeded but not required to be bit-identical — the bootstrap
#: resamples and the stochastic rollout both draw. 0.005 AP is an order of
#: magnitude below the smallest difference this phase draws a conclusion from.
AP_TOLERANCE = 0.005


def _published(record: dict) -> dict:
    return record["metrics"]["task_published_label"]


def compare_cell(recorded: dict, rerun: dict | None) -> dict[str, Any]:
    """One cell's recorded artifact against its re-run.

    A difference in `n_rows` or prevalence is NOT drift — it means the two runs
    scored different data, which is a worse finding and must not be reported as
    agreement with a small delta.
    """
    rec = _published(recorded)
    rec_ap = rec["systems"]["world_model_calibrated"]["auc_pr"]
    if rerun is None:
        return {"recorded_ap": rec_ap, "rerun_ap": None, "delta_ap": None,
                "reproduces": None, "gained_ci": None, "reason": "not re-run"}

    new = _published(rerun)
    new_sys = new["systems"]["world_model_calibrated"]
    new_ap = new_sys["auc_pr"]
    gained_ci = new_sys.get("auc_pr_bootstrap") is not None
    out = {"recorded_ap": rec_ap, "rerun_ap": new_ap, "delta_ap": abs(new_ap - rec_ap),
           "gained_ci": gained_ci, "reason": ""}

    if rec.get("n_rows") != new.get("n_rows"):
        out["reproduces"] = False
        out["reason"] = f"different rows ({rec.get('n_rows')} vs {new.get('n_rows')}) — not the same evaluation"
    elif abs((rec.get("prevalence_natural") or 0) - (new.get("prevalence_natural") or 0)) > 1e-9:
        out["reproduces"] = False
        out["reason"] = (f"different prevalence ({rec.get('prevalence_natural')} vs "
                         f"{new.get('prevalence_natural')}) — not the same evaluation")
    elif out["delta_ap"] > AP_TOLERANCE:
        out["reproduces"] = False
        out["reason"] = f"AP moved {out['delta_ap']:.4f}, past the {AP_TOLERANCE} tolerance"
    else:
        out["reproduces"] = True
    return out


def drift_markdown(rows: list[tuple[str, str, dict]]) -> str:
    judged = [r for _, _, r in rows if r["reproduces"] is not None]
    drifted = [r for r in judged if r["reproduces"] is False]
    if not judged:
        # "All 0 cells reproduce" is a positive claim from no evidence.
        head = f"**No cell has been re-run yet.** {len(rows)} recorded cell(s) are listed for comparison."
    elif drifted:
        head = f"**{len(drifted)} of {len(judged)} re-run cells do not reproduce.**"
    else:
        head = f"**All {len(judged)} re-run cells reproduce** within {AP_TOLERANCE} AP, on identical rows."
    lines = [head, "",
             "| cell | split | recorded AP | re-run AP | Δ | reproduces | gained CI | note |",
             "|---|---|---:|---:|---:|:--:|:--:|---|"]
    for label, split, r in rows:
        f = lambda x: "—" if x is None else f"{x:.4f}"
        verdict = {True: "yes", False: "**no**", None: "—"}[r["reproduces"]]
        ci = {True: "yes", False: "no", None: "—"}[r["gained_ci"]]
        lines.append(f"| {label} | {split} | {f(r['recorded_ap'])} | {f(r['rerun_ap'])} | "
                     f"{f(r['delta_ap'])} | {verdict} | {ci} | {r['reason'] or ''} |")
    lines += ["", f"A re-run is seeded but not bit-identical — the bootstrap resamples and the rollout "
                  f"draws — so agreement is judged at {AP_TOLERANCE} AP, an order of magnitude below the "
                  "smallest difference this phase draws a conclusion from. A row marked **no** for "
                  "*different rows* or *different prevalence* did not drift: it scored different data, "
                  "which is the worse finding."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="experiments/runs")
    ap.add_argument("--splits", default="val,test,holdout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = Path(args.runs)

    rows = []
    for recorded_dir in sorted(runs.glob("xeval_*")):
        label = recorded_dir.name.removeprefix("xeval_")
        for split in args.splits.split(","):
            rec_path = recorded_dir / "artifacts" / "metrics" / split / "benchmark.json"
            if not rec_path.exists():
                continue
            new_path = runs / f"repro_{label}" / "artifacts" / "metrics" / split / "benchmark.json"
            new = json.loads(new_path.read_text()) if new_path.exists() else None
            rows.append((label, split, compare_cell(json.loads(rec_path.read_text()), new)))

    text = drift_markdown(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()

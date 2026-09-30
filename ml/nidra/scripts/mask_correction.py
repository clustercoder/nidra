"""What D145's fix did to each recorded cross-evaluation cell.

The transition receives no gradient on features the scaler dropped, and the
rollout was feeding that untrained output back in, compounding it into the state
the frozen head reads (D145, §3.37). Every CTU and cross-dataset cell in Run 9
was scored under the defect.

`q_maskall` re-scores each one into `mask_<label>/` rather than over
`repro_<label>/`, for the same reason the reproduction re-run was kept separate:
if the numbers move, the published figures must stay matched to the artifacts
that produced them. This reports the movement.

The correction does not go one way — it helped some cells and hurt others, which
is what an untrained signal projected through untrained weights should do — so
nothing here averages the deltas. The split is the finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: Below this the cell's conclusions do not change. Set at the head-training
#: seed spread §3.29 measured (sd 0.047) rather than at a bootstrap tolerance:
#: the question is "does this move a conclusion", not "is this reproducible".
RECONSIDER_DELTA = 0.05


def _published(bench: dict) -> dict:
    return bench["metrics"]["task_published_label"]


def _ap(table: dict, system: str) -> float | None:
    s = table["systems"].get(system)
    return None if s is None else s["auc_pr"]


def compare_cell(before: dict, after: dict | None) -> dict[str, Any]:
    """One cell's pre-fix artifact against its re-score.

    `reconsider` is True only when the fix moved the published arm enough to
    change what the cell says. A moved oracle or a changed row count means the
    two runs are not the same evaluation, which is a worse finding than drift
    and is reported as such rather than as a delta.
    """
    b = _published(before)
    out: dict[str, Any] = {
        "before_calibrated": _ap(b, "world_model_calibrated"),
        "before_raw": _ap(b, "world_model"),
        "after_calibrated": None, "after_raw": None,
        "delta_calibrated": None, "delta_raw": None,
        "beat_oracle_before": None, "beat_oracle_after": None,
        "reconsider": None, "note": "",
    }
    ob, pb = _ap(b, "oracle_true_future"), _ap(b, "persistence")
    if ob is not None and out["before_raw"] is not None:
        out["beat_oracle_before"] = out["before_raw"] > ob

    if after is None:
        out["note"] = "not re-scored yet"
        return out

    a = _published(after)
    oa, pa = _ap(a, "oracle_true_future"), _ap(a, "persistence")
    out["after_calibrated"], out["after_raw"] = _ap(a, "world_model_calibrated"), _ap(a, "world_model")
    out["delta_calibrated"] = out["after_calibrated"] - out["before_calibrated"]
    out["delta_raw"] = out["after_raw"] - out["before_raw"]
    if oa is not None and out["after_raw"] is not None:
        out["beat_oracle_after"] = out["after_raw"] > oa

    if b.get("n_rows") != a.get("n_rows"):
        out["reconsider"] = False
        out["note"] = f"different rows ({b.get('n_rows')} vs {a.get('n_rows')}) — not the same evaluation"
    elif ob is not None and oa is not None and abs(oa - ob) > 1e-9:
        # The oracle scores TRUE future states, whose dropped slots are already
        # exactly zero. The mask cannot touch it. If it moved, something other
        # than the fix differs and the delta does not mean what it looks like.
        out["reconsider"] = False
        out["note"] = f"oracle moved ({ob:.4f} → {oa:.4f}) — the re-score differs in more than the fix"
    elif pb is not None and pa is not None and abs(pa - pb) > 1e-9:
        out["reconsider"] = False
        out["note"] = f"persistence moved ({pb:.4f} → {pa:.4f}) — the re-score differs in more than the fix"
    else:
        out["reconsider"] = abs(out["delta_calibrated"]) >= RECONSIDER_DELTA
    return out


def correction_markdown(rows: list[tuple[str, str, dict]]) -> str:
    done = [r for _, _, r in rows if r["after_calibrated"] is not None]
    if not done:
        head = (f"**No cell has been re-scored yet.** {len(rows)} cell(s) are listed "
                "for comparison.")
    else:
        up = sum(1 for r in done if r["delta_calibrated"] > 0)
        down = sum(1 for r in done if r["delta_calibrated"] < 0)
        big = [r for r in done if r["reconsider"]]
        head = (f"**{len(done)} of {len(rows)} cells re-scored: {up} up, {down} down.** "
                f"{len(big)} moved the published arm by at least {RECONSIDER_DELTA} AP. "
                "The deltas are not averaged: the fix removed an untrained signal that "
                "helped some cells and hurt others, and the split is the finding.")

    lines = [head, "",
             "| cell | split | AP before | AP after | Δ cal | Δ raw | beats oracle (before → after) | note |",
             "|---|---|---:|---:|---:|---:|:--:|---|"]
    f = lambda x: "—" if x is None else f"{x:.4f}"
    sg = lambda x: "—" if x is None else f"{x:+.4f}"
    yn = lambda x: {True: "yes", False: "no", None: "—"}[x]
    for label, split, r in rows:
        lines.append(f"| {label} | {split} | {f(r['before_calibrated'])} | {f(r['after_calibrated'])} | "
                     f"{sg(r['delta_calibrated'])} | {sg(r['delta_raw'])} | "
                     f"{yn(r['beat_oracle_before'])} → {yn(r['beat_oracle_after'])} | {r['note']} |")

    flipped = [(l, s, r) for l, s, r in rows
               if r["beat_oracle_after"] is not None and r["beat_oracle_before"] != r["beat_oracle_after"]]
    lines += ["", "### Cells whose oracle verdict changed", ""]
    if flipped:
        for l, s, r in flipped:
            lines.append(f"- **{l}/{s}** — {yn(r['beat_oracle_before'])} → {yn(r['beat_oracle_after'])}")
    else:
        lines.append("None. Whatever makes a cell beat its oracle, the phantom drift was not it "
                     "— which is what the pre-registered test in §3.37 predicted it would not be.")

    suspect = [(l, s, r) for l, s, r in rows if "moved" in r["note"] or "different rows" in r["note"]]
    if suspect:
        lines += ["", "### Re-scores that differ in more than the fix", ""]
        for l, s, r in suspect:
            lines.append(f"- **{l}/{s}** — {r['note']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="experiments/runs")
    ap.add_argument("--splits", default="val,test,holdout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = Path(args.runs)

    rows = []
    for before_dir in sorted(runs.glob("repro_*")):
        label = before_dir.name.removeprefix("repro_")
        for split in args.splits.split(","):
            bp = before_dir / "artifacts" / "metrics" / split / "benchmark.json"
            if not bp.exists():
                continue
            apath = runs / f"mask_{label}" / "artifacts" / "metrics" / split / "benchmark.json"
            after = json.loads(apath.read_text()) if apath.exists() else None
            rows.append((label, split, compare_cell(json.loads(bp.read_text()), after)))

    text = correction_markdown(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()

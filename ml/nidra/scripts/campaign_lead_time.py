"""Stage-transition lead time from the benchmark's score dumps.

    python -m nidra.scripts.campaign_lead_time --config config/default.yaml \\
        --splits test,holdout --out reports/tables/campaign_lead_time.md

Answers the PS item "predict before compromise completes" in the only form
CIC-IDS2017 can support: for every multi-phase campaign on one host, was that
host flagged before the phase that completes the campaign began?
(`nidra.eval.campaign_metrics` has the definitions.)

Re-runs no model. The scores and the frozen threshold are read back from the
`benchmark_scores.npz` / `benchmark.json` pair `nidra.eval.benchmark` writes;
the stage labels come from the split table the benchmark was built from. The
published arm is scored at its validation-frozen threshold; every other system
is reported only threshold-free (its highest score before completion), because
the world model's threshold is not theirs (DECISIONS.md D148).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nidra.eval.campaign_metrics import campaign_lead_report, episode_spans

logger = logging.getLogger(__name__)

PUBLISHED_SYSTEM = "world_model_calibrated"
#: npz keys that are labels or bookkeeping, not per-row system scores
NON_SCORE_KEYS = {"y_published", "y_detect", "weight", "stratum", "cluster", "host_id", "origin_ts",
                  "inside_episode", "minutes_to_onset", "risk_k_calibrated", "risk_k_raw", "future_is_attack"}


def score_systems(dump: dict[str, np.ndarray]) -> list[str]:
    return sorted(k for k, v in dump.items()
                  if k not in NON_SCORE_KEYS and not k.startswith("onset_") and np.ndim(v) == 1)


def split_report(episodes, dump: dict[str, np.ndarray], threshold: float, window_seconds: int,
                 max_gap_minutes: int, pre_onset_minutes: int) -> dict[str, Any]:
    """The published arm at its threshold, plus every system's highest score
    before completion on the same campaigns."""
    if PUBLISHED_SYSTEM not in dump:
        raise KeyError(f"score dump has no {PUBLISHED_SYSTEM!r}; systems: {score_systems(dump)}")
    kw = dict(window_seconds=window_seconds, max_gap_minutes=max_gap_minutes, pre_onset_minutes=pre_onset_minutes)
    report = campaign_lead_report(episodes, dump["host_id"], dump["origin_ts"], dump[PUBLISHED_SYSTEM],
                                  threshold, **kw)
    peaks: dict[str, list[float | None]] = {}
    for system in score_systems(dump):
        other = campaign_lead_report(episodes, dump["host_id"], dump["origin_ts"], dump[system], threshold, **kw)
        peaks[system] = [c["max_score_before_completion"] for c in other["campaigns"]]
    return {**report, "system": PUBLISHED_SYSTEM, "max_score_before_completion_by_system": peaks}


def _hhmm(ts: int, utc_offset_hours: int) -> str:
    return datetime.fromtimestamp(ts + utc_offset_hours * 3600, tz=timezone.utc).strftime("%H:%M")


def markdown(reports: dict[str, dict[str, Any]], utc_offset_hours: int = -3) -> str:
    out = ["### Stage-transition lead time — flagged before the campaign's completing phase?", "",
           f"Times are capture-local (UTC{utc_offset_hours:+d}). A flag during an earlier phase is that phase "
           "recognised in time to act before the completing one; it is not a forecast that names the "
           "completing phase.", "",
           "| split | host | phases (onset stage) | completing phase | flagged before | lead (min) | "
           "max score before | windows scored / expected | completing phase detected (latency) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for split, rep in reports.items():
        for c in rep["campaigns"]:
            phases = ", ".join(f"{_hhmm(p['onset_ts'], utc_offset_hours)} {p['stage']}" for p in c["phases"])
            flag = (f"yes, {_hhmm(c['flag_ts'], utc_offset_hours)}" if c["flagged_before_completion"] else "no")
            lead = f"{c['stage_transition_lead_s'] / 60:.0f}" if c["stage_transition_lead_s"] is not None else "—"
            peak = f"{c['max_score_before_completion']:.3f}" if c["max_score_before_completion"] is not None else "—"
            det = (f"yes ({c['completing_phase_latency_min']:.0f} min)" if c["completing_phase_detected"] else "no")
            out.append(f"| {split} | {c['host']} | {phases} | {_hhmm(c['completing_onset_ts'], utc_offset_hours)} "
                       f"{c['completing_stage']} | {flag} | {lead} | {peak} | "
                       f"{c['scored_windows_before_completion']} / {c['expected_windows_before_completion']} | {det} |")
    out.append("")
    for split, rep in reports.items():
        out.append(f"- **{split}**: {rep['n_flagged_before_completion']} of {rep['n_campaigns_scored']} "
                   f"multi-phase campaigns flagged before completion at threshold {rep['threshold']:.3f}; "
                   f"{rep['n_single_episode_campaigns']} single-episode campaigns not scored.")
    return "\n".join(out)


def main() -> None:
    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.utils.config import load_config, resolve_path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="test,holdout")
    ap.add_argument("--metrics-dir", default=None, help="default: the config's artifacts.metrics_dir")
    ap.add_argument("--max-gap-minutes", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    window_seconds, _, _ = geometry_from_config(cfg)
    eval_cfg = cfg.get("eval", {})
    merge_gap = int(eval_cfg.get("episode_merge_gap_windows", 5))
    pre_onset = int(eval_cfg.get("benchmark_set", {}).get("pre_onset_minutes", 30))
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else resolve_path(cfg, cfg["artifacts"]["metrics_dir"])
    splits = build_all_splits(cfg)

    reports = {}
    for split in args.splits.split(","):
        record = json.loads((metrics_dir / split / "benchmark.json").read_text())
        with np.load(metrics_dir / split / "benchmark_scores.npz", allow_pickle=False) as z:
            dump = {k: z[k] for k in z.files}
        episodes = episode_spans(getattr(splits, split), window_seconds, merge_gap)
        reports[split] = split_report(episodes, dump, float(record["threshold"]), window_seconds,
                                      args.max_gap_minutes, pre_onset)
        logger.info("%s: %d of %d campaigns flagged before completion", split,
                    reports[split]["n_flagged_before_completion"], reports[split]["n_campaigns_scored"])

    text = markdown(reports)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        out.with_suffix(".json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"wrote {out}")
    print(text)


if __name__ == "__main__":
    main()

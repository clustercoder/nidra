"""Write the baseline manifest: a provenance record of the artifacts and
published numbers that exist at a given commit, before they are superseded.

    python -m nidra.scripts.write_baseline_manifest --config config/default.yaml \
        --out experiments/BASELINE_MANIFEST.json

The manifest never overwrites an existing file unless --force is passed:
a baseline is a historical record, not a rolling one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidra.utils.config import load_config, resolve_path
from nidra.utils.provenance import experiment_record, file_digest, write_json


def _published_metrics(cfg: dict) -> dict:
    """Pull the headline rows out of the committed metrics JSONs so the
    manifest is self-contained."""
    metrics_dir = resolve_path(cfg, cfg["artifacts"]["metrics_dir"])
    out: dict = {}
    for split in ("test", "holdout"):
        entry: dict = {}
        for name in ("baselines", "ablations", "calibration", "lead_time"):
            path = metrics_dir / split / f"{name}.json"
            if not path.exists():
                continue
            entry[f"{name}_sha256"] = file_digest(path)
            payload = json.loads(path.read_text())
            if name == "baselines":
                entry["run_params"] = {k: payload.get(k) for k in (
                    "n_samples", "n_trajectories", "max_eval_samples", "n_eval_rows",
                    "ensemble_seeds", "risk_pooling", "risk_threshold", "config_hash", "git_commit")}
                entry["baselines"] = {
                    k: {m: v.get(m) for m in ("auc_pr", "f1", "precision", "recall", "fpr", "tp", "fp", "tn", "fn")}
                    for k, v in payload.get("baselines", {}).items() if isinstance(v, dict) and "auc_pr" in v
                }
            if name == "lead_time":
                for section in ("raw", "ensemble", "ensemble_calibrated"):
                    if section in payload:
                        entry[f"lead_time_{section}"] = {
                            k: payload[section].get(k) for k in ("median_lead_time_s", "n_episodes", "n_no_warning", "fraction_no_warning")
                        }
        out[split] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists — a baseline manifest is not overwritten without --force")

    cfg = load_config(args.config)
    pooling = {
        "method": cfg["rollout"].get("risk_pooling_method"),
        "quantile": cfg["rollout"].get("risk_pooling_quantile"),
        "head_reduction": cfg["rollout"].get("risk_pooling_head_reduction"),
        "n_samples_per_member": cfg["rollout"].get("n_samples_per_member"),
    }
    weights_dir = resolve_path(cfg, cfg["artifacts"]["weights_dir"])
    calib_path = weights_dir / "risk_calibration.json"
    calibration = json.loads(calib_path.read_text()) if calib_path.exists() else None
    if calibration is not None:
        calibration = {k: v for k, v in calibration.items() if k != "params"} | {"params_present": "params" in calibration}

    record = experiment_record(
        cfg, stage=args.label, pooling=pooling, calibration=calibration,
        metrics=_published_metrics(cfg), hash_processed=True,
        extra={"label": args.label},
    )
    write_json(out, record)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Run a recorded training experiment: derive a config from a base config plus
overrides, point its artifact directories at experiments/runs/<label>/,
train the requested stages, and write a provenance record.

    python -m nidra.scripts.run_experiment --label geomA_L15K3 \
        --set windowing.context_length=15 --set windowing.horizon_length=3 \
        --set windowing.min_windows_per_host=18 --set labels.risk_threshold_windows=3 \
        --seeds 0 --stages dynamics,heads --epochs 30

The derived config is written to experiments/runs/<label>/config.yaml and
is what the run loads (so `_config_hash` is the hash of the config actually
used). The processed-table cache stays the canonical one; only weights,
scaler and metrics land under the run directory. Nothing under
artifacts/ is touched unless --label production is given explicitly.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from nidra.utils.config import PROJECT_ROOT, load_config
from nidra.utils.provenance import experiment_record, write_json

logger = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"


def _coerce(value: str) -> Any:
    """CLI override values: YAML-parse so 15 -> int, 0.5 -> float, true -> bool,
    [0,1] -> list, null -> None."""
    return yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """`a.b.c=value` dotted assignments on a deep copy."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be key.path=value")
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return out


def derive_run_config(base_path: str | None, label: str, overrides: list[str], run_dir: Path) -> Path:
    base = load_config(base_path)
    cfg = {k: v for k, v in base.items() if not k.startswith("_")}
    cfg = apply_overrides(cfg, overrides)
    if label != "production":
        cfg["artifacts"] = {
            **cfg["artifacts"],
            "root": str(run_dir / "artifacts"),
            "weights_dir": str(run_dir / "artifacts" / "weights"),
            "scaler_dir": str(run_dir / "artifacts" / "scaler"),
            "metrics_dir": str(run_dir / "artifacts" / "metrics"),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def copy_run_artifacts(src_run: Path, weights_dir: Path, scaler_dir: Path, seeds: list[int]) -> None:
    """Seed a run directory with another run's dynamics checkpoints and scaler
    so a heads-only stage compares recipes on identical stage-1 weights."""
    src_weights = src_run / "artifacts" / "weights"
    src_scaler = src_run / "artifacts" / "scaler"
    if not src_weights.is_dir() or not src_scaler.is_dir():
        raise FileNotFoundError(f"--init-from {src_run.name}: expected {src_weights} and {src_scaler}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for name in (f"model_seed_{seed}.pt", f"model_seed_{seed}_metadata.json"):
            if not (src_weights / name).exists():
                raise FileNotFoundError(f"--init-from {src_run.name}: missing {src_weights / name}")
            shutil.copy2(src_weights / name, weights_dir / name)
    for item in src_scaler.iterdir():
        if item.is_file():
            shutil.copy2(item, scaler_dir / item.name)
    logger.info("initialised weights/scaler from %s", src_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--seeds", default="0", help="comma-separated")
    parser.add_argument("--stages", default="dynamics,heads,onset",
                        help="comma-separated subset of dynamics,heads,onset,gru_baseline")
    parser.add_argument("--epochs", type=int, default=None, help="dynamics epochs override")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--init-from", default=None, metavar="LABEL",
                        help="copy weights + scaler from experiments/runs/LABEL before running (heads-only "
                             "recipe comparisons on one dynamics checkpoint)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(args.threads)

    run_dir = RUNS_DIR / args.label if args.label != "production" else PROJECT_ROOT
    cfg_path = derive_run_config(args.base_config, args.label, args.overrides, run_dir if args.label != "production" else RUNS_DIR / "production")
    cfg = load_config(cfg_path)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.init_from:
        copy_run_artifacts(RUNS_DIR / args.init_from, Path(cfg["artifacts"]["weights_dir"]),
                           Path(cfg["artifacts"]["scaler_dir"]), seeds)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    t0 = time.time()
    logger.info("experiment %s: config %s (hash %s) seeds %s stages %s", args.label, cfg_path, cfg["_config_hash"], seeds, stages)

    from nidra.train.train_dynamics import prepare_training_data, train_one_seed
    from nidra.train.train_heads import train_heads_for_seed

    windowed, scaler = prepare_training_data(cfg, args.max_train_samples, args.max_val_samples)
    logger.info("data ready in %.0fs: train %d (pos %d) val %d (pos %d); dropped features %s",
                time.time() - t0, len(windowed["train"].X), int(windowed["train"].risk_label.sum()),
                len(windowed["val"].X), int(windowed["val"].risk_label.sum()), scaler.dropped_features)
    head_data = None
    if "heads" in stages or "onset" in stages:
        # Heads are functions of one state: train and select them on EVERY row
        # of the split, not the dynamics' windowed subsample (train/head_data.py).
        from nidra.train.head_data import build_head_arrays
        from nidra.train.pipeline import build_all_splits
        splits = build_all_splits(cfg)
        head_data = {"train": build_head_arrays(splits.train, scaler), "val": build_head_arrays(splits.val, scaler)}
        del splits
        logger.info("head data: train %s val %s (%.0fs)", head_data["train"].summary(), head_data["val"].summary(), time.time() - t0)

    results: dict[str, Any] = {"seeds": {}}
    for seed in seeds:
        seed_result: dict[str, Any] = {}
        if "dynamics" in stages:
            meta = train_one_seed(cfg, seed, args.epochs, windowed, scaler, "cpu")
            seed_result["dynamics"] = {k: v for k, v in meta.items() if k != "history"}
            seed_result["dynamics_history"] = meta["history"]
            logger.info("seed %d dynamics done at %.0fs: best epoch %s, %s", seed, time.time() - t0,
                        meta["best_epoch"], json.dumps(meta["best_val_free_running"], default=float)[:300])
        if "heads" in stages:
            hmeta = train_heads_for_seed(cfg, seed, windowed, scaler, "cpu", head_data=head_data)
            seed_result["heads"] = {k: v for k, v in hmeta.items() if k not in ("history", "heads_history")}
            seed_result["heads_history"] = hmeta.get("heads_history")
            logger.info("seed %d heads done at %.0fs", seed, time.time() - t0)
        if "onset" in stages:
            from nidra.train.train_onset import train_onset_head_for_seed
            ometa = train_onset_head_for_seed(cfg, seed, windowed, scaler, "cpu", head_data=head_data)
            seed_result["onset"] = {k: v for k, v in ometa.items() if k != "history"}
            seed_result["onset_history"] = ometa.get("history")
            logger.info("seed %d onset head done at %.0fs: best val AP by horizon %s", seed, time.time() - t0,
                        ometa["best_val_ap_natural_by_horizon"])
        results["seeds"][str(seed)] = seed_result

    if "gru_baseline" in stages:
        # The conventional GRU classifier baseline on the same data (seed 0 only — it is a baseline).
        from nidra.eval.gru_classifier import train_gru_classifier
        from nidra.train.pipeline import scale_arrays
        X_tr, _ = scale_arrays(windowed["train"], scaler)
        X_va, _ = scale_arrays(windowed["val"], scaler)
        gcfg = cfg.get("baselines", {}).get("gru_classifier", {})
        out_path = Path(cfg["artifacts"]["weights_dir"]) / "gru_classifier.pt"
        results["gru_baseline"] = train_gru_classifier(
            X_tr, windowed["train"].risk_label.astype(int), X_va, windowed["val"].risk_label.astype(int), out_path,
            epochs=int(gcfg.get("epochs", 8)), hidden=int(gcfg.get("hidden", 64)), input_noise=float(gcfg.get("input_noise", 0.0)),
            seed=0, threads=args.threads)
        logger.info("gru baseline done at %.0fs: %s", time.time() - t0, {k: results["gru_baseline"][k] for k in ("best_epoch", "best_val_auc_pr")})
        del X_tr, X_va

    record = experiment_record(
        cfg, stage="train", seeds=seeds, metrics=results,
        extra={"label": args.label, "note": args.note, "stages": stages, "epochs_override": args.epochs,
               "init_from": args.init_from,
               "n_train_samples": int(len(windowed["train"].X)), "n_val_samples": int(len(windowed["val"].X)),
               "wall_seconds": time.time() - t0},
    )
    out = (run_dir if args.label != "production" else RUNS_DIR / "production") / "record.json"
    write_json(out, record)
    logger.info("wrote %s (%.0fs)", out, time.time() - t0)


if __name__ == "__main__":
    main()

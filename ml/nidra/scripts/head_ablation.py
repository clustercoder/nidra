"""Train several risk-head variants on ONE frozen dynamics run.

    python -m nidra.scripts.head_ablation --base-config config/ctu13.yaml \
        --init-from ctu_dyn --label ctu_heads --seeds 0 \
        --variants state state,hidden hidden state,hidden,delta,logvar

The point of the ablation is to attribute a change to the head and nothing
else, so every variant sits on identical frozen dynamics, identical data and
an identical scaler. Running them as separate `run_experiment` invocations
would do that too — and would also repeat the data preparation and the
per-seed encoder context (a 1.5M-row pass) once per variant. Here both are
built once and shared.

Each variant still lands in its own run directory, `runs/<label>__<variant>/`,
with its own config.yaml and artifacts, so every downstream tool (benchmark,
report_tables, plots) treats it as an ordinary run.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from nidra.data.schema import FEATURE_ORDER
from nidra.models.build import risk_head_components
from nidra.scripts.run_experiment import RUNS_DIR, copy_run_artifacts, derive_run_config
from nidra.utils.config import load_config
from nidra.utils.provenance import experiment_record, write_json

logger = logging.getLogger(__name__)


def variant_tag(components: tuple[str, ...]) -> str:
    """Directory-safe name for a component set, e.g. `state+hidden`."""
    return "+".join(components)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--init-from", required=True, metavar="LABEL",
                        help="the dynamics run every variant sits on")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="comma-separated component lists, e.g. state state,hidden")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--stages", default="heads", help="heads and/or onset")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(args.threads)
    seeds = [int(s) for s in args.seeds.split(",")]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    t0 = time.time()

    variants: list[tuple[str, ...]] = []
    for spec in args.variants:
        cfg_probe = {"model": {"risk_head": {"components": [c.strip() for c in spec.split(",") if c.strip()]}}}
        variants.append(risk_head_components(cfg_probe))
    logger.info("head ablation %s: %d variants %s on dynamics run %s",
                args.label, len(variants), [variant_tag(v) for v in variants], args.init_from)

    # ---- data, prepared once for every variant -----------------------------
    from nidra.train.head_data import build_head_arrays
    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.train.train_dynamics import prepare_training_data
    from nidra.train.train_heads import train_heads_for_seed

    # A throwaway config just to prepare data: same base, same overrides, and
    # the FIRST variant's head so the scaler/regime checks see this run's
    # settings. Its artifacts directory is the init-from copy.
    prep_dir = RUNS_DIR / f"{args.label}__prep"
    prep_cfg_path = derive_run_config(args.base_config, args.label, args.overrides, prep_dir)
    prep_cfg = load_config(prep_cfg_path)
    copy_run_artifacts(RUNS_DIR / args.init_from,
                       Path(prep_cfg["artifacts"]["weights_dir"]), Path(prep_cfg["artifacts"]["scaler_dir"]), seeds)
    windowed, scaler = prepare_training_data(prep_cfg, args.max_train_samples, args.max_val_samples)
    splits = build_all_splits(prep_cfg)
    head_data = {"train": build_head_arrays(splits.train, scaler), "val": build_head_arrays(splits.val, scaler)}
    keep = ["host_id", "window_ts", *FEATURE_ORDER]
    head_tables = {name: getattr(splits, name)[keep].copy() for name in ("train", "val")}
    del splits
    _, L, _ = geometry_from_config(prep_cfg)
    logger.info("data ready in %.0fs: head train %s val %s", time.time() - t0,
                head_data["train"].summary(), head_data["val"].summary())

    results: dict = {"label": args.label, "init_from": args.init_from, "variants": {}}
    out_path = RUNS_DIR / args.label / "head_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for components in variants:
        tag = variant_tag(components)
        run_dir = RUNS_DIR / f"{args.label}__{tag}"
        overrides = [*args.overrides, f"model.risk_head.components=[{','.join(components)}]"]
        cfg_path = derive_run_config(args.base_config, f"{args.label}__{tag}", overrides, run_dir)
        cfg = load_config(cfg_path)
        copy_run_artifacts(RUNS_DIR / args.init_from,
                           Path(cfg["artifacts"]["weights_dir"]), Path(cfg["artifacts"]["scaler_dir"]), seeds)

        per_seed = {}
        for seed in seeds:
            if "heads" in stages:
                meta = train_heads_for_seed(cfg, seed, windowed, scaler, "cpu",
                                            head_data=head_data, head_tables=head_tables)
                per_seed[str(seed)] = {
                    "val_auc_pr_natural": meta["heads_best_val_auc_pr_natural"],
                    "val_auc_pr": meta["heads_best_val_auc_pr"],
                    "best_epoch": meta["heads_best_epoch_risk"],
                    "input_dim": meta["risk_head_input_dim"],
                }
            if "onset" in stages:
                from nidra.train.train_onset import train_onset_head_for_seed
                ometa = train_onset_head_for_seed(cfg, seed, windowed, scaler, "cpu", head_data=head_data)
                per_seed.setdefault(str(seed), {})["onset_val_ap"] = ometa["best_val_ap_natural_by_horizon"]
            logger.info("%s seed %d done at %.0fs: %s", tag, seed, time.time() - t0, per_seed[str(seed)])

        results["variants"][tag] = {"components": list(components), "run_dir": str(run_dir), "seeds": per_seed}
        write_json(out_path, results)

    record = experiment_record(prep_cfg, stage="head_ablation",
                               extra={"label": args.label, "init_from": args.init_from,
                                      "variants": [variant_tag(v) for v in variants],
                                      "seeds": seeds, "context_length": L,
                                      "seconds": round(time.time() - t0, 1)})
    write_json(RUNS_DIR / args.label / "provenance.json", record)
    logger.info("head ablation %s complete in %.0fs -> %s", args.label, time.time() - t0, out_path)
    print(json.dumps({k: v["seeds"] for k, v in results["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()

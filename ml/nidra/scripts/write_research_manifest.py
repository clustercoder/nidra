"""Baseline manifest for a research phase: what existed, and on what machine,
before anything new was trained.

    python -m nidra.scripts.write_research_manifest \
        --out experiments/RUN9_BASELINE_MANIFEST.json --baseline-ref baseline-run8

Distinct from `write_baseline_manifest.py`, which records the METRICS a
baseline published. This records the INPUTS: the commit and tag the phase
started from, the digest of every raw dataset file (both datasets), the
digest of every shipped artifact, the environment, and the hardware — so
that "we could not reproduce it" can be answered with a diff rather than a
guess.

Never overwrites without --force: a baseline is a historical record.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from nidra.utils.config import PROJECT_ROOT, load_config, resolve_path
from nidra.utils.provenance import file_digest, file_stat, write_json


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _hardware() -> dict:
    def sysctl(key: str) -> str | None:
        try:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": sysctl("hw.ncpu"),
        "memory_bytes": sysctl("hw.memsize"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_mps_available"] = bool(torch.backends.mps.is_available())
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_default_threads"] = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover - torch is a hard dependency in practice
        info["torch_error"] = str(exc)
    return info


def _environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "shap", "matplotlib", "pyyaml"):
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
    }


def _raw_dataset_inventory(cfg: dict) -> dict:
    """Digest every raw input file the configured days name, per dataset."""
    from nidra.train.pipeline import _flow_dirs, day_format

    dataset_cfg = cfg["dataset"]
    dirs = _flow_dirs(dataset_cfg)
    packets_dir = Path(dataset_cfg.get("packets_dir") or ".").expanduser()
    out: dict = {}
    for day_key, day_meta in dataset_cfg["days"].items():
        fmt = day_format(day_meta)
        path = dirs[fmt] / day_meta["file"]
        entry = {"format": fmt, "role": day_meta.get("role"), **(file_stat(path) or {"path": str(path), "missing": True})}
        if path.exists():
            entry["sha256"] = file_digest(path)
        if day_meta.get("packets"):
            ppath = packets_dir / day_meta["packets"]
            entry["packets"] = file_stat(ppath) or {"path": str(ppath), "missing": True}
            if ppath.exists():
                entry["packets"]["sha256"] = file_digest(ppath)
        out[day_key] = entry
    return out


def _artifact_inventory(cfg: dict) -> dict:
    """Digest the shipped artifacts of the baseline being preserved."""
    out: dict = {}
    for key in ("weights_dir", "scaler_dir"):
        directory = resolve_path(cfg, cfg["artifacts"][key])
        if not directory.exists():
            out[key] = {"path": str(directory), "missing": True}
            continue
        out[key] = {
            "path": str(directory),
            "files": {
                f.name: {"bytes": f.stat().st_size, "sha256": file_digest(f)}
                for f in sorted(directory.iterdir()) if f.is_file()
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-ref", default="baseline-run8",
                        help="git tag or SHA of the immutable baseline this phase must stay comparable to")
    parser.add_argument("--configs", nargs="+", default=["config/default.yaml", "config/ctu13.yaml"])
    parser.add_argument("--label", default="run9-ctu13")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists — a baseline manifest is not overwritten without --force")

    configs: dict = {}
    datasets: dict = {}
    for path in args.configs:
        cfg = load_config(path)
        configs[path] = {
            "config_hash": cfg["_config_hash"],
            "lineage": cfg.get("_config_lineage"),
            "feature_regime": cfg.get("features", {}).get("regime", "full"),
            "train_days": cfg.get("splits", {}).get("train_days"),
            "test_days": cfg.get("splits", {}).get("test_days"),
            "holdout_days": cfg.get("splits", {}).get("holdout_days"),
        }
        datasets[path] = _raw_dataset_inventory(cfg)

    base_cfg = load_config(args.configs[0])
    record = {
        "label": args.label,
        "written_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "baseline_ref": args.baseline_ref,
            "baseline_sha": _git("rev-list", "-n", "1", args.baseline_ref),
            "worktree_clean": _git("status", "--porcelain") == "",
            "tags": (_git("tag", "-l") or "").split("\n"),
        },
        "environment": _environment(),
        "hardware": _hardware(),
        "configs": configs,
        "raw_datasets": datasets,
        "baseline_artifacts": _artifact_inventory(base_cfg),
    }
    write_json(out, record)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

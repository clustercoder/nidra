"""What the frozen stage head sees, per stage, that the risk head does not.

    python -m nidra.scripts.stage_head_diagnostic \\
        --config experiments/runs/ctu_heads__state+hidden/config.yaml --split val

A cross-host supervised probe reaches ROC 0.970 on `ctu_4:recon` where the
risk head scores chance (reports/CTU13_MODEL_IMPROVEMENT). One difference
between them is the objective: the probe is fit on that stage one-vs-rest,
while the risk head is fit on `risk_label` — any attack in the next K windows,
every stage pooled into one positive class dominated by whichever stage has
the most examples.

NIDRA already trains a second head on the stage label. If the STAGE head finds
recon while the risk head does not, the signal is inside the model and the
published composite is simply not using it, which is a fixable thing. If
neither finds it, the pooled objective is not the explanation.

Observed states only — this scores the heads where they were trained, so a
gap here is about the objective and not about rollout error.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def one_vs_rest(scores: np.ndarray, labels: np.ndarray, stage: str) -> dict:
    """AP and ROC for `stage` against everything else, plus the base rate."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = (np.asarray(labels) == stage).astype(int)
    base = float(y.mean()) if len(y) else 0.0
    if y.sum() == 0 or (y == 0).sum() == 0:
        return {"stage": stage, "n": int(y.sum()), "base_rate": base,
                "auc_pr": float("nan"), "roc_auc": float("nan"), "lift": float("nan")}
    ap = float(average_precision_score(y, scores))
    return {"stage": stage, "n": int(y.sum()), "base_rate": base, "auc_pr": ap,
            "roc_auc": float(roc_auc_score(y, scores)),
            "lift": ap / base if base > 0 else float("nan")}


def diagnostic_markdown(rows: list[dict], split: str) -> str:
    out = [f"### Frozen head diagnostic per stage — observed states, **{split}**", "",
           "The stage head's own class probability against that stage one-vs-rest, beside the risk head's "
           "score on the same rows. Observed states only: both heads are scored where they were trained, "
           "so a gap is about the objective rather than about rollout error.", "",
           "| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |",
           "|---|---|---|---|---|---|---|---|"]
    def f(v, nd=3):
        return "—" if v is None or v != v else f"{v:.{nd}f}"

    def lift(v):
        return "—" if v is None or v != v else f"{v:.0f}×"

    for r in rows:
        out.append(
            f"| {r['stage']} | {r['n']} | {r['base_rate']:.5f} | {f(r['auc_pr'])} | {lift(r['lift'])} | "
            f"{f(r['roc_auc'])} | {f(r.get('risk_auc_pr'))} | {f(r.get('risk_roc_auc'))} |")
    return "\n".join(out)


def main() -> None:
    from nidra.data.normalize import FeatureScaler
    from nidra.data.schema import FEATURE_ORDER, STAGE_LABELS
    from nidra.eval.benchmark import load_models
    from nidra.train.head_context import build_head_context
    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.utils.config import load_config, resolve_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    _, L, _ = geometry_from_config(cfg)
    df = getattr(build_all_splits(cfg), args.split).sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    models = load_models(cfg, [int(s) for s in args.seeds.split(",")])
    states = torch.from_numpy(scaler.transform(df[FEATURE_ORDER].to_numpy(dtype="float32")).astype("float32"))

    risk, stage_probs = [], []
    for m in models:
        ctx = build_head_context(df, scaler, m, L)
        parts = {"state": states, "hidden": torch.from_numpy(ctx.hidden),
                 "delta": torch.from_numpy(ctx.delta), "logvar": torch.from_numpy(ctx.logvar)}
        with torch.no_grad():
            from nidra.models.heads import TrajectoryRiskHead
            logits = (m.risk_head(**{k: v for k, v in parts.items() if k in m.risk_head.components})
                      if isinstance(m.risk_head, TrajectoryRiskHead) else m.risk_head(states))
            risk.append(torch.sigmoid(logits.squeeze(-1)).numpy())
            stage_probs.append(torch.softmax(m.stage_head(states), dim=-1).numpy())
    risk = np.mean(risk, axis=0)
    stage_probs = np.mean(stage_probs, axis=0)

    labels = df["stage_label"].to_numpy()
    rows = []
    for j, stage in enumerate(STAGE_LABELS):
        if stage == "benign":
            continue
        r = one_vs_rest(stage_probs[:, j], labels, stage)
        rr = one_vs_rest(risk, labels, stage)
        rows.append({**r, "risk_auc_pr": rr["auc_pr"], "risk_roc_auc": rr["roc_auc"]})
        logger.info("%s: n=%d stage-head AP %.4f ROC %.3f | risk-head AP %.4f ROC %.3f",
                    stage, r["n"], r["auc_pr"], r["roc_auc"], rr["auc_pr"], rr["roc_auc"])

    md = diagnostic_markdown([r for r in rows if r["n"] > 0], args.split)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

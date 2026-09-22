"""Can the two frozen heads be combined at inference to recover the stages the
risk head misses?

    python -m nidra.scripts.head_fusion_screen --config <run>/config.yaml \\
        --split val --out reports/tables/ctu_head_fusion_val.md

`stage_head_diagnostic` established that the signal for the stages the risk
head ranks below chance is already inside the model: on CTU validation the
stage head reaches ROC 0.854 on c2 and 0.638 on recon where the risk head sits
at 0.434 and 0.320. Both heads are frozen and already trained, so combining
their scores costs nothing and breaks no invariant — it adds no parameter and
fits nothing on the states it scores.

The question this answers is whether a scalar fusion rule can actually harvest
that. It cannot be assumed: the composite's AP is dominated by whichever stage
carries most of the positives, and a rule that helps a 24-window stage while
dragging a 172-window one loses on the metric that is reported.

Rules are fixed and parameter-free on purpose. A weighted blend with the
weight fitted here would be a model selected on the split it is scored on,
which is the one thing §33 forbids; if a fixed rule wins, a fitted one is
worth doing properly on train and confirming on val.

Screening only. Observed states, one seed, no bootstrap — the output decides
whether a rollout-level system is worth building, and nothing reads it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

#: name -> (risk, nonbenign) -> fused score. Parameter-free by design.
FUSION_RULES: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "risk head alone (published)": lambda r, s: r,
    "stage 1-P(benign) alone": lambda r, s: s,
    "max": np.maximum,
    "mean": lambda r, s: 0.5 * (r + s),
    "noisy-or": lambda r, s: 1.0 - (1.0 - r) * (1.0 - s),
    "geometric mean": lambda r, s: np.sqrt(np.clip(r, 1e-9, 1.0) * np.clip(s, 1e-9, 1.0)),
}

BASELINE_RULE = "risk head alone (published)"


@dataclass(frozen=True)
class FusionRow:
    """One rule's score on one label set."""

    rule: str
    label_set: str
    n_positive: int
    ap: float
    roc: float

    def delta_ap(self, baseline: "FusionRow") -> float:
        return self.ap - baseline.ap


def score_rules(risk: np.ndarray, nonbenign: np.ndarray, y: np.ndarray, label_set: str) -> list[FusionRow]:
    """Every rule against one binary label vector."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    risk = np.asarray(risk, dtype="float64")
    nonbenign = np.asarray(nonbenign, dtype="float64")
    y = np.asarray(y).astype(int)
    if risk.shape != nonbenign.shape or risk.shape != y.shape:
        raise ValueError(f"shape mismatch: risk {risk.shape}, nonbenign {nonbenign.shape}, y {y.shape}")
    if y.sum() == 0 or (y == 0).sum() == 0:
        raise ValueError(f"{label_set}: needs both classes, got {y.sum()} positives of {len(y)}")
    return [FusionRow(rule=name, label_set=label_set, n_positive=int(y.sum()),
                      ap=float(average_precision_score(y, fn(risk, nonbenign))),
                      roc=float(roc_auc_score(y, fn(risk, nonbenign))))
            for name, fn in FUSION_RULES.items()]


def any_rule_wins(rows: list[FusionRow], baseline_rule: str = BASELINE_RULE) -> bool:
    """Does any fusion rule beat the published head on the composite label?
    Single-head rows are not fusion and do not count as a win."""
    base = next((r for r in rows if r.rule == baseline_rule), None)
    if base is None:
        raise ValueError(f"no baseline row {baseline_rule!r}")
    singles = {BASELINE_RULE, "stage 1-P(benign) alone"}
    return any(r.ap > base.ap for r in rows if r.rule not in singles)


def fusion_markdown(rows: list[FusionRow], split: str, per_stage: list[FusionRow] | None = None) -> str:
    by_set: dict[str, list[FusionRow]] = {}
    for r in rows:
        by_set.setdefault(r.label_set, []).append(r)

    out = [f"### Frozen-head fusion screen — observed states, **{split}**", "",
           "Both heads are already trained and frozen; these rules add no parameter and fit nothing "
           "on the rows they score. Screening only — one seed, no bootstrap, observed states. A rule "
           "that wins here earns a rollout-level implementation and a confirmation run; it is not "
           "itself a result.", ""]
    for label_set, group in by_set.items():
        base = next(r for r in group if r.rule == BASELINE_RULE)
        out += [f"**{label_set}** — {base.n_positive} positives", "",
                "| rule | AP | ΔAP vs published | ROC |", "|---|---|---|---|"]
        for r in sorted(group, key=lambda q: -q.ap):
            d = r.delta_ap(base)
            out.append(f"| {r.rule} | {r.ap:.4f} | {'—' if r.rule == BASELINE_RULE else f'{d:+.4f}'} | {r.roc:.4f} |")
        out.append("")
        out.append(f"Verdict: {'a fusion rule beats the published head' if any_rule_wins(group) else '**no fusion rule beats the published head**'}.")
        out.append("")

    if per_stage:
        out += ["**Per stage, one-vs-rest.** Where the two heads disagree, a scalar rule lands between "
                "them rather than above either — the risk head's confident scores on the majority stage "
                "outrank the stage head's correct ordering of the minority one.", "",
                "| stage | positives | " + " | ".join(FUSION_RULES) + " |",
                "|---|---|" + "---|" * len(FUSION_RULES)]
        by_stage: dict[str, dict[str, FusionRow]] = {}
        for r in per_stage:
            by_stage.setdefault(r.label_set, {})[r.rule] = r
        for stage, d in by_stage.items():
            cells = [f"{d[name].roc:.3f}" if name in d else "—" for name in FUSION_RULES]
            n = next(iter(d.values())).n_positive
            out.append(f"| {stage} | {n} | " + " | ".join(cells) + " |")
        out += ["", "ROC, not AP: at these base rates AP is dominated by a handful of rows."]
    return "\n".join(out)


def main() -> None:
    import torch

    from nidra.data.normalize import FeatureScaler
    from nidra.data.schema import FEATURE_ORDER, STAGE_LABELS
    from nidra.eval.benchmark import load_models
    from nidra.train.head_context import build_head_context
    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.utils.config import load_config, resolve_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    _, L, _ = geometry_from_config(cfg)
    df = getattr(build_all_splits(cfg), args.split).sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    model = load_models(cfg, [args.seed])[0]

    states = torch.from_numpy(scaler.transform(df[FEATURE_ORDER].to_numpy(dtype="float32")).astype("float32"))
    ctx = build_head_context(df, scaler, model, L)
    parts = {"state": states, "hidden": torch.from_numpy(ctx.hidden.copy()),
             "delta": torch.from_numpy(ctx.delta.copy()), "logvar": torch.from_numpy(ctx.logvar.copy())}
    components = getattr(model.risk_head, "components", ("state",))
    with torch.no_grad():
        logits = model.risk_head(**{k: v for k, v in parts.items() if k in components})
        risk = torch.sigmoid(logits).numpy().ravel()
        nonbenign = 1.0 - torch.softmax(model.stage_head(states), dim=-1).numpy()[:, STAGE_LABELS.index("benign")]

    rows = score_rules(risk, nonbenign, df["risk_label"].to_numpy(), "composite risk label")
    stage_labels = df["stage_label"].to_numpy()
    per_stage: list[FusionRow] = []
    for stage in sorted(set(stage_labels) - {"benign"}):
        y = (stage_labels == stage).astype(int)
        if y.sum() == 0:
            continue
        per_stage += score_rules(risk, nonbenign, y, stage)

    md = fusion_markdown(rows, args.split, per_stage=per_stage)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        logger.info("wrote %s", args.out)
    print(md)


if __name__ == "__main__":
    main()

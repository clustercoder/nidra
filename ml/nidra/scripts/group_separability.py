"""Is this attack group separable AT ALL in the feature space it is scored in?

    python -m nidra.scripts.group_separability --config experiments/runs/ctu_heads__state+hidden/config.yaml \\
        --split val --out reports/tables/ctu_group_separability.md

Every head tried in this phase scores `ctu_4:c2` (Rbot command-and-control,
56 windows, 6 episodes) at AP 0.001 and ROC 0.47 — at or below chance — while
`ctu_6:c2` goes from 0.478 to 0.999 ROC when the head gains history. Before
concluding anything about the model, it is worth knowing whether those windows
are distinguishable from the capture's benign traffic at all in 32 flow
features at Δ=60 s.

This fits a supervised probe DIRECTLY on the group's one-vs-rest label, with
host-grouped cross-validation so no host appears in both folds, and reports
its AP. That is an upper bound in the same sense `oracle_true_future` is one:
it uses information no deployed system has (the labels of the split it scores)
and exists to bound what is achievable, never to produce a forecast. It is a
diagnostic and is never a selection signal — nothing in the pipeline reads it.

A group whose probe AP is near its base rate is invisible in this
representation, and no head architecture will find it. A group whose probe AP
is high and whose forecast AP is low is a model failure worth chasing.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Small and regularised on purpose. The question is whether the signal is
#: there, not how well an unconstrained model can memorise 56 windows.
PROBE_KWARGS = dict(max_iter=400, learning_rate=0.08, max_depth=3,
                    min_samples_leaf=20, l2_regularization=1.0, class_weight="balanced")


@dataclass(frozen=True)
class GroupProbe:
    group: str
    n_positive: int
    n_rows: int
    base_rate: float
    probe_ap: float
    probe_roc: float
    n_folds: int

    @property
    def lift(self) -> float:
        return self.probe_ap / self.base_rate if self.base_rate > 0 else float("nan")

    @property
    def separable(self) -> bool:
        """A probe that cannot reach three times the base rate on labels it was
        handed is not finding the group."""
        return self.lift >= 3.0


def host_grouped_folds(hosts: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Fold assignment by host, so a host's windows never straddle a fold.
    Hosts carrying positives are spread across folds first, otherwise a fold
    can end up with no positive at all and an undefined AP."""
    rng = np.random.default_rng(seed)
    hosts = np.asarray(hosts)
    uniq = np.unique(hosts)
    pos_hosts = np.unique(hosts[np.asarray(y) == 1])
    neg_hosts = np.setdiff1d(uniq, pos_hosts)
    assign: dict = {}
    for i, h in enumerate(rng.permutation(pos_hosts)):
        assign[h] = i % n_folds
    for i, h in enumerate(rng.permutation(neg_hosts)):
        assign[h] = i % n_folds
    fold_of = np.array([assign[h] for h in hosts])
    return [np.where(fold_of == f)[0] for f in range(n_folds)]


def probe_group(X: np.ndarray, y: np.ndarray, hosts: np.ndarray, group: str,
                n_folds: int = 5, seed: int = 0) -> GroupProbe:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    base = float(y.mean()) if len(y) else 0.0
    scores = np.zeros(len(y), dtype="float64")
    used = 0
    for test_idx in host_grouped_folds(hosts, y, n_folds, seed):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(test_idx) == 0 or y[train_idx].sum() == 0 or len(np.unique(y[train_idx])) < 2:
            continue
        clf = HistGradientBoostingClassifier(random_state=seed, **PROBE_KWARGS).fit(X[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        used += 1
    ap = float(average_precision_score(y, scores)) if y.sum() and used else float("nan")
    roc = float(roc_auc_score(y, scores)) if y.sum() and used else float("nan")
    return GroupProbe(group=group, n_positive=int(y.sum()), n_rows=int(len(y)), base_rate=base,
                      probe_ap=ap, probe_roc=roc, n_folds=used)


def probes_markdown(probes: list[GroupProbe], split: str, forecast_ap: dict[str, float] | None = None) -> str:
    out = [f"### Attack-group separability probe — **{split}**", "",
           "A supervised probe fit directly on each group's one-vs-rest label, host-grouped 5-fold. "
           "It reads the labels of the split it scores, so it is an upper bound in the same sense the "
           "oracle is — a diagnostic, never a system, and never a selection signal.", "",
           "| group | positives | base rate | probe AP | lift | probe ROC | separable | forecast AP |",
           "|---|---|---|---|---|---|---|---|"]
    for p in sorted(probes, key=lambda q: -q.n_positive):
        fa = (forecast_ap or {}).get(p.group)
        out.append(f"| {p.group} | {p.n_positive} | {p.base_rate:.5f} | {p.probe_ap:.3f} | {p.lift:.0f}× | "
                   f"{p.probe_roc:.3f} | {'yes' if p.separable else '**no**'} | "
                   f"{'—' if fa is None else f'{fa:.3f}'} |")
    return "\n".join(out)


def main() -> None:
    from nidra.data.normalize import FeatureScaler
    from nidra.data.schema import FEATURE_ORDER
    from nidra.train.pipeline import build_all_splits
    from nidra.utils.config import load_config, resolve_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--min-positives", type=int, default=5)
    parser.add_argument("--benchmark", default=None, help="a benchmark.json to read forecast AP from")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    df = getattr(build_all_splits(cfg), args.split)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    X = scaler.transform(df[FEATURE_ORDER].to_numpy(dtype="float32")).astype("float32")
    hosts = df["host_id"].to_numpy()
    capture = df["split_group"].to_numpy() if "split_group" in df.columns else np.full(len(df), "all")
    stage = df["stage_label"].to_numpy()

    keys = [f"{c}:{s}" for c, s in zip(capture, stage)]
    probes = []
    for group in sorted({k for k in keys if not k.endswith(":benign")}):
        y = np.array([k == group for k in keys], dtype=int)
        if y.sum() < args.min_positives:
            continue
        probes.append(probe_group(X, y, hosts, group))
        logger.info("%s: %d positives, probe AP %.4f (%.0fx base)", group, probes[-1].n_positive,
                    probes[-1].probe_ap, probes[-1].lift)

    forecast_ap = None
    if args.benchmark:
        rec = json.loads(open(args.benchmark).read())
        per = rec.get("metrics", rec).get("per_attack_group", {})
        forecast_ap = {g: v.get("systems", {}).get("world_model", {}).get("auc_pr") for g, v in per.items()} if per else None

    md = probes_markdown(probes, args.split, forecast_ap)
    print(md)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()

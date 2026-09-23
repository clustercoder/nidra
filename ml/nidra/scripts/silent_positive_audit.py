"""How many forecast positives carry no information in the state at all?

    python -m nidra.scripts.silent_positive_audit --config <run>/config.yaml \\
        --splits train,val --out reports/tables/ctu_silent_positives.md

`risk_label[t]` is 1 when an attack starts within the horizon, so a positive
row need not contain any attack traffic — it is the window BEFORE. If the host
happens to be silent in that window, its 45 features sit on the scaler's silent
state, which is the same vector every other silent window has. Those rows are
bit-identical to thousands of negatives, so no function of the state can order
them above those negatives. History can; the state cannot.

That makes this a ceiling, not a complaint. Within the floor stratum the best
achievable AP from the state alone is the stratum's own prevalence, and any
state-only head that appears to beat it on a split is reading something that
is not in the state — worth checking for, which is why the number is written
down rather than asserted.

The audit is per split and reports both the strict floor (all features within
tolerance of the silent state) and the weaker `is_active == 0`, because a
backward-looking delta or slope can be non-zero in a silent window when the
PREVIOUS window was not: that trace of history inside the state vector is the
difference between the two counts.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Scaled-space tolerance for "this row is the silent state". Tight: the point
#: is exact indistinguishability, not approximate quietness.
FLOOR_ATOL = 1e-5


@dataclass(frozen=True)
class SilentAudit:
    """One split's accounting of positives that carry no state information."""

    split: str
    n_rows: int
    n_positive: int
    n_preonset: int            # risk_label == 1 and stage_label == benign
    n_preonset_inactive: int   # ... and is_active == 0
    n_floor_positive: int      # ... and every feature at the silent state
    n_floor_rows: int          # all rows at the silent state, both classes

    @property
    def floor_prevalence(self) -> float:
        """The best AP any function of the STATE can reach inside the floor
        stratum: every row there is the same vector, so the only ranking
        available is a constant one."""
        return self.n_floor_positive / self.n_floor_rows if self.n_floor_rows else float("nan")

    @property
    def share_of_positives(self) -> float:
        return self.n_floor_positive / self.n_positive if self.n_positive else float("nan")


def audit_split(scaled: np.ndarray, is_active: np.ndarray, risk_label: np.ndarray,
                stage_label: np.ndarray, zero_state: np.ndarray, split: str) -> SilentAudit:
    scaled = np.asarray(scaled, dtype="float32")
    if scaled.shape[1] != len(zero_state):
        raise ValueError(f"{split}: {scaled.shape[1]} features against a {len(zero_state)}-feature silent state")
    y = np.asarray(risk_label).astype(int)
    pre = (y == 1) & (np.asarray(stage_label) == "benign")
    at_floor = np.all(np.isclose(scaled, np.asarray(zero_state, dtype="float32"), atol=FLOOR_ATOL), axis=1)
    return SilentAudit(split=split, n_rows=int(len(y)), n_positive=int(y.sum()), n_preonset=int(pre.sum()),
                       n_preonset_inactive=int((pre & (np.asarray(is_active) == 0)).sum()),
                       n_floor_positive=int((pre & at_floor).sum()), n_floor_rows=int(at_floor.sum()))


@dataclass(frozen=True)
class FloorProbe:
    """What a head's scores are actually doing inside the floor stratum.

    Every row there is the same state vector, so a state-only head produces one
    constant and lands exactly on the ceiling. A history-aware head produces a
    different score per row, and the question is what that variation encodes.
    Two decompositions answer it:

    `host_mean_*` replaces every row's score by its host's mean, discarding
    everything the head said about WHICH window. If that alone reproduces the
    ranking, the head is recognising the host.

    `within_*` restricts to the infected host, where host identity is constant
    and only the timing question remains — the one advance warning actually
    asks. A within-host lift near 1.0 means no timing signal at all.
    """

    n_rows: int
    n_positive: int
    ap: float
    roc: float
    host_mean_ap: float
    host_mean_roc: float
    n_positive_hosts: int
    within_host_rows: int
    within_host_prevalence: float
    within_host_ap: float
    within_host_roc: float
    #: ROC computed inside EACH positive host and averaged over them. With one
    #: infected host it equals `within_host_roc`; with several it differs,
    #: because pooling their rows leaves a between-host component in the
    #: ranking and that component is host identity again, one level down.
    per_host_roc_macro: float = float("nan")

    @property
    def ceiling(self) -> float:
        """A state-only head's AP here: the stratum's prevalence."""
        return self.n_positive / self.n_rows if self.n_rows else float("nan")

    @property
    def lift_over_ceiling(self) -> float:
        return self.ap / self.ceiling if self.ceiling else float("nan")

    @property
    def within_host_lift(self) -> float:
        p = self.within_host_prevalence
        return self.within_host_ap / p if p else float("nan")

    @property
    def is_constant(self) -> bool:
        """The head emitted one value over the whole stratum, which is all a
        state-only head CAN do here. It carries neither host identity nor
        timing, and calling that "no host identity" would read as praise."""
        return abs(self.lift_over_ceiling - 1.0) < 1e-6 and abs(self.roc - 0.5) < 1e-6

    @property
    def is_host_identity(self) -> bool:
        """Did the head buy its lift by recognising the host rather than the
        moment? True when collapsing to host means keeps the ranking AND the
        within-host ordering carries nothing."""
        if self.is_constant:
            return False
        return bool(self.host_mean_roc >= 0.9 and self.within_host_lift < 1.1)

    @property
    def verdict(self) -> str:
        if self.is_constant:
            return "constant — at the ceiling, as a state-only head must be"
        if self.is_host_identity:
            return "**host identity**"
        return "carries timing signal"


def floor_stratum_probe(scores, y, hosts, floor_mask) -> FloorProbe:
    """Decompose a head's ranking over the masked rows into host identity and
    timing.

    The floor stratum is the motivating case — there the state is provably
    uninformative, so anything the head does is one or the other. Nothing in
    the arithmetic is specific to it, and passing an all-true mask asks the
    same question of a whole split, which is worth doing for any result that
    rests on a single infected host.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.asarray(scores, dtype="float64")[np.asarray(floor_mask)]
    yy = np.asarray(y).astype(int)[np.asarray(floor_mask)]
    hh = np.asarray(hosts)[np.asarray(floor_mask)]
    if len(yy) == 0 or yy.sum() == 0 or (yy == 0).sum() == 0:
        raise ValueError(f"floor stratum needs both classes, got {yy.sum()} positives of {len(yy)}")

    means = {h: s[hh == h].mean() for h in np.unique(hh)}
    host_mean = np.array([means[h] for h in hh])
    pos_hosts = np.unique(hh[yy == 1])
    k = np.isin(hh, pos_hosts)
    both = yy[k].sum() > 0 and (yy[k] == 0).sum() > 0

    per_host = []
    for h in pos_hosts:
        m = hh == h
        if yy[m].sum() and (yy[m] == 0).sum():
            per_host.append(roc_auc_score(yy[m], s[m]))
    macro = float(np.mean(per_host)) if per_host else float("nan")

    return FloorProbe(
        n_rows=int(len(yy)), n_positive=int(yy.sum()),
        ap=float(average_precision_score(yy, s)), roc=float(roc_auc_score(yy, s)),
        host_mean_ap=float(average_precision_score(yy, host_mean)),
        host_mean_roc=float(roc_auc_score(yy, host_mean)),
        n_positive_hosts=int(len(pos_hosts)), within_host_rows=int(k.sum()),
        within_host_prevalence=float(yy[k].mean()) if k.sum() else float("nan"),
        within_host_ap=float(average_precision_score(yy[k], s[k])) if both else float("nan"),
        within_host_roc=float(roc_auc_score(yy[k], s[k])) if both else float("nan"),
        per_host_roc_macro=macro)


def probe_markdown(probes: dict[str, FloorProbe], split: str, stratum: str = "floor") -> str:
    where = ("inside the floor stratum" if stratum == "floor" else "over the whole split")
    preamble = ("Every row in the stratum is the same state vector. A state-only head can only emit a "
                "constant, so it lands exactly on the ceiling; a history-aware head varies, and these "
                "two columns say what the variation is."
                if stratum == "floor" else
                "The same decomposition applied to every row, not only the uninformative ones. Here a "
                "state-only head has real features to work with, so its columns are informative too "
                "and the comparison between the two heads is the point.")
    out = [f"### What a head's scores encode {where} — **{split}**", "",
           preamble + " **host-mean** replaces each score by its host's mean, "
           "keeping only host identity. **within-host** restricts to the infected host(s), where "
           "identity is constant and only the timing question remains — which is the question advance "
           "warning asks.", "",
           "| head | rows | positives | positive hosts | AP | lift over ceiling | ROC | host-mean AP | "
           "host-mean ROC | within-host prevalence | within-host AP | within-host lift | within-host ROC | "
           "per-host ROC (macro) | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, p in probes.items():
        out.append(f"| {name} | {p.n_rows} | {p.n_positive} | {p.n_positive_hosts} | {p.ap:.6f} | "
                   f"{p.lift_over_ceiling:.2f}× | {p.roc:.4f} | {p.host_mean_ap:.4f} | {p.host_mean_roc:.4f} | "
                   f"{p.within_host_prevalence:.4f} | {p.within_host_ap:.4f} | {p.within_host_lift:.2f}× | "
                   f"{p.within_host_roc:.4f} | {p.per_host_roc_macro:.4f} | {p.verdict} |")
    out += ["", "A lift that disappears within the host is the head recognising *who*, not *when*. With more "
                "than one infected host the pooled **within-host** columns still carry a between-host "
                "component — host identity again, one level down — so **per-host ROC (macro)**, computed "
                "inside each host and averaged, is the figure to read there. With a single infected host "
                "the two are identical."]
    return "\n".join(out)


def audit_markdown(audits: list[SilentAudit]) -> str:
    out = ["### Positives that carry no information in the state", "",
           "`risk_label` marks the windows BEFORE an attack starts, and a host that is silent in such a "
           "window produces the scaler's silent state — the same 45 numbers every other silent window "
           "has. Those rows cannot be ordered above the negatives they are identical to by any function "
           "of the state; only the host's history distinguishes them.", "",
           "| split | rows | positives | pre-onset | of those, silent | at the exact floor | floor stratum | "
           "state-only AP ceiling there | share of all positives |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in audits:
        out.append(f"| {a.split} | {a.n_rows} | {a.n_positive} | {a.n_preonset} | {a.n_preonset_inactive} | "
                   f"{a.n_floor_positive} | {a.n_floor_rows} | {a.floor_prevalence:.6f} | "
                   f"{100 * a.share_of_positives:.1f}% |")
    out += ["", "The ceiling is the stratum's own prevalence. A state-only head that beats it on a split is "
                "reading something the state does not contain, so the number is also a leak check."]
    return "\n".join(out)


def main() -> None:
    from nidra.data.normalize import FeatureScaler
    from nidra.data.schema import FEATURE_ORDER
    from nidra.train.pipeline import build_all_splits
    from nidra.utils.config import load_config, resolve_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--probe-heads", default=None,
                        help="comma-separated name=config pairs; score each head on the LAST split and "
                             "decompose its floor-stratum ranking into host identity and timing")
    parser.add_argument("--probe-stratum", default="floor", choices=("floor", "all"),
                        help="'floor' asks what the head does where the state is uninformative; "
                             "'all' asks the same of the whole split")
    parser.add_argument("--probe-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    zero = scaler.zero_state_scaled().astype("float32")
    splits = build_all_splits(cfg)

    audits = []
    for name in args.splits.split(","):
        t = getattr(splits, name.strip())
        raw = t[FEATURE_ORDER].to_numpy(dtype="float32")
        a = audit_split(scaler.transform(raw).astype("float32"), raw[:, FEATURE_ORDER.index("is_active")],
                        t["risk_label"].to_numpy(), t["stage_label"].to_numpy(), zero, name.strip())
        logger.info("%s: %d/%d positives at the silence floor (%.1f%%), ceiling AP %.6f",
                    a.split, a.n_floor_positive, a.n_positive, 100 * a.share_of_positives, a.floor_prevalence)
        audits.append(a)

    if args.probe_heads:
        probes = {}
        last = args.splits.split(",")[-1].strip()
        for pair in args.probe_heads.split(","):
            name, _, cfg_path = pair.partition("=")
            probes[name.strip()] = _probe_head(cfg_path.strip(), last, args.seed, args.probe_stratum)
        pmd = probe_markdown(probes, last, stratum=args.probe_stratum)
        if args.probe_out:
            from pathlib import Path
            Path(args.probe_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.probe_out).write_text(pmd + "\n")
            logger.info("wrote %s", args.probe_out)
        print(pmd)
        print()

    md = audit_markdown(audits)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
        logger.info("wrote %s", args.out)
    print(md)


def _probe_head(config_path: str, split: str, seed: int, stratum: str = "floor") -> FloorProbe:
    """Score one run's frozen risk head on the split's observed states and
    decompose what it does inside the floor stratum."""
    import torch

    from nidra.data.normalize import FeatureScaler
    from nidra.data.schema import FEATURE_ORDER
    from nidra.eval.benchmark import load_models
    from nidra.train.head_context import build_head_context
    from nidra.train.pipeline import build_all_splits, geometry_from_config
    from nidra.utils.config import load_config, resolve_path

    cfg = load_config(config_path)
    _, L, _ = geometry_from_config(cfg)
    scaler = FeatureScaler.load(*FeatureScaler.default_paths(resolve_path(cfg, cfg["artifacts"]["scaler_dir"])))
    model = load_models(cfg, [seed])[0]
    df = getattr(build_all_splits(cfg), split).sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    scaled = scaler.transform(df[FEATURE_ORDER].to_numpy(dtype="float32")).astype("float32")
    states = torch.from_numpy(scaled)
    ctx = build_head_context(df, scaler, model, L)
    parts = {"state": states, "hidden": torch.from_numpy(ctx.hidden.copy()),
             "delta": torch.from_numpy(ctx.delta.copy()), "logvar": torch.from_numpy(ctx.logvar.copy())}
    components = getattr(model.risk_head, "components", ("state",))
    with torch.no_grad():
        scores = torch.sigmoid(model.risk_head(**{k: v for k, v in parts.items()
                                                  if k in components})).numpy().ravel()
    mask = (np.ones(len(df), dtype=bool) if stratum == "all" else
            np.all(np.isclose(scaled, scaler.zero_state_scaled().astype("float32"), atol=FLOOR_ATOL), axis=1))
    return floor_stratum_probe(scores, df["risk_label"].to_numpy(), df["host_id"].to_numpy(), mask)


if __name__ == "__main__":
    main()

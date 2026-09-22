"""Markdown tables from benchmark.json records — the numbers in
REAL_DATA_RESULTS.md and the model card are generated here, never typed.

    python -m nidra.scripts.report_tables --run experiments/runs/production \\
        --splits val,test,holdout [--historical experiments/BASELINE_MANIFEST_delta30_run7.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_ORDER = [
    "world_model_calibrated", "world_model", "world_model_deterministic", "noised_persistence",
    "isotropic_noise_persistence", "persistence", "ridge_two_lag", "oracle_true_future",
    "lr_current_state", "lr_flattened_history", "gbdt_current_state", "gru_classifier",
]
SYSTEM_LABEL = {
    "world_model_calibrated": "NIDRA (stochastic rollout, calibrated)",
    "world_model": "NIDRA (stochastic rollout)",
    "world_model_deterministic": "NIDRA (deterministic rollout)",
    "noised_persistence": "persistence + learned noise (mean disabled)",
    "isotropic_noise_persistence": "persistence + isotropic noise",
    "persistence": "persistence (risk head on S_t)",
    "ridge_two_lag": "ridge two-lag dynamics + risk head",
    "oracle_true_future": "oracle: risk head on the true future",
    "lr_current_state": "logistic regression on S_t",
    "lr_flattened_history": "logistic regression on the L-window history",
    "gbdt_current_state": "gradient-boosted trees on S_t",
    "gru_classifier": "GRU sequence classifier",
}


def _load(path: Path) -> dict[str, Any]:
    rec = json.loads(path.read_text())
    return rec["metrics"] if "metrics" in rec and isinstance(rec["metrics"], dict) and "task_published_label" in rec["metrics"] else rec


def _f(x, nd=3) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def systems_table(m: dict, split: str) -> str:
    pub = m["task_published_label"]["systems"]
    det = m["task_A_detection"]["systems"]
    onset = m["task_B_onset_forecast"]
    thr = m["operating_point"]["threshold_used"]
    lines = [f"**{split}** — {m['eval_set']['n_rows']} scored rows, natural prevalence {m['eval_set']['natural_prevalence_published']:.5f}, "
             f"operating point `{m['operating_point']['pooling_key']}`, threshold {thr:.3f} (selected on val).",
             "", "| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |",
             "|---|---|---|---|---|---|---|---|"]
    for name in SYSTEM_ORDER:
        if name not in pub:
            continue
        r = pub[name]
        ci = r.get("auc_pr_bootstrap")
        ci_s = "—"
        if ci:
            ci_s = f"[{_f(ci['ci_low'])}, {_f(ci['ci_high'])}]"
            if ci.get("n_positive_clusters", 99) < 5:
                ci_s += f" ({ci['n_positive_clusters']} positive episodes: not informative)"
        at = r["at_threshold"]
        o5 = onset.get("5", {}).get("systems", {}).get(name, {}).get("auc_pr")
        o15 = onset.get("15", {}).get("systems", {}).get(name, {}).get("auc_pr")
        lines.append(f"| {SYSTEM_LABEL.get(name, name)} | {_f(r['auc_pr'])} | {ci_s} | {_f(r['roc_auc'])} | "
                     f"{_f(at['precision'], 2)} / {_f(at['recall'], 2)} / {_f(at['f1'], 2)} | {_f(r.get('false_alarms_per_hour'), 2)} | "
                     f"{_f(det.get(name, {}).get('auc_pr'))} | {_f(o5)} / {_f(o15)} |")
    oh5 = onset.get("5", {}).get("systems", {}).get("onset_head_direct", {}).get("auc_pr")
    oh15 = onset.get("15", {}).get("systems", {}).get("onset_head_direct", {}).get("auc_pr")
    if oh5 is not None or oh15 is not None:
        lines.append(f"| onset head (explicit supervision, Task B only) | — | — | — | — | — | — | {_f(oh5)} / {_f(oh15)} |")
    mand = m.get("task_published_label_at_mandated_threshold", {}).get("systems", {}).get("world_model_calibrated")
    if mand:
        at = mand["at_threshold"]
        lines.append("")
        lines.append(f"At the mandated 0.75 threshold on the calibrated score: P {_f(at['precision'], 2)} / R {_f(at['recall'], 2)} / F1 {_f(at['f1'], 2)}, "
                     f"{int(at['n_alerts'])} alerts.")
    return "\n".join(lines)


def attribution_table(m: dict, split: str) -> str:
    lines = [f"**{split}** — paired episode-bootstrap difference in natural-prevalence AP (published label).", "",
             "| comparison | ΔAP | 95% CI |", "|---|---|---|"]
    for k, v in m["attribution_published_label"].items():
        lines.append(f"| {k} | {_f(v['point'], 3, ) if isinstance(v, dict) else '—'} | [{_f(v['ci_low'])}, {_f(v['ci_high'])}] |")
    sf = m["state_forecast"]
    lines += ["", "State forecast skill vs persistence (MSE, kept features, all origins / active origins): " +
              ", ".join(f"{k} {_f(v)}" for k, v in sf["skill_vs_persistence"].items()) +
              f"; NIDRA vs ridge {_f(sf.get('skill_vs_ridge'))}."]
    return "\n".join(lines)


def horizon_table(m: dict, split: str) -> str:
    per = m["task_C_progression"]["per_horizon"]
    lines = [f"**{split}** — per-horizon: is t+k an attack window? (natural prevalence)", "",
             "| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |",
             "|---|---|---|---|---|---|---|---|"]
    for e in per:
        lines.append(f"| {e['k']} | {e['minutes_ahead']:.0f} | {e['base_rate_natural']:.5f} | {_f(e['auc_pr_natural'])} | "
                     f"{_f(e.get('auc_pr_natural_deterministic'))} | {_f(e.get('auc_pr_natural_oracle'))} | {_f(e['brier'], 5)} | "
                     f"{_f(e.get('stage_top1_accuracy_on_attack_futures'), 2)} ({e.get('n_attack_futures', 0)}) |")
    return "\n".join(lines)


def onset_table(m: dict, split: str) -> str:
    onset = m["task_B_onset_forecast"]
    lines = [f"**{split}** — Task B: an episode begins within h minutes (origins outside any episode).", "",
             "| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |", "|---|---|---|---|---|---|---|---|"]
    for h, tab in onset.items():
        s = tab["systems"]
        g = lambda n: _f(s.get(n, {}).get("auc_pr"))
        lines.append(f"| {h} | {tab['n_pos']} | {g('world_model_calibrated')} | {g('persistence')} | {g('ridge_two_lag')} | {g('gbdt_current_state')} | {g('lr_flattened_history')} | {g('onset_head_direct')} |")
    return "\n".join(lines)


def episode_table(m: dict, split: str) -> str:
    pe = m["per_episode"]["world_model_calibrated"]
    lines = [f"**{split}** — per episode at the selected threshold ({pe['threshold']:.3f}), {pe['m_consecutive']} consecutive windows: "
             f"{pe['n_episodes']} episodes, {pe['n_warned_pre_onset']} warned before onset, {pe['n_detected_within_episode']} alerted inside the episode; "
             f"median lead {pe['median_lead_time_s']} s, median latency {pe['median_latency_min']} min.", "",
             "| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |", "|---|---|---|---|---|---|"]
    for e in pe["episodes"]:
        lines.append(f"| {e['episode']} | {e['length_windows']} | {e['n_pre_onset_rows']} | {_f(e['max_score_pre_onset'])} | "
                     f"{'yes' if e['warned_pre_onset'] else 'no'} ({_f(e['lead_time_s'], 0)}) | {'yes' if e['detected_within_episode'] else 'no'} ({_f(e['latency_min'], 1)}) |")
    return "\n".join(lines)


#: A group whose positives all sit on one host cannot distinguish "learned the
#: behaviour" from "learned the host". Every group on the CTU validation
#: captures is in this state, and so was Run 8's Friday Bot-C2 result, so the
#: number needs the caveat attached to it rather than kept in a footnote.
SINGLE_HOST_MARK = "¹"


def attack_group_table(m: dict, split: str) -> str:
    """Per-attack-group AP, with the host count beside it.

    `_per_attack_group` has always recorded `n_hosts`; nothing rendered it,
    which is how a single-host group's AP could be read as a generalisation
    result. It is rendered here and marked.
    """
    groups = m.get("per_attack_group") or {}
    if not groups:
        return f"_{split}: no per-group breakdown in this benchmark_"
    lines = ["| group | family | positives | episodes | hosts | world model AP | oracle AP | "
             "persistence AP | state skill vs persistence |", "|---|---|---|---|---|---|---|---|---|"]
    single = False
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1].get("n_positive_rows", 0)):
        sysd = g.get("systems", {})
        def ap(key):
            # summarize_scores writes `auc_pr`; the tables elsewhere call it AP.
            return _f(sysd.get(key, {}).get("auc_pr"), 3)
        one = g.get("n_hosts") == 1
        single = single or one
        lines.append(f"| `{name}` | {g.get('family') or '—'} | {g.get('n_positive_rows', '—')} | "
                     f"{g.get('n_episodes', '—')} | {g.get('n_hosts', '—')}{SINGLE_HOST_MARK if one else ''} | "
                     f"{ap('world_model')} | {ap('oracle_true_future')} | {ap('persistence')} | "
                     f"{_f(g.get('state_skill_vs_persistence'), 3)} |")
    if single:
        lines += ["", f"{SINGLE_HOST_MARK} All of this group's positives are on a single host, so its AP does "
                      "not separate the attack's behaviour from that host's identity. Treat it as a "
                      "within-host result until a capture with two infected hosts in the same stage says "
                      "otherwise."]
    return "\n".join(lines)


def host_identity_table(m: dict, split: str) -> str:
    """Per system: is the ranking the host, or the moment?

    Both corpora put each attack group's positives on one host, so an
    aggregate AP cannot tell those apart. Within-host ROC can, and it is
    prevalence-independent, so it is the column to compare across splits and
    datasets. Within-host lift is not comparable that way — the base rate
    beside it differs — and is shown for reading a single row, not for
    ranking rows against each other.
    """
    hi = m.get("host_identity_published_label") or {}
    if not hi:
        return f"_{split}: no host/timing decomposition in this benchmark_"
    lines = ["| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | "
             "within-host AP | within-host lift | within-host ROC | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in SYSTEM_ORDER:
        if name not in hi:
            continue
        h = hi[name]
        lift = _f(h.get("within_host_lift"), 2)
        verdict = "**host identity**" if h.get("is_host_identity") else "carries timing signal"
        lines.append(f"| {name} | {_f(h.get('ap'))} | {_f(h.get('roc'))} | {_f(h.get('host_mean_roc'), 4)} | "
                     f"{h.get('n_positive_hosts', '—')} | {_f(h.get('within_host_prevalence'), 4)} | "
                     f"{_f(h.get('within_host_ap'))} | {lift if lift == '—' else lift + '×'} | "
                     f"{_f(h.get('within_host_roc'), 4)} | {verdict} |")
    lines += ["", "Within-host ROC is the column that survives the single-host confound: it asks, on the "
                  "infected host alone, whether the system orders the attack windows above that host's "
                  "own benign ones."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run dir holding artifacts/metrics/<split>/benchmark.json, or the metrics dir itself")
    parser.add_argument("--splits", default="val,test,holdout")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.run)
    metrics_dir = root / "artifacts" / "metrics" if (root / "artifacts" / "metrics").exists() else root
    sections = []
    for split in args.splits.split(","):
        path = metrics_dir / split / "benchmark.json"
        if not path.exists():
            sections.append(f"_{split}: no benchmark.json at {path}_")
            continue
        m = _load(path)
        sections += [f"### Systems — {split}", systems_table(m, split), "", f"### Attribution — {split}", attribution_table(m, split), "",
                     f"### Horizon — {split}", horizon_table(m, split), "", f"### Onset forecasting — {split}", onset_table(m, split), "",
                     f"### Episodes — {split}", episode_table(m, split), "",
                     f"### Per attack group — {split}", attack_group_table(m, split), "",
                     f"### Host identity vs timing — {split}", host_identity_table(m, split), ""]
    text = "\n".join(sections)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

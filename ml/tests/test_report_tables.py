"""The Markdown tables in REAL_DATA_RESULTS.md and the model card are generated
from benchmark records, never typed — so the generator must read the record
shape `nidra.eval.benchmark` writes and must flag the intervals that are not
informative (fewer than five positive episodes)."""

from __future__ import annotations

import json

from nidra.scripts import report_tables as rt


def _system(ap: float, clusters: int = 12) -> dict:
    return {
        "auc_pr": ap, "roc_auc": 0.9, "false_alarms_per_hour": 0.5,
        "auc_pr_bootstrap": {"ci_low": ap - 0.1, "ci_high": ap + 0.1, "n_positive_clusters": clusters},
        "at_threshold": {"precision": 0.8, "recall": 0.6, "f1": 0.69, "n_alerts": 40},
    }


def _record(clusters: int = 12) -> dict:
    systems = {"world_model_calibrated": _system(0.7, clusters), "world_model": _system(0.7, clusters),
               "persistence": _system(0.6, clusters), "ridge_two_lag": _system(0.5, clusters)}
    onset_tab = {"n_pos": 3, "systems": {"world_model_calibrated": {"auc_pr": 0.01}, "persistence": {"auc_pr": 0.02},
                                          "onset_head_direct": {"auc_pr": 0.03}}}
    return {
        "eval_set": {"n_rows": 100, "natural_prevalence_published": 0.001},
        "operating_point": {"threshold_used": 0.7, "pooling_key": "mean|q=-|max"},
        "task_published_label": {"systems": systems},
        "task_published_label_at_mandated_threshold": {"systems": {"world_model_calibrated": _system(0.7)}},
        "task_A_detection": {"systems": {k: {"auc_pr": 0.8} for k in systems}},
        "task_B_onset_forecast": {"5": onset_tab, "15": onset_tab},
        "task_C_progression": {"per_horizon": [
            {"k": 1, "minutes_ahead": 1.0, "base_rate_natural": 0.001, "auc_pr_natural": 0.8, "brier": 0.001,
             "auc_pr_natural_deterministic": 0.79, "auc_pr_natural_oracle": 0.9,
             "stage_top1_accuracy_on_attack_futures": 0.7, "n_attack_futures": 10}]},
        "attribution_published_label": {"world_model - persistence": {"point": 0.1, "ci_low": 0.0, "ci_high": 0.2}},
        "state_forecast": {"skill_vs_persistence": {"world_model_deterministic": 0.4}, "skill_vs_ridge": 0.05},
        "per_episode": {"world_model_calibrated": {
            "threshold": 0.7, "m_consecutive": 2, "n_episodes": 1, "n_warned_pre_onset": 1, "n_detected_within_episode": 1,
            "median_lead_time_s": 120.0, "median_latency_min": 0.0,
            "episodes": [{"episode": "h@1", "length_windows": 5, "n_pre_onset_rows": 3, "max_score_pre_onset": 0.8,
                          "warned_pre_onset": True, "lead_time_s": 120.0, "detected_within_episode": True, "latency_min": 0.0}]}},
    }


def test_every_table_renders_from_a_benchmark_record():
    m = _record()
    for fn in (rt.systems_table, rt.attribution_table, rt.horizon_table, rt.onset_table, rt.episode_table):
        text = fn(m, "test")
        assert text.startswith("**test**")
        assert "|" in text


def test_systems_table_reports_the_onset_head_at_both_horizons_and_the_mandated_threshold():
    text = rt.systems_table(_record(), "test")
    assert "onset head (explicit supervision, Task B only) | — | — | — | — | — | — | 0.030 / 0.030 |" in text
    assert "At the mandated 0.75 threshold" in text
    assert "@ 1 min" not in text


def test_few_positive_episodes_are_flagged_as_not_informative():
    text = rt.systems_table(_record(clusters=2), "holdout")
    assert "(2 positive episodes: not informative)" in text
    assert "not informative" not in rt.systems_table(_record(clusters=12), "test")


def test_load_accepts_a_wrapped_provenance_record(tmp_path):
    rec = {"stage": "benchmark", "metrics": _record()}
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(rec))
    assert "task_published_label" in rt._load(path)
    path.write_text(json.dumps(_record()))
    assert "task_published_label" in rt._load(path)


def test_attack_group_table_marks_a_single_host_group():
    """Every CTU validation group and Run 8's Friday Bot-C2 have their
    positives on one host; an AP rendered without that is read as a
    generalisation result."""
    from nidra.scripts.report_tables import attack_group_table
    m = {"per_attack_group": {
        "ctu_4:c2": {"n_positive_rows": 23, "n_episodes": 6, "n_hosts": 1, "family": "Rbot",
                     "state_skill_vs_persistence": 0.12,
                     "systems": {"world_model": {"auc_pr": 0.001}, "oracle_true_future": {"auc_pr": 0.002},
                                 "persistence": {"auc_pr": 0.0009}}},
        "ctu_6:exfil": {"n_positive_rows": 122, "n_episodes": 4, "n_hosts": 3, "family": "Menti",
                        "systems": {"world_model": {"auc_pr": 0.97}}}}}
    md = attack_group_table(m, "val")
    assert "ctu_4:c2" in md and "ctu_6:exfil" in md
    assert "0.001" in md and "0.970" in md          # the AP key is auc_pr, not ap
    assert "1¹" in md                                   # marked
    assert "3¹" not in md and "| 3 |" in md             # not marked
    assert "single host" in md
    # ordered by positives, so the bigger group comes first
    assert md.index("ctu_6:exfil") < md.index("ctu_4:c2")


def test_attack_group_table_without_single_host_groups_has_no_footnote():
    from nidra.scripts.report_tables import attack_group_table
    m = {"per_attack_group": {"g": {"n_positive_rows": 5, "n_hosts": 4, "systems": {}}}}
    md = attack_group_table(m, "val")
    assert "single host" not in md


def test_attack_group_table_handles_a_benchmark_without_groups():
    from nidra.scripts.report_tables import attack_group_table
    assert "no per-group breakdown" in attack_group_table({}, "test")


def test_host_identity_table_reports_the_within_host_column():
    """The decomposition's whole point is that aggregate AP cannot separate
    'learned the family' from 'learned the host'; the table has to carry the
    column that can."""
    from nidra.scripts.report_tables import host_identity_table
    m = {"host_identity_published_label": {
        "world_model": {"ap": 0.489, "roc": 0.744, "host_mean_roc": 0.9995, "n_positive_hosts": 2,
                        "within_host_prevalence": 0.7579, "within_host_ap": 0.8951,
                        "within_host_lift": 1.18, "within_host_roc": 0.6621, "is_host_identity": False},
        "persistence": {"ap": 0.3, "roc": 0.6, "host_mean_roc": 0.999, "n_positive_hosts": 2,
                        "within_host_prevalence": 0.7579, "within_host_ap": 0.76,
                        "within_host_lift": 1.0, "within_host_roc": 0.50, "is_host_identity": True}}}
    md = host_identity_table(m, "val")
    assert "0.6621" in md and "1.18×" in md
    assert "**host identity**" in md and "carries timing signal" in md
    assert "Within-host ROC" in md


def test_host_identity_table_handles_a_benchmark_without_it():
    from nidra.scripts.report_tables import host_identity_table
    assert "no host/timing decomposition" in host_identity_table({}, "test")

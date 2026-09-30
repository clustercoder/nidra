"""The scorecard is the phase's headline artifact, so a cell that silently
borrows another regime's number, or a missing column rendered as a plausible
one, would be the worst kind of error this project can make.
"""

from __future__ import annotations

import json

import pytest

from nidra.scripts.cross_dataset_scorecard import _ci, _f, _host_identity_fields, read_benchmark, row_for


def _record(ap=0.5, system="world_model_calibrated", host_identity=True, n_hosts=1):
    m = {
        "task_published_label": {
            "n_rows": 1000, "prevalence_natural": 0.003,
            "systems": {system: {"auc_pr": ap, "roc_auc": 0.8,
                                 "at_threshold": {"precision": 0.4, "recall": 0.6, "f1": 0.48},
                                 "false_alarms_per_hour": 1.32,
                                 "alerts_per_hour": 5.16,
                                 "active_benign_false_alarm_rate": 0.000365,
                                 "auc_pr_bootstrap": {"ci_low": 0.4, "ci_high": 0.6, "n_positive_clusters": 7}},
                        "persistence": {"auc_pr": 0.3},
                        "gru_classifier": {"auc_pr": 0.45},
                        "oracle_true_future": {"auc_pr": 0.9}}},
        "task_B_onset_forecast": {"5": {"systems": {system: {"auc_pr": 0.01}}}},
        "per_episode": {"world_model_calibrated": {"n_episodes": 10, "n_warned_pre_onset": 3}},
        "state_forecast": {"skill_vs_persistence": {"mean": 0.2}, "skill_vs_ridge": 0.1},
    }
    if host_identity:
        m["host_identity_published_label"] = {
            system: {"n_positive_hosts": n_hosts, "within_host_roc": 0.66,
                     "within_host_prevalence": 0.75, "host_mean_roc": 0.9995}}
    return {"split": "test", "metrics": m}


def test_row_carries_the_host_decomposition():
    r = row_for(_record())
    assert r["n_positive_hosts"] == 1
    assert r["within_host_roc"] == pytest.approx(0.66)
    assert r["host_mean_roc"] == pytest.approx(0.9995)


def test_a_benchmark_without_the_block_gets_no_column_not_a_number():
    """Benchmarks predating the block must show a gap. Inventing a value here
    would put a fabricated number in the headline table."""
    r = row_for(_record(host_identity=False))
    assert r["within_host_roc"] is None
    assert r["n_positive_hosts"] is None
    assert _f(r["within_host_roc"]) == "—"


def test_the_decomposition_is_read_for_the_row_s_own_system():
    m = _record(system="world_model")["metrics"]
    assert _host_identity_fields(m, "world_model")["within_host_roc"] == pytest.approx(0.66)
    assert _host_identity_fields(m, "persistence")["within_host_roc"] is None


def test_best_baseline_excludes_the_model_and_the_oracle():
    """An oracle counted as a baseline would make every row look beaten."""
    r = row_for(_record())
    assert r["best_baseline"] == "gru_classifier"
    assert r["best_baseline_ap"] == pytest.approx(0.45)


def test_ci_marks_a_thin_cluster_count():
    assert _ci({"auc_pr_bootstrap": {"ci_low": 0.1, "ci_high": 0.2, "n_positive_clusters": 3}}).endswith("*")
    assert not _ci({"auc_pr_bootstrap": {"ci_low": 0.1, "ci_high": 0.2, "n_positive_clusters": 9}}).endswith("*")
    assert _ci(None) == ""


def test_missing_benchmark_reads_as_none_not_as_zero(tmp_path):
    assert read_benchmark(tmp_path / "nope.json") is None


def test_unreadable_benchmark_reads_as_none(tmp_path):
    p = tmp_path / "benchmark.json"
    p.write_text("{ truncated")
    assert read_benchmark(p) is None


def test_nan_renders_as_a_gap():
    assert _f(float("nan")) == "—"
    assert _f(None) == "—"


def test_scorecard_end_to_end_marks_single_host_rows(tmp_path, monkeypatch, capsys):
    import sys

    from nidra.scripts import cross_dataset_scorecard as mod

    for label, n_hosts in (("cic_core", 4), ("ctu", 1)):
        d = tmp_path / label / "artifacts" / "metrics" / "test"
        d.mkdir(parents=True)
        (d / "benchmark.json").write_text(json.dumps(_record(n_hosts=n_hosts)))
    out = tmp_path / "scorecard.md"
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(tmp_path), "--splits", "test",
                                      "--map", "CIC:CIC:cic_core", "--map", "CTU:CTU:ctu",
                                      "--out", str(out)])
    mod.main()
    text = out.read_text()
    assert "within-host ROC" in text
    assert "1¹" in text and "| 4 |" in text
    assert "single host" in text
    assert "0.6600" in text
    # the JSON sidecar carries the same fields for downstream use
    side = json.loads((out.parent / "scorecard.json").read_text())
    assert side["ctu/test"]["within_host_roc"] == pytest.approx(0.66)


def test_false_alarms_per_hour_is_the_per_hour_field_not_the_per_row_one():
    """The scorecard read `active_benign_false_alarm_rate` — a fraction of
    active-benign ROWS — and printed it under an FA/h heading, where it rounded
    to 0.00 and made every regime look silent. The real rate was 1.32/h."""
    r = row_for(_record())
    assert r["false_alarms_per_hour"] == pytest.approx(1.32)
    assert r["active_benign_fa_rate"] == pytest.approx(0.000365)
    assert r["alerts_per_hour"] == pytest.approx(5.16)


def test_both_false_alarm_columns_reach_the_table(tmp_path, monkeypatch):
    import sys

    from nidra.scripts import cross_dataset_scorecard as mod

    d = tmp_path / "r" / "artifacts" / "metrics" / "test"
    d.mkdir(parents=True)
    (d / "benchmark.json").write_text(json.dumps(_record()))
    out = tmp_path / "s.md"
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(tmp_path), "--splits", "test",
                                      "--map", "A:B:r", "--out", str(out)])
    mod.main()
    text = out.read_text()
    assert "FA/h" in text and "FA rate on active benign" in text
    assert "1.32" in text and "0.00036" in text


def test_a_not_run_row_has_the_same_column_count_as_a_real_one(tmp_path, monkeypatch):
    """A short filler row silently shifts every column after it."""
    import sys

    from nidra.scripts import cross_dataset_scorecard as mod

    d = tmp_path / "r" / "artifacts" / "metrics" / "test"
    d.mkdir(parents=True)
    (d / "benchmark.json").write_text(json.dumps(_record()))
    out = tmp_path / "s.md"
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(tmp_path), "--splits", "test",
                                      "--map", "A:B:r", "--map", "A:B:missing", "--out", str(out)])
    mod.main()
    rows = [l for l in out.read_text().splitlines() if l.startswith("| A |")]
    assert len(rows) == 2
    assert rows[0].count("|") == rows[1].count("|")
    header = next(l for l in out.read_text().splitlines() if l.startswith("| Training"))
    assert header.count("|") == rows[0].count("|")


def _with_calibration(record, raw_brier=0.063, cal_brier=0.0056):
    """The calibration block as the benchmark actually writes it: an empty bin
    carries `null`, not a zero frequency."""
    record["metrics"]["calibration_published_label"] = {
        "raw": {"brier_natural": raw_brier, "reliability": [
            {"bin_center": 0.05, "n": 900, "weighted_n": 9900.0, "observed_frequency_natural": 0.002},
            {"bin_center": 0.95, "n": 10, "weighted_n": 100.0, "observed_frequency_natural": 0.95},
            {"bin_center": 0.45, "n": 0, "weighted_n": 0.0, "observed_frequency_natural": None}]},
        "calibrated": {"brier_natural": cal_brier, "reliability": [
            {"bin_center": 0.05, "n": 990, "weighted_n": 9950.0, "observed_frequency_natural": 0.006},
            {"bin_center": 0.95, "n": 5, "weighted_n": 50.0, "observed_frequency_natural": 0.90}]}}
    return record


def test_the_row_carries_brier_and_ece_for_both_arms():
    """§36 item 19 and §32 criterion 8. Without these in the scorecard, the
    only calibration evidence lives inside per-run benchmark JSON."""
    r = row_for(_with_calibration(_record()))
    assert r["brier_raw"] == pytest.approx(0.063)
    assert r["brier_calibrated"] == pytest.approx(0.0056)
    assert r["ece_raw"] == pytest.approx((9900 * 0.048 + 100 * 0.0) / 10000)
    assert r["ece_calibrated"] is not None


def test_the_constant_base_rate_brier_travels_with_them():
    """A Brier of 0.0056 at a prevalence of 0.006 is not a good Brier, and the
    reader cannot know that without the constant a lazy predictor would score."""
    r = row_for(_with_calibration(_record()))
    p = (9950 * 0.006 + 50 * 0.90) / 10000
    assert r["base_rate_from_bins"] == pytest.approx(p)
    assert r["brier_constant"] == pytest.approx(p * (1 - p))


def test_a_run_without_calibration_gets_none_not_zero():
    r = row_for(_record())
    assert r["brier_raw"] is None and r["ece_calibrated"] is None and r["brier_constant"] is None


def test_an_all_empty_reliability_does_not_report_a_confident_zero():
    rec = _record()
    rec["metrics"]["calibration_published_label"] = {
        "calibrated": {"brier_natural": 0.1, "reliability": [
            {"bin_center": 0.05, "n": 0, "weighted_n": 0.0, "observed_frequency_natural": None}]}}
    r = row_for(rec)
    assert r["brier_calibrated"] == pytest.approx(0.1)
    assert r["ece_calibrated"] != r["ece_calibrated"]      # nan
    assert r["brier_constant"] != r["brier_constant"]      # nan


def test_the_row_carries_the_deterministic_system_beside_the_oracle():
    """§25. `oracle_true_future` is the frozen head on the TRUE future under a
    single-trajectory readout; `world_model_calibrated` pools ~200 stochastic
    trajectories and then applies Platt. Comparing those two is not a ratio to
    an upper bound, and in 7 of 14 measured cells it exceeds 1. The matched
    comparison is against `world_model_deterministic`, so it has to be in the
    row for the oracle column to be readable at all."""
    rec = _record()
    rec["metrics"]["task_published_label"]["systems"]["world_model_deterministic"] = {"auc_pr": 0.21}
    r = row_for(rec)
    assert r["deterministic_ap"] == pytest.approx(0.21)
    assert r["oracle_ap"] == pytest.approx(0.9)


def test_a_run_without_the_deterministic_system_gets_none():
    assert row_for(_record())["deterministic_ap"] is None


def test_gaps_are_listed_with_the_path_that_was_missing(tmp_path, monkeypatch, capsys):
    """Fourteen all-dash rows in one scorecard, of which eight were absent by
    design (transfer arms have no val split) and six were crashes nobody
    noticed. The table renders both identically. It cannot know which is which
    — but it can say where it looked, which makes the difference auditable
    instead of invisible."""
    import sys

    from nidra.scripts import cross_dataset_scorecard as sc

    runs = tmp_path / "runs"
    (runs / "arm_a" / "artifacts" / "metrics" / "test").mkdir(parents=True)
    (runs / "arm_a" / "artifacts" / "metrics" / "test" / "benchmark.json").write_text(
        json.dumps(_with_calibration(_record())))
    out = tmp_path / "scorecard.md"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--runs", str(runs), "--splits", "test,holdout",
        "--map", "A:A:arm_a", "--map", "B:B:arm_b", "--out", str(out)])
    sc.main()

    text = out.read_text()
    assert "3 of 4" in text                       # arm_a/holdout, arm_b/test, arm_b/holdout
    assert "arm_b/artifacts/metrics/test" in text
    assert "arm_a/artifacts/metrics/holdout" in text
    assert "arm_a/artifacts/metrics/test" not in text.split("### Gaps")[1]   # the one that IS there


def test_a_scorecard_with_no_gaps_says_nothing_about_them(tmp_path, monkeypatch):
    import sys

    from nidra.scripts import cross_dataset_scorecard as sc

    runs = tmp_path / "runs"
    (runs / "arm_a" / "artifacts" / "metrics" / "test").mkdir(parents=True)
    (runs / "arm_a" / "artifacts" / "metrics" / "test" / "benchmark.json").write_text(
        json.dumps(_with_calibration(_record())))
    out = tmp_path / "sc.md"
    monkeypatch.setattr(sys, "argv", ["prog", "--runs", str(runs), "--splits", "test",
                                      "--map", "A:A:arm_a", "--out", str(out)])
    sc.main()
    assert "### Gaps" not in out.read_text()


def test_the_published_system_is_bootstrapped():
    """§33 requires confidence intervals, and `world_model_calibrated` is the
    system every headline AP in this phase is quoted from. It was absent from
    BOOTSTRAP_SYSTEMS while ten other systems were in it, so the scorecard's
    `AP [95% CI]` column rendered a bare AP in every cell.

    The uncalibrated `world_model` is not a substitute: Platt is monotone
    within a horizon but the composite takes `max` ACROSS horizons afterwards,
    so the two can rank rows differently — §3.24 found calibration flipping two
    oracle comparisons for exactly that reason."""
    from nidra.eval.benchmark import BOOTSTRAP_SYSTEMS
    assert "world_model_calibrated" in BOOTSTRAP_SYSTEMS
    assert "world_model" in BOOTSTRAP_SYSTEMS

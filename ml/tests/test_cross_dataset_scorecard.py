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
                                 "active_benign_false_alarm_rate": 1.5,
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

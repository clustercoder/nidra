"""The natural-prevalence benchmark: evaluation-set construction, weighted
metrics, operating-point selection on validation, and the end-to-end CLI on
synthetic fixtures (structure and invariants, not real-data numbers)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nidra.data.labels import attach_risk_label, label_stage_table
from nidra.data.schema import CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS
from nidra.data.splits import SplitResult, temporal_train_val_split
from nidra.data.windowize import build_state_rows
from nidra.eval import benchmark as bench_mod
from nidra.eval.eval_set import build_eval_set
from nidra.eval.metrics_natural import (
    best_f1_threshold,
    bootstrap_ap,
    bootstrap_ap_difference,
    confusion_at,
    weighted_ap,
)
from nidra.explain.shap_runner import build_shap_background, save_background
from nidra.train.pipeline import build_windowed_splits, fit_scaler
from nidra.train.train_dynamics import train_one_seed
from nidra.train.train_heads import train_heads_for_seed
from tests.conftest import full_cfg_dict
from tests.fixtures.synth import make_synthetic_flows, make_synthetic_packets


def _day(portscan_start: int, n_hosts: int = 3, n_windows: int = 200) -> pd.DataFrame:
    flows = make_synthetic_flows(n_hosts=n_hosts, n_windows=n_windows, portscan_start_window=portscan_start, portscan_len=10)
    packets = make_synthetic_packets(flows)
    states = build_state_rows(flows, packets, window_seconds=WINDOW_SECONDS)
    stage_table, _ = label_stage_table(flows)
    return attach_risk_label(states, stage_table, horizon_k=HORIZON_LENGTH, window_seconds=WINDOW_SECONDS)


# ---------------------------------------------------------------------------
# eval set
# ---------------------------------------------------------------------------

def test_eval_set_strata_weights_reproduce_the_full_split():
    table = _day(100)
    ev = build_eval_set(table, CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS, "test",
                        caps={"active_negative": 40, "silent_negative": 10}, seed=0)
    counts = ev.stratum_counts
    # every positive and every pre-onset row is kept
    assert counts["positive"]["sampled"] == counts["positive"]["full"] > 0
    assert counts["pre_onset"]["sampled"] == counts["pre_onset"]["full"] > 0
    assert counts["active_negative"]["sampled"] <= 40
    # weights sum to the number of candidate origins in the full split
    total_full = sum(c["full"] for c in counts.values())
    assert abs(ev.weight.sum() - total_full) < 1e-6
    # natural prevalence from weights equals the split's candidate prevalence
    prev = counts["positive"]["full"] / total_full
    assert abs(np.average(ev.y_published, weights=ev.weight) - prev) < 1e-9
    # forecasting rows exclude every in-episode origin
    assert not ev.inside_episode[ev.mask_forecast].any()
    assert ev.onset_labels[3][ev.mask_forecast].sum() > 0
    # an onset label at a longer horizon is a superset of a shorter one
    assert (ev.onset_labels[10] >= ev.onset_labels[3]).all()
    # clusters: attack-related rows share an episode key, benign rows are keyed by host
    assert ev.cluster[ev.y_published == 1][0].count("@") == 1


def test_eval_set_pre_onset_rows_are_before_the_onset_and_not_attacks():
    table = _day(120)
    ev = build_eval_set(table, CONTEXT_LENGTH, HORIZON_LENGTH, WINDOW_SECONDS, "test", seed=1)
    pre = ev.stratum == "pre_onset"
    assert pre.any()
    assert (ev.minutes_to_onset[pre] <= 30).all() and (ev.minutes_to_onset[pre] > 0).all()
    assert (ev.arrays.stage_label[pre] == "benign").all()


# ---------------------------------------------------------------------------
# weighted metrics
# ---------------------------------------------------------------------------

def test_weighted_ap_equals_ap_on_the_expanded_population():
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 30)
    s = rng.random(50) + 0.5 * y
    w = np.where(y == 1, 1.0, 4.0)      # each negative stands for four
    y_full = np.concatenate([y[y == 1], np.repeat(y[y == 0], 4)])
    s_full = np.concatenate([s[y == 1], np.repeat(s[y == 0], 4)])
    assert abs(weighted_ap(y, s, w) - weighted_ap(y_full, s_full, None)) < 1e-9
    c = confusion_at(y, s, 0.9, w)
    assert abs(c["fp"] - (s_full[y_full == 0] >= 0.9).sum()) < 1e-9


def test_best_f1_threshold_is_on_the_curve_and_bootstrap_intervals_contain_the_point():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    w = np.ones(400)
    thr, f1 = best_f1_threshold(y, s, w)
    assert 0 <= thr <= 1 and abs(confusion_at(y, s, thr, w)["f1"] - f1) < 1e-6
    clusters = np.array([f"c{i // 10}" for i in range(400)])
    b = bootstrap_ap(y, s, w, clusters, n_resamples=50)
    assert b["ci_low"] <= b["point"] <= b["ci_high"]
    d = bootstrap_ap_difference(y, s, s, w, clusters, n_resamples=20)
    assert abs(d["point"]) < 1e-12 and abs(d["ci_low"]) < 1e-12


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

@pytest.fixture
def benchmark_artifacts(tmp_path, monkeypatch):
    cfg = full_cfg_dict(tmp_path)
    cfg["training_data"] = {"max_train_samples": 600, "max_val_samples": 300}
    cfg["eval"]["episode_merge_gap_windows"] = 5
    scaler_dir = tmp_path / "scaler"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    train_day = _day(60)          # episode at windows 60-69: stays in train
    val_day = _day(100, n_hosts=2)
    test_day = _day(100)
    train = train_day
    val = val_day
    splits = SplitResult(train=train, val=val, test=test_day, holdout=pd.DataFrame())
    windowed = build_windowed_splits(splits)
    scaler = fit_scaler(windowed["train"])
    scaler.save(scaler_dir / "feature_scaler.json", scaler_dir / "scaler_metadata.json")
    benign_mask = windowed["train"].stage_label == "benign"
    save_background(build_shap_background(scaler.transform(windowed["train"].X[benign_mask, -1, :]), n_centroids=10),
                    scaler_dir / "shap_background.npy")
    train_one_seed(cfg, seed=0, epochs_override=1, windowed=windowed, scaler=scaler, device="cpu")
    train_heads_for_seed(cfg, seed=0, windowed=windowed, scaler=scaler, device="cpu")
    monkeypatch.setattr(bench_mod, "build_all_splits", lambda cfg: splits)
    return cfg, tmp_path


def test_benchmark_selects_on_val_then_freezes_for_test(benchmark_artifacts):
    cfg, tmp_path = benchmark_artifacts
    op_path = tmp_path / "weights" / "operating_point.json"
    val_rec = bench_mod.run(cfg, "val", [0], n_samples=6, out_dir=tmp_path / "metrics" / "val", operating_point_path=op_path,
                            select_op=True, caps={"active_negative": 60, "silent_negative": 20}, n_resamples=10, chunk=64,
                            gru_classifier=None)
    assert op_path.exists()
    op = json.loads(op_path.read_text())
    assert op["selected_on"] == "val"
    assert op["pooling_key"] in op["candidates"]
    assert 0.0 <= op["threshold"]["f1_optimal_calibrated"] <= 1.0
    assert len(op["calibration"]["params_by_k"]) == HORIZON_LENGTH

    # selecting on test must be refused
    with pytest.raises(ValueError):
        bench_mod.run(cfg, "test", [0], 6, tmp_path / "m", op_path, True, None, 5, 64, None)

    test_rec = bench_mod.run(cfg, "test", [0], n_samples=6, out_dir=tmp_path / "metrics" / "test", operating_point_path=op_path,
                             select_op=False, caps={"active_negative": 60, "silent_negative": 20}, n_resamples=10, chunk=64,
                             gru_classifier=None)
    m = test_rec["metrics"]
    assert m["operating_point"]["source"].startswith("loaded from")
    assert m["operating_point"]["pooling"] == op["pooling"]
    systems = m["task_published_label"]["systems"]
    for name in ("world_model", "world_model_deterministic", "persistence", "noised_persistence",
                 "isotropic_noise_persistence", "ridge_two_lag", "oracle_true_future",
                 "lr_current_state", "lr_flattened_history", "gbdt_current_state", "world_model_calibrated"):
        assert name in systems, name
        assert np.isfinite(systems[name]["auc_pr"])
        assert "active_benign_false_alarm_rate" in systems[name]
    assert "auc_pr_bootstrap" in systems["world_model"]
    assert "world_model - persistence" in m["attribution_published_label"]
    assert set(m["task_B_onset_forecast"]) == {"1", "3", "5", "10", "15", "30"}
    assert len(m["task_C_progression"]["per_horizon"]) == HORIZON_LENGTH
    assert "skill_vs_ridge" in m["state_forecast"]
    assert m["per_episode"]["world_model_calibrated"]["n_episodes"] >= 1
    # provenance travels with the numbers
    assert test_rec["geometry"]["window_seconds"] == WINDOW_SECONDS
    assert test_rec["checkpoints"]["weights"]["seed_0"]["sha256"] is not None
    assert (tmp_path / "metrics" / "test" / "benchmark.json").exists()
    assert (tmp_path / "metrics" / "test" / "benchmark_scores.npz").exists()
    # the sampled-prevalence AP is not the natural one
    wm = systems["world_model"]
    assert wm["auc_pr"] != wm["auc_pr_unweighted"] or wm["n_pos"] == 0


# ---------------------------------------------------------------------------
# operating point under a horizon extension (K=10 check on a K=6 operating point)
# ---------------------------------------------------------------------------

def test_calibration_extends_to_a_longer_horizon_by_repeating_the_last_params():
    from nidra.eval.operating_point import extend_calibration_to_horizon
    params = [{"a": 1.0 + k / 10, "b": -0.1 * k, "n": 10, "degenerate": False} for k in range(6)]
    ext = extend_calibration_to_horizon(params, 10)
    assert len(ext) == 10
    assert ext[:6] == params
    for k in range(6, 10):
        assert ext[k]["a"] == params[-1]["a"] and ext[k]["b"] == params[-1]["b"]
        assert ext[k]["extended_from_k"] == 6
    assert extend_calibration_to_horizon(params, 6) == params
    assert extend_calibration_to_horizon(params, 3) == params[:3]


def test_apply_operating_point_accepts_a_longer_horizon_and_says_so():
    from nidra.eval.operating_point import apply_operating_point
    from nidra.eval.systems import ScoreBundle
    rng = np.random.default_rng(0)
    K = 10
    bundle = ScoreBundle(K=K, risk_traj=rng.uniform(size=(4, 12, K)).astype("float16"), n_members=1, n_samples_per_member=12)
    op = {"pooling": {"method": "mean", "quantile": None, "horizon_reduction": "max"},
          "calibration": {"params_by_k": [{"a": 1.0, "b": 0.0, "n": 10, "degenerate": False}] * 6}}
    applied = apply_operating_point(bundle, op)
    assert applied["risk_k_calibrated"].shape == (4, K)
    assert applied["calibration_horizons_extended"] is True
    same_k = apply_operating_point(ScoreBundle(K=6, risk_traj=rng.uniform(size=(4, 12, 6)).astype("float16"),
                                               n_members=1, n_samples_per_member=12), op)
    assert same_k["calibration_horizons_extended"] is False

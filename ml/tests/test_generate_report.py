from nidra.scripts.generate_report import plot_baseline_comparison


def test_plot_baseline_comparison_ignores_non_metrics_entries(tmp_path):
    """baselines.json can carry provenance entries (e.g.
    calibration_fit_metadata) alongside real {f1, auc_pr} rows once
    calibration is present — this must not crash the plot."""
    baselines = {
        "world_model": {"f1": 0.5, "auc_pr": 0.8},
        "persistence": {"f1": 0.3, "auc_pr": 0.6},
        "calibration_fit_metadata": {"a": 1.5, "b": 0.2, "n": 4000, "degenerate": False},
    }
    plot_baseline_comparison(baselines, tmp_path, "baseline_comparison.png", "test")
    assert (tmp_path / "baseline_comparison.png").exists()


def test_plot_baseline_comparison_handles_no_provenance_entries(tmp_path):
    baselines = {
        "world_model": {"f1": 0.5, "auc_pr": 0.8},
        "persistence": {"f1": 0.3, "auc_pr": 0.6},
    }
    plot_baseline_comparison(baselines, tmp_path, "baseline_comparison.png", "test")
    assert (tmp_path / "baseline_comparison.png").exists()

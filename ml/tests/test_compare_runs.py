"""Cross-run comparison table: the harness that turns N benchmark.json
records into one ranked Markdown table.

The ablations in the CTU-13 phase (head variants, feature regimes,
cross-dataset transfer) all have the same shape — several runs, one split,
one system — and the comparison has to be assembled from the recorded
artifacts rather than retyped. It also has to make it hard to rank variants
on a split that must not be used for selection, which is what the
non-selection-split warning is for.
"""

from __future__ import annotations

import json

import pytest

from nidra.scripts.compare_runs import (
    SELECTION_SPLIT,
    RunRow,
    comparison_markdown,
    load_run_metrics,
    rank_rows,
    run_metrics_path,
    variant_row,
)


def _metrics(ap: float, ci: tuple[float, float], roc: float = 0.8, n_clusters: int = 40,
             margin: float = 0.05, margin_ci: tuple[float, float] = (0.01, 0.09)) -> dict:
    sysentry = {
        "auc_pr": ap,
        "roc_auc": roc,
        "auc_pr_bootstrap": {"ci_low": ci[0], "ci_high": ci[1], "n_positive_clusters": n_clusters},
        "at_threshold": {"precision": 0.7, "recall": 0.4, "f1": 0.51},
        "false_alarms_per_hour": 1.5,
        "n_pos": 120,
    }
    systems = {name: dict(sysentry) for name in
               ("world_model_calibrated", "world_model", "persistence", "persistence_rollout", "oracle_true_future")}
    systems["persistence"] = {**sysentry, "auc_pr": ap - margin}
    systems["persistence_rollout"] = {**sysentry, "auc_pr": ap - margin}
    systems["oracle_true_future"] = {**sysentry, "auc_pr": ap + 0.2}
    return {
        "eval_set": {"n_rows": 20171, "natural_prevalence_published": 0.0003, "n_positive": 120},
        "operating_point": {"pooling_key": "mean@0.85/max", "threshold_used": 0.71},
        "task_published_label": {"systems": systems},
        "task_A_detection": {"systems": systems},
        "task_B_onset_forecast": {"5": {"systems": {"world_model_calibrated": {"auc_pr": 0.02}}}},
        "attribution_published_label": {
            "world_model - persistence": {"point": margin, "ci_low": margin_ci[0], "ci_high": margin_ci[1]},
        },
    }


def _write_run(root, label: str, split: str, metrics: dict):
    d = root / label / "artifacts" / "metrics" / split
    d.mkdir(parents=True, exist_ok=True)
    (d / "benchmark.json").write_text(json.dumps({"stage": "benchmark", "split": split, "metrics": metrics}))
    return root / label


class TestPaths:
    def test_metrics_path_follows_the_run_layout(self, tmp_path):
        assert run_metrics_path(tmp_path / "ctu_heads__state", "val") == (
            tmp_path / "ctu_heads__state" / "artifacts" / "metrics" / "val" / "benchmark.json")

    def test_a_missing_record_is_named_not_silently_skipped(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ctu_heads__state"):
            load_run_metrics(tmp_path / "ctu_heads__state", "val")

    def test_loading_unwraps_the_metrics_block(self, tmp_path):
        run = _write_run(tmp_path, "a", "val", _metrics(0.5, (0.4, 0.6)))
        assert load_run_metrics(run, "val")["eval_set"]["n_rows"] == 20171


class TestRow:
    def test_a_row_carries_the_point_estimate_and_its_interval(self):
        row = variant_row("state+hidden", _metrics(0.52, (0.41, 0.63)))
        assert row.label == "state+hidden"
        assert row.ap == pytest.approx(0.52)
        assert (row.ci_low, row.ci_high) == pytest.approx((0.41, 0.63))

    def test_the_margin_over_the_reference_is_read_from_the_paired_bootstrap(self):
        row = variant_row("state", _metrics(0.5, (0.4, 0.6), margin=0.07, margin_ci=(-0.01, 0.15)))
        assert row.margin == pytest.approx(0.07)
        assert (row.margin_ci_low, row.margin_ci_high) == pytest.approx((-0.01, 0.15))

    def test_a_margin_interval_that_includes_zero_is_flagged_not_hidden(self):
        row = variant_row("state", _metrics(0.5, (0.4, 0.6), margin=0.07, margin_ci=(-0.01, 0.15)))
        assert row.margin_excludes_zero is False
        assert variant_row("s", _metrics(0.5, (0.4, 0.6), margin_ci=(0.01, 0.09))).margin_excludes_zero is True

    def test_an_interval_from_too_few_episodes_is_marked_uninformative(self):
        assert variant_row("s", _metrics(0.5, (0.1, 0.9), n_clusters=3)).few_episodes is True
        assert variant_row("s", _metrics(0.5, (0.4, 0.6), n_clusters=40)).few_episodes is False

    def test_the_oracle_gap_is_carried_so_head_quality_is_visible(self):
        row = variant_row("s", _metrics(0.5, (0.4, 0.6)))
        assert row.oracle_ap == pytest.approx(0.7)
        assert row.oracle_gap == pytest.approx(0.2)

    def test_a_system_absent_from_the_record_is_an_error_naming_the_system(self):
        m = _metrics(0.5, (0.4, 0.6))
        del m["task_published_label"]["systems"]["world_model_calibrated"]
        with pytest.raises(KeyError, match="world_model_calibrated"):
            variant_row("s", m)


class TestRanking:
    def test_rows_rank_by_point_estimate_descending(self):
        rows = [variant_row("lo", _metrics(0.30, (0.2, 0.4))), variant_row("hi", _metrics(0.55, (0.4, 0.7)))]
        assert [r.label for r in rank_rows(rows)] == ["hi", "lo"]

    def test_ranking_does_not_mutate_the_input(self):
        rows = [variant_row("lo", _metrics(0.30, (0.2, 0.4))), variant_row("hi", _metrics(0.55, (0.4, 0.7)))]
        rank_rows(rows)
        assert [r.label for r in rows] == ["lo", "hi"]


class TestMarkdown:
    def test_the_table_names_the_split_it_was_built_from(self):
        md = comparison_markdown([variant_row("a", _metrics(0.5, (0.4, 0.6)))], "val",
                                 "world_model_calibrated", "persistence")
        assert "val" in md and "world_model_calibrated" in md

    def test_every_variant_appears_as_a_row(self):
        rows = [variant_row("state", _metrics(0.40, (0.3, 0.5))), variant_row("state+hidden", _metrics(0.55, (0.4, 0.7)))]
        md = comparison_markdown(rows, "val", "world_model_calibrated", "persistence")
        assert "| state |" in md and "| state+hidden |" in md

    def test_an_uninformative_interval_says_so_in_the_table(self):
        md = comparison_markdown([variant_row("a", _metrics(0.5, (0.1, 0.9), n_clusters=2))], "val",
                                 "world_model_calibrated", "persistence")
        assert "2 positive episodes" in md

    def test_a_non_selection_split_carries_a_visible_warning(self):
        md = comparison_markdown([variant_row("a", _metrics(0.5, (0.4, 0.6)))], "test",
                                 "world_model_calibrated", "persistence")
        assert "not a selection split" in md.lower()
        assert SELECTION_SPLIT == "val"

    def test_the_selection_split_carries_no_such_warning(self):
        md = comparison_markdown([variant_row("a", _metrics(0.5, (0.4, 0.6)))], "val",
                                 "world_model_calibrated", "persistence")
        assert "not a selection split" not in md.lower()


class TestRunRowIsImmutable:
    def test_rows_are_frozen(self):
        row = variant_row("a", _metrics(0.5, (0.4, 0.6)))
        assert isinstance(row, RunRow)
        with pytest.raises(Exception):
            row.ap = 0.9

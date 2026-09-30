"""The matrix exists to satisfy two §33 rules at once — a provenance record per
experiment, and no hidden failures — so its job is to describe runs accurately
rather than flatteringly."""

from __future__ import annotations

import json

from nidra.scripts.experiment_matrix import collect, corpora_of, matrix_markdown, read_record, split_counts


def _record(days, **kw):
    base = {"label": "r", "stage": "train", "git_commit": "abc12345deadbeef",
            "geometry": {"window_seconds": 60, "context_length_L": 30, "horizon_length_K": 6},
            "dataset": {"name": "prose", "feature_regime": "cross_core", "days": days},
            "stages": ["dynamics"], "ensemble_seeds": [0], "wall_seconds": 3600.0}
    base.update(kw)
    return base


def test_combined_run_is_not_reported_as_one_corpus():
    """The record's `name` concatenates both corpora, and splitting it on the
    first parenthesis reported the combined runs as CIC-only."""
    days = {"tuesday": {"format": "cicflowmeter", "role": "train"},
            "ctu_1": {"format": "ctu_binetflow", "role": "train"}}
    assert corpora_of({"name": "CIC-IDS2017 (csvs) + CTU-13 (binetflow)", "days": days}) == "CIC-IDS2017+CTU-13"
    assert corpora_of({"name": "x", "days": {"a": {"format": "ctu_binetflow"}}}) == "CTU-13"


def test_corpora_falls_back_to_the_name_when_no_day_is_recognised():
    assert corpora_of({"name": "something else", "days": {"a": {"format": "mystery"}}}) == "something else"


def test_split_counts_use_the_effective_roles():
    days = {"a": {"role": "train"}, "b": {"role": "train"}, "c": {"role": "val"},
            "d": {"role": "test"}, "e": {"role": "holdout"}}
    # "train" and "test" both start with t; 2 train and 3 test days rendered
    # as "2t/3t" told a reader nothing.
    assert split_counts({"days": days}) == "2tr 1va 1te 1ho"


def test_a_run_without_a_record_is_listed_not_dropped(tmp_path):
    """A directory with no record must be visible. Dropping it silently is
    exactly the hiding the rule forbids."""
    (tmp_path / "with_record").mkdir()
    (tmp_path / "with_record" / "record.json").write_text(json.dumps(_record({})))
    (tmp_path / "no_record").mkdir()
    rows, bare = collect(tmp_path)
    assert [r.label for r in rows] == ["r"]
    assert bare == ["no_record"]
    assert "no_record" in matrix_markdown(rows, bare)


def test_mixed_geometries_are_flagged(tmp_path):
    """Two rows at different geometries are not comparable, and a table that
    does not say so invites exactly that comparison."""
    for name, L in (("a", 30), ("b", 15)):
        d = tmp_path / name
        d.mkdir()
        rec = _record({}, label=name)
        rec["geometry"]["context_length_L"] = L
        (d / "record.json").write_text(json.dumps(rec))
    rows, bare = collect(tmp_path)
    md = matrix_markdown(rows, bare)
    assert "2 geometries" in md
    assert "not comparable" in md


def test_a_zero_second_run_is_shown_as_such(tmp_path):
    """The onset 2x2 silently 'completed' in 0 s because of a shell glob bug.
    A 0m row is the evidence that happened; it must not round away."""
    d = tmp_path / "z"
    d.mkdir()
    (d / "record.json").write_text(json.dumps(_record({}, wall_seconds=0.4)))
    rows, _ = collect(tmp_path)
    assert rows[0].wall == "0m"


def test_unreadable_record_is_treated_as_missing(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "record.json").write_text("{ truncated")
    assert read_record(d) is None


def test_wall_switches_to_hours_when_long(tmp_path):
    d = tmp_path / "long"
    d.mkdir()
    (d / "record.json").write_text(json.dumps(_record({}, wall_seconds=7200.0)))
    rows, _ = collect(tmp_path)
    assert rows[0].wall == "2.0h"

"""Config inheritance.

The CTU-13 phase needs a config per training regime (CIC-only, CTU-only,
each transfer direction, combined, each feature regime). They differ in the
dataset and splits blocks and must be IDENTICAL everywhere else — a copied
400-line config that drifts in one hyperparameter turns a controlled
comparison into two unrelated runs.
"""

from __future__ import annotations

import textwrap

import pytest

from nidra.utils.config import load_config


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_child_inherits_and_overrides_by_deep_merge(tmp_path):
    _write(tmp_path, "base.yaml", """
        model: {n_features: 45, encoder: {hidden_size: 128, num_layers: 2}}
        splits: {train_days: [a, b]}
    """)
    child = _write(tmp_path, "child.yaml", """
        extends: base.yaml
        model: {encoder: {hidden_size: 256}}
        splits: {train_days: [c]}
    """)
    cfg = load_config(child)
    assert cfg["model"]["encoder"]["hidden_size"] == 256
    assert cfg["model"]["encoder"]["num_layers"] == 2      # untouched key survives
    assert cfg["model"]["n_features"] == 45
    assert cfg["splits"]["train_days"] == ["c"]            # lists replace, never merge


def test_extends_chains(tmp_path):
    _write(tmp_path, "a.yaml", "x: 1\ny: 1\nz: 1\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\ny: 2\n")
    c = _write(tmp_path, "c.yaml", "extends: b.yaml\nz: 3\n")
    cfg = load_config(c)
    assert (cfg["x"], cfg["y"], cfg["z"]) == (1, 2, 3)


def test_the_hash_covers_the_whole_resolved_config_not_just_the_child(tmp_path):
    base = _write(tmp_path, "base.yaml", "x: 1\n")
    child = _write(tmp_path, "child.yaml", "extends: base.yaml\ny: 2\n")
    before = load_config(child)["_config_hash"]
    base.write_text("x: 99\n")
    after = load_config(child)["_config_hash"]
    assert before != after, "a change to the parent must change the child's provenance hash"


def test_the_resolved_lineage_is_recorded(tmp_path):
    _write(tmp_path, "base.yaml", "x: 1\n")
    child = _write(tmp_path, "child.yaml", "extends: base.yaml\n")
    cfg = load_config(child)
    assert [p.split("/")[-1] for p in cfg["_config_lineage"]] == ["base.yaml", "child.yaml"]


def test_a_cycle_is_refused_rather_than_hanging(tmp_path):
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_a_missing_parent_is_refused(tmp_path):
    child = _write(tmp_path, "child.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_config(child)


def test_a_config_without_extends_is_unchanged(tmp_path):
    path = _write(tmp_path, "plain.yaml", "x: 1\n")
    cfg = load_config(path)
    assert cfg["x"] == 1 and "extends" not in cfg


def test_a_mapping_can_declare_that_it_replaces_rather_than_merges(tmp_path):
    # `dataset.days` is a mapping, and a CTU-only config that merged with the
    # CIC day list would quietly train on both datasets.
    _write(tmp_path, "base.yaml", """
        dataset:
          root: /base
          days: {monday: {file: m.csv}, tuesday: {file: t.csv}}
    """)
    child = _write(tmp_path, "child.yaml", """
        extends: base.yaml
        dataset:
          days:
            _replace: true
            ctu_1: {file: 1.binetflow}
    """)
    cfg = load_config(child)
    assert set(cfg["dataset"]["days"]) == {"ctu_1"}
    assert cfg["dataset"]["root"] == "/base"      # sibling keys still inherit
    assert "_replace" not in cfg["dataset"]["days"]

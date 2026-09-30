"""Training sample caps must come from the config, not only from CLI flags.

The full train split is ~6.9M [L,F] float32 windows at production scale
(~35GB to materialize), so the cap is applied during windowing. It used to be
reachable ONLY via --max-train-samples, which meant the documented full-scale
command (`--config config/default.yaml`, no flags) was OOM-killed before the
first epoch while the published checkpoints record 500000/50000 — the
documented command could not produce the documented numbers.
"""

from pathlib import Path

import yaml

from nidra.train.train_dynamics import resolve_sample_caps

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_config_caps_are_used_when_no_cli_flags_are_given():
    cfg = {"training_data": {"max_train_samples": 500000, "max_val_samples": 50000}}
    assert resolve_sample_caps(cfg, None, None) == (500000, 50000)


def test_cli_flags_override_config_caps():
    cfg = {"training_data": {"max_train_samples": 500000, "max_val_samples": 50000}}
    assert resolve_sample_caps(cfg, 40000, 8000) == (40000, 8000)


def test_each_cli_flag_overrides_independently():
    cfg = {"training_data": {"max_train_samples": 500000, "max_val_samples": 50000}}
    assert resolve_sample_caps(cfg, 1000, None) == (1000, 50000)
    assert resolve_sample_caps(cfg, None, 2000) == (500000, 2000)


def test_a_config_without_the_section_stays_uncapped():
    # Backward compatible: an older config simply behaves as before.
    assert resolve_sample_caps({}, None, None) == (None, None)


def test_an_explicit_null_config_cap_means_uncapped():
    cfg = {"training_data": {"max_train_samples": None, "max_val_samples": None}}
    assert resolve_sample_caps(cfg, None, None) == (None, None)


def test_shipped_configs_carry_the_caps_their_published_checkpoints_were_trained_under():
    expected = {
        "default.yaml": (500000, 50000),
        "default_logvar15.yaml": (500000, 50000),
        "mvp_2017.yaml": (40000, 8000),
    }
    for name, (train, val) in expected.items():
        cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
        assert resolve_sample_caps(cfg, None, None) == (train, val), name

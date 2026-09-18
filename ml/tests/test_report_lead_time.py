"""Lead time is the project's headline result, and its plot was never being
produced.

`lead_time.json` nests its numbers under `raw` / `calibrated` / `ensemble` /
`ensemble_calibrated`, but plot_lead_time_distribution was handed the whole
file and read `lead_time_distribution_s` off the TOP level, where it does not
exist. The lookup returned None every time, the function logged "no
lead-time distribution to plot" and returned, and the report silently came
out without the figure. Every other plot in this script is passed its inner
dict; this one was not.

Which section gets plotted matters: `ensemble` is the pooled statistic
NidraPredictor actually serves and the one the published lead-time figures
come from, so it is preferred over the single-seed `raw` section.
"""

from __future__ import annotations

import pytest

from nidra.scripts.generate_report import select_lead_time_section


def _section(median, dist, n_episodes=10, frac=0.1):
    return {
        "median_lead_time_s": median,
        "lead_time_distribution_s": list(dist),
        "n_episodes": n_episodes,
        "fraction_no_warning": frac,
    }


def test_prefers_the_pooled_ensemble_section():
    """The served statistic, not the single-seed approximation."""
    payload = {
        "split": "test",
        "raw": _section(100.0, [100.0, 200.0]),
        "ensemble": _section(31830.0, [29970.0, 32970.0, 31830.0]),
    }
    label, section = select_lead_time_section(payload)
    assert label == "ensemble"
    assert section["median_lead_time_s"] == 31830.0


def test_falls_back_to_raw_when_no_ensemble_was_run():
    """A single-seed run (no --use-ensemble) still gets its plot."""
    payload = {"split": "test", "raw": _section(100.0, [100.0, 200.0])}
    label, section = select_lead_time_section(payload)
    assert label == "raw"
    assert section["median_lead_time_s"] == 100.0


def test_never_silently_prefers_a_calibrated_section():
    """Calibration is measured to make quantile-pooled scores worse and is
    not what the headline reports — it must not be picked over the raw or
    ensemble sections just because it is present."""
    payload = {
        "split": "test",
        "raw": _section(100.0, [100.0]),
        "calibrated": _section(9.0, [9.0]),
        "ensemble_calibrated": _section(8.0, [8.0]),
    }
    label, _ = select_lead_time_section(payload)
    assert label == "raw"


def test_section_with_an_empty_distribution_is_not_selected():
    """A section where no episode was ever warned has nothing to plot;
    prefer one that does rather than reporting an empty histogram."""
    payload = {
        "split": "test",
        "ensemble": _section(None, []),
        "raw": _section(100.0, [100.0, 150.0]),
    }
    label, section = select_lead_time_section(payload)
    assert label == "raw"
    assert section["lead_time_distribution_s"] == [100.0, 150.0]


def test_returns_none_when_there_is_nothing_plottable():
    assert select_lead_time_section({"split": "test", "seed": 0}) == (None, None)
    assert select_lead_time_section({"split": "test", "raw": _section(None, [])}) == (None, None)


def test_ignores_non_dict_metadata_keys():
    """The file also carries scalars like split/seed/n_samples — indexing
    into those would raise, not degrade."""
    payload = {"split": "test", "seed": 0, "n_samples": 200,
               "risk_pooling": {"method": "quantile"},
               "raw": _section(100.0, [100.0])}
    label, _ = select_lead_time_section(payload)
    assert label == "raw"

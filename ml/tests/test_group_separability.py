"""The supervised separability probe: can anything see this attack group?"""

from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.group_separability import GroupProbe, host_grouped_folds, probe_group, probes_markdown


def _probe(**kw) -> GroupProbe:
    base = dict(group="ctu_4:c2", n_positive=56, n_rows=60000, base_rate=0.001,
                probe_ap=0.02, probe_roc=0.8, n_folds=5)
    return GroupProbe(**{**base, **kw})


class TestVerdict:
    def test_lift_is_the_probe_against_the_base_rate(self):
        assert _probe(probe_ap=0.02, base_rate=0.001).lift == pytest.approx(20.0)

    def test_a_probe_near_the_base_rate_is_not_separable(self):
        assert _probe(probe_ap=0.0015, base_rate=0.001).separable is False

    def test_a_probe_well_above_it_is(self):
        assert _probe(probe_ap=0.02, base_rate=0.001).separable is True

    def test_a_zero_base_rate_does_not_divide_by_zero(self):
        assert np.isnan(_probe(base_rate=0.0).lift)


class TestFolds:
    def test_no_host_straddles_a_fold(self):
        hosts = np.array([f"h{i % 12}" for i in range(240)])
        y = (np.arange(240) % 17 == 0).astype(int)
        folds = host_grouped_folds(hosts, y, n_folds=4)
        seen: dict = {}
        for f, idx in enumerate(folds):
            for h in np.unique(hosts[idx]):
                assert seen.setdefault(h, f) == f

    def test_every_row_lands_in_exactly_one_fold(self):
        hosts = np.array([f"h{i % 9}" for i in range(180)])
        folds = host_grouped_folds(hosts, np.zeros(180, dtype=int), n_folds=3)
        assert sorted(np.concatenate(folds)) == list(range(180))

    def test_positive_hosts_are_spread_before_negative_ones(self):
        """Otherwise a fold can hold no positive and its AP is undefined."""
        hosts = np.array([f"h{i}" for i in range(20)] * 5)
        y = np.isin(hosts, ["h0", "h1", "h2", "h3"]).astype(int)
        folds = host_grouped_folds(hosts, y, n_folds=4, seed=0)
        assert all(y[idx].sum() > 0 for idx in folds)


class TestProbe:
    def test_a_linearly_separable_group_is_found(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 6)).astype("float32")
        hosts = np.array([f"h{i % 20}" for i in range(600)])
        y = (X[:, 0] > 1.6).astype(int)
        p = probe_group(X, y, hosts, "g")
        assert p.separable and p.probe_roc > 0.9

    def test_a_label_unrelated_to_the_features_is_not(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(600, 6)).astype("float32")
        hosts = np.array([f"h{i % 20}" for i in range(600)])
        y = (rng.random(600) < 0.05).astype(int)
        assert probe_group(X, y, hosts, "g").separable is False

    def test_it_reports_the_counts_it_was_given(self):
        X = np.zeros((50, 3), dtype="float32")
        y = np.array([1] * 5 + [0] * 45)
        p = probe_group(X, y, np.array([f"h{i % 10}" for i in range(50)]), "g")
        assert p.n_positive == 5 and p.n_rows == 50 and p.base_rate == pytest.approx(0.1)


class TestMarkdown:
    def test_groups_are_listed_by_size(self):
        md = probes_markdown([_probe(group="small", n_positive=6), _probe(group="big", n_positive=100)], "val")
        assert md.index("| big |") < md.index("| small |")

    def test_an_unseparable_group_is_marked(self):
        assert "**no**" in probes_markdown([_probe(probe_ap=0.0011, base_rate=0.001)], "val")

    def test_the_forecast_column_is_filled_when_supplied(self):
        md = probes_markdown([_probe(group="g")], "val", {"g": 0.123})
        assert "0.123" in md

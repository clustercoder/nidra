"""The supervised separability probe: can anything see this attack group?"""

from __future__ import annotations

import numpy as np
import pytest

from nidra.scripts.group_separability import (GroupProbe, host_grouped_folds, positive_hosts, probe_group,
                                              probes_markdown)


def _probe(**kw) -> GroupProbe:
    base = dict(group="ctu_4:c2", n_positive=56, n_positive_hosts=4, n_rows=60000, base_rate=0.001,
                probe_ap=0.02, probe_roc=0.8, n_folds=5)
    return GroupProbe(**{**base, **kw})


class TestVerdict:
    def test_lift_is_the_probe_against_the_base_rate(self):
        assert _probe(probe_ap=0.02, base_rate=0.001).lift == pytest.approx(20.0)

    def test_a_probe_near_the_base_rate_is_not_separable(self):
        assert _probe(probe_ap=0.0015, base_rate=0.001).separable is False

    def test_high_lift_with_a_below_chance_ranking_is_not_separable(self):
        """The measured ctu_4:c2 case: 34x lift at a 0.02% base rate with
        ROC 0.443 — a few positives at the very top of a ranking that is worse
        than chance overall."""
        assert _probe(probe_ap=0.008, base_rate=0.00023, probe_roc=0.443).separable is False

    def test_both_conditions_together_are_separable(self):
        assert _probe(probe_ap=0.008, base_rate=0.00023, probe_roc=0.83).separable is True

    def test_a_probe_well_above_it_is(self):
        assert _probe(probe_ap=0.02, base_rate=0.001).separable is True

    def test_a_zero_base_rate_does_not_divide_by_zero(self):
        assert np.isnan(_probe(base_rate=0.0).lift)
        assert _probe(base_rate=0.0).separable is False

    def test_a_probe_that_could_not_be_scored_claims_nothing(self):
        p = _probe(probe_ap=float("nan"), probe_roc=float("nan"), n_folds=0)
        assert p.separable is False and p.n_folds == 0


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


class TestSingleHostGroups:
    """Every attack group on the CTU validation captures has its positives on
    ONE host. A host-grouped split cannot be formed: the only fold with a
    positive test host has no positive to train on, and every other fold has
    no positive to score. Scoring those rows at a default 0.0 put every
    positive at the bottom and produced a confident ROC of 0.10 for every
    group — a bug that looked exactly like a finding."""

    def _one_host(self, informative: bool):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(400, 5)).astype("float32")
        hosts = np.array(["victim"] * 40 + [f"h{i % 12}" for i in range(360)])
        y = np.zeros(400, dtype=int)
        y[:40] = 1
        if informative:
            X[:40, 0] += 4.0
        return X, y, hosts

    def test_it_is_flagged_host_leaky_rather_than_scored_as_if_grouped(self):
        X, y, hosts = self._one_host(True)
        p = probe_group(X, y, hosts, "g")
        assert p.n_positive_hosts == 1 and p.host_leaky is True

    def test_an_informative_single_host_group_is_found_not_inverted(self):
        X, y, hosts = self._one_host(True)
        p = probe_group(X, y, hosts, "g")
        assert p.probe_roc > 0.9, "the old default-zero scoring inverted this to ~0.1"

    def test_an_uninformative_one_is_near_chance_not_near_zero(self):
        X, y, hosts = self._one_host(False)
        p = probe_group(X, y, hosts, "g")
        assert 0.3 < p.probe_roc < 0.7

    def test_positive_hosts_are_reported(self):
        hosts = np.array(["a", "a", "b", "c"])
        assert list(positive_hosts(hosts, np.array([1, 0, 1, 0]))) == ["a", "b"]


class TestMarkdown:
    def test_groups_are_listed_by_size(self):
        md = probes_markdown([_probe(group="small", n_positive=6), _probe(group="big", n_positive=100)], "val")
        assert md.index("| big |") < md.index("| small |")

    def test_an_unseparable_group_is_marked(self):
        assert "**no**" in probes_markdown([_probe(probe_ap=0.0011, base_rate=0.001)], "val")

    def test_a_host_leaky_row_says_so(self):
        md = probes_markdown([_probe(n_positive_hosts=1, host_leaky=True)], "val")
        assert "host-leaky" in md

    def test_an_unscorable_group_prints_a_dash_not_a_number(self):
        md = probes_markdown([_probe(probe_ap=float("nan"), probe_roc=float("nan"), n_folds=0)], "val")
        assert "| — |" in md


class TestTransferProbe:
    """Fit on one split's rows for the group's stage, score another split's
    group. The within-split probe on CTU validation has to fall back to
    row-stratified folds because every group's positives sit on one host, and
    it can then answer with host identity. This one cannot: the fit never sees
    the scored host."""

    def _data(self, transferable: bool):
        rng = np.random.default_rng(5)
        X_fit = rng.normal(size=(500, 5)).astype("float32")
        y_fit = np.zeros(500, dtype=int); y_fit[:60] = 1
        X_fit[:60, 0] += 3.0
        X_score = rng.normal(size=(300, 5)).astype("float32")
        y_score = np.zeros(300, dtype=int); y_score[:30] = 1
        if transferable:
            X_score[:30, 0] += 3.0           # same behaviour, different host
        else:
            X_score[:30, 1] += 3.0           # a different signature entirely
        return X_fit, y_fit, X_score, y_score, np.array(["victim"] * 30 + [f"h{i}" for i in range(270)])

    def test_a_transferable_signature_is_found(self):
        from nidra.scripts.group_separability import transfer_probe
        X_f, y_f, X_s, y_s, hosts = self._data(True)
        assert transfer_probe(X_f, y_f, X_s, y_s, "g", hosts).probe_roc > 0.9

    def test_a_signature_that_does_not_transfer_is_not(self):
        from nidra.scripts.group_separability import transfer_probe
        X_f, y_f, X_s, y_s, hosts = self._data(False)
        p = transfer_probe(X_f, y_f, X_s, y_s, "g", hosts)
        assert p.probe_roc < 0.7 and p.separable is False

    def test_it_is_never_marked_host_leaky(self):
        from nidra.scripts.group_separability import transfer_probe
        X_f, y_f, X_s, y_s, hosts = self._data(True)
        assert transfer_probe(X_f, y_f, X_s, y_s, "g", hosts).host_leaky is False

    def test_a_stage_absent_from_the_fit_split_claims_nothing(self):
        from nidra.scripts.group_separability import transfer_probe
        X_f, y_f, X_s, y_s, hosts = self._data(True)
        p = transfer_probe(X_f, np.zeros_like(y_f), X_s, y_s, "g", hosts)
        assert np.isnan(p.probe_roc) and p.n_folds == 0


def test_markdown_states_the_verdict_is_one_sided():
    """A failed probe is not evidence of unlearnability. `ctu_4:c2` was written
    up as unreachable on a probe ROC of 0.443 and NIDRA's own stage head then
    reached 0.864 on it, so the table has to say which direction it reads in."""
    probe = GroupProbe(group="g:c2", n_positive=23, n_positive_hosts=1, base_rate=0.0002,
                       probe_ap=0.008, probe_roc=0.443, n_rows=65690, n_folds=5, host_leaky=True)
    md = probes_markdown([probe], split="val")
    assert "one-sided" in md
    assert "unproven" in md and "unlearnable" in md
    assert "**no**" in md


def test_separable_is_false_but_documented_as_unproven():
    probe = GroupProbe(group="g:c2", n_positive=23, n_positive_hosts=1, base_rate=0.0002,
                       probe_ap=0.008, probe_roc=0.443, n_rows=65690, n_folds=5, host_leaky=True)
    assert probe.separable is False
    assert "False is not evidence" in GroupProbe.separable.__doc__

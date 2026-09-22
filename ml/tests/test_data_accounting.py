"""How much of the dataset the model actually saw.

"Trained on the whole dataset" is a claim with several possible meanings, and
the roadmap asks for all of them separately: raw rows, canonical per-host
states, states eligible as a sequence origin, states sampled per epoch, and
total state exposures over training. The arithmetic connecting them is what
this pins; the counts themselves come from the cached tables.
"""

from __future__ import annotations

import pytest

from nidra.scripts.data_accounting import SplitAccounting, accounting_markdown, coverage_table


def _acc(**kw) -> SplitAccounting:
    base = dict(split="train", captures=3, raw_flows=1_400_000, canonical_states=1_420_000,
                active_states=900_000, eligible_origins=1_300_000, positive_origins=4_000,
                sampled_per_epoch=500_000, epochs=20, context_length=30)
    return SplitAccounting(**{**base, **kw})


class TestArithmetic:
    def test_exposures_count_every_window_of_every_sampled_sequence(self):
        # a sequence is L windows, and the model reads all of them each epoch
        assert _acc().total_state_exposures == 500_000 * 20 * 30

    def test_coverage_is_the_sampled_fraction_of_eligible_origins(self):
        assert _acc().origin_coverage == pytest.approx(500_000 / 1_300_000)

    def test_coverage_is_capped_at_one_when_nothing_was_subsampled(self):
        assert _acc(sampled_per_epoch=2_000_000).origin_coverage == 1.0

    def test_coverage_of_an_empty_split_is_zero_not_a_division_error(self):
        assert _acc(eligible_origins=0).origin_coverage == 0.0

    def test_epochs_of_resampling_can_cover_more_than_one_pass(self):
        # 20 epochs x 500k draws from a 1.3M pool: most origins are seen at
        # least once even though each epoch sees 38% of them.
        assert _acc().expected_origin_touches == pytest.approx(20 * 500_000 / 1_300_000)

    def test_a_split_never_subsampled_reports_full_coverage_and_n_touches(self):
        a = _acc(eligible_origins=400_000, sampled_per_epoch=400_000)
        assert a.origin_coverage == 1.0
        assert a.expected_origin_touches == pytest.approx(20.0)

    def test_prevalence_is_over_eligible_origins(self):
        assert _acc().origin_prevalence == pytest.approx(4_000 / 1_300_000)

    def test_a_frozen_record(self):
        with pytest.raises(Exception):
            _acc().split = "val"


class TestMarkdown:
    def test_every_split_is_a_row(self):
        md = accounting_markdown([_acc(split="train"), _acc(split="val", sampled_per_epoch=50_000)])
        assert "| train |" in md and "| val |" in md

    def test_the_columns_the_roadmap_asks_for_are_present(self):
        md = accounting_markdown([_acc()])
        for header in ("raw flows", "canonical states", "eligible origins", "sampled / epoch",
                       "state exposures"):
            assert header in md

    def test_large_counts_are_grouped_so_they_can_be_read(self):
        assert "1,420,000" in accounting_markdown([_acc()])


class TestCoverageTable:
    def test_it_totals_across_splits(self):
        rows = [_acc(split="train", canonical_states=1_000), _acc(split="val", canonical_states=500)]
        totals = coverage_table(rows)
        assert totals["canonical_states"] == 1_500
        assert totals["splits"] == 2

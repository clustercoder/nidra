"""Cross-dataset feature regimes.

CTU-13 cannot produce 13 of the 45 canonical features (11 packet-level ones,
`iat_max`, and the retransmission delta), and computes the six TCP flag
ratios from a different quantity than CIC-IDS2017 does. A regime is the
named, auditable answer to "which features may a cross-dataset experiment
use" — FEATURE_ORDER stays 45 wide and the excluded columns are declared
`drop` in the scaler, which is the mechanism the pipeline already has for a
feature the model must not read.
"""

from __future__ import annotations

import numpy as np
import pytest

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import (
    FEATURE_ORDER,
    FEATURE_REGIMES,
    regime_dropped_features,
    regime_kinds,
)


class TestRegimeDefinitions:
    def test_full_regime_drops_nothing(self):
        assert regime_dropped_features("full") == []

    def test_cross_core_drops_exactly_what_ctu_cannot_produce(self):
        dropped = set(regime_dropped_features("cross_core"))
        assert dropped == set(FEATURE_ORDER[15:26]) | {"iat_max", "d_retrans_rate"}
        assert len(dropped) == 13

    def test_cross_strict_also_drops_the_tcp_flag_features(self):
        strict = set(regime_dropped_features("cross_strict"))
        core = set(regime_dropped_features("cross_core"))
        assert core < strict
        assert strict - core == {
            "syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "psh_ratio", "urg_ratio",
            "d_syn_ratio", "slope3_syn_ratio",
        }

    def test_regimes_keep_the_activity_indicator(self):
        for name in FEATURE_REGIMES:
            assert "is_active" not in regime_dropped_features(name)

    def test_regime_sizes(self):
        assert len(FEATURE_ORDER) - len(regime_dropped_features("cross_core")) == 32
        assert len(FEATURE_ORDER) - len(regime_dropped_features("cross_strict")) == 24

    def test_unknown_regime_is_refused(self):
        with pytest.raises(ValueError, match="unknown feature regime"):
            regime_kinds("whatever")

    def test_every_dropped_name_is_a_real_feature(self):
        for name in FEATURE_REGIMES:
            assert set(regime_dropped_features(name)) <= set(FEATURE_ORDER)


class TestScalerHonoursTheRegime:
    def _fit(self, regime: str) -> FeatureScaler:
        rng = np.random.default_rng(0)
        X = np.abs(rng.normal(size=(400, len(FEATURE_ORDER)))) + 0.5
        return FeatureScaler(kinds=regime_kinds(regime)).fit(X, active_mask=np.ones(400, dtype=bool))

    def test_masked_features_are_zero_after_transform(self):
        scaler = self._fit("cross_core")
        rng = np.random.default_rng(1)
        out = scaler.transform(np.abs(rng.normal(size=(10, len(FEATURE_ORDER)))) + 1.0)
        for name in regime_dropped_features("cross_core"):
            assert np.allclose(out[:, FEATURE_ORDER.index(name)], 0.0), name

    def test_model_mask_excludes_them(self):
        scaler = self._fit("cross_core")
        assert scaler.model_mask.sum() == 32
        assert set(scaler.dropped_features) >= set(regime_dropped_features("cross_core"))

    def test_the_regime_is_recorded_as_a_declared_drop_reason(self):
        scaler = self._fit("cross_core")
        assert scaler.drop_reason_["ttl_mean"] == "declared"

    def test_full_regime_keeps_informative_features(self):
        scaler = self._fit("full")
        assert scaler.model_mask.sum() == len(FEATURE_ORDER)

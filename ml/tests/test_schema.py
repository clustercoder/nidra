from nidra.data.schema import FEATURE_ORDER, FEATURE_INDEX, LOG1P_FEATURES, validate_feature_dict, validate_state_array_width
import pytest


def test_exactly_45_features():
    assert len(FEATURE_ORDER) == 45


def test_feature_order_is_deterministic_list_not_set():
    assert isinstance(FEATURE_ORDER, list)
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))


def test_feature_index_matches_order():
    for i, name in enumerate(FEATURE_ORDER):
        assert FEATURE_INDEX[name] == i


def test_log1p_features_are_valid_feature_names():
    for name in LOG1P_FEATURES:
        assert name in FEATURE_ORDER


def test_validate_feature_dict_passes_on_exact_match():
    d = {name: 0.0 for name in FEATURE_ORDER}
    validate_feature_dict(d)  # should not raise


def test_validate_feature_dict_fails_on_missing():
    d = {name: 0.0 for name in FEATURE_ORDER[:-1]}
    with pytest.raises(ValueError):
        validate_feature_dict(d)


def test_validate_feature_dict_fails_on_extra():
    d = {name: 0.0 for name in FEATURE_ORDER}
    d["not_a_real_feature"] = 1.0
    with pytest.raises(ValueError):
        validate_feature_dict(d)


def test_validate_state_array_width():
    validate_state_array_width(45)
    with pytest.raises(ValueError):
        validate_state_array_width(44)

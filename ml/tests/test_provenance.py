

def test_effective_role_comes_from_splits_not_the_annotation():
    """`dataset.days[].role` is hand-written and had drifted: ctu_4 and ctu_6
    are annotated "test" and are the validation captures. A record that
    misstates which split a day landed in is worse than no record."""
    from nidra.utils.provenance import _effective_roles
    cfg = {"splits": {"train_days": ["ctu_1"], "val_days": ["ctu_4", "ctu_6"],
                      "test_days": ["ctu_8"], "holdout_days": ["ctu_12"]}}
    roles = _effective_roles(cfg)
    assert roles["ctu_4"] == "val" and roles["ctu_6"] == "val"
    assert roles["ctu_1"] == "train" and roles["ctu_8"] == "test" and roles["ctu_12"] == "holdout"


def test_a_carved_training_day_is_marked_as_both():
    from nidra.utils.provenance import _effective_roles
    cfg = {"splits": {"train_days": ["monday", "tuesday", "ctu_1"],
                      "val_carve_train_days": ["monday", "tuesday"]}}
    roles = _effective_roles(cfg)
    assert roles["monday"] == "train+val_carve"
    assert roles["ctu_1"] == "train"


def test_a_day_in_no_split_list_is_unused():
    from nidra.utils.provenance import _effective_roles
    assert _effective_roles({"splits": {"train_days": ["a"]}}).get("ctu_7") is None

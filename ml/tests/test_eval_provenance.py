"""Every metrics file run_eval writes must record the parameters that
produced it.

Without this, artifacts/metrics/*.json is a set of bare numbers: there is no
way for a reviewer to tell whether a committed 0.835 came from the published
200-rollouts-per-member run or from a cheap 20-rollout smoke run, and the two
differ by more than the rounding shown in the README. The headline table and
the committed artifacts silently disagreed for exactly this reason.

A wrong stamp is worse than no stamp, and the first version of this was
wrong: it read the pooling keys from the top level of the config, where they
do not live, so it silently recorded the `"mean"` default for runs that were
actually quantile-pooled. The keys live under `rollout:`. The defence is
structural — run_eval's own forecast calls and this stamp now read through
one helper — plus the guard test below, which asserts against the real
shipped config so a schema move fails loudly instead of degrading to a
plausible-looking default.
"""

from __future__ import annotations

import numpy as np

from nidra.eval.run_eval import build_run_params, resolve_pooling
from nidra.utils.config import load_config


def _cfg(**rollout):
    base = {"risk_pooling_method": "quantile", "risk_pooling_quantile": 0.85,
            "risk_pooling_head_reduction": "before_pooling"}
    base.update(rollout)
    return {"rollout": base, "eval": {"risk_threshold": 0.75, "forecast_chunk_size": 500}}


def test_reads_pooling_from_the_rollout_section():
    pooling = resolve_pooling(_cfg())
    assert pooling["method"] == "quantile"
    assert pooling["quantile"] == 0.85
    assert pooling["head_reduction"] == "before_pooling"


def test_top_level_pooling_keys_are_not_mistaken_for_config():
    """The exact bug: keys at the top level are not where pooling lives, and
    must not be picked up as though they were."""
    stray = {"rollout": {}, "eval": {}, "risk_pooling_method": "quantile",
             "risk_pooling_quantile": 0.85}
    assert resolve_pooling(stray)["method"] == "mean"


def test_the_shipped_config_is_reported_as_quantile_pooled():
    """Guard against schema drift: config/default.yaml is what produced every
    committed number, and it is quantile-pooled. If this ever reports 'mean',
    either the config moved or the reader broke — both must fail here rather
    than be written into a metrics file as fact."""
    pooling = resolve_pooling(load_config(None))
    assert pooling["method"] == "quantile", "shipped config must be reported as quantile-pooled"
    assert pooling["quantile"] == 0.85


def test_records_the_parameters_that_change_the_numbers():
    params = build_run_params(_cfg(), seed=0, split_name="test", n_samples=200,
                              max_eval_samples=4000, ensemble_seeds=[0, 1, 2, 3, 4],
                              n_eval_rows=4000)

    assert params["split"] == "test"
    assert params["seed"] == 0
    assert params["n_samples"] == 200
    assert params["max_eval_samples"] == 4000
    assert params["n_eval_rows"] == 4000
    assert params["ensemble_seeds"] == [0, 1, 2, 3, 4]
    # 200 rollouts per member x 5 members is what the served statistic pools
    assert params["n_trajectories"] == 1000
    assert params["risk_pooling"]["method"] == "quantile"
    assert params["risk_pooling"]["quantile"] == 0.85
    assert params["risk_pooling"]["head_reduction"] == "before_pooling"
    assert params["risk_threshold"] == 0.75
    assert params["forecast_chunk_size"] == 500


def test_single_seed_run_reports_its_own_trajectory_count():
    params = build_run_params(_cfg(), seed=3, split_name="holdout", n_samples=50,
                              max_eval_samples=None, ensemble_seeds=None, n_eval_rows=17)
    assert params["ensemble_seeds"] is None
    assert params["n_trajectories"] == 50
    assert params["max_eval_samples"] is None
    assert params["n_eval_rows"] == 17


def test_params_are_json_serializable():
    """numpy scalars leak in from array lengths and make json.dump raise —
    the whole point is that these land in a file."""
    import json

    params = build_run_params(_cfg(), seed=np.int64(0), split_name="test", n_samples=np.int64(20),
                              max_eval_samples=np.int64(500), ensemble_seeds=None,
                              n_eval_rows=np.int64(500))
    json.dumps(params)  # must not raise
    assert isinstance(params["seed"], int)
    assert isinstance(params["n_eval_rows"], int)


def test_defaults_are_filled_when_config_omits_everything():
    """An older config must still produce a complete, readable record rather
    than a KeyError mid-run."""
    params = build_run_params({}, seed=0, split_name="val", n_samples=10,
                              max_eval_samples=100, ensemble_seeds=None, n_eval_rows=100)
    assert params["risk_pooling"]["method"] == "mean"
    assert params["risk_threshold"] is None

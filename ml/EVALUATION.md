# Evaluation

All numbers cited here are from `REAL_DATA_RESULTS.md` §Run 2 — this
document explains the harness and metric definitions; that one is the
source of truth for actual measured results.

## Running it

```bash
cd ml
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000
```

Writes `baselines.json`, `ablations.json`, `calibration.json`,
`lead_time.json` to `<metrics_dir>/<split>/` — **the split-specific
subdirectory is load-bearing**: running `test` then `holdout` against the
same `metrics_dir` used to silently overwrite the first split's files with
the second's (a real bug hit and fixed while producing Run 2's results; see
`REAL_DATA_RESULTS.md` and `tests/test_run_eval.py::test_run_eval_does_not_clobber_metrics_across_splits`).

`--n-samples` controls rollout trajectories per evaluation sample (cost
scales with this — it's the expensive step). `--max-eval-samples` caps the
evaluated/baseline-training set size via
`nidra.data.dataset.subsample_stratified_by_risk` (stratified: every
positive `risk_label` sample is kept, negatives are randomly filled) — pass
`0` to disable capping and evaluate the full split (hundreds of thousands of
samples; slow).

## The four mandated baselines (`nidra/eval/baselines.py`)

1. **LR on `S_t`** — logistic regression on the current 45-dim scaled
   state. Mandated by the problem statement.
2. **LR on flattened `S[t-L..t]`** (1,350 features for L=30) — **the
   baseline that matters**. History without a learned transition operator.
   Beating this is the real test of whether the transition model earns its
   place.
3. **Persistence** (`Ŝ[t+k]=S_t` for all k) — scored with the *same frozen
   risk head* the world model uses, isolating the transition model's
   contribution from a head-quality difference.
4. **Oracle** — frozen heads on the *true* future state. Upper bound; no
   forecasting error, only head error.

## State forecast nRMSE (`nidra/eval/metrics.py::state_nrmse`)

Per-feature, per-horizon `RMSE / feature_scale`. **`feature_scale` should
always be `FeatureScaler.reference_std_`** (a per-feature std computed once
over the full training population, in the same scaled+log1p units
everything is compared in) — never the std of the eval batch itself, which
was the root cause of a real bug: many of the 45 features are structurally
near-constant on large slices of this dataset (packet aggregates on a
then-flow-only day, rare-event ratios like `urg_ratio`), so a small,
stratified eval batch's own std collapses toward zero for them, and
dividing by that near-zero number inflated nRMSE by orders of magnitude
(154,775 / 783,843 in the pre-fix run) for reasons having nothing to do with
forecast quality. `run_eval.py` passes `scaler.reference_std_` through to
every ablation that computes nRMSE; the floor is `0.05` (5% of one training
IQR) rather than the old `1e-8`. Post-fix, values are legible single digits
(test: 6.52 world model vs. 5.71 persistence).

**If you see an nRMSE value above ~50, something regressed** — check that
`feature_scale` is actually being passed (not silently falling back to
per-batch std) before assuming the model got worse.

## Lead time (`nidra/eval/metrics.py::lead_time_for_episode`/`lead_time_distribution`)

Defined exactly once: earliest **sustained** (`m=2` consecutive windows)
crossing of `threshold=0.75` strictly before the true attack-stage onset,
per attack episode. Always report the **median and full distribution**,
never the maximum (a single lucky episode is not a result) — and always
report `fraction_no_warning`, the recall side of the same coin. Computed
against the FULL (unsampled) split, not the capped `eval_arrays`, since it
needs each attacked host's complete chronological sequence — the cost is
naturally bounded by the small number of hosts ever attacked.

`threshold=0.75` and `m=2` are protected claims from the project spec — do
not tune them to make lead-time numbers look better. Run 2's test-split
result (0 of 10 episodes warned) is a real finding about calibration, not a
threshold-choice artifact; see `REAL_DATA_RESULTS.md`.

## Ablations (`nidra/eval/ablations.py`) — the falsification suite

Each is a real experiment with a real possible failure (project Rule 3): if
persistence matches the model, or time-shuffle doesn't collapse
performance, that is the reported result, never engineered away.

- **Persistence ablation**: replaces the transition model with `mu=0`
  (copy-forward), scored with the same frozen risk head. Expects a
  collapse relative to the full world model; `auc_collapse` is reported
  either way.
- **Time-shuffle ablation**: permutes window order within each input
  sequence. Expects a collapse (the encoder can no longer read a trend); no
  collapse means the model may be using per-window features only.
- **Horizon curve**: AUC-PR and nRMSE vs. `k`. Smooth degradation is
  expected; a flat curve (`flat_curve_leakage_warning`) is a leakage red
  flag, not a good result.
- **Surprise signal**: one-step forecast error, benign vs. pre-attack
  windows (`future_is_attack.any(axis=1)` but the origin window itself is
  benign — the regime-change case). A rise before onset is a positive
  secondary signal a classifier structurally cannot produce.

## Calibration (`nidra/eval/calibration.py`, `nidra/eval/calibrate.py`)

Per-horizon Brier score + reliability diagram (predicted probability bin →
observed frequency), computed by `nidra/eval/calibration.py`. Computed on
whichever split is passed — pass validation data if this is meant to
inform any downstream calibration fitting, test/holdout data only for
final reporting, never both roles at once.

**Post-hoc recalibration** (`nidra/eval/calibrate.py`, fit via
`python -m nidra.scripts.fit_calibration`) exists, is unit-tested, and is
wired into `run_eval.py` (which always computes and reports a
`world_model_calibrated` comparison row/`calibration_recalibrated` section
whenever `<weights_dir>/risk_calibration.json` is present) — but it is
**off by default in `NidraPredictor`** (`apply_calibration=False`), because
whether it helps is **checkpoint-dependent, not a fixed property of the
technique**: measured against the MVP-scale ensemble
(`artifacts_mvp_2017/`), a correctly base-rate-respecting Platt fit
*reduces* recall at threshold=0.75 (dropped to 0.000 at every horizon,
pooled-ensemble-verified); measured again against the full-scale ensemble
(`artifacts/`, `config/default.yaml`, 500k/50k samples), the same technique
*mildly helps* recall on both test and holdout (pooled-ensemble-verified:
~1%→2-9%), never hurting it. **Both findings are real; neither
generalizes to the other checkpoint.** See `MODEL_CARD.md` and
`REAL_DATA_RESULTS.md` (Run 2 addendum and Run 3) for the full root-cause
explanations. Never refit either checkpoint's calibration with
`class_weight="balanced"` to make the recall number look better — that
would manufacture confidence the underlying signal doesn't support
specifically to clear the mandated threshold, which this document's own
rule above (don't tune 0.75/m=2 to make numbers look better) already
forbids one level up.

**Standing methodological note (added after Run 3): always verify a
calibrated recall/lead-time claim against the real pooled-ensemble path
before citing it.** `run_eval.py`'s `world_model_calibrated` baseline row
and `lead_time.json`'s `"calibrated"` section apply the pooled-ensemble-fit
calibration to a **single seed's own raw output** (`applied_to_single_seed_
approximation: true` in the metadata) — a documented shortcut for a quick
per-seed sanity check, not the real `NidraPredictor` serving statistic. At
MVP scale this approximation agreed in direction with the real
pooled-ensemble check. At full scale it did not just differ in magnitude —
it looked *qualitatively* better than reality (single-seed lead-time:
9-of-10 test episodes warned; real pooled-ensemble recall: 2.4%). Before
trusting either file's calibrated numbers as a serving-behavior claim,
re-verify with `ensemble_world_model_forecast` directly (pattern: see
`REAL_DATA_RESULTS.md` Run 3's calibration section) — this cost nothing
more than writing one script, and it caught what would otherwise have been
a materially overstated recommendation.

## Behavioral regimes (`nidra/explain/regimes.py`) — descriptive, not predictive

K-means over the encoder's latent hidden state (`h_t`). `discover_regimes`'s
signature structurally cannot accept a label array
(`test_discover_regimes_never_sees_labels` enforces this) — regimes are
discovered from representation geometry alone. `regime_risk_profile`
computes each cluster's risk rate only *after* clustering, for
interpretation. Not used anywhere in the prediction path.

## Reality overlay (`nidra/scripts/reality_overlay.py`)

```bash
python -m nidra.scripts.reality_overlay --config config/mvp_2017.yaml \
    --weights-dir artifacts_mvp_2017/weights --scaler-dir artifacts_mvp_2017/scaler \
    --split test --out-json reality_overlay.json --out-png reports/reality_overlay.png
```

Generates one forecast from observations up to `origin_ts` only (no future
data — `NidraPredictor.forecast()` structurally never receives `Y`), then
overlays it against what was subsequently actually observed at that host.
Reports `state_nrmse_scaled` per horizon as the primary metric (unit-
normalized, comparable across features) — a supplementary
`raw_abs_error_per_feature` is also written but is explicitly illustrative
only: `flow_duration_var`/`iat_var`-family features are squared-microsecond
quantities with a naturally huge raw dynamic range, so a single blended
raw-unit RMSE across all 45 features would be dominated by whichever
happens to be large in a given window — the same scaled-vs-raw-units
pitfall behind the nRMSE bug above, caught here before it could mislead a
report (a first pass showed RMSE values in the 10^11-10^14 range purely
from this).

## Report/artifact generation (`nidra/scripts/generate_report.py`)

```bash
python -m nidra.scripts.generate_report --config config/mvp_2017.yaml \
    --test-metrics-dir artifacts_mvp_2017/metrics/test \
    --holdout-metrics-dir artifacts_mvp_2017/metrics/holdout \
    --reports-dir reports --metadata-dir artifacts_mvp_2017/metadata
```

Reads already-computed metrics/weight-metadata JSON and renders
`reports/*.png` + `artifacts*/metadata/{model,dataset,experiment_config}.json`
— never computes a metric itself. A plot is skipped (logged, not
fabricated) when its input file is missing.

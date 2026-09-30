# Evaluation

This document explains the harnesses and metric definitions.
`REAL_DATA_RESULTS.md` is the source of truth for measured results — **Run 8
(the natural-prevalence benchmark below) is the current, best-supported set of
numbers**. Everything from the second section onward describes the older
`run_eval.py` harness (Runs 1–7, Δ=30 s, balanced evaluation subsample); it is
kept because those runs are still in the results document, not because its
numbers describe the shipped system.

## The primary benchmark (Run 8, current): `nidra.eval.benchmark`

```bash
cd ml
# 1. validation selects the operating point and writes artifacts/weights/operating_point.json
python -m nidra.eval.benchmark --split val --select-operating-point --n-samples 60 --n-resamples 300
# 2. test and holdout only LOAD it (a run without one uses config pooling + the mandated 0.75 and says so)
python -m nidra.eval.benchmark --split test --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split holdout --n-samples 60 --n-resamples 300
# tables and figures from the records
python -m nidra.scripts.report_tables --run . --splits val,test,holdout
python -m nidra.scripts.plot_benchmark --run . --splits test,holdout --out reports/run8
```

Writes `<metrics_dir>/<split>/benchmark.json` (plus `benchmark_scores.npz` with every
system's per-row scores). Each record starts with the provenance block from
`nidra/utils/provenance.py` — git commit, config hash, dataset file digests, geometry,
seeds, checkpoint sha256, operating point, caps, eval seed — so a number can be traced
to exactly what produced it. `--set key=value` overrides (used for the K-extension check)
are recorded in the file as `config_overrides`.

### Evaluation set (`nidra/eval/eval_set.py`)

Stratified, weighted back to the split's **natural prevalence**:

| stratum | what | sampled | weight |
|---|---|---|---|
| positive | every origin whose published label is 1 (attack within K windows) | all | 1 |
| pre_onset | every origin ≤ `pre_onset_minutes` (30) before an episode onset | all | 1 |
| active_negative | benign origins whose current state is active (`is_active=1`) | ≤ `active_negative_cap` (15,000) | N_full / n_sampled |
| silent_negative | benign origins whose current state is silent | ≤ `silent_negative_cap` (5,000) | N_full / n_sampled |

Every metric is computed with those weights, so precision, F1, false-alarm rate and AP
are the ones a deployment at the split's prevalence would see (test 0.0033, holdout
0.00034). The balanced 4,000-row subsample of Runs 1–7 (prevalence ≈0.46) is a different
quantity; the two are never mixed in one table.

Episodes are attack runs on one host merged across gaps of ≤ `episode_merge_gap_windows`
(5). The geometry (`nidra/data/onset.py`) is shared by the eval set, the onset-target
training arrays and the per-episode report, and keys never cross an episode boundary.

### Tasks

- **Task A — published label** (primary): attack on this host within the next K windows,
  origin anywhere. Natural-prevalence AP with an **episode-cluster bootstrap** interval
  (resampling episodes, not rows, so the 87–92% of positives that sit inside an already
  running episode do not count as independent evidence). `n_positive_clusters` is
  reported; when it is < 5 the interval is wide by construction and the summary marks it
  with `*`. A row-level interval is reported alongside for reference only.
- **Task A′ — detection**: is the origin state itself attack-labelled. What a classifier
  measures; reported so the two are never conflated.
- **Task B — onset within h minutes** (h ∈ {1, 3, 5, 10, 15, 30}): origin strictly outside
  any episode, positive iff the next episode on the host begins within h. This is the
  forecasting question the problem statement asks; the dataset's attacks start without a
  same-host precursor phase, so it has very few positives and every system is near the
  prevalence floor — reported, not tuned.
- **Task C — per horizon k**: attack at exactly `t+k`, plus stage top-1 at `t+k`,
  ridge/persistence state skill at `t+k`.
- **Attribution**: paired episode-bootstrap AP differences for
  `world_model − {persistence, persistence + learned noise, isotropic noise, deterministic
  rollout, ridge two-lag, LR, GBDT, GRU classifier}`. The sign and interval of
  `world_model − persistence` is the test of whether the transition model contributes
  anything beyond the head on `S_t`.
- **State-forecast skill**: `1 − MSE(model) / MSE(persistence)` on the deterministic
  rollout, over the features the scaler keeps (constant and duplicate training columns are
  dropped from the model and the metric alike); ridge two-lag is the
  linear reference. Also the empirical coverage of the 90% band.
- **Per episode**: earliest sustained crossing before onset (lead time, minutes), or
  latency after onset, at the frozen threshold and at the mandated 0.75.

### Operating point (`nidra/eval/operating_point.py`)

Chosen on validation only, written to `artifacts/weights/operating_point.json`, loaded
by both the test/holdout benchmark and `NidraPredictor`:

1. pooling statistic over sampled trajectories — nine candidates (mean, median, quantiles
   0.5–0.95, max, P(trajectory > 0.5)) scored on validation Task A AP;
2. per-horizon Platt calibration fit on validation with the natural weights;
3. threshold: the validation F1-optimal one on the calibrated score. The mandated 0.75 is
   reported alongside in every record; neither is ever chosen on test.

### Baselines and ablations scored in the same run

Persistence (frozen risk head on `S_t`), persistence + the model's learned noise (mean
disabled), isotropic-noise persistence, deterministic world model (no noise), ridge
two-lag dynamics rolled out K steps, oracle on the true future states, LR on `S_t`, LR on
the flattened L-window history, GBDT on `S_t`, the GRU sequence classifier trained by the
`gru_baseline` stage, and (`onset_head_direct`) the explicit onset head in the Task B
tables. All classifier baselines are fit on a 500k-row stratified training sample and
scored on the identical evaluation rows.

---

## The legacy harness (`run_eval.py`, Runs 1–7, superseded)

Everything below describes the balanced-subsample harness the earlier runs were
measured with. It still runs, and its metric definitions are still correct for what
they measure, but its numbers are not comparable to the benchmark above.

### Running it

Against the shipped production model. This needs no dataset download — the
weights, scaler and windowed tables are all committed under `artifacts/`,
and these are the flags the published numbers were produced with:

```bash
cd ml
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 \
    --split test    --n-samples 200 --max-eval-samples 4000 --use-ensemble
python -m nidra.eval.run_eval --config config/default.yaml --seed 0 \
    --split holdout --n-samples 200 --max-eval-samples 4000 --use-ensemble
```

The harness examples further down this document use `config/mvp_2017.yaml`,
the reduced-scale config kept for fast iteration and for provenance of Run 2.
Its metrics are committed but **its weights are not** (see `.gitignore` — it
is a superseded variant), so those commands require training that config's
ensemble first. Anything you want to run against the shipped model uses
`config/default.yaml`.

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
python -m nidra.scripts.reality_overlay --config config/default.yaml \
    --weights-dir artifacts/weights --scaler-dir artifacts/scaler \
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
python -m nidra.scripts.generate_report --config config/default.yaml \
    --test-metrics-dir artifacts/metrics/test \
    --holdout-metrics-dir artifacts/metrics/holdout \
    --reports-dir reports --metadata-dir artifacts/metadata
```

Reads already-computed metrics/weight-metadata JSON and renders
`reports/*.png` + `artifacts*/metadata/{model,dataset,experiment_config}.json`
— never computes a metric itself. A plot is skipped (logged, not
fabricated) when its input file is missing.

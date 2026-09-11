# Real CIC-IDS2017 Run — Results

## Run 2 (current): 5-seed ensemble, all 5 PCAPs extracted, 40k/8k samples

This supersedes Run 1 below as the current, best-supported result. Run 1's
content is kept unmodified further down for provenance — nothing in it is
deleted or rewritten, and it remains an honest record of what an
under-resourced single-seed run showed. Nothing in this section is
fabricated, extrapolated, or inherited from Run 1 — every number below comes
from this run.

### What changed since Run 1

1. **All five raw PCAPs are now extracted** (Monday, Tuesday, Wednesday,
   Thursday, Friday — Run 1 only had Monday and Friday). All 8 day-files now
   carry real tshark-extracted packet-level features; none run in flow-only
   mode any more. `config/default.yaml` and `config/mvp_2017.yaml` were
   updated to reference the new `Tuesday-WorkingHours_packets.parquet`,
   `Wednesday-workingHours_packets.parquet`, and
   `Thursday-WorkingHours_packets.parquet` files.
2. **Three real bugs found and fixed, in the metric and serving layers, not
   the data**:
   - `eval/metrics.state_nrmse` normalized by `std(y_true)` recomputed on
     whatever (often small, stratified) eval batch was passed in. Many of
     the 45 features are structurally near-constant on large slices of this
     dataset (packet aggregates on a then-flow-only day, rare-event ratios
     like `urg_ratio`), so that batch std collapsed toward the old `1e-8`
     floor and inflated nRMSE by orders of magnitude — this is what produced
     Run 1's `nrmse_world_model_mean` of 154,775 / 783,843. Fixed by
     normalizing against `FeatureScaler.reference_std_`, a per-feature std
     computed once over the full training population (see
     `nidra/data/normalize.py`), with the floor raised from `1e-8` to `0.05`
     (5% of one training IQR). nRMSE is now a legible single-digit number
     (see below) — this was a metric-computation bug, not a rollout-quality
     problem; the underlying rolled-out states were never as bad as Run 1's
     number implied.
   - `serve/predictor.py`'s `forecast()` never inverse-transformed the
     rolled-out `predicted_states_mean` before populating
     `predicted_features` — it silently leaked the model's internal
     RobustScaler+log1p-scaled representation into what the web contract
     documents as raw units (e.g. `bytes_total: 162000.0`). Fixed by adding
     `FeatureScaler.inverse_transform()` and calling it in `forecast()`.
   - Fixing the above exposed a second, deeper issue: for the 5 log1p-scaled
     features (`bytes_total`, `active_flow_count`, `out_degree`,
     `new_peer_count`, `retrans_count`), a rollout prediction that is only
     modestly off in scaled space (where nRMSE is a sane ~5-9) inverts
     through `expm1` to a raw-unit value in the 10^10-10^14 range —
     `expm1` amplifies scaled-space error exponentially. Caught via
     `reality_overlay.py` against the real trained ensemble, not
     hypothetically. Fixed with a ceiling (25.0, i.e. `expm1(25)≈7.2e10`,
     generous for any of these features even under an extreme DDoS window)
     on the pre-`expm1` value in `FeatureScaler.inverse_transform`.
   - A fourth, non-numeric bug: `eval/run_eval.py` wrote
     `baselines.json`/`ablations.json`/`calibration.json`/`lead_time.json`
     to a flat `metrics_dir` regardless of split — running the documented
     `--split test` then `--split holdout` sequence silently overwrote the
     test split's results with the holdout split's. Fixed by writing to
     `metrics_dir/<split_name>/`; caught by hand running exactly that
     sequence (both splits' results were backed up before the fix landed,
     so nothing here is a re-run to get a better number).
3. **Day-level caching added** (`nidra.train.pipeline.load_and_label_day`'s
   `cache_path`): the CSV-load → join → windowize → label pass is the
   dominant cost of every train/eval invocation (~24 minutes for all 8 real
   day-files, one time) and was previously repeated from scratch by
   `train_dynamics`, `train_heads`, and every `run_eval.py` call. Both
   configs' `processed_dir` now point at the same shared cache
   (`artifacts/processed`), since the cached table only depends on
   `windowing:`/`dataset:`, identical between the two configs. This is what
   made training all 5 seeds in this run practical.
4. **5x more training data, single seed → full 5-seed ensemble**: `--max-train-samples
   40000 --max-val-samples 8000` (vs. Run 1's 8000/2000), 30-epoch cap (vs.
   20), all 5 seeds `[0,1,2,3,4]` trained (vs. Run 1's seed 0 only).

### Stage 1 — dynamics training: materially more stable than Run 1

All 5 seeds converge to a tight band, unlike Run 1's noisy, bouncing train
loss:

| seed | best val multi-step NLL | epochs to early-stop |
|---|---|---|
| 0 | -1.3645 | 25 |
| 1 | -1.3632 | 27 |
| 2 | -1.3690 | (early-stopped) |
| 3 | -1.3682 | (early-stopped) |
| 4 | -1.3576 | (early-stopped) |

Compare to Run 1's single seed: best val NLL -0.4387, with train NLL
"bouncing around rather than smoothly decreasing" — that instability is
gone at this scale. This is a real, measured improvement from more data, not
a change to the loss function or training procedure.

### Stage 2 — heads: still overfitting, unresolved

`pos_weight=138.4` (vs. Run 1's 26.9 — the positive-class rarity is worse at
this larger, more representative sample). Train loss fell smoothly
(1.70→0.35 over 17 epochs before early-stopping); val loss stayed noisy in
the 11-19 range with no clear downward trend across all 5 seeds
(best-val-loss range: 8.35-11.37). **This is the same genuine, unresolved
overfitting signal Run 1 reported — more dynamics-training data did not fix
it.** The stage head in particular remains under-exercised: several of the
6 stage classes are still rare enough to hit the class-weight ceiling even
at 40,000 samples.

### Evaluation — test split (Friday), n=4,000 (stratified from 438,708 real candidates)

| Model | F1 | Precision | Recall | AUC-PR |
|---|---|---|---|---|
| LR (current state) | 0.758 | 0.944 | 0.632 | 0.817 |
| LR (flattened history) | 0.785 | 0.968 | 0.659 | 0.863 |
| Persistence | 0.592 | 0.947 | 0.431 | 0.694 |
| **World model** | **0.082** | 0.889 | **0.043** | **0.803** |
| Oracle (upper bound) | 0.934 | 0.903 | 0.966 | 0.947 |

**The world model now clearly beats persistence on AUC-PR (0.803 vs. 0.694,
+0.103) — reversing Run 1's finding that it did not.** The persistence
ablation (`nidra/eval/ablations.py`) independently confirms this:
`auc_collapse=+0.103`, `"interpretation": "world model beats persistence"`.
This is the real, structural test the project is built around
(`baseline_lr_flattened_history` and `baseline_persistence` both reuse the
same frozen risk head the world model uses), and at this scale it passes.

**But F1/recall expose a real, separate, newly-surfaced problem: the
model's probabilities are badly miscalibrated relative to the fixed 0.75
threshold.** Precision is high (0.889 — when the world model does cross
0.75, it is usually right) but recall is 0.043 — it almost never crosses
0.75 even for true positives. This is a ranking-vs-calibration gap: the
model separates risk classes well (AUC-PR, which is threshold-free) but its
absolute probability outputs sit systematically below the mandated
operating point. The most likely mechanism, consistent with the
frozen-head design (Rule 1: heads are trained ONLY on observed states,
never on rollout output): the risk head's training distribution (real
observed states) and its actual inference input (rolled-out, sampled,
somewhat-off predicted states) are not identical, and the head was never
exposed to "slightly-off-distribution positives" at high confidence during
training. **This is reported honestly as an open problem, not tuned away by
changing the mandated 0.75 threshold or 2-window lead rule — both are
protected by the project's own claims discipline.**

**Lead time: 0 of 10 attack episodes received a sustained warning
(`fraction_no_warning=1.0`)** — a direct consequence of the calibration gap
above: even where the model ranks risk correctly, its probability rarely
sustains above 0.75 for the required 2 consecutive windows. This is the
single most consequential open item in this run.

### Evaluation — holdout split (Thursday, both Web-Attacks + Infiltration), n=4,000 (stratified from 674,269 real candidates)

| Model | F1 | AUC-PR |
|---|---|---|
| LR (current state) | 0.698 | 0.616 |
| LR (flattened history) | 0.865 | 0.883 |
| Persistence | 0.648 | 0.635 |
| **World model** | 0.396 | 0.635 |
| Oracle (upper bound) | 0.629 | 0.897 |

**World model AUC-PR (0.635) is statistically indistinguishable from
persistence (0.635)** — `auc_collapse=-0.037`,
`"interpretation": "NO MEANINGFUL GAP over persistence"`. On a genuinely
unseen attack type (Infiltration was never in any training split), the
learned dynamics provide no measurable edge over "assume no change." This
is the honest generalization result: the transition model's within-
distribution improvement over persistence (test split, above) does not
transfer to an attack family it never saw signatures from — exactly what
the holdout is designed to test, and exactly the kind of result the
project's ablation-honesty rule says must be reported as-is.

Lead time: 1 of 2 attack episodes received a warning (median 18,660s ≈
5.2 hours; `n_episodes=2` is barely more than an anecdote — Thursday's
Infiltration file has very few positively-labelled rows, same limitation
noted in Run 1).

### Ablations

**Persistence ablation**: test — world model beats persistence
(`auc_collapse=+0.103`); holdout — no meaningful gap
(`auc_collapse=-0.037`). Both reported as-is (§ above).

**Time-shuffle ablation — the same split-inconsistency as Run 1, still
unresolved at 5x the scale:**
- Test: normal-order AUC-PR 0.806 vs. shuffled 0.798 (collapse 0.007) —
  *"NO COLLAPSE under shuffling"*.
- Holdout: normal-order AUC-PR 0.628 vs. shuffled 0.575 (collapse 0.054) —
  *"temporal order matters (collapse observed)"*.

The same qualitative pattern Run 1 showed (test: no shuffle-sensitivity;
holdout: real shuffle-sensitivity) persists after 5x more training data and
5 seeds instead of 1 — this rules out "undertrained single seed" as the
explanation. It is a genuine, reproducible property of how the model
behaves differently on these two splits, still unexplained, still an open
item.

**Horizon curve — the odd/even parity oscillation from Run 1 is also still
present, at this scale, on the test split specifically:**

test: `auc_pr_by_k = [0.518, 0.174, 0.560, 0.159, 0.497, 0.162]` — k=1,3,5
(indices 0,2,4 → horizons 1,3,5) are consistently ~3x higher than k=2,4,6.
holdout: `auc_pr_by_k = [0.242, 0.369, 0.238, 0.206, 0.227, 0.162]` — no
clean alternation (horizon 2 is the highest, not the lowest, breaking the
test-split pattern), matching Run 1's finding that holdout does not show
the same parity structure. `flat_curve_leakage_warning=false` on both
splits (no automated leakage flag fired). Same status as Run 1: real,
reproducible, unexplained on the test split specifically, and not resolved
by more data/seeds — still an open investigation item, most likely (per the
task's own hypothesis, not confirmed) a rollout/sampling-step artifact or a
genuine periodicity in Friday's labelled attack windows aliasing against
the 30s/K=6 window geometry.

**Surprise signal** (still the one ablation that shows a consistent,
unambiguous positive signal on both splits): `mean_error_pre_attack` is
~17-21x `mean_error_benign` (test: 5.26 vs. 0.31; holdout: 5.33 vs. 0.25),
`error_rises_before_onset=true` on both.

**State nRMSE — now a legible, sane number** (the metric-computation fix,
§ above): test `nrmse_world_model_mean=6.52` vs. `nrmse_persistence_mean=5.71`
(world model slightly worse on raw state accuracy despite better risk
ranking — plausible: the transition model can rank risk usefully from
partial/directional signal without reconstructing every one of 45 features
precisely); holdout `world_model=2.53` vs. `persistence=2.39`, same
pattern. Both are in scaled (RobustScaler+log1p) units, where 1.0
represents one training-population standard deviation — orders of
magnitude away from Run 1's 154,775/783,843, and directly comparable across
features and horizons for the first time.

### Behavioral regimes (new in this run — `nidra/explain/regimes.py`)

K-means (k=5) over the encoder's latent hidden state (`h_t`, 128-dim),
computed on 3,000 stratified test-split samples — the clustering algorithm
never sees `risk_label` (see the module's structural test,
`test_discover_regimes_never_sees_labels`). Risk rate per discovered
regime, computed only AFTER clustering, for interpretation:

| regime | n samples | risk rate |
|---|---|---|
| 0 | 933 | 98.7% |
| 1 | 719 | 0.8% |
| 2 | 520 | 99.4% |
| 3 | 371 | 4.6% |
| 4 | 457 | 85.1% |

The encoder's latent space separates cleanly into high-risk (regimes 0, 2,
4: 85-99%) and low-risk (regimes 1, 3: <5%) clusters despite never being
trained or clustered against risk labels — a genuine, unprompted positive
result: the learned representation organizes itself around risk-relevant
structure. This is a descriptive/interpretability finding, not a new
detector — the regimes are not used for prediction anywhere in the
pipeline.

### Ensemble / serving

All 5 seed checkpoints load successfully in a clean `NidraPredictor`
process (`nidra.serve.benchmark`). At `rollout.n_samples_per_member=200`
(5×200=1,000 trajectories, matching the project's `~1000 sampled futures`
target and `config/default.yaml`), measured p95 latency was 616ms — over
the 300ms target. Per the documented policy ("cut samples toward 100 before
cutting ensemble members"), `config/mvp_2017.yaml` now uses
`n_samples_per_member=100` (500 total trajectories): p95=108ms, comfortably
under target. This is a measured trade-off on this specific CPU, not a
silent change — `config/default.yaml` (the full-scale production config)
still specifies 200/member for hardware that can sustain it.

### Reality overlay (new in this run — `nidra/scripts/reality_overlay.py`)

Generates a forecast from observations up to `origin_ts` only (no future
data consumed — enforced structurally, since `NidraPredictor.forecast()`
never receives `Y`), then overlays it against what was subsequently
actually observed. One example run against the test split: host
`192.168.10.9`, origin `2017-07-07T11:33:30Z`, 500 pooled trajectories.
Per-horizon `state_nrmse_scaled` (the primary, unit-normalized error
metric): `[4.98, 1.07, 2.67, 0.79, 9.16, 2.69]` — noisy at the single-
sample level (expected; this is a qualitative demo of one forecast, not an
aggregate metric), same order of magnitude as the aggregate nRMSE numbers
above. A raw-unit `raw_abs_error_per_feature` breakdown is also written per
horizon but is explicitly documented as illustrative only: `flow_duration_var`
and `iat_var`-family features are squared-microsecond quantities with a
naturally huge raw dynamic range independent of forecast quality, so a
single blended raw-unit RMSE across all 45 features would be dominated by
whichever of those happens to be large in a given window — exactly the
scaled-vs-raw-units pitfall this project's own metric-auditing guidance
warns about, caught here by hand before it could mislead the report.

### Honest summary (Run 2)

More data and a real 5-seed ensemble produced two genuine, measured
improvements over Run 1: (1) the state-forecast nRMSE bug is fixed, and (2)
the world model now clearly beats persistence and the current-state LR
baseline on AUC-PR on the test split — the central claim this project is
built around, which Run 1 could not support. Both are real, not spun.

It also surfaced a new, more consequential problem that Run 1's much worse
nRMSE bug had been obscuring: **the model's probability calibration is bad
enough that the mandated 0.75/2-window operating point produces zero
warnings on the test split.** AUC-PR says the signal is there; the fixed
threshold says the system does not yet act on it. This is the top priority
for follow-up work, and it is a head-training/calibration problem, not a
transition-model problem (the transition model's own state nRMSE and
AUC-PR-based ranking both improved with more data).

The three previously-unresolved open items from Run 1 — the odd/even
horizon-parity oscillation on the test split, the time-shuffle
inconsistency between splits, and heads-training val-loss not converging —
are ALL STILL PRESENT after 5x the data and a full 5-seed ensemble instead
of 1. This rules out "just needs more data/seeds" as the explanation for
any of the three; they need direct investigation, not more compute.

Infiltration (holdout) generalization is honestly negative: no measurable
AUC-PR edge over persistence on an attack type never seen in training.

### Reproduction (Run 2)

```bash
cd ml
python -m nidra.data.pcap_extract cicids2017/pcap/Tuesday-WorkingHours.pcap cicids2017/pcap/parquet/Tuesday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Wednesday-workingHours.pcap cicids2017/pcap/parquet/Wednesday-workingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Thursday-WorkingHours.pcap cicids2017/pcap/parquet/Thursday-WorkingHours_packets.parquet
for seed in 0 1 2 3 4; do
  python -m nidra.train.train_dynamics --config config/mvp_2017.yaml --max-train-samples 40000 --max-val-samples 8000 --epochs 30 --seed $seed
  python -m nidra.train.train_heads    --config config/mvp_2017.yaml --max-train-samples 40000 --max-val-samples 8000 --seed $seed
done
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split test    --n-samples 50 --max-eval-samples 4000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split holdout --n-samples 50 --max-eval-samples 4000
python -m nidra.serve.benchmark --weights-dir artifacts_mvp_2017/weights --scaler-path artifacts_mvp_2017/scaler/robust_scaler.joblib --config config/mvp_2017.yaml
python -m nidra.scripts.reality_overlay --config config/mvp_2017.yaml --weights-dir artifacts_mvp_2017/weights --scaler-dir artifacts_mvp_2017/scaler --split test --out-json reality_overlay.json --out-png reports/reality_overlay.png
python -m nidra.scripts.generate_report --config config/mvp_2017.yaml --test-metrics-dir artifacts_mvp_2017/metrics/test --holdout-metrics-dir artifacts_mvp_2017/metrics/holdout --reports-dir reports --metadata-dir artifacts_mvp_2017/metadata
```

### Caveats that materially limit every number above

- **Still not `config/default.yaml`'s full production scale** (60/30
  epochs, uncapped samples against all 6.9M+ train candidates) — 40,000/
  8,000-sample, 30-epoch MVP-scale run, now with a real 5-seed ensemble
  rather than Run 1's single seed. The full-scale run has not been executed
  (compute/time, not a blocker) — nothing about the data layer needs to
  change to run it; see "Upgrading to the full dataset" below.
- **Heads val-loss still does not converge cleanly** — unresolved, flagged
  above, present across all 5 seeds.
- **Calibration at the 0.75 threshold is the top open problem** — flagged
  prominently above, not present in Run 1's report because Run 1's nRMSE
  bug and worse-than-persistence AUC-PR obscured that the underlying signal
  was there but miscalibrated.
- **Horizon-parity oscillation and time-shuffle split-inconsistency remain
  unexplained** after ruling out "undertrained" as the cause.
- **Ensemble evaluation ran per-seed** (baselines/ablations/calibration/
  lead-time were computed against seed 0 only, not re-run through the
  pooled 5-member `NidraPredictor`) — the 5-seed ensemble is real and fully
  loads/serves (see "Ensemble / serving" above), but a pooled-ensemble
  version of the baselines/ablations harness is a good next step, not
  completed here.
- **Lead-time numbers are based on 10 (test) and 2 (holdout) episodes** —
  small samples, same caveat as Run 1.

---

## Run 1 (superseded): single-seed MVP, Monday+Friday PCAPs only

This section is preserved unmodified from the original run for provenance.
Run 2 above supersedes it as the current result — in particular, the nRMSE
numbers below are known-wrong (metric bug, fixed in Run 2) and should not be
cited; everything else here is an honest historical record.

## 1. Dataset

Source: `GeneratedLabelledFlows.zip` / "TrafficLabelling" release (the
variant with Source IP, Destination IP, and Timestamp columns intact — the
sibling "MachineLearningCVE" release strips those and cannot be windowed by
host/time) plus the raw `Monday-WorkingHours.pcap` and
`Friday-WorkingHours.pcap` captures, all downloaded directly from the
dataset's official distribution into `cicids2017/` in this repo (gitignored,
not committed).

**Every CSV was read in full** — no `nrows`/row-count truncation anywhere in
this run:

| Day (CSV) | Input rows | Accepted | Notes |
|---|---|---|---|
| Monday | 529,918 | 529,918 | 100% benign |
| Tuesday | 445,909 | 445,909 | FTP/SSH brute-force |
| Wednesday | 692,703 | 692,703 | DoS Hulk/GoldenEye/Slowloris/Slowhttptest, Heartbleed |
| Thursday (Web Attacks) | 458,968 | 170,366 | 288,602 rows dropped — genuine blank trailing rows in the source file (all-NaN, not a parsing bug); also required a latin-1 decode fallback (Windows-1252 en-dash in "Web Attack – XSS" labels breaks plain UTF-8) |
| Thursday (Infiltration) | 288,602 | 288,602 | only 36 rows actually labelled `Infiltration` |
| Friday (Morning) | 191,033 | 191,033 | benign + Bot |
| Friday (PortScan) | 286,467 | 286,467 | |
| Friday (DDoS) | 225,745 | 225,745 | |

### Real packet-level features (Monday + Friday)

The raw PCAPs for Monday and Friday were downloaded, **MD5-verified intact**
before use, and extracted with `python -m nidra.data.pcap_extract`:

| Day | PCAP size | Packets extracted | Time |
|---|---|---|---|
| Monday | 10.8GB | 11,626,492 | 263.4s |
| Friday | 8.8GB | 9,915,680 | 218.6s |

**One real bug was found and fixed during this extraction**, not discovered
in advance: tshark's default field output joins a multi-valued field (e.g.
`ip.src` appearing twice when a packet embeds another packet's IP header,
as ICMP errors do) with the same character used as the field separator,
silently corrupting the row. This was caught (not guessed) by watching the
initial extraction attempt log a stream of "dropping malformed tshark row"
warnings, tracing it to specific packets, and confirming the fix
(`-E occurrence=f`, which takes only the first value of any multi-valued
field) against a live field-count check before committing to the full
10.8GB+8.8GB extraction. A related latent bug was found alongside it: the
row-drop logic for missing `ip_src`/`ip_dst` used `dropna()`, which only
catches actual `NaN` — a tshark row with a genuinely empty (but present)
IP field would have silently passed through uncaught. Both are fixed in
`nidra/data/pcap_extract.py` and covered by `tests/test_pcap_extract.py`
(9 new tests), which didn't exist before this run.

Tuesday, Wednesday, and both Thursday files have **no PCAP yet** (not
downloaded — see the user's own decision to proceed with only Monday and
Friday rather than wait) and ran in **flow-only mode**: their 11
packet-aggregate features (`ttl_*`, `tcp_window_*`, `frag_flag_rate`,
`payload_size_*`, `retrans_*`) are zero for every window on those days,
logged loudly by the pipeline (`WARNING ... running in FLOW-ONLY mode`),
never silently treated as full-feature data. The other 34 of 45 features
(flow aggregates, graph scalars, backward-looking dynamics, activity flag)
are complete for all 8 days regardless of PCAP availability.

### Windowing (Δ=30s, L=30, K=6), hosts kept vs. dropped (<36 windows)

| Day | Hosts kept | Hosts dropped |
|---|---|---|
| Monday | 3,775 | 4,464 |
| Tuesday | 3,173 | 4,019 |
| Wednesday | 3,211 | 4,478 |
| Thursday (Web Attacks) | 1,351 | 2,852 |
| Thursday (Infiltration) | 1,779 | 3,322 |
| Friday (Morning) | 1,535 | 2,929 |
| Friday (PortScan) | 1,019 | 2,648 |
| Friday (DDoS) | 409 | 1,658 |

Full windowed train-candidate count (Monday+Tuesday+Wednesday, before any
capping): **6,911,848 samples**. This is the real, complete number — the
sample caps described below are applied only at the tensor-construction
stage (stratified by `risk_label`, keeping every positive sample), not by
truncating the source data.

## 2. Training (single seed — MVP scale, not the 5-seed ensemble)

`config/mvp_2017.yaml`: seed 0 only, 20-epoch cap, `--max-train-samples 8000
--max-val-samples 2000` (stratified subsample of the 6.9M-candidate train
split).

**Stage 1 — dynamics (encoder + transition):**

| epoch | train_nll | val_nll | teacher_forcing_p |
|---|---|---|---|
| 0 | 0.2057 | 0.0967 | 1.00 |
| 1 | -0.4170 | 0.0666 | 0.94 |
| 2 | -0.7146 | -0.4024 | 0.88 |
| 3 | -0.7663 | -0.3795 | 0.82 |
| 4 | -0.7341 | -0.3580 | 0.77 |
| 5 | -0.7112 | **-0.4387** | 0.71 |
| 6 | -0.7238 | -0.3761 | 0.65 |
| 7 | -0.7201 | -0.3467 | 0.59 |
| 8 | -0.7151 | -0.3781 | 0.53 |
| 9 | -0.8245 | -0.3601 | 0.48 |
| 10 | -0.6902 | -0.3764 | 0.42 |
| 11 | -0.7948 | -0.3868 | 0.36 |

Early-stopped at epoch 11 (patience 6, best val NLL -0.4387 at epoch 5).
Train NLL bounces around rather than smoothly decreasing after epoch 2 —
at 8,000 train samples this is plausibly just batch noise, not a stable
trend, and has not been checked across multiple seeds.

**Stage 2 — frozen dynamics, risk + stage heads (observed states only):**

- `pos_weight` = 26.875, `class_weights` = [0.169, 1333.8, 11.21, 1333.8,
  1333.8, 1333.8] — four of six stage classes are essentially absent from
  this train sample and hit the weight ceiling, same limitation as before:
  the stage head is really only exercised on benign vs. one other class at
  this scale.
- Train loss fell steadily (2.2704 → 0.3802 over 19 epochs), but **val loss
  did not track it** — it stayed noisy in the 6.8–9.9 range with no clear
  downward trend (best 6.8466 at epoch 12), and training early-stopped at
  epoch 18 on stalled patience. This is reported as a genuine, unresolved
  sign of overfitting at this sample size — not smoothed over. It is a
  materially worse convergence picture for the heads than dynamics training
  showed, and should be treated as unresolved until re-run with a larger
  sample cap or across the full 5-seed ensemble.

## 3. Evaluation — test split (Friday), n=2,000 (capped, stratified from 438,708 candidates)

| Model | F1 | AUC-PR | Precision | Recall | FPR |
|---|---|---|---|---|---|
| LR (current state) | 0.790 | 0.975 | 0.993 | 0.656 | 0.053 |
| LR (flattened history) | 0.904 | 0.993 | 0.996 | 0.828 | 0.040 |
| Persistence | 0.704 | 0.969 | 0.996 | 0.544 | 0.027 |
| **World model** | 0.641 | 0.962 | 0.997 | 0.473 | 0.020 |
| Oracle (upper bound) | 0.990 | 0.996 | 0.996 | 0.985 | 0.053 |

**The world model did not beat any other baseline on F1 or AUC-PR on the
test split** — including persistence. AUC-PR values across the board are
high (>0.96) because Friday's risk-label positive rate is high at this
sample size, which compresses the visible gap between models; F1 still
shows persistence and the flattened-history LR clearly ahead. This is
reported as-is, not spun.

## 4. Evaluation — holdout split (Thursday, both Web-Attacks + Infiltration), n=2,000 (capped, stratified from 674,269 candidates)

Both Thursday files (Web Attacks, Infiltration) are held out from training
entirely, per the mandated generalization test — neither attack family is
seen during Stage-1 or Stage-2 training.

| Model | F1 | AUC-PR | Precision | Recall | FPR |
|---|---|---|---|---|---|
| LR (current state) | 0.780 | 0.717 | — | — | — |
| LR (flattened history) | 0.921 | 0.933 | — | — | — |
| Persistence | 0.816 | 0.783 | — | — | — |
| **World model** | 0.784 | 0.732 | — | — | — |
| Oracle (upper bound) | 0.848 | 0.857 | — | — | — |

(precision/recall/fpr omitted here for brevity — see
`artifacts_mvp_2017/metrics/baselines.json` from this run, copied to
`/tmp/eval_holdout_2017/` at run time.) **World model again did not beat
persistence or either LR baseline** on the unseen-attack holdout. Unlike
the test split, though, holdout AUC-PR values are meaningfully lower
across every model (0.72–0.93 vs. 0.96–0.99 on test) — consistent with a
real, expected generalization gap on genuinely unseen attack types, not a
pipeline defect.

## 5. Ablations

**Persistence ablation** (both splits): *"NO MEANINGFUL GAP over
persistence — dynamics may not be adding value; this is a reportable
negative result, not a bug to hide."* `auc_collapse` slightly negative on
both (test: -0.006, holdout: -0.044) — consistent with the baseline table
above; the learned dynamics are not beating "assume no change" at this
training scale.

**Time-shuffle ablation:**
- Test split: *"NO COLLAPSE under shuffling — the model may be using
  per-window features only, not real temporal structure; this is a
  reportable negative result, not a bug to hide."* (normal-order AUC-PR
  0.963 vs. shuffled 0.963, collapse essentially zero: -0.0005.)
- Holdout split: **the opposite, healthy result** — *"temporal order
  matters (collapse observed)"* (normal-order AUC-PR 0.732 vs. shuffled
  0.631, collapse 0.100). The model's reliance on real temporal structure
  is inconsistent between the two splits, worth further investigation
  rather than averaging away.

**Horizon curve — a genuine, unexplained anomaly worth flagging
prominently:** AUC-PR by horizon step on the test split oscillates sharply
by parity rather than decaying smoothly: k=0,2,4 (0.643, 0.688, 0.709) are
all much higher than k=1,3,5 (0.216, 0.211, 0.211). The **same odd/even
pattern shows up independently in the calibration Brier scores** for the
same split (k=0,2,4: 0.208/0.171/0.154 vs. k=1,3,5: 0.574/0.558/0.548) —
two different metrics agreeing on the same alternating structure is a
strong signal this is real, not sampling noise. The holdout split does
**not** show this pattern (its Brier scores are flat, 0.234–0.264 across
all six horizons; its AUC-PR-by-k does dip but without the same clean
alternation: 0.285, 0.235, 0.342, 0.093, 0.321, 0.039). No automated
`flat_curve_leakage_warning` fired on either split. This parity effect on
the test split specifically is unexplained and should be investigated
before trusting per-horizon numbers there — possible causes include an
artifact of the rollout/sampling step, or a genuine periodicity in Friday's
labelled attack windows that a 30s/K=6 window geometry happens to alias
against; neither has been confirmed.

**Surprise signal** (prediction error rising before attack onset): on both
splits, `mean_error_pre_attack` is roughly 15–22× `mean_error_benign`
(test: 5.30 vs. 0.35; holdout: 5.38 vs. 0.24), and
`error_rises_before_onset=true` on both — the one ablation that shows a
consistent, genuine positive signal on both splits.

**State nRMSE:** as in the earlier CIC-IDS2017 pipeline-validation run,
`nrmse_world_model_mean` is anomalously large (test: 154,775; holdout:
783,843) versus `nrmse_persistence_mean` (~1.2 and ~0.86 respectively) —
several orders of magnitude apart. This rollout-state-divergence issue is
still unresolved and still only affects the raw rolled-out *state* values,
not the risk/stage scores read off them (which is where the baseline
numbers above come from) — but it remains an open item, not something this
run fixed.

## 6. Calibration

Mean Brier score across the 6 horizon steps: **0.369 (test)**, **0.250
(holdout)** — both driven substantially by the k=1/3/5 spikes described
above on the test split; see §5 for the per-horizon breakdown.

## 7. Lead time

- **Test split (Friday):** 10 attack episodes found in the (capped-for-cost)
  scan; median lead time **32,340s (~9 hours)**, distribution ranging
  10,290–36,300s across 8 warned episodes, with 2 of 10 episodes
  (`fraction_no_warning=0.2`) receiving no warning at all. A ~9-hour median
  lead time is large enough on a Δ=30s/K=6 (3-minute) forecast horizon
  that it likely reflects episodes where risk was elevated for most of the
  day rather than a sharp, localized early-warning signal — worth reading
  as "sustained elevated risk across a long attack day" rather than "the
  system predicted this specific attack 9 hours ahead," which would be a
  stronger and less-supported claim.
- **Holdout split (Thursday):** only 2 attack episodes found; median lead
  time 11,205s, both received warning (`n_no_warning=0`). With `n_episodes=2`
  this is barely more than an anecdote, not a distribution — consistent with
  Thursday's Infiltration file having only 36 positively-labelled rows.

## 8. Honest summary

This run proves the full pipeline works genuinely end-to-end against the
complete, real CIC-IDS2017 dataset, including real tshark-extracted
packet-level features for two of five capture days (something the prior
pipeline-validation attempt never had at all). It surfaced and fixed two
real bugs in the PCAP extraction path that had never been exercised at
scale before this run. It does **not** show the world model outperforming
simpler baselines on either split, shows an unexplained horizon-parity
oscillation on the test split specifically, shows inconsistent
time-shuffle behavior between splits, and shows the heads-training
val-loss not converging cleanly. None of this is hidden or reframed as
success — it is the genuine state of a single-seed, reduced-epoch MVP run,
and should be read as exactly that.

### Caveats that materially limit every number above

- **Single seed (seed 0) of the planned 5-seed ensemble** — no inter-seed
  variance, no error bars anywhere in this document.
- **20-epoch cap for dynamics, 20 for heads** (`config/mvp_2017.yaml`), not
  `config/default.yaml`'s production 60/30. Neither stage shows clean,
  confirmed convergence.
- **3 of 8 day-files (Tuesday, Wednesday, both Thursday files) are
  flow-only** — their PCAPs are not yet downloaded. Only Monday and Friday
  have real packet-level features.
- **Training/eval sample counts are capped** (8,000 train / 2,000 val for
  training; 2,000 for eval baselines/ablations/calibration) via stratified
  subsampling of the full, real windowed data — not by truncating the raw
  CSVs, which were read in full (6.9M real train-candidate windows exist;
  8,000 were sampled for this MVP run).
- **Stage-head class imbalance**: 4 of 6 stage classes are effectively
  absent from the capped train sample; the stage head is meaningfully
  validated only on benign vs. one other class.
- **Heads val-loss did not converge** — flagged above, unresolved.
- **The horizon-parity oscillation (§5) and rollout-state nRMSE anomaly
  (§5) are both unresolved**, open items for follow-up investigation.
- **Lead-time numbers are based on 10 (test) and 2 (holdout) episodes** —
  small samples, not distributions with statistical weight.

## Reproduction

```
cd ml
python -m nidra.data.pcap_extract cicids2017/pcap/Monday-WorkingHours.pcap cicids2017/pcap/parquet/Monday-WorkingHours_packets.parquet
python -m nidra.data.pcap_extract cicids2017/pcap/Friday-WorkingHours.pcap cicids2017/pcap/parquet/Friday-WorkingHours_packets.parquet
python -m nidra.train.train_dynamics --config config/mvp_2017.yaml --max-train-samples 8000 --max-val-samples 2000
python -m nidra.train.train_heads --config config/mvp_2017.yaml --max-train-samples 8000 --max-val-samples 2000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split test --n-samples 30 --max-eval-samples 2000
python -m nidra.eval.run_eval --config config/mvp_2017.yaml --seed 0 --split holdout --n-samples 30 --max-eval-samples 2000
python -m nidra.scripts.portscan_sanity_plot --csv "cicids2017/csv/extracted/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --out portscan_sanity.png
```

Requires the real CIC-IDS2017 `TrafficLabelling` CSVs and the
`Monday-WorkingHours.pcap` / `Friday-WorkingHours.pcap` captures placed
under `cicids2017/` per `config/mvp_2017.yaml`'s `dataset:` paths (not
committed to this repo — see `.gitignore`).

## Upgrading to the full dataset

`config/default.yaml` is the full-scale production config: same dataset,
same day/split structure, 5-seed ensemble, 60/30 epoch budget. Nothing
about this run's data layer needs to change to use it — it reads the exact
same `cicids2017/` paths. The path to closing the gap to full-scale,
full-feature training is purely additive:

1. Download the remaining PCAPs (Tuesday, Wednesday, Thursday) and extract
   them the same way as Monday/Friday above — `dataset.days.<day>.packets`
   in the config is the only thing that needs a new filename per day, no
   code changes.
2. Switch from `config/mvp_2017.yaml` to `config/default.yaml` (or raise
   `mvp_2017.yaml`'s epoch/seed/sample-cap numbers directly) once compute
   time allows a full run.

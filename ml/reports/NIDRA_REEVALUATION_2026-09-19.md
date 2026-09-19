# NIDRA — first-principles re-evaluation

Scope: the worktree at commit `36801e5` (`claude/model-metrics-improvement-c81859`), the five shipped checkpoints in `ml/artifacts/weights/` (metadata `git_commit c8785f2`, `config_hash 5f50ab5570322547`), the committed processed tables, and `ml/config/default.yaml`. Every number marked **measured** was produced in this session by scratchpad scripts that import the project's own loaders and model code unmodified; nothing in the repository was changed. Scripts and raw outputs: `scratchpad/audit/` (`0x_*.py`, `out/*.json`, `out/*.log`).

Verdict up front, because the rest of the document is long:

1. My earlier conclusion — *"the risk head is the ceiling; fix its training first"* — is **falsified as stated**. Retraining the head under every regime I tried (balanced sampling, focal loss, longer/shorter training, separate selection) moves test AUC-PR by −0.18 to −0.01 and holdout by −0.04 to +0.09, i.e. inside the noise of a 6-episode training set. The head is not undertrained; it is *under-supervised* (409 positive windows, one host, six attack episodes), and more optimisation only memorises them.
2. What does move the head is **preprocessing**: the shipped `RobustScaler` is an identity map (`center_=0, scale_=1` for all 45 features, because it was fitted on a population that is 98% all-zero rows), so a dozen features saturate the ±10 clip in 45–90% of active windows. A 45→64→1 MLP on the *same* current-state input with a real scaler and input-noise augmentation reaches test AUC-PR 0.908 (natural-prevalence 0.231) versus the shipped head's 0.701 (0.060) — and versus the full production world model's 0.960 (0.132). So the single largest measured gain in this audit came from a two-line preprocessing fix, not from the model.
3. The world model's headline advantage over persistence (0.960 vs 0.701) is **not learned dynamics**. Setting the transition mean to zero and keeping only the model's predicted noise plus the same quantile pooling reproduces it exactly (0.956; paired bootstrap difference +0.006 [−0.008, +0.020]); on the holdout split removing the drift is *better* (0.784 vs 0.688, and tenfold at natural prevalence). The learned drift `mu` adds nothing to ranking on familiar attacks and hurts on unfamiliar ones; the heteroscedastic variance head plus the 85th-percentile statistic is doing all the work. **Addendum (§18): this is a property of the Δ=30 artifact, not of the architecture — after re-windowing at Δ=60 and retraining one seed, the drift contributes +0.09 AUC-PR on both splits with confidence intervals excluding zero, and the GRU beats the ridge two-lag baseline on state forecasting (skill 0.39 vs 0.30).**
4. The evaluation itself overstates everything by roughly an order of magnitude: the eval sample has 46% positive prevalence against a natural 0.42%, and 89–93% of its negatives are the literal all-zero state vector. At natural prevalence the production model's AUC-PR is 0.13 on test and 0.013 on holdout, and its precision at the mandated 0.75 threshold is 0.12 / 0.015 — 29% / 25% of *active* benign windows score above 0.75.
5. The data has a hard representational defect: CIC-IDS2017 flow timestamps have **minute** resolution, the window is **30 s**, so every `:30` window is structurally empty (`is_active = 0.0000` on test and holdout). Half of all positives are these empty windows; every attack episode fragments into one-window pieces; persistence is wrong by construction at odd horizons; and the "world model beats persistence on NRMSE" result is this parity artifact (a *period-2* persistence baseline matches the model; a ridge regression on two lags beats it at every horizon).
6. `risk_label = 1` is 95% "attack already in progress". Genuinely pre-onset positives number 24 (train), 78 (test), 24 (holdout), from 6 / 16 / 5 true onsets. No architecture can learn approach-to-compromise dynamics from six onsets on one host; the honest forecasting claim available on this dataset is much narrower than the one currently made.

---

## 1. Current architecture and pipeline (as built, not as documented)

| Stage | What actually happens | Source |
|---|---|---|
| Flows | CICFlowMeter CSVs (TrafficLabelling), `Timestamp` at **minute** resolution (`4/7/2017 9:20`; no row carries seconds) | `flow_load.py`, raw CSVs |
| Packets | tshark parquet per day, microsecond timestamps, all 5 PCAPs present | `pcap_extract.py`, `~/nidra/cicids2017/pcap/parquet` |
| Windowing | Δ=30 s per `src_ip`; flow aggregates keyed to the flow's start minute → only `:00` windows get flows; packet aggregates windowed at 30 s then **left-joined onto flow windows**, so packet data in `:30` windows is discarded; gap-fill emits all-zero rows from host-min to host-max | `windowize.py:227,278` |
| State | 45 features; `reciprocity` constant (=1), `rst_ratio`/`frag_flag_rate` ≈ always 0, `is_active` redundant; 9 pairs with \|ρ\|>0.9 (e.g. `iat_var`≡`flow_duration_var`) | `feature_audit.csv` |
| Scaling | `log1p` on 5 features, then `RobustScaler` fitted on a 500k stratified sample of which 98% are zero rows → `center_=0, scale_=1` everywhere → **identity**, then clip ±10 | `artifacts/scaler/robust_scaler.joblib` |
| Labels | stage per (`src_ip`, window) = most severe label of flows *sent* by that host; `risk_label[t]=1` if any attack window in `(t, t+6]` (3 min) — includes windows already inside an attack; victims of DoS/brute force never get an attack label | `labels.py` |
| Splits | Mon+Tue+Wed train (val = trailing 15% of Wed by timestamp), Fri test, Thu holdout. All train positives are on one host, `172.16.0.1`. 1,656 hosts appear in both train and test | `splits.py`, `split_facts.json` |
| Sampling | 500k train / 50k val, stratified: every positive kept (409 / 54), negatives at 10.8% | `dataset.py` |
| Dynamics | 2-layer GRU(128) → MLP → diagonal-Gaussian **delta**, logvar ∈ [−6, 3]; K=6 unrolled NLL with scheduled sampling; **validation NLL uses teacher forcing p=1.0**, so checkpoint selection never sees free-running error | `losses.py`, `train_dynamics.py:184` |
| Heads | 45→64→1 risk MLP and 45→64→6 stage MLP on the raw state (not the GRU state); joint loss, `pos_weight` 1221, stage class weights up to 83,333 for absent classes, one global gradient clip; selected on that weighted val loss → **epoch 0 or 1 selected for all 5 seeds** (val AUC-PR 0.014–0.025) | `train_heads.py`, `model_seed_*_metadata.json` |
| Serving statistic | 5 members × 200 stochastic trajectories; every head scores every trajectory; mean over heads, **0.85-quantile over the 1000 trajectories**, max over k | `risk_pooling.py`, `baselines.py` |
| Calibration | Platt per k on val; maps p=0.999 to ≈0.27, so nothing crosses 0.75 (published `_calibrated` rows are F1 0.011 / 0.000) | `risk_calibration.json` |
| Threshold / q | 0.75 fixed by the problem statement; **q=0.85 chosen by inspecting test and holdout F1** (config comment, `rollout:` block) | `config/default.yaml` |

---

## 2. Verified baseline

Headline artifacts: `ml/artifacts/metrics/{test,holdout}/baselines.json` (`ensemble_world_model` row), produced by `run_eval.py --seed 0 --n-samples 200 --max-eval-samples 4000 --ensemble-seeds 0..4` at commit `c8785f2`. Reproduced here on the identical 4,000-row stratified sample (same RNG seed 0 in `subsample_stratified_by_risk`); the rollout RNG is not seeded by `run_eval`, so third-decimal differences are expected.

| Quantity | Test (Friday) | Holdout (Thursday) | Validation |
|---|---|---|---|
| AUC-PR, published | **0.960** | **0.682** | never computed for the world model |
| AUC-PR, reproduced (sampled prevalence) | 0.960 | 0.681 | see §15 |
| **AUC-PR at natural prevalence** (measured) | **0.132** | **0.013** | |
| ROC-AUC (measured; not in pipeline) | 0.973 | 0.950 | |
| Precision / Recall / F1 @0.75, published | 0.964 / 0.855 / 0.906 | 0.731 / 0.800 / 0.764 | — |
| Precision @0.75 at natural prevalence (measured) | **0.117** | **0.015** | |
| Threshold | 0.75 (mandated) | 0.75 | — |
| Positive prevalence: eval sample / natural | 1,853/4,000 = 46.3% / 1,853/438,708 = **0.42%** | 245/4,000 = 6.1% / 245/674,269 = **0.036%** | 54/4,000 (capped) / 54/575,302 |
| Negatives that are the all-zero vector | 90.6% | 92.8% | — |
| Attack episodes: reported / contiguous runs / true (gap ≤5 min) | 10 (= attacked hosts) / 696 / **16** | 2 / 96 / **5** | — / 13 / 2 |
| Forecast horizon | K=6 × 30 s = 3 min | same | same |
| Lead time (published, ensemble) | median 4,320 s, 8/10 "episodes" warned; ceiling-bound (Run 7) | 18,210 s, 1/2 | — |
| Rollout trajectories | 5 members × 200 = 1,000 | same | — |
| Pooling | head-mean, then 0.85-quantile, then max over k | same | — |
| Calibration | Platt per k (val); degenerate at 0.75 | same | — |
| Seeds | eval subsample seed 0; ensemble seeds 0–4; rollout RNG unseeded | | |
| Head val AUC-PR at the selected epoch (from metadata) | seeds 0–4: 0.014, 0.025, 0.018, 0.014, 0.014 | | |
| Dynamics val NLL (teacher-forced) | −1.23 (seed 0) | | |

Discrepancies resolved:
- The root `REAL_DATA_RESULTS.md` reports Run 1 (single seed, F1 0.641, "3 of 8 day files flow-only"). Both statements are stale: `ml/REAL_DATA_RESULTS.md` Run 7 is current, and all five packet parquets exist and are baked into the committed processed tables.
- `lead_time.json` counts "episodes" as attacked hosts (`_host_onset_ts` = first attack window per host), not attack episodes.
- The persistence ablation's NRMSE (model 3.10 vs persistence 4.16) reproduces (2.96 vs 4.20 at k=1) and is explained in §7: persistence alternates 4.2 / 1.97 / 4.16 / 1.82 / 4.14 / 1.95 across k because of the `:30` artifact.
- The per-horizon AUC-PR in `ablations.json` (0.25 → 0.16) uses per-step `future_is_attack` labels at a far lower positive rate; it is not comparable to the headline 0.96 and was never meant to be.

---

## 3. Error decomposition (test split, exact published sample, **measured**)

`AUC-PR` = sampled prevalence (comparable to the published numbers); `AP-nat` = natural prevalence via negative re-weighting (×203.5); `R@.75` = recall at the mandated threshold; `F1*nat` = best F1 at natural prevalence (threshold chosen post hoc, so an upper bound). Full table with precision, ROC-AUC and operating points: `out/decomposition_test.json`.

| Row | System | AUC-PR | AP-nat | ROC | R@.75 | F1*nat |
|---|---|---|---|---|---|---|
| E | LR on S_t (published `lr_current_state`, 4k train) | 0.766 | 0.117 | 0.657 | 0.494 | 0.166 |
| E | LR on flattened history (4k) | 0.731 | 0.076 | 0.612 | 0.385 | 0.192 |
| E | LR on S_t, 500k train | 0.559 | 0.053 | 0.362 | 0.264 | 0.137 |
| E | GBDT (regularised) on S_t, 500k | 0.731 | 0.059 | 0.641 | 0.172 | 0.130 |
| — | **shipped head on S_t = persistence** | **0.701** | 0.060 | 0.566 | 0.166 | 0.152 |
| A | LR (4k) on **true** future, max_k | 0.937 | 0.083 | 0.962 | 0.981 | 0.153 |
| A | GBDT on true future, max_k | 0.921 | 0.105 | 0.954 | 0.566 | 0.119 |
| B | **shipped head on true future = oracle** | 0.878 | 0.087 | 0.871 | 0.577 | 0.134 |
| C | LR (4k) on deterministic rollout, max_k | 0.885 | 0.113 | 0.849 | 0.667 | 0.188 |
| C | LR on stochastic-mean state, max_k | 0.922 | 0.124 | 0.910 | 0.718 | 0.197 |
| D/G | shipped head on deterministic rollout, max_k | 0.870–0.879 | 0.080 | 0.83 | 0.10–0.14 | 0.17–0.20 |
| H | **production: 5×200 stochastic, q=0.85, max_k** | **0.960** | **0.132** | 0.973 | 0.859 | 0.252 |
| F | **noised persistence: mu≡0, model's own logvar, q=0.85** | **0.956** | 0.129 | 0.968 | 0.507 | 0.240 |
| H' | production restricted to own-head/after-pooling (like-for-like with F) | 0.955 | 0.116 | 0.970 | 0.861 | 0.245 |
| F | persistence + isotropic noise σ=0.1 / 0.3 / 1.0, q=0.85 | 0.68 / 0.65 / 0.71 | 0.05–0.10 | 0.5–0.6 | — | 0.13–0.15 |
| I | production with mean pooling | 0.955 | 0.108 | 0.974 | **0.034** | 0.214 |
| I | q = 0.5 / 0.75 / 0.9 / 0.95 / 0.99 | 0.609 / 0.925 / 0.963 / 0.966 / 0.965 | 0.03 / 0.11 / 0.13 / 0.14 / 0.13 | | 0.06 / 0.45 / 0.93 / 0.96 / 0.99 | |
| I | max pooling; P(traj>0.5); P(traj>0.75) | 0.949; 0.959; 0.962 | 0.09; 0.11; 0.12 | | 1.00; 0.04; 0.01 | |
| — | production + Platt (published `_calibrated`) | 0.958 | 0.120 | 0.974 | 0.030 | 0.248 |

Episode-block bootstrap (300 resamples of the 16 true episodes and the negatives): production 0.951 [0.866, 0.980]; noised persistence 0.946 [0.869, 0.976]; **paired difference +0.006 [−0.008, +0.020]**; at natural prevalence production 0.135 [0.044, 0.241] vs LR-on-S_t 0.127 [0.084, 0.195].

Answers to the six questions:

1. **Is the classifier the bottleneck?** Partly, and only in a specific sense. The shipped head ranks worse than a 4k-row logistic regression on the same input (0.701 vs 0.766) and on the true future (0.878 vs 0.937). But the production statistic (0.960) already exceeds *any* classifier on the true future, so the head's ranking is not what limits the headline. What limits the head is its input representation (§6, §8), not its capacity or loss.
2. **Is the world model the bottleneck?** The *mean* dynamics contribute nothing measurable: F (mu≡0) equals H. The deterministic rollout does lift a classifier from 0.766 to 0.885, but §7 shows that gain is the period-2 alternation (restoring the `:00` state from a `:30` origin), and on active `:00` origins the deterministic rollout is no better than persistence (0.933 vs 0.936).
3. **Is stochastic rollout providing genuine predictive information?** Yes — but it is the **predicted variance**, not the predicted trajectory. Isotropic noise of any scale does not reproduce the effect (0.65–0.71); the model's state-dependent logvar does (0.956). The logvar head, conditioned on the GRU summary, has effectively become a second classifier ("how volatile do I expect this host to be"), and the 0.85-quantile of the head applied to the resulting spread is a dilation of the head's decision surface in the directions the variance model considers plausible.
4. **Is pooling producing an artificial boost?** Pooling changes the *threshold behaviour* dramatically (recall at 0.75: 0.03 mean vs 0.86 at q=0.85 vs 1.00 max) but AUC-PR only modestly (0.955 mean vs 0.960). Under the identity scaler the trajectory noise is enormous relative to ratio features (std up to e^{1.5}=4.5 on features bounded in [0, 0.5]), so the 0.85-quantile is a statistic of the head's response to essentially unphysical states. It is monotone in the head's output, which is why it preserves ranking; it is not a calibrated probability of anything.
5. **Is calibration helping or hurting?** Neither for ranking (0.958 vs 0.960); it destroys the mandated operating point because it truthfully reports that, at the validation prevalence, nothing deserves p ≥ 0.75. The calibration is right and the threshold is wrong for this prevalence.
6. **Is persistence unexpectedly strong?** Persistence is *understated* by the current comparison: the head-on-S_t persistence (0.701) is a weak classifier on a broken input; `LR on S_t` (0.766) is the fairer persistence and, at natural prevalence, is statistically tied with the production model (0.127 vs 0.135).

### 3b. Holdout split (Thursday: web attacks + infiltration; 245 positives, natural prevalence 0.036%; **measured**)

| Row | System | AUC-PR | AP-nat | ROC | R@.75 | F1*nat |
|---|---|---|---|---|---|---|
| E | LR on S_t (4k, published) | 0.751 | 0.130 | 0.904 | 0.780 | 0.192 |
| E | LR on flattened history (4k, published) | 0.865 | 0.068 | 0.945 | 0.853 | 0.159 |
| E | LR on S_t, 500k | 0.645 | 0.246 | 0.721 | 0.641 | 0.372 |
| — | shipped head on S_t = persistence | 0.618 | 0.141 | 0.694 | 0.433 | 0.212 |
| A | LR (500k) on true future | 0.874 | **0.364** | 0.952 | 0.918 | 0.499 |
| B | shipped head on true future = oracle | 0.773 | 0.347 | 0.953 | 0.604 | 0.497 |
| C | LR (500k) on deterministic rollout | 0.574 | **0.015** | 0.766 | 0.629 | 0.056 |
| D/G | shipped head on deterministic rollout | 0.69–0.70 | 0.030 | 0.86 | 0.48 | 0.12–0.14 |
| H | **production (q=0.85)** | **0.681** | **0.013** | 0.950 | 0.796 | 0.037 |
| F | **noised persistence (mu≡0), q=0.85** | **0.784** | **0.129** | 0.965 | 0.604 | 0.196 |
| H' | production, like-for-like with F | 0.688 | 0.014 | 0.954 | 0.784 | 0.037 |
| F | persistence + isotropic noise σ=0.3, mean-pool | 0.607 | 0.238 | 0.670 | 0.420 | 0.322 |
| I | production with mean pooling | 0.784 | 0.031 | 0.981 | 0.020 | 0.094 |
| I | q = 0.5 / 0.75 / 0.9 / 0.95 / 0.99 | 0.61 / 0.70 / 0.67 / 0.72 / 0.75 | 0.03 / 0.02 / 0.01 / 0.02 / 0.02 | | | |
| — | production + Platt | 0.675 | 0.013 | 0.952 | 0.020 | 0.038 |
| ENS | 1 member → 5 members; 10 → 200 traj/member | 0.626 → 0.681; 0.687 → 0.681 | | | | |

Per host (holdout, vs all sampled negatives): attacker (web attacks) production 0.51 / LR 0.55 / regularised GBDT 0.81; infiltrated `192.168.10.8`: 0.56 / 0.72 / 0.31. 25.2% of active benign windows score ≥ 0.75 under the production statistic.

On the unseen attack types the learned drift is **harmful**: removing it raises AUC-PR from 0.688 to 0.784 and natural-prevalence AP from 0.014 to 0.129; a linear classifier reading the true future gets 0.364 AP-nat, reading the *predicted* future gets 0.015 — worse than reading the current state (0.246). The transition model, trained on one attacker's dynamics, forecasts benign active hosts toward attack-like states.

---

## 4. Data, label and split problems (**measured**)

### 4.1 Window/timestamp mismatch — the load-bearing defect
- Every `Timestamp` string in the TrafficLabelling CSVs is 13–14 characters (`4/7/2017 9:20`); none carries seconds. `windowize.py:107` says "DD/MM/YYYY HH:MM:SS" — wrong.
- Consequence at Δ=30 s: `is_active` at `:30` windows is **0.0000** on test and holdout (1.0% on train, from a handful of oddly-formatted Monday rows); 0 of 697 test attack windows are at `:30`; the attacker `172.16.0.1` is active in 36% of `:00` windows and 0% of `:30` windows.
- Because gap-fill emits a zero row between every pair of real minutes, every attack episode is chopped into 1-window runs (train: 200/200; test: 695/696; holdout: 96/96). `find_episodes`, `episode_id`, lead-time onsets and the "episodes" counts all operate on these fragments.
- Half of all `risk_label=1` windows are these empty `:30` windows (50.0% of test positives). Their 45-dim state is zero except for the six backward `d_*` and four `slope3_*` features, which hold the *negated previous minute*; that is how a classifier "detects" them.
- Re-windowing at Δ=60 s with the unmodified pipeline (`07_rewindow60.py`, cache in scratchpad): active rate rises from 4.8% to 9.3% (test), the parity oscillation in persistence error disappears (RMSE by k: 2.42, 2.34, 2.33, 2.33, 2.40, 2.41 vs 4.20, 1.97, 4.16, 1.82, 4.14, 1.95), episodes become contiguous (holdout: 21- and 40-minute runs), and the natural-prevalence detectability of positives is unchanged or better (LR-4k holdout AP-nat 0.286 vs 0.183; LR-500k 0.409 vs 0.246; test 0.070 vs 0.064). Sampled-prevalence AUC-PR drops (test 0.44 vs 0.68) because the artificially easy `:30` positives are gone — which is the point.
- The packet aggregates (11 features) are computed over 30 s but attributed to a 60 s flow window, and the `:30` half is dropped by the left join. At Δ=60 this inconsistency disappears too.

### 4.2 The label is detection, not forecasting
`risk_label=1` composition relative to true episodes (attack minutes merged with ≤5 min gaps):

| Split | Positives | Inside a running episode | ≤3 min before onset | True onsets | Windows ≤10 / ≤30 min pre-onset (currently labelled 0) |
|---|---|---|---|---|---|
| train | 438 | **94.5%** | 5.5% (24 windows) | 6 | 70 / 150 |
| val | 54 | 77.8% | 22.2% (12) | 2 | 40 / 120 |
| test | 1,882 | **95.9%** | 4.1% (78) | 16 | 248 / 656 |
| holdout | 274 | 91.2% | 8.8% (24) | 5 | 77 / 149 |

Nothing at 5/10/15/30 min is a positive under the current label by construction (K=6 windows = 3 min). The 24 train pre-onset windows are on the attacker host, which is inactive between attacks: 4.9% of them are active windows, 48% of the `:00` ones are the exact zero vector. **There is no observable precursor in the attacker's own traffic** in this dataset; attacks start abruptly from scripted tools. The two multi-stage sequences that do exist (Thursday: Dropbox download → internal portscan on `192.168.10.8`; Friday: portscan → DDoS from `172.16.0.1`, 27 minutes apart) are both labelled attack throughout, so they are not available as benign-looking precursors either.

### 4.3 Supervision is one host
All 409 training positives are `172.16.0.1` (Kali attacker). The val split's 54 positives are one Heartbleed episode on the same host. Test positives are 7 internal bot hosts + the C2 server + the attacker; holdout positives are the attacker (web attacks) and `192.168.10.8` (infiltration). Labels are keyed on `src_ip` only, so the DoS/brute-force victim (`192.168.10.50`) never receives an attack label although its inbound traffic is the most dynamic thing on Tuesday and Wednesday.

### 4.4 Splits
- Day-based train/test/holdout is sound for attack episodes; 1,656 hosts are shared between train and test and 1,714 between train and holdout (the same enterprise network), so host-signature learning is possible for the attacker, which appears in every split with attack labels. Per-host test results (§11) show the production model is *worst* on that host (AUC-PR 0.41) and best on the bot hosts (0.83–0.87), so host memorisation is not what drives the headline.
- Validation is the trailing 15% of Wednesday: 54 positives, one attack type, one host. Every checkpoint- and calibration-selection decision in the pipeline rests on it.
- `risk_label` for the last 6 train windows before the val cut is computed from val windows (labels are attached before splitting). Negligible in size, worth closing.

---

## 5. Evaluation problems (**measured**)

1. **Prevalence.** The stratified cap keeps every positive and 2,147 random negatives → 46% prevalence on test, 6% on holdout, against 0.42% and 0.036% natural. AUC-PR is prevalence-dependent; the published 0.960 becomes 0.132 at natural prevalence (bootstrap [0.04, 0.24]).
2. **Negative composition.** 90.6% of sampled test negatives and 92.8% of holdout negatives are the all-zero vector; only 201 test negatives are active windows. The published precision of 0.964 is precision against zero vectors. On the 201 active negatives the production score averages 0.417 and **29.4% exceed 0.75**. Scaled to the full test day (~48k active benign windows) that is ~14k false alarms against ~1.9k positives.
3. **Operating-point leakage.** `q=0.85` was selected by test and holdout F1 (documented in the config). The threshold 0.75 is mandated, but a pooling statistic chosen to make 0.75 work on the test set is a tuned threshold by another name. §15 reports what validation alone would select.
4. **Lead time** measures available history (Run 7 already says so); "episodes" are hosts; the 2-consecutive-window rule at Δ=30 spans an empty `:30` window by construction.
5. **NRMSE** normalises by a reference std computed on a 98%-zero population and floored at 0.05; the model-vs-persistence comparison is dominated by parity (§7).
6. **Time-shuffle ablation** cannot collapse: the production score depends on the last window through the head, and on history only through the variance model; the "no collapse on test" finding is structural, not evidence about temporal learning.
7. **Per-horizon AUC-PR** in the ablations uses a different label (`future_is_attack[k]`) from the headline; the two are reported side by side as if comparable.

---

## 6. Risk-head diagnosis (**measured**, `06_heads.py`; train = the exact 500k/50k sample the shipped heads saw)

| Classifier on S_t (unless noted) | Test AUC-PR / AP-nat | Holdout AUC-PR / AP-nat |
|---|---|---|
| shipped heads (ensemble mean) | 0.701 / 0.060 | 0.618 / 0.141 |
| LR balanced, 500k | 0.559 / 0.053 | 0.645 / 0.246 |
| GBDT balanced, 500k, unregularised | 0.412 / 0.031 | 0.430 / 0.044 |
| MLP, `pos_weight`=n_neg/n_pos, select val AP (shipped regime, risk-only) | 0.553 / 0.054 | 0.660 / 0.369 |
| MLP, `pos_weight`, **1 epoch** (what the shipped selection effectively picked) | 0.687 / 0.066 | 0.584 / 0.236 |
| MLP balanced 1:10, select val AP | 0.633 / 0.067 | 0.707 / 0.368 |
| MLP balanced, last epoch | 0.520 / 0.052 | 0.667 / 0.378 |
| MLP focal γ=2 | 0.552 / 0.057 | 0.630 / 0.367 |
| MLP balanced + dropout 0.2 + wd 1e-3 / hidden 256×2 | ≈ 0.55–0.63 | ≈ 0.63–0.70 |
| **MLP balanced + proper scaling** (signed log1p on 15 heavy-tailed features, z-score fitted on *active* train rows) | **0.777 / 0.126** | **0.767 / 0.387** |
| **+ input noise σ=0.1** | 0.869 / 0.214 | 0.816 / 0.323 |
| **+ input noise σ=0.3** | **0.908 / 0.231** | **0.830 / 0.444** |
| + noise 0.3, **trained on active windows only**, zero state → 0 (3 seeds) | 0.927–0.931 / 0.210–0.213 | 0.798–0.804 / 0.315–0.337 |
| MLP balanced on GRU `h_t` (seed-0 frozen encoder) | 0.958 / 0.120 | 0.914 / 0.219 |
| MLP balanced on `[S_t, h_t]` | 0.928 / 0.133 | 0.846 / 0.240 |
| MLP balanced on `h_t` of all 5 encoders (640-d) | 0.732 / 0.061 | 0.833 / 0.403 |
| LR on flattened history (60k) | 0.339 / 0.031 | 0.601 / **0.548** |
| GBDT on flattened history (60k) | 0.419 / 0.058 | 0.508 / 0.412 |
| for reference: production world model | 0.960 / 0.132 | 0.681 / 0.013 |

What this says:
- **Loss engineering does nothing reliable.** Balanced batches, focal loss, selection metric, regularisation, capacity: all land within ±0.1 of the shipped head on both splits, with test and holdout moving in opposite directions. The epoch-0/1 selection I criticised earlier turns out to be the *best* test result among `pos_weight` runs, because longer training memorises the 409 attacker windows and transfers worse to the bot hosts.
- **Preprocessing is the lever.** The identity scaler leaves `flow_duration_mean` clipped at +10 in 82% of active rows, `ttl_mean` in 90%, `tcp_window_mean` in 88%, `iat_*` in 81–82%, `payload_size_*` in 63–69% (`feature_audit.csv`). A real scaler alone: +0.08/+0.15 (test/holdout); with input-noise augmentation: +0.21/+0.21 over the shipped head, and the best natural-prevalence AP of anything measured on test (0.231, vs 0.132 for the full production pipeline).
- **Hard negatives fix the operating point, not the ranking.** Training only on the 21,856 active windows leaves AP essentially unchanged (0.93 / 0.80) but cuts the share of active benign windows scoring ≥0.75 from 15.4% / 13.0% to 0.5% / 0.0% (test / holdout); results are stable across three seeds (±0.005 AP).
- **`h_t` is informative but not for false positives.** The frozen GRU summary reaches 0.958 at sampled prevalence yet only 0.120 at natural prevalence: it ranks positives above zero-vectors superbly and no better than the head among active benign windows. If the head is to read `h_t`, it needs the hard negatives in training (§9) to be worth anything operationally.
- Every number in this table is fragile: 409 positives from 6 episodes on one host, 54 validation positives from one episode. The test/holdout disagreement for flattened history (0.339 vs 0.601) is the same instability the published numbers show.

Which head architecture to keep: the current 45→64→1 on the *state* is the right shape for the frozen-head invariant and for SHAP. Keep it; fix its input.

---

## 7. World-model diagnosis (**measured**, `05_dynamics.py`, test split; RMSE in the model's own clipped units, mean over features)

| Predictor of S_{t+k} | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | MSE skill vs persistence |
|---|---|---|---|---|---|---|---|
| persistence S_t | 4.20 | 1.97 | 4.16 | 1.82 | 4.14 | 1.95 | 0 |
| **period-2 persistence** (S_{t+k} = S_{t+k−2}) | 1.98 | 1.97 | 1.80 | 1.82 | 1.94 | 1.95 | 0.65 |
| all-zero | 2.76 | 2.76 | 2.76 | 2.76 | 2.75 | 2.76 | 0.28 |
| **world model, deterministic (5-member mean)** | 1.65 | 1.60 | 1.69 | 1.67 | 1.74 | 1.72 | 0.73 |
| world model, seed 0 | 1.74 | 1.71 | 1.83 | 1.79 | 1.89 | 1.90 | 0.69 |
| **ridge on [S_t, S_{t−1}] → S_{t+k}** (fit on the same 500k) | **1.39** | **1.46** | **1.53** | **1.53** | **1.60** | **1.60** | **0.78** |

- Holdout repeats the pattern: persistence 1.97/0.96/1.96/0.94/1.96/1.06; period-2 0.97–1.06; world model 0.87→0.96; **ridge 0.70→0.90** at k=1→6; on attack-stage origins the world model's MSE (5.20) is worse than period-2 persistence (4.23) and ridge (3.26).
- The published "model beats persistence" (NRMSE 3.10 vs 4.16) is the odd-k half of the alternation. Against period-2 persistence the GRU's advantage is 1.65 vs 1.98 at k=1, and **a linear two-lag regression beats the GRU at every horizon**.
- Per feature at k=1, the GRU is *worse* than period-2 persistence on `active_flow_count`, `psh_ratio`, `fin_ratio`, `dst_port_entropy`, `d_dst_port_entropy` and the two constant features; it is better only on the `d_*`/`slope3_*` columns, which are deterministic functions of the history it is given.
- By origin: MSE on `:00` origins 3.12 (persistence 11.9), on `:30` origins 2.52 (persistence 9.1); on attack-stage origins 5.89 vs 24.3. The model's skill is concentrated where persistence is wrong by parity.
- `corr(true Δ, predicted Δ)` at k=1 = 0.91, mean |predicted Δ| = 1.50 vs |true Δ| = 1.71 — again parity: the model has learned "the state flips to/from zero next window".
- **Uncertainty is over-dispersed**: the 5–95% trajectory band covers 96–99% of truths (target 90%); mean trajectory std 0.77 → 1.72 across k versus mean |error| 0.5–0.7; 23.5% of (row, k, feature) cells have std > 2. This over-dispersion is precisely what makes the 0.85-quantile trick work (§3).
- Checkpoint selection uses teacher-forced validation NLL (`train_dynamics.py:184`, `teacher_forcing_p=1.0`), so free-running divergence is never measured during training; the cosine schedule (T_max=60) never anneals because early stopping fires at epoch ~16.
- At Δ=60 (no retraining possible without code changes; measured with the linear baselines only): persistence RMSE is flat across k (2.42–2.41), and the ridge two-lag model retains 30% MSE skill over persistence. So real, learnable, non-artifact dynamics exist at the minute scale — the question the retrain in §14 answers is whether a GRU captures more of it than a linear model.

Which architectural variants are worth testing, given the above: a **linear/ridge multi-horizon model** (it is currently the best forecaster and costs nothing); an **MLP transition on [S_t, S_{t−1}, S_{t−2}]** (Markov-3, no recurrence); the current GRU with free-running validation; and a direct multi-horizon head. Temporal-conv/Transformer encoders are not justified until something beats ridge.

---

## 8. Feature / preprocessing diagnosis (**measured**, `feature_audit.csv`, `09_feature_ablation.py`)

- **Scaler**: identity (`center_=0, scale_=1`, all 45). Root cause: `RobustScaler` fitted on 500k rows of which 97.9% are all-zero → 25th and 75th percentiles both 0 → IQR 0 → sklearn substitutes 1. `reference_std` (NRMSE normaliser) is therefore the raw std of a 98%-zero population, floored at 0.05 for 11 features.
- **Clip saturation** (share of *active* train rows at ±10): `ttl_mean` 0.90, `tcp_window_mean` 0.88, `flow_duration_mean` 0.82, `iat_max` 0.82, `iat_mean` 0.82, `payload_size_p95` 0.69, `payload_size_var` 0.68, `payload_size_mean` 0.63, `flow_duration_var` 0.45, `iat_var` 0.45, `slope3_iat_var` 0.45, `d_iat_var` 0.44. Twelve features are effectively binary. Nine of the eleven packet features "restored" by the join fix are among them, which is why that fix moved the headline only modestly.
- **Zero inflation**: on active rows, `rst_ratio` 99.8% zero, `frag_flag_rate` 99.9%, `local_clustering_coeff` 96%, `fin_ratio` 95.5%, `ttl_var` 91%, `psh_ratio` 89%; `reciprocity` ≡ 1 and `is_active` ≡ 1 on active rows (constant).
- **Redundancy** (|ρ|>0.9 on active rows): `flow_duration_var`≡`iat_var` (1.00), `flow_duration_mean`~`iat_max` (0.99), `iat_mean`~`iat_max` (0.99), `payload_size_mean`~`p95` (0.97), `out_degree`~`dst_ip_entropy` (0.96), `payload_size_var`~`p95` (0.94), `ttl_mean`~`tcp_window_mean` (0.92, both clipped to a constant).
- **Univariate signal** (AUC attack vs benign-active, train→test): `psh_ratio` 0.95→0.91, `active_flow_count` 0.97→0.79, `bytes_total` 0.93→0.83, `pkts_per_flow_mean` 0.91→0.83, `d_iat_var` 0.91→0.81, the `iat/flow_duration` family 0.89–0.91→0.80–0.83. Volume and push-flag features separate attacks; the graph and packet families add little (`out_degree`, `dst_ip_entropy` 0.5 train → 0.78 test — they separate *Friday's* attacks but not training's).
- **Group ablation** (GBDT on S_t, 500k, unregularised — itself overfit, so read the direction not the level): dropping `d_*/slope3_*` collapses holdout AP from 0.43 to 0.05; `only d_*/slope3_*` matches `all 45`; `only is_active` scores 0.68 on test, above `all 45` (0.41). Permutation importance is concentrated in `d_iat_var`, `active_flow_count`, `d_out_degree`; 27 features have zero importance. The delta features matter because they carry the previous minute into the empty `:30` windows — an artifact of §4.1, not a discovery.
- **Transform tests** (via the head experiments): signed log1p on the 15 heavy-tailed features + z-score on active rows: +0.08/+0.15 AUC-PR on the head; with noise +0.21/+0.21. Winsorisation/quantile transforms were not separately tested — the log1p+z-score result is large enough that the next test should be *which* transform, at Δ=60, not *whether*.

Recommended feature changes (all cheap, none change the claim): fit the scaler on active rows (zero rows map to a fixed point); log1p every count/duration/variance feature (15, not 5); drop `reciprocity`, `is_active` (as a model input; keep for masking), `rst_ratio`, `frag_flag_rate`, and one of each redundant pair; recompute the packet aggregates over the same window as the flows.

---

## 9. Sampling diagnosis (**measured**)

- 500k cap = all 409 positives + 10.8% of negatives. Of the 240 windows in the 30 minutes before a true onset on the attacking host, **126 (52.5%) are in the training sample**; 497 of the attacker's 1,349 windows are present.
- Training samples whose K-step horizon contains a benign→attack transition: **232 of 500,000** (0.05%). Since the transition target is a delta in feature space and the attacker is inactive before attacking, almost all of these are "zero → attack-level burst" jumps, which no dynamics model can anticipate from an all-zero history.
- Random vs attack-adjacent vs episode-balanced sampling therefore cannot be distinguished on this dataset by their effect on *onset* forecasting; they can only change how the dynamics model spends capacity between the 98% zero rows and the 2% active rows. The measurable version of this question is "does the dynamics model fit active-window transitions better when zero rows are down-weighted" (§14, E5). Class-distribution effects on the *head* are covered in §6: balanced sampling changed nothing reliable.
- What would change the picture is more positive *hosts*: labelling both endpoints (victim state during DoS/brute force) roughly doubles supervision and adds an internal host to training. This is a label-definition decision (§16), not a sampling knob.

---

## 10. Uncertainty / rollout diagnosis (**measured**, §3 rows F, H, I; §7 coverage)

What `q=0.85` is: for a given origin, the score is the largest value v such that at least 15% of the 1,000 sampled futures have `head(state_k) ≥ v` at some horizon k. Under the identity scaler the sampled futures are S_t + drift + N(0, σ²(h_t)) with σ up to 4.5 on features whose entire range is [0, 0.5], so the futures are not plausible network states; the statistic measures the head's response 1.04σ into the variance model's preferred directions.

Attribution of the ensemble improvement over persistence (test AUC-PR):

| Component | AUC-PR | Δ |
|---|---|---|
| head on S_t (persistence) | 0.701 | — |
| + learned drift, deterministic (D) | 0.870 | +0.17, but +0.00 on active `:00` origins (0.933 vs 0.936) → parity artifact |
| + model noise, drift **removed**, q=0.85 (F) | 0.956 | +0.26 |
| + drift restored (H', like-for-like) | 0.955 | −0.001 |
| + 5 heads on each trajectory (before-pooling) | 0.960 | +0.005 |
| trajectories/member 10 → 50 → 200 | 0.950 → 0.957 → 0.960 | variance reduction of the quantile estimate |
| isotropic noise instead of the model's logvar | 0.65–0.71 | the *structure* of the variance is what matters |

On holdout the attribution is starker: drift removal *improves* ranking (0.688 → 0.784) and natural-prevalence AP tenfold (0.014 → 0.129).

So: **(A) learned dynamics: ≈0 on test, negative on unseen attacks; (B) uncertainty modelling: ≈ all of the test gain; (C) aggregation statistic: converts B into a threshold-crossing score but adds little ranking.** The ensemble's contribution (§16) is mostly more trajectories and more heads averaging the same statistic (seed diversity: 0.905 → 0.955; head-voting on one member: 0.905 → 0.927).

Is B a legitimate forecasting signal? It is a legitimate statistic of the model's predictive distribution, and it does depend on history through `h_t`. It is not the story the project tells (a forecasted trajectory reaching compromise). The defensible statement is: *the dynamics model's predicted volatility, read through the frozen head, ranks risk better than the head on the current state.* Whether that survives a real scaler and Δ=60 is the retrain question (§14, E4).

---

## 11. Baseline comparison (same split, same label, same 4,000 rows; **measured**)

Test split. "AP-nat" is the number that would matter in deployment.

| System | AUC-PR | AP-nat | Notes |
|---|---|---|---|
| persistence (shipped head on S_t) | 0.701 | 0.060 | the published "persistence" |
| LR on S_t, 4k (published) | 0.766 | 0.117 | |
| LR on S_t, 500k | 0.559 | 0.053 | more negatives → worse; unstable |
| GBDT on S_t, 500k, regularised | 0.731 | 0.059 | unregularised: 0.412 (memorises) |
| LR on flattened 15-min history, 4k / 60k | 0.731 / 0.339 | 0.076 / 0.031 | history hurts on test, helps on holdout (0.548 nat) |
| GBDT on flattened history, 60k | 0.419 | 0.058 | |
| MLP on S_t, proper scaling + noise (new) | 0.908 | **0.231** | stable across 3 seeds (±0.005) |
| MLP on frozen GRU `h_t` (new) | 0.958 | 0.120 | a "temporal classifier" using the project's own encoder |
| **production world model** | **0.960** | 0.132 | statistically tied with LR-on-S_t at natural prevalence |
| noised persistence (mu≡0) | 0.956 | 0.129 | |
| oracle: shipped head on true future | 0.878 | 0.087 | *below* the production statistic |
| LR (4k) on true future | 0.937 | 0.083 | |

Per host (test, production model vs LR on S_t, AUC-PR against all sampled negatives): attacker `172.16.0.1` 0.41 vs 0.54; C2 server `205.174.165.73` 0.38 vs 0.63; bot hosts `192.168.10.{5,8,9,14,15}` 0.83–0.87 vs 0.49–0.56; the three hosts with ≤8 positives are noise. The world model's win is concentrated on the long, periodic bot-beaconing episodes (144–176 minutes each), where "this host's variance is high and its state is active" is the whole signal.

Does NIDRA add value *specifically through forecasting future state dynamics*? On this evidence, no: the value it adds over the best simple baseline comes from the variance model + pooling (§10), and a head with a correct scaler captures more of the natural-prevalence signal without any rollout. A GRU-hidden-state classifier (the "temporal classifier" the user asked about) matches the production ranking at sampled prevalence with no rollout at all. The forecasting contribution is currently unmeasurable because (i) the state forecast is dominated by the parity artifact and (ii) the label contains almost no pre-onset windows.

Holdout (natural-prevalence AP): LR on S_t 500k 0.246; shipped head 0.141; noised persistence 0.129; LR on flattened history 0.068; **production 0.013**; the scaled+noise head 0.31–0.44. Every simple baseline beats the production statistic on the unseen attack types at natural prevalence. Pre-onset / per-episode evaluation: §12.2.

---

## 12. Pre-onset forecasting, per-episode behaviour, and generalisation

### 12.1 Attack-family and episode transfer (**measured**, head on S_t with the fixed input transform + noise, stage 2 only)

Within the training days there are four merged attack episodes on the attacker host: FTP-Patator (Tue 09:31 local, 97 positive windows), a 6-window fragment, SSH-Patator (Tue 14:06, 130), and the Wednesday DoS block (09:45, 176; slowloris/slowhttptest/Hulk/GoldenEye merge at a 5-min gap tolerance). Leave-one-episode-out, evaluated on the held-out episode's positives against **all 21,474 active benign training windows** (base rate 0.4–0.8%):

| Held out | AP vs active negatives (base rate) | lift | recall @0.5 | same model on test / holdout (sampled AUC-PR) |
|---|---|---|---|---|
| FTP-Patator | 0.094 (0.004) | ×23 | 0.49 | 0.927 / 0.838 |
| SSH-Patator | 0.182 (0.006) | ×30 | 0.95 | 0.847 / 0.816 |
| Wednesday DoS block | 0.050 (0.008) | ×6 | 0.47 | 0.941 / 0.793 |
| train on DoS only → all brute force | 0.018 (0.011) | ×2 | 0.27 | 0.943 / 0.775 |
| train on brute force only → DoS | 0.050 (0.008) | ×6 | 0.47 | 0.941 / 0.793 |

Cross-family transfer against hard negatives is between ×2 and ×6 over the base rate — close to nothing — while the same models score 0.94 on the published test sample. The test sample cannot see this because its negatives are zero vectors and its positives are dominated by seven bot hosts whose "signal" is *any* periodic activity on an otherwise silent internal host. The classifier learns attack *signatures* of the training families plus "active host on a quiet network"; it does not learn attack dynamics that transfer between families.

Per host on test (§11): the production model is worst on the one host it trained on (`172.16.0.1`, 0.41) and best on hosts it never saw attack (bots, 0.83–0.87). That is consistent with the bot signal being "activity + volatility", not anything learned from the attacker.

### 12.2 Full-timeline evaluation (**measured**, test: 22,836 origins = every window of the 10 attacked hosts, the 8 busiest benign hosts, and 40 random hosts re-weighted ×51 to stand in for the other 2,011; ensemble 5×100 trajectories)

AUC-PR by label (weighted; the "published" row reproduces §3's natural-prevalence 0.132 independently):

| Label (n positives) | production q.85 | mean-pool | deterministic | persistence | oracle | LR S_t | GBDT S_t | LR flat | GBDT flat | GBDT [S_t,h_t] |
|---|---|---|---|---|---|---|---|---|---|---|
| published `risk_label` (1,853) | **0.134** | 0.087 | 0.059 | 0.047 | 0.074 | 0.043 | 0.027 | 0.015 | 0.051 | 0.044 |
| during-attack detection (682) | 0.057 | 0.047 | 0.033 | 0.043 | 0.038 | 0.033 | 0.033 | 0.008 | 0.028 | 0.040 |
| pre-onset ≤3 min, during-attack excluded (78) | 0.003 | 0.005 | 0.003 | 0.000 | 0.014 | 0.007 | 0.000 | 0.011 | 0.001 | 0.000 |
| pre-onset ≤5 min (130) | 0.005 | 0.005 | 0.003 | 0.000 | 0.009 | 0.009 | 0.000 | 0.011 | 0.002 | 0.000 |
| pre-onset ≤10 min (248) | 0.012 | 0.010 | 0.005 | 0.001 | 0.006 | 0.015 | 0.000 | 0.009 | 0.006 | 0.000 |
| pre-onset ≤15 min (350) | 0.019 | 0.013 | 0.007 | 0.001 | 0.006 | 0.015 | 0.001 | 0.007 | 0.005 | 0.000 |
| pre-onset ≤30 min (627) | 0.037 | 0.024 | 0.014 | 0.002 | 0.006 | 0.018 | 0.001 | 0.007 | 0.005 | 0.001 |

The base rate for the pre-onset labels is 0.4–3% of non-attack windows; every system, including the oracle that reads the true future, sits at or below it. **There is no measurable pre-onset forecasting signal at any horizon from 3 to 30 minutes, for any model.** The one place the production statistic is above the pack (≤30 min, 0.037) is the bot hosts, whose "pre-onset" windows are ordinary internal-host activity that the statistic already flags.

Per true episode (16), threshold 0.75 with the 2-consecutive-window rule, "warned" = crossing in the 30 min before onset, "detected" = crossing inside the episode; false-alarm rate = share of *all* windows on the 8 busiest benign hosts that cross:

| System (thr) | warned pre-onset | detected in episode | median latency after onset | **FA rate, busy benign hosts** | FA rate, random hosts |
|---|---|---|---|---|---|
| production q.85 (0.75) | 8/16 | 7/16 | 0 min | **78.0%** | 1.1% |
| mean-pool (0.75) | 0/16 | 0/16 | — | 1.6% | 0.0% |
| max-pool (0.75) | 14/16 | 15/16 | 0 | 100% | 94% |
| deterministic rollout (0.75) | 1/16 | 4/16 | 105 min | 16.4% | 0.3% |
| persistence, head on S_t (0.75) | 0/16 | 5/16 | 88 min | 24.0% | 0.7% |
| oracle (0.75) | 10/16 | 13/16 | 0.5 min | 66.0% | 3.7% |
| LR on S_t (0.5) | 11/16 | 12/16 | 1.25 min | 56.4% | 2.1% |
| GBDT on S_t (0.5) | 3/16 | 5/16 | 1 min | 10.7% | 0.1% |
| GBDT flattened history (0.5) | 3/16 | 9/16 | 1 min | 1.5% | 0.02% |

The published "8 of 10 episodes warned, zero false alarms at the mandated threshold" reproduces here as 8/16 warned — every one of them a bot host on which the statistic is ≥0.75 essentially permanently — at the cost of firing on 78% of the busiest benign hosts' windows. The 8 "warned" episodes and the 78% are the same phenomenon. Detection latency where it detects is 0 min (the first attack minute), i.e. it is a detector with a permissive threshold.

Dynamics "surprise" by distance to onset (k=1 MSE, world model deterministic vs persistence): busy benign hosts 5.1 / 33.4; attacked hosts >30 min before onset 5.4 / 36.1; 10–30 min 4.7 / 33.3; 3–10 min 5.0 / 28.5; ≤3 min 7.2 / 33.8; inside episode 5.3 / 35.9. The error is flat until the last three minutes, where the onset itself is inside the K-window; the published "surprise signal" (error rises before onset) is the onset entering the target, not a precursor. Persistence's 33–36 everywhere is the parity artifact again.

Holdout timeline (**measured**, 17,891 origins: 2 attacked hosts, 8 busiest benign hosts, 40 random hosts re-weighted ×59; 5 true episodes):

| Label (n) | production | mean-pool | deterministic | persistence | oracle | LR S_t | GBDT S_t | LR flat | GBDT flat | GBDT [S_t,h_t] |
|---|---|---|---|---|---|---|---|---|---|---|
| published `risk_label` (245) | 0.031 | 0.047 | 0.051 | 0.197 | 0.364 | 0.050 | 0.079 | 0.405 | **0.423** | 0.275 |
| during-attack detection (83) | 0.017 | 0.059 | 0.063 | 0.386 | 0.256 | 0.020 | 0.072 | 0.236 | **0.573** | 0.161 |
| pre-onset ≤3 / ≤10 / ≤30 min (24 / 77 / 149) | .003 / .008 / .018 | .002 / .009 / .014 | .001 / .009 / .011 | 0 / 0 / .003 | .001 / .001 / .012 | .001 / .001 / .002 | 0 / 0 / 0 | 0 / 0 / .001 | .002 / .018 / .047 | 0 / .012 / .022 |

| System (thr) | warned pre-onset | detected | median latency | **FA rate, busy benign hosts** | FA random hosts |
|---|---|---|---|---|---|
| production q.85 (0.75) | 4/5 | 4/5 | 0.5 min | **89.7%** | 0.5% |
| mean-pool (0.75) | 0/5 | 0/5 | — | 1.9% | 0.0% |
| deterministic (0.75) | 2/5 | 2/5 | 7.75 min | 14.5% | 0.07% |
| persistence (0.75) | 2/5 | 2/5 | 7.5 min | 18.0% | 0.4% |
| oracle (0.75) | 4/5 | 4/5 | 6.25 min | 57.7% | 2.0% |
| LR S_t (0.5) | 4/5 | 4/5 | 2.5 min | 69.1% | 1.0% |
| GBDT flattened history (0.5) | 4/5 | 2/5 | 7.75 min | 1.1% | 0.0% |

On the unseen attack day the production statistic is the worst ranking system on every label except max-pooling, fires on 90% of the busy benign hosts' windows, and its four "warned" episodes are again hosts on which it is permanently above threshold. The flattened-history GBDT is the best detector here (0.57 during-attack, 1% false alarms) — the same LR-flat-wins-on-holdout pattern the project already reports, now visible at natural host prevalence. Dynamics error by distance to onset is flat (WM 3.6–6.9 across all bins; persistence 18–42, parity-dominated).

---

## 13. Highest-impact opportunities

Ranked by expected **legitimate** improvement in out-of-sample risk ranking at natural prevalence and in the defensibility of the world-model claim — not by ease. "Changes claim" = whether the project's stated mechanism changes.

| # | Intervention | Expected metric impact | Evidence | Compute | Leak/overfit risk | Changes claim |
|---|---|---|---|---|---|---|
| A1 | **Honest evaluation harness**: natural-prevalence weighting, active-negative composition, per-episode bootstrap, pre-onset labels at 3/6/15/30 min, period-2 persistence, validation-only q/threshold | Headline AUC-PR 0.96 → ≈0.13 (test, natural); precision@0.75 0.96 → ≈0.12; all comparisons become meaningful | §3, §5, bootstrap | minutes | removes leakage (q chosen on test) | No — it corrects what is claimed |
| A2 | **Δ = 60 s windows** (match timestamp resolution); recompute packet aggregates on the same window | Removes the parity artifact; sampled AUC-PR falls (easy `:30` positives vanish), natural-prevalence AP flat-to-better (measured with linear models: holdout 0.18→0.29, 0.25→0.41); state-forecast baselines become honest | §4.1, §7, `11_delta60_compare.py` | re-window 30 min + full retrain | none | No, but every recorded number is invalidated (Δ change) |
| A3 | **Real scaler**: fit on active rows, log1p all 15 heavy-tailed features, drop constants/duplicates | Head AUC-PR +0.08/+0.15 (test/holdout) measured; dynamics: 12 currently-binary features become informative (unmeasured, expected large) | §6, §8 | retrain | none | No |
| A4 | **Input-noise augmentation for the head** (scaled space, σ≈0.3), balanced batches, separate optimisers/selection | +0.13/+0.06 on top of A3 (test 0.908, AP-nat 0.231 vs production 0.132); stable across seeds | §6 | minutes | small (val has 54 positives — select on AP with early stop, not on loss) | No |
| A5 | **Validation-only operating point**, calibrated score reported at the mandated 0.75 | F1@0.75 changes (§15); removes a documented test-set choice | config comment | none | removes leakage | No |
| B1 | **Retrain dynamics at Δ=60 with A3, free-running validation loss, period-2/ridge/MLP-Markov baselines** | Unknown; ridge has 0.30 skill at Δ=60 — the GRU must beat it or be replaced | §7 | hours | none | Possibly — if the GRU loses to ridge, the "learned dynamics" claim narrows to "learned volatility" |
| B2 | **Variance objective**: β-NLL or logvar cap, target 90% coverage | Calibrated bands; production statistic's advantage over the head may shrink (that would be the honest number) | §7, §10 | retrain | none | Sharpens it |
| B3 | **Head trained on active windows** (hard negatives), zero state → fixed low score | Active-benign false-alarm rate at 0.75: 15% → <1% (measured), ranking unchanged | `hard-negative` run | minutes | none | No |
| B4 | **Head reads `[S_t, h_t]`** from the frozen encoder | Sampled AUC-PR up (0.93–0.96), natural-prevalence AP not (0.12–0.13) unless combined with B3 | §6 | minutes | encoder becomes a second path for temporal signal | Yes — persistence baseline must then use the same head; record in `DECISIONS.md` |
| B5 | **Both-endpoint labels** (victim state during DoS/brute force) | Doubles positive windows, adds internal hosts to training; per-host holdout transfer likely improves | §4.3 | re-label + retrain heads | label semantics change — must be a decision, not a tune | Yes (what "risk" means) |
| B6 | **Multi-horizon risk targets** at Δ=60 (3/6/15/30 min), reported per horizon | Makes the "forecast" a measured curve; expect near-base-rate at ≥15 min | §4.2 | heads only | none | Sharpens it |
| C1 | Attack-adjacent / episode-balanced dynamics sampling | Small: 232 transitions exist in 500k | §9 | retrain | none | No |
| C2 | TCN/Transformer encoder | Not justified until B1 shows the GRU beats ridge | §7 | hours | overfit (409 positives) | No |
| C3 | More ensemble members / trajectories | 10→200 traj: +0.01; 1→5 members: +0.05; diminishing | §3 | linear | none | No |
| C4 | More attack episodes (a second dataset or synthetic replays) | The only route to a pre-onset claim; out of scope for this build by `CLAUDE.md` | §4.2 | large | new | Yes |

**Tier A** = A1–A5: high confidence, and A1/A2/A3 are prerequisites for measuring anything else. **Tier B** = B1–B6: promising, each needs the experiment in §14. **Tier C** = speculative.

---

## 14. Experimental roadmap (cheapest disconfirming experiment first; one change per experiment)

**E0 — Honest evaluation (no retraining).**
Change: `nidra/eval/metrics.py` gains prevalence-weighted AP/PR curves and per-episode bootstrap; `run_eval.py` reports both sampled and natural prevalence, builds the eval negatives as *all active windows* + a random sample of inactive ones, adds pre-onset labels at 3/6/15/30 min from merged episodes, adds period-2 persistence and ridge baselines to `ablations.py`, and selects q/threshold on val only. `lead_time_runner.py` reports detection latency after onset and the ceiling ratio. Split: unchanged. Metric: everything above. If the hypothesis "the headline is prevalence-inflated" is true: test AP-nat ≈ 0.13, precision@0.75 ≈ 0.12, active-benign FA rate ≈ 29%. If false: AP-nat > 0.5. Invalidates: the scorecard's precision/F1 rows and the persistence-ablation text either way.

**E1 — Head input transform + augmentation on the existing frozen dynamics (stage 2 only, ~10 min).**
Change: `RiskHead` gets a fixed, non-trainable preprocessing layer (signed log1p on the heavy-tailed indices, mean/std fitted on active train rows, stored with the scaler); `train_heads.py`: balanced batches, Gaussian input noise σ=0.3, risk and stage heads on separate optimisers/clips, selection on val AP with patience. Config: `train_heads.input_noise`, `train_heads.sampling`. Metric: head-on-S_t and production-statistic AP-nat on test/holdout (E0 harness). Expected if true: head 0.06 → ≈0.2 AP-nat on test (measured 0.231 in the scratchpad version); production statistic ≥ that. If false: production statistic *drops* because the rollout noise, now viewed through the log/z transform, lands off-manifold — that result would say the pooling statistic depends on the identity scaler and must be re-derived. Invalidates: "the head is the ceiling" if the production statistic does not follow the head up.

**E2 — Δ=60 + real scaler + dead-feature removal, full retrain (hours).**
Change: `config/default.yaml` `windowing.window_seconds: 60` (L=30 → 30 min history, K=6 → 6 min; or L=15/K=3 to keep the PRD's spans — choose one and say so); `normalize.py` fits on active rows; `schema.py` `LOG1P_FEATURES` = 15 heavy-tailed features, drop `reciprocity`, `rst_ratio`, `frag_flag_rate`, `is_active` as inputs (schema version bump; `FEATURE_ORDER` length assert updated in one place); `windowize.py` packet aggregation over `window_seconds` (already parameterised — verify the left-join no longer discards). `train_dynamics.py` validation with `teacher_forcing_p=0` alongside the current one. Split: same days. Metric: state-forecast skill vs persistence, period-2 and ridge at k=1..6; production-statistic AP-nat. Expected if "the GRU learns real minute-scale dynamics": skill > 0.30 (ridge) and per-feature skill positive on active-window features; production AP-nat > E1's head-alone. If false: skill ≤ ridge → E2b: replace the transition with ridge/MLP-Markov and re-measure; the world-model claim becomes "learned predictive distribution", not "learned trajectory". Invalidates every number in `REAL_DATA_RESULTS.md` (Δ change) — by design.

**E3 — Attribution at Δ=60 (no extra training).** Re-run `04_decomposition.py` logic: noised persistence vs production, deterministic vs stochastic, by origin activity. Expected if the variance model is the mechanism: mu≡0 still matches production. If the drift now matters: gap > bootstrap CI. Decides whether B2 (variance objective) or the transition model is the next investment.

**E4 — Variance objective sweep (retrain stage 1, 3–4 runs).** `logvar_max ∈ {0, 1, 1.5}` and β-NLL (β=0.5). Metric: 90% band coverage, state skill, production AP-nat. Expected if the pooling gain is over-dispersion: coverage → 90% and the production-vs-head gap shrinks toward zero; if the variance carries real signal, the gap survives calibration.

**E5 — Hard-negative head and the `h_t` question (stage 2 only).** Train the head on active windows only (B3), then on `[S_t, h_t]` (B4), each with E1's regime. Metric: AP-nat and false-alarm rate on active benign windows, per host. Expected if `h_t` carries operational signal: AP-nat up *and* FA down; if it only separates positives from zero vectors (as measured now), AP-nat flat — then keep the head on `S_t` and preserve the cleaner claim.

**E6 — Leave-one-attack-out within train (stage 2 only).** Hold out each Tue/Wed attack type (FTP, SSH, slowloris, slowhttptest, Hulk, GoldenEye, Heartbleed) when training the head; evaluate on the held-out episode's windows + that day's active negatives. Expected if the head learns "attack-like traffic": AP well above base rate for held-out types; if it memorises signatures: near base rate for the dissimilar ones (DoS ↔ brute force). This is the cheapest measurement of the generalisation claim and needs no test data.

**E7 — Both-endpoint labels (decision + stage 2 + eval).** Measure the added positives/hosts first (Tue/Wed victims); retrain heads; compare per-host holdout. Document the semantic change in `DECISIONS.md` before looking at metrics.

**E8 — Multi-horizon targets (stage 2 + eval, after E2).** Heads for P(attack within 3/6/15/30 min) applied to the rolled-out states at the matching k; report the curve. Expected: ≥15-min AP near base rate on this dataset (no precursors); the honest published figure.

Experiments not worth running now: TCN/Transformer encoders (C2), temporal GNN (out of scope), sampling curricula for dynamics (C1) before E2 shows dynamics are learnable at all.

---

## 15. Expected metric ranges — **estimates**, not measurements

Anchors: measured test AP-nat today = 0.13 (production), 0.23 (scaled+noise head, stage-2 only); holdout AP-nat 0.44 for the same head. Ranges are my judgement from those anchors and the linear-baseline results at Δ=60; the test/holdout disagreement (a factor of 2–3 for the same model) is the dominant uncertainty and no experiment below will shrink it — only more attack episodes would.

| After | Test AUC-PR (sampled) | Test AP-nat | Holdout AP-nat | State skill vs ridge (Δ=60) | Notes |
|---|---|---|---|---|---|
| E0 only | 0.96 (unchanged) | 0.13 | 0.1–0.2 | — | numbers become honest, not better |
| E1 | 0.90–0.97 | 0.18–0.28 | 0.3–0.5 | — | head alone measured at 0.23 / 0.44 |
| E2 (Δ=60 + scaler), head alone | not comparable (new positives) | 0.15–0.35 | 0.3–0.5 | GRU: −0.1 to +0.15 relative to ridge | the honest world-model test |
| E2 + production statistic | | ≥ head alone if the variance signal is real; else ≈ head | | | E3 decides |
| E5 (hard negatives) | | flat | flat | | FA at 0.75 on active benign: 15% → <1% (measured at Δ=30) |
| E8 pre-onset ≥15 min | | ≈ base rate (0.005–0.05) | ≈ base rate | | expected negative result; publish it |

Pre-onset at 3–6 min after E2: 0.05–0.2 AP-nat is plausible only for the bot hosts (periodic beaconing already under way); for the attacker host, ≈ base rate.

### 15.1 Operating point chosen on validation only (**measured**, `10_operating_point.py`; capped val set = 54 positives, natural prevalence 0.0094%)

| Pooling | val AP-nat | val best-F1 (nat) | thr* | test AP-nat | test F1@thr* (nat) | test F1@0.75 (nat) | holdout AP-nat | holdout F1@thr* | holdout F1@0.75 |
|---|---|---|---|---|---|---|---|---|---|
| mean | 0.021 | 0.036 | 0.76 | 0.108 | 0.051 | 0.054 | 0.031 | 0.010 | 0.013 |
| P(traj > 0.5) | 0.021 | 0.036 | 0.79 | 0.113 | 0.054 | 0.063 | 0.034 | 0.010 | 0.012 |
| q=0.75 | 0.002 | 0.017 | 0.96 | 0.114 | 0.097 | 0.224 | 0.018 | 0.041 | 0.044 |
| **q=0.85 (shipped)** | 0.002 | 0.005 | 0.15 | 0.132 | 0.113 | **0.206** | 0.013 | 0.014 | **0.029** |
| q=0.95 | 0.002 | 0.005 | 0.49 | 0.137 | 0.114 | 0.169 | 0.015 | 0.015 | 0.024 |
| max | 0.003 | 0.007 | 1.00 | 0.089 | 0.166 | 0.009 | 0.011 | 0.022 | 0.001 |

Validation would select mean or P(>0.5) pooling with a threshold near 0.76–0.79 — the setting the project measured and rejected because its test F1 at 0.75 was 0.010. The shipped q=0.85 scores worst of all on validation and best on test at 0.75, which is what selection on the test set looks like. **At natural prevalence, the published test F1 of 0.906 becomes 0.206 with the shipped (test-chosen) operating point and 0.05–0.11 with any validation-chosen one; holdout 0.029 and ≈0.01.** The validation split (one Heartbleed episode) cannot support any operating-point decision; E0 must include a validation set with more than one attack episode (e.g. hold out one of the four Tue/Wed episodes by time, not the trailing 15% of Wednesday).

---

## 16. What should NOT be done

- Do not fine-tune heads on rolled-out states, and do not let the head read the rollout's *own* variance as a feature at training time (it would learn the pooling trick directly). The frozen-head invariant is the one thing that keeps the statistic interpretable.
- Do not choose `q`, thresholds, `logvar_max` or trajectory counts by test/holdout metrics again. The config already documents one such choice; E0 replaces it.
- Do not keep Δ=30 s and "smooth over" the `:30` windows (interpolation, carrying forward the previous minute, dropping them). Any of those leaks the next minute's flow timing or manufactures dynamics. Change Δ.
- Do not fit the scaler on the zero rows — but do keep the zero rows in the dynamics data (silent windows are real state); they just must not define the scale.
- Do not switch to CSE-CIC-IDS2018 in this build (`CLAUDE.md`); do not train on Thursday. Note separately (§13 C4) that the pre-onset claim is data-limited, not model-limited.
- Do not add `h_t` to the head to beat LR-on-flattened-history on holdout unless E5 shows a natural-prevalence gain; the sampled-prevalence gain is real but operationally empty.
- Do not report lead time in hours or minutes from the current runner; report detection latency after onset and the ratio to the available-history ceiling.
- Do not use the unregularised GBDT (or any high-capacity tabular model) as a "strong baseline" on 409 positives; it scores 0.41 and proves only that it memorises. Regularised GBDT or LR are the fair strong baselines here.
- Do not compare NRMSE to naive persistence without period-2 persistence beside it, and never across a Δ change.
- Do not relabel to raise metrics. Both-endpoint labels (E7) are defensible on their merits; adopt them as a decision, before the metrics are computed.

---

## 17. Revised recommendation

**On my previous conclusion.** "The risk head is the ceiling and should be fixed first" was wrong in its diagnosis and half-right in its prescription:
- Wrong: the head's *training regime* (pos_weight, joint stage loss, epoch-0 selection) is not what limits it. Every regime change I tested landed within noise, and the epoch-0 head is among the best on test. The oracle-below-LR observation that led me there is real (0.878 vs 0.937) but irrelevant to the headline, which the pooling statistic already pushes above any true-future classifier.
- Half-right: the head's *input* limits it, and fixing the scaler plus noise augmentation is the single largest measured gain in this audit (test AP-nat 0.06 → 0.23; holdout 0.14 → 0.44). That is a preprocessing fix that happens to be applied at the head, not a head fix.
- What I missed entirely: the 30-second window against minute-resolution timestamps, the identity scaler, the prevalence-inflated evaluation, and the fact that the world model's advantage is its variance head, not its dynamics. Each of these is larger than anything about the head.

**Optimal strategy, in order.**

1. **Make the numbers true before making them better** (E0). Every current comparison — model vs baseline, model vs persistence, calibrated vs raw — is computed on a 46%-prevalence sample whose negatives are 90% zero vectors, with a pooling statistic picked on the test set. Nothing can be optimised legitimately on that harness. This will lower the headline by roughly an order of magnitude and that is the correct outcome.
2. **Fix the two data defects together and retrain once** (E2 = A2 + A3): Δ=60 s and a scaler fitted on active rows with log1p on all heavy-tailed features. State it as a Δ change that invalidates prior artifacts, per `CLAUDE.md`. Add period-2 persistence and ridge as mandatory dynamics baselines and free-running validation NLL at the same time so the retrained model is judged on what it does at inference.
3. **Apply the head fixes** (E1/A4, B3) inside that retrain: fixed input transform, noise augmentation, balanced batches, separate selection, hard negatives. Keep the head on the state; keep it frozen.
4. **Re-run the attribution** (E3, E4). If mu≡0 still matches the production statistic after Δ=60 and real scaling, the transition mean is not earning its place: replace it with the simplest model that beats ridge, or reframe NIDRA's mechanism as *learned predictive uncertainty over host state*, which is what the evidence currently supports and is still a world-model formulation. If the drift now contributes, the claim stands and the roadmap continues to E5–E8.
5. **Reframe the claim to what CIC-IDS2017 can support**, regardless of the outcome of 4: state forecasting skill at minute scale (measurable), risk ranking at natural prevalence with a stated false-alarm rate (measurable), detection latency within episodes (measurable), and an explicit statement that pre-onset warning beyond ~3 minutes is unsupported because the dataset contains six training onsets on one host with no observable precursor traffic. A team that says this unprompted is more credible than one whose 0.96 evaporates under the first question about prevalence.

**What I would not do:** spend any more effort on head losses, ensembling, trajectory counts, calibration variants or encoder architectures on the current representation. The measured headroom there is zero to negative, and every hour on it is an hour on an artifact.

*The Δ=60 retrain (E2/E3's first data point) finished after the document was first delivered; §18 below reports it and revises point 4 of this section.*

---

## 18. Addendum — single-seed retrain at Δ=60 s (**measured**, `12_delta60_retrain.py`, `out60/retrain60_results.json`)

Setup: the project's own `prepare_training_data` → `train_one_seed` → `train_heads_for_seed`, code unmodified, config `window_seconds: 60` (L=30 → 30 min history, K=6 → 6 min horizon), 500k/50k stratified sample (205 positives), seed 0 only, artifacts in the scratchpad. Everything else — including the identity scaler and the epoch-0 head selection (val AUC-PR 0.044) — is as shipped. Dynamics early-stopped at epoch 21 (best teacher-forced val NLL −0.988 at epoch 13; the curve is flat from epoch 3). Evaluated on the Δ=60 capped sets (test 941 positives, natural prevalence 0.51%; holdout 122, 0.041%), 200 trajectories. Sampled-prevalence AUC-PR is **not** comparable to the Δ=30 tables (different positives); the within-Δ=60 attribution and the natural-prevalence AP are.

State forecasting (RMSE, mean over features; skill = 1 − MSE/MSE_persistence):

| Predictor | test k=1 → k=6 | test skill | holdout k=1 → k=6 | holdout skill | MSE on active origins (test / holdout) |
|---|---|---|---|---|---|
| persistence | 2.42 → 2.41 (flat) | 0 | 1.43 → 1.61 | 0 | 14.8 / 12.0 |
| period-2 persistence | 2.34 → 2.41 | 0.015 | 1.44 → 1.61 | −0.002 | — |
| ridge on [S_t, S_{t−1}] | 2.04 → 2.09 | 0.295 | 1.22 → 1.39 | 0.273 | 10.0 / 7.7 |
| **GRU world model, deterministic** | **1.66 → 1.99** | **0.391** | **1.04 → 1.31** | **0.383** | **8.4 / 6.0** |

The parity artifact is gone (period-2 ≡ persistence), the error now grows monotonically with horizon, and **the GRU beats the linear two-lag model at every horizon on both splits**, including on active origins. Trajectory std grows 0.64 → 1.53 across k (a forecast cone), against 0.77 → 1.72 at Δ=30 where k=1 was already parity-inflated.

Risk ranking, single seed × 200 trajectories:

| System | test AUC-PR | test AP-nat | test active-only AUC-PR | FA active benign ≥0.75 | holdout AUC-PR | holdout AP-nat | holdout active-only | FA |
|---|---|---|---|---|---|---|---|---|
| head on S_t (persistence) | 0.552 | 0.108 | 0.784 | 2.9% | 0.461 | 0.313 | 0.570 | 2.5% |
| head on true future (oracle) | 0.805 | 0.171 | — | — | 0.688 | 0.354 | — | — |
| head on deterministic rollout | 0.651 | 0.107 | — | — | 0.509 | 0.286 | — | — |
| **noised persistence (mu≡0), q=0.85** | 0.795 | 0.140 | 0.878 | 9.4% | 0.610 | 0.276 | 0.682 | 9.5% |
| **world model, q=0.85** | **0.880** | 0.164 | 0.905 | 10.8% | **0.714** | 0.283 | 0.756 | 10.2% |
| **world model, mean pooling** | **0.903** | **0.170** | **0.911** | **2.0%** | **0.758** | **0.326** | **0.786** | **0.0%** |
| LR (4k) on S_t | 0.439 | 0.070 | — | — | 0.614 | 0.286 | — | — |
| LR (4k) on true future / on det rollout | 0.829 / 0.483 | 0.150 / 0.069 | — | — | 0.849 / 0.612 | 0.311 / 0.195 | — | — |

Episode-block bootstrap (300 resamples; 15 test episodes, 5 holdout): **world model q.85 − noised persistence = +0.086 [+0.058, +0.126] (test), +0.096 [+0.033, +0.163] (holdout)**; world model mean − persistence = +0.348 [+0.279, +0.432], +0.270 [+0.058, +0.463].

What changes:
- **§3 answer 2 and §10 attribution (A) are revised.** At Δ=30 the drift contributed nothing (test) or harm (holdout); at Δ=60 it contributes +0.09 AUC-PR on both splits with intervals excluding zero, on top of the variance/pooling effect, and the world model now beats its own oracle less by pooling and more by forecasting (deterministic-rollout head 0.651 vs persistence 0.552). The transition model earns its place once the window matches the data.
- **§17 point 4 resolves in favour of keeping the transition model.** E2 is confirmed as the first thing to do; the "learned volatility" reframing is not needed.
- **Mean pooling is now the best statistic on both splits** (0.903 / 0.758) and has the lowest false-alarm rate on active benign windows (2.0% / 0.0%, versus 29% / 25% for the Δ=30 production statistic). The quantile trick was compensating for the artifact. Recall at 0.75 under mean pooling is low (0.03 / 0.30), so the operating point must be set on validation with a calibrated score — the head's probabilities, not the pooling rule, are what need fixing at the threshold.
- Still true and still to do: the head is the weak link (persistence 0.55; still epoch-0 selected with val AUC-PR 0.044), the scaler is still the identity, and the label is still 95% during-attack. The stacking of A3/A4 on this retrain is E2's second step, not measured here. Single seed, 200 trajectories, 5 holdout episodes: treat the holdout interval as wide.

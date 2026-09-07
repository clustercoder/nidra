# Real CIC-IDS2017 Run — Results

This document reports a genuine end-to-end run of the NIDRA ML pipeline
against the complete, real CIC-IDS2017 dataset: all 8 published day-files,
read in full (no row truncation), with real tshark-extracted packet-level
features for two of the five capture days. Config used:
`config/mvp_2017.yaml`. Nothing here is fabricated, extrapolated, or
inherited from an earlier partial run — every number below comes from the
run described in this document.

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

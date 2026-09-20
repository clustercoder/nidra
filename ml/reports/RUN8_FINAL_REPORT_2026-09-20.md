# NIDRA — Run 8 final report (2026-09-20)

The roadmap of 2026-09-19 (`reports/NIDRA_REEVALUATION_2026-09-19.md`) was implemented in
the worktree in the mandated order. This report lists what changed, what was run, what
was measured, and what remains. Every number here is copied from a record under
`artifacts/metrics/` or `experiments/runs/`; the tables in `reports/run8/benchmark_tables.md`
are generated from the same records. Nothing was tuned on test or holdout; nothing was
relabelled; no attack class is omitted.

## 1. Files changed

Since the frozen baseline (`git tag baseline-delta30-run7`): 105+ files, ~9,600 insertions
(`git diff --stat baseline-delta30-run7..HEAD`). Grouped:

| area | files |
|---|---|
| data | `nidra/data/{join,windowize,splits,dataset,normalize,labels,schema}.py` (Δ=60 canonical, flow/packet fusion, validation cut with pre-onset margin, feature-specific scaler); new `onset.py` (episode geometry, onset targets), `attack_mapping.py`, `audit.py`, `preprocessing_audit.py` |
| models | `heads.py` (+`OnsetHead`), `transition.py` (+optional two-lag linear skip, off), `world_model.py`, `risk_pooling.py` (numpy pooling shared by eval and serving) |
| training | `losses.py` (β-NLL, MSE aux, sample weights, free-running metrics), `train_dynamics.py` (free-running selection), `train_heads.py` (rewritten recipe + independent selection), new `head_data.py`, `train_onset.py`, `pipeline.py` |
| evaluation | new `benchmark.py`, `eval_set.py`, `metrics_natural.py`, `operating_point.py`, `systems.py`, `state_metrics.py`, `episode_metrics.py`, `gru_classifier.py` |
| explainability | new `explain/forecast_attribution.py` (IG + deletion faithfulness); `counterfactual.py` |
| serving | `serve/predictor.py` (operating point, risk curve, progression → ATT&CK, `forecast_batch`, forecast attributions), `serve/benchmark.py` |
| offline pipeline | new `cli/forecast.py`, `cli/report_html.py` |
| provenance / scripts | new `utils/provenance.py`, `scripts/run_experiment.py`, `report_tables.py`, `plot_benchmark.py`, `write_baseline_manifest.py`; `demo_forecast.py` |
| config | `ml/config/default.yaml` (geometry, loss, head recipe, onset, splits, benchmark caps), repo `config/default.yaml` (Δ=60, scaler path) |
| docs | `REAL_DATA_RESULTS.md` (Run 8 + rewritten scorecard), `MODEL_CARD.md` (rewritten), `README.md` (root; results rewritten), `ml/README.md` + `ML_README.md`, `EVALUATION.md`, `TRAINING.md`, `ARCHITECTURE.md`, `PRODUCTION_RUN_GUIDE.md`, `CLAUDE.md`, `DECISIONS.md` (D102–D115), `docs/ATTACK_MAPPING.md`, `reports/{DATASET_ASSESSMENT,NTRO_COMPLIANCE}_2026-09-20.md`, `artifacts/metrics/README.md` |
| web | site copy and charts follow Δ=60 and the measured numbers (`hero.tsx`, `experiments.tsx`, `evidence-teaser.tsx`, `console-mock.tsx`, `forecast-cone-svg.tsx`, `how-nidra-works.tsx`, `narrative-carousel.tsx`, `world-model-stages.tsx`, `hero-art.tsx`, `capture-upload.tsx`); `scripts/make_demo_replay.py` reads geometry and pooling from the served model |
| tests | 24 new/extended test modules (attack mapping, onset, head data, onset head, CLI, forecast attribution, benchmark, operating point, splits margin, predictor surfaces, report tables, figures, …) |
| artifacts | `artifacts/weights/` (5 checkpoints + 5 onset heads + GRU baseline + `operating_point.json`), `artifacts/scaler/feature_scaler.json` + audit, `artifacts/metrics/{val,test,holdout,test_K10}/`, `experiments/runs/*/{config.yaml,record.json,…}`, `experiments/BASELINE_MANIFEST_delta30_run7.json` |

## 2. Experiments run (all recorded under `ml/experiments/runs/<label>/{config.yaml,record.json,artifacts/}`)

| label | what | outcome |
|---|---|---|
| `geomA_L15K3` | Δ=60, L=15/K=3, plain NLL, seed 0, 30 epochs, Δ=30-era head recipe | geometry arm A |
| `geomB_L30K6` | Δ=60, L=30/K=6, plain NLL, seed 0, early-stopped at 23 | geometry arm B |
| `headsA_new`, `headsA_bal_nonoise`, `headsA_imb_noise`, `headsA_bal_r5_lr3`, `headsB_imb_noise` | head-recipe comparison on one checkpoint (`--init-from`) | D108: imbalanced + input noise, natural-AP selection |
| `var_plain`, `var_linskip`, `var_betanll`, `var_mseaux`, `var_nonsilent`, `var_linskip_betanll`, `var_betanll_nonsilent` (+ `_h2` uniform head pass) | Stage-5 dynamics variants at L=15/K=3, 20 epochs, seed 0 | D113: β-NLL 0.5 kept |
| `production` | L=30/K=6, β-NLL, 5 seeds × 24 epochs, heads + onset head on every split row, GRU classifier baseline | the shipped artifacts |
| `lodo_without_wednesday`, `lodo_without_tuesday` | leave-one-day-out retrains (seed 0) | generalisation, §Generalisation |

Every benchmark record (`artifacts/metrics/<split>/benchmark.json` per run) carries git
commit, config hash, dataset digests, geometry, seeds, checkpoint sha256, operating point,
caps and the eval seed.

## 3. Commands used

```bash
# data (once): canonical Δ=60 tables, fusion fix, audit
python -m nidra.data.audit --config config/default.yaml --out artifacts/metadata/data_audit_w60.json
# recorded experiments
python -m nidra.scripts.run_experiment --label geomA_L15K3 --seeds 0 --stages dynamics,heads --epochs 30 \
    --set windowing.context_length=15 --set windowing.horizon_length=3 --set windowing.min_windows_per_host=18 --set labels.risk_threshold_windows=3
python -m nidra.scripts.run_experiment --label geomB_L30K6 --seeds 0 --stages dynamics,heads --epochs 30
python -m nidra.scripts.run_experiment --label headsA_imb_noise --base-config experiments/runs/geomA_L15K3/config.yaml --init-from geomA_L15K3 \
    --stages heads --set train_heads.risk_sampling=imbalanced --set train_heads.input_noise=0.3 --set train_heads.selection_metric=val_auc_pr_natural
python -m nidra.scripts.run_experiment --label var_betanll --base-config experiments/runs/geomA_L15K3/config.yaml --seeds 0 --stages dynamics,heads --epochs 20 --set train_dynamics.beta_nll=0.5 ...
python -m nidra.scripts.run_experiment --label production --seeds 0,1,2,3,4 --stages dynamics,heads,onset,gru_baseline --epochs 24
# benchmark: validation selects and freezes the operating point; test/holdout only load it
python -m nidra.eval.benchmark --split val --select-operating-point --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split test --n-samples 60 --n-resamples 300
python -m nidra.eval.benchmark --split holdout --n-samples 60 --n-resamples 300
# horizon extension check and tables/figures
python -m nidra.eval.benchmark --split test --n-samples 60 --set windowing.horizon_length=10 --set labels.risk_threshold_windows=10 --set windowing.min_windows_per_host=40 --out-dir artifacts/metrics/test_K10
python -m nidra.scripts.report_tables --run . --splits val,test,holdout
python -m nidra.scripts.plot_benchmark --run . --splits test,holdout --out reports/run8
# offline pipeline and demo
python -m nidra.cli.forecast --csv <CICFlowMeter.csv> [--pcap capture.pcap] --out out/
python -m nidra.cli.report_html --run out/
python -m nidra.scripts.demo_forecast --day friday_portscan
python -m nidra.data.attack_mapping --write
pytest tests/ -q
```

## 4. Final architecture

```
flows (CICFlowMeter CSV) + packets (tshark parquet)
  → per-host per-minute state, 45 features (15 flow, 11 packet, 8 graph, 11 dynamics),
    silent minutes emitted as is_active=0 states                    nidra/data/windowize.py
  → feature-specific scaling fit on active TRAIN rows                nidra/data/normalize.py
  → GRU(2×128) encoder over L=30 minutes → h_t                       nidra/models/encoder.py
  → transition: μ, log σ² of the next-state delta (β-NLL 0.5)        nidra/models/transition.py
  → K=6 recursive stochastic rollout, own predictions fed back        nidra/models/world_model.py
  → frozen heads on every sampled future state: risk (45→64→1),
    stage (45→64→6); onset head (45→64→6 horizons) on S_t            nidra/models/heads.py
  → pooling statistic + per-horizon Platt + threshold, all frozen
    on validation (operating_point.json)                              nidra/eval/operating_point.py
  → forecast(): risk curve with band, progression → ATT&CK tactics,
    attributions (SHAP on S_t, IG through the rollout + faithfulness) nidra/serve/predictor.py
```


## 5. Final metrics (natural prevalence, 5-seed ensemble, operating point frozen on validation)

| | Historical (Run 7: Δ=30, balanced subsample, pooling tuned on test) | Corrected protocol — strongest baseline (Run 8) | **Final NIDRA (Run 8)** |
|---|:---:|:---:|:---:|
| Test AP | 0.960 | 0.164 (GRU sequence classifier) | **0.058** [0.018, 0.199] |
| Test P / R / F1 | 0.964 / 0.855 / 0.906 | 0.49 / 0.03 / 0.05 | 0.89 / 0.03 / 0.06 |
| Holdout AP | 0.682 | 0.443 (LR on 30-min history) | **0.439** [0.000, 0.768] |
| Holdout P / R / F1 | 0.731 / 0.800 / 0.764 | 0.68 / 0.53 / 0.60 | 0.85 / 0.32 / 0.46 |
| Validation AP (2 episodes) | — | 0.756 (persistence + learned noise) | 0.726 |
| False alarms / h, test / holdout (all hosts) | — | — | 0.48 / 0.86 |
| Episodes warned before onset | "8 of 10" | 0 | 0 of 15 / 0 of 5 |
| State-forecast skill vs persistence, test / holdout | — | 0.565 / 0.595 (ridge two-lag) | **0.587 / 0.616** |

Thresholds: validation F1-optimal 0.718 (Run 8), mandated 0.75 (Run 7); at 0.75 Run 8 is
within 0.01 F1 of the numbers above on every split.

## 6. Baseline comparisons (AP on the published label)

| system | val | test | holdout |
|---|---|---|---|
| NIDRA (stochastic, calibrated) | 0.726 | 0.058 | 0.439 |
| NIDRA deterministic | 0.748 | 0.081 | 0.380 |
| persistence + learned noise | 0.756 | 0.057 | 0.383 |
| persistence + isotropic noise | 0.734 | 0.064 | 0.293 |
| persistence (risk head on S_t) | 0.749 | 0.065 | 0.295 |
| ridge two-lag + risk head | 0.693 | 0.083 | 0.275 |
| oracle (head on the true future) | 0.770 | 0.117 | 0.400 |
| LR on S_t | 0.432 | 0.033 | 0.241 |
| LR on the 30-window history | 0.519 | 0.036 | 0.443 |
| GBDT on S_t | 0.688 | 0.024 | 0.294 |
| GRU sequence classifier | 0.706 | 0.164 | 0.382 |

## 7. World-model contribution (paired episode-bootstrap ΔAP, 300 resamples)

| comparison | val | test | holdout |
|---|---|---|---|
| world model − persistence | −0.024 [−0.045, +0.052] | −0.007 [−0.026, +0.011] | +0.143 [−0.000, +0.283] |
| world model − persistence + learned noise | −0.030 [−0.048, +0.031] | +0.001 [−0.011, +0.021] | +0.055 [−0.001, +0.147] |
| world model − isotropic noise | −0.008 [−0.016, +0.051] | −0.006 [−0.023, +0.013] | +0.146 [−0.000, +0.305] |
| world model − deterministic | −0.023 [−0.053, +0.013] | −0.023 [−0.048, −0.003] | +0.058 [−0.014, +0.169] |
| deterministic − persistence | −0.001 [−0.020, +0.058] | +0.016 [−0.007, +0.043] | +0.085 [+0.000, +0.147] |
| world model − ridge two-lag | +0.033 [−0.022, +0.131] | −0.025 [−0.074, −0.000] | +0.163 [−0.003, +0.339] |
| world model − GRU classifier | +0.020 [−0.023, +0.115] | −0.106 [−0.293, +0.020] | +0.057 [−0.081, +0.174] |
| state skill vs persistence: NIDRA / ridge | 0.672 / 0.653 | 0.587 / 0.565 | 0.616 / 0.595 |

Reading: the transition model contributes measurably at the **state** level on every
split (5–6 % lower MSE than a linear two-lag model, 59–67 % lower than persistence); at the
**risk** level it contributes on Thursday's unseen families (+0.14, lower bound −0.00) and
not on Friday's, where the per-state risk head does not recognise the attack states at all
(§10).

## 8. Forecasting results by horizon (AP for "attack at t+k")

| k (min) | val NIDRA / oracle | test NIDRA / oracle | holdout NIDRA / oracle | stage top-1 on attack futures (val / test / holdout) |
|---|---|---|---|---|
| 1 | 0.832 / 0.894 | 0.063 / 0.082 | 0.582 / 0.400 | 0.76 / 0.00 / 0.04 |
| 2 | 0.855 / 0.905 | 0.048 / 0.078 | 0.505 / 0.393 | 0.73 / 0.00 / 0.00 |
| 3 | 0.834 / 0.893 | 0.042 / 0.080 | 0.454 / 0.369 | 0.71 / 0.00 / 0.00 |
| 4 | 0.855 / 0.906 | 0.036 / 0.078 | 0.404 / 0.436 | 0.69 / 0.00 / 0.00 |
| 5 | 0.844 / 0.880 | 0.030 / 0.080 | 0.352 / 0.451 | 0.56 / 0.00 / 0.00 |
| 6 | 0.819 / 0.878 | 0.024 / 0.078 | 0.285 / 0.416 | 0.42 / 0.00 / 0.00 |

Task B (episode begins within h min, origin outside any episode): NIDRA 0.001–0.038 (val),
0.003–0.016 (test), 0.000–0.003 (holdout); the explicit onset head 0.000–0.115; LR on the
history is the best system at 0.04–0.08 on test/holdout. Training positives at
1 / 3 / 5 / 10 / 15 / 30 min: 3 / 9 / 15 / 29 / 39 / 69. Horizon extension to K=10: §9.

## 9. Generalisation results

- **Every evaluation family is unseen in training** by construction: test = Bot,
  PortScan, DDoS; holdout = Web Brute Force / XSS / SQLi, Infiltration; validation =
  SSH-Patator, Heartbleed; training = FTP-Patator, DoS ×4.
- **Holdout (leave-attack-family-out, Thursday)**: AP 0.439 vs persistence 0.295, 3 of 5
  episodes found (Web attacks at 1 and 29 min, Infiltration at 8 min), Web-attack minutes
  scored 0.78–0.94 at the median, Infiltration 0.00 (0.75 at the 90th percentile).
- **Test (Friday)**: AP 0.058; Bot-C2 windows (833 of 946 positives) scored ≈ 0 by every
  head-based system; DDoS scored 1.00 from its first window; PortScan partially.
- **Leave-one-day-out retrains**: <<LODO_SHORT>>
- **Horizon extension (K = 10 on test)**: rollout stable to ten steps, state skill 0.56–0.64 at every k (ridge 0.53–0.61); risk-level AP 0.062 vs persistence 0.066 (ΔAP −0.004 [−0.021, +0.016]); per-horizon AP 0.059 → 0.020 while the oracle stays at 0.085 — depth costs resolution, the head is the ceiling (`artifacts/metrics/test_K10/`).

## 10. Diagnosis (why the risk-level margin is where it is)

1. The oracle — the same frozen head on the *true* future — is 0.117 on test and 0.770 on
   validation: no forecaster can exceed what its head recognises.
2. The head on S_t has ROC-AUC **0.37** on Friday: it scores Friday's attack minutes below
   the all-zero silent state that carries 87 % of the split's weight. Trained on 179
   positives from two families on one attacker host, it does not transfer to Botnet C2 on
   internal workstations.
3. A GRU sequence classifier on the identical rows reaches ROC-AUC **0.976** (AP 0.164):
   the 30-window history carries family-general signal a per-state head cannot read. The
   fix — a head on the encoder state at each rollout step, trained on observed pairs and
   frozen — keeps every invariant and is the first open item; it needs a family-transfer
   selection split (leave-one-day-out), which is why it was not adopted in this run.
4. Task B is bounded by the data: 3 / 9 / 15 same-host onset precursors at 1 / 3 / 5 min in
   2.27 M training origins.

## 11. ATT&CK mapping status

Implemented and generated: `nidra/data/attack_mapping.py` → `docs/ATTACK_MAPPING.md`
(stage → tactic: recon TA0043/TA0007, initial_access TA0001/TA0006, lateral TA0008/TA0007,
c2 TA0011, exfil TA0040/TA0010 with the DoS/DDoS note; all 14 CIC-IDS2017 labels →
techniques, e.g. T1595.001, T1046, T1110.001, T1133, T1190, T1021, T1071.001, T1498.001,
T1499.002/003), consistency-tested, attached per horizon to every `forecast()` under
`attack_mapping`, with the progression labelled "projected stage sequence
(model-internal)". Limitation: stage top-1 on attack futures is 0.00 on test and holdout
because `recon`, `c2` and `lateral` never occur in training — the mapping is correct, the
stage forecast on those days is not a finding.

## 12. Explainability status

Four mechanisms, all in `explain()`: KernelSHAP on the current state (risk head, 100
k-means benign centroids), SHAP on the projected stage, temporal saliency from a future
feature to the input windows, and **signed integrated-gradient attributions of the pooled
forecast** over every (window, feature) cell with a **deletion faithfulness check**
(drop-top-m vs random-m, `faithful` flag and `completeness_gap` reported;
`tests/test_forecast_attribution.py`). Counterfactuals remain "model-internal what-if".

## 13. Offline demo status

- **Serving latency** (`python -m nidra.serve.benchmark`, K=6, 5 members × 200 trajectories,
  M1 CPU, 20 calls): median **114.6 ms**, mean 117.6 ms, p95 126.8 ms, max 162.7 ms —
  target 300 ms, pass. `NidraPredictor` loads `operating_point.json` at construction and
  refuses one not selected on validation; `forecast()` returns the per-horizon risk curve
  (calibrated and raw, trajectory band, P(attack within horizon)), the projected stage
  sequence with ATT&CK tactics/techniques per horizon, and the operating point in force.
- **Web console fixture** (`scripts/make_demo_replay.py` → `web/src/fixtures/demo-replay.json`,
  regenerated from the Run 8 artifacts through the same `load_predictor` the inference
  worker uses). The Friday Botnet episode the fixture used to replay is the benchmark's
  documented failure: on the Run 8 model it is a flat zero for all 48 windows (0 of 36
  attack-labelled windows scored). The fixture now replays the **Thursday Infiltration**
  episode on workstation 192.168.10.8 — the held-out day, a stage (`lateral`) with no
  training examples — anchored 12 minutes before the 15:04 onset: 12 quiet windows at 0.00,
  the observed risk crossing 0.75 one minute after onset and the forecast's peak horizon
  reaching 0.71–0.91 through the scan (4 windows where the forecast crosses one window
  before the observation does; on this single episode P 1.00 / R 0.14 at the served 0.718
  over 48 windows; the three quiet peers on the same /24 peak at 0.54). The fixture states why this
  episode was chosen and where the Botnet numbers are, and the console's stage label for it
  is model-internal by construction.
- **Offline pipeline** (`python -m nidra.cli.forecast --csv <Friday-PortScan CSV> --out …`):
  286,467 CICFlowMeter flows → 85,497 host-minute states for 3,667 hosts (flow-only mode —
  packet features zero, warned loudly, since a CSV carries no packets), 3,000 randomly
  capped origins scored with 300 trajectories each in 100 s, 2 above the threshold (1.0
  alerts per hour of capture), `forecasts.csv` / `alerts.json` (10 fully explained alerts
  with ATT&CK mapping) / `summary.json` with ground truth (6 PortScan episodes on
  172.16.0.1, 0 alerted at the random cap) and `report.html`
  (`reports/run8/offline_portscan/`). This is a pipeline demonstration, not an
  evaluation: the random origin cap leaves 3 positives in 3,000 rows.
- **Environment note**: `scripts/make_demo_replay.py` validates every forecast against the
  backend's Pydantic `Forecast` schema, so it needs `pydantic` in the ML environment
  (installed into `ml/.venv` for this run; it is already a declared backend dependency).

## 14. NTRO compliance matrix

`reports/NTRO_COMPLIANCE_2026-09-20.md` — 16 rows, each with the implementing module, the
demonstrating command/artifact, and the measured status; rows 2, 4, 12 and 16 carry the
open scientific questions and say so.

## 15. Remaining limitations

1. No advance warning demonstrated on any split (dataset has no same-host precursors).
2. The per-state risk head does not transfer to Friday's families; the world model cannot
   outrun its head. The history-aware head is designed, not built.
3. The risk-level margin over persistence is not resolved from zero on any single split;
   the state-level margin is.
4. Stage forecasting is undefined for stages absent from training (Friday, Thursday).
5. Validation has two episodes; every validation interval is uninformative.
6. Sampled noise costs AP on test and gains on holdout; the stochastic path is served.
7. One dataset (CIC-IDS2017); CTU-13 is the identified next candidate; 2018 excluded.
8. Hardware caps: 500k/50k dynamics subsample, 60 trajectories per member in benchmarks,
   one M1 laptop.
9. The Δ=30 web fixture and hero claim were replaced (§13); the console's stage names
   should not be rendered as findings on days whose stages are absent from training.

# NTRO PS 26153 — compliance matrix (2026-09-20)

Each row names the requirement, what implements it, and **how it is demonstrated** — a
command that runs and an artifact it writes. "Implemented" without a demonstrating artifact
is marked as such. Numbers are in `REAL_DATA_RESULTS.md` (Run 8), `MODEL_CARD.md` and
`artifacts/metrics/<split>/benchmark.json` (the production run's records; the tables in
`reports/run8/benchmark_tables.md` are generated from them).

| # | Requirement | Implementation | Demonstrated by | Status |
|---|---|---|---|---|
| 1 | Network-state representation | 45-feature per-host state at Δ=60 s: 15 flow, 11 packet, 8 graph scalars, 11 dynamics (`nidra/data/schema.py`, `windowize.py`); feature-specific preprocessing fit on train only (`normalize.py`) | `python -m nidra.data.audit` → `artifacts/metadata/data_audit_w60.json`; `preprocessing_audit.md` next to every scaler | done |
| 2 | Temporal dynamics P(S_{t+1} \| S_t) | GRU encoder → diagonal-Gaussian delta transition (`nidra/models/transition.py`), β-NLL multi-step unrolled loss with scheduled sampling, selected on free-running validation NLL | per-epoch free-running NLL/MSE/skill/coverage in `model_seed_*_metadata.json`; `state_forecast` block of every `benchmark.json` | **measured**: state-forecast skill vs persistence 0.672 / 0.587 / 0.616 (val / test / holdout), 5–6 % lower MSE than a ridge two-lag linear model on every split; 90 % band coverage 0.98 |
| 3 | K-step simulation | recursive stochastic rollout, 200 trajectories/member × 5 members in serving, 60 in benchmarks (`WorldModel.rollout`) | `risk_curve` and `predicted_features` per horizon in every `forecast()`; `task_C_progression.per_horizon` in `benchmark.json`; K=10 extension check in `artifacts/metrics/test_K10/` | done |
| 4 | Future attack likelihood | frozen risk head on rolled-out states, pooled (median) and calibrated at the validation operating point | primary metric: natural-prevalence AP with episode-bootstrap CI (`task_published_label`); Task B onset tables (`task_B_onset_forecast`) | **measured**: AP 0.058 [0.018, 0.199] test, 0.439 [0.000, 0.768] holdout; ΔAP vs persistence −0.007 [−0.026, +0.011] test, +0.143 [−0.000, +0.283] holdout; Task B at the prevalence floor for every system (3 / 9 / 15 training onsets within 1 / 3 / 5 min) — the dataset limitation, reported |
| 5 | Attack progression | frozen stage head on rolled-out states → projected stage sequence → ATT&CK tactics (`nidra/data/attack_mapping.progression_summary`) | `progression` in every `forecast()`; `stage_top1_accuracy_on_attack_futures` per horizon in `benchmark.json` | done — six-bucket taxonomy; stage top-1 on attack futures 0.76 → 0.42 (val, k=1→6), 0.00 on test and holdout because `recon`/`c2`/`lateral` never occur in training (limitation, labelled model-internal) |
| 6 | MITRE ATT&CK mapping | curated tables `nidra/data/attack_mapping.py`, generated `docs/ATTACK_MAPPING.md`, consistency test | `python -m nidra.data.attack_mapping --write`; `attack_mapping` per horizon in `forecast()` | done — presentation mapping, not inference |
| 7 | Explainability | SHAP on the current state (risk), SHAP on the projected stage, temporal saliency, and integrated-gradient attributions of the forecast with a deletion faithfulness check (`nidra/explain/forecast_attribution.py`) | `explain()` output (`forecast_attributions.faithfulness`); `tests/test_forecast_attribution.py` | done |
| 8 | Flow + packet telemetry | CICFlowMeter CSV + tshark packet parquet fused per (host, minute) (`windowize._fuse_flow_and_packet_windows`) | `data_audit_w60.json` packet/flow feature population rates on active rows | done |
| 9 | PCAP / CSV ingestion, offline | `python -m nidra.cli.forecast --pcap … / --csv …` → forecasts.csv, alerts.json, summary.json; `python -m nidra.cli.report_html` | `tests/test_cli_forecast.py`; the offline run on the real Friday PortScan CSV in Run 8 §8.8 (`reports/run8/offline_portscan/`) | done |
| 10 | Offline inference, no network | predictor loads committed artifacts; no service required | `python -m nidra.scripts.demo_forecast`; the CLI above | done |
| 11 | Logistic-regression benchmark | LR on S_t and LR on the flattened history, plus GBDT, ridge two-lag, GRU sequence classifier, persistence and noised persistence (`nidra/eval/systems.py`) | every `benchmark.json`, `task_published_label.systems` | done |
| 12 | Unseen-attack generalisation | every evaluation family is unseen in training by construction (test: Bot/PortScan/DDoS; holdout: Web×3/Infiltration); leave-one-day-out retrains (`experiments/runs/lodo_*`) | holdout `benchmark.json`; LODO records (Run 8 §8.6) | **measured**: transfers to Thursday's families (AP 0.439, 3 of 5 episodes found) and, in the leave-one-day-out retrain, to Wednesday's DoS/Heartbleed (AP 0.451, best system); not to Friday's Botnet C2 (per-state head ranks it below silence; oracle 0.117) nor cleanly to Tuesday's Patator in the second LODO run (head fires on benign traffic) — the diagnosis is in Run 8 §8.7 |
| 13 | Working demo | web console (`web/`, fixture regenerated from the Run 8 artifacts by `scripts/make_demo_replay.py`), offline CLI + HTML report, `demo_forecast` | Run 8 §8.8 | done |
| 14 | Reproducible configuration | `config/default.yaml` is the complete training configuration; every run writes a provenance record (git commit, config hash, dataset digests, geometry, seeds, checkpoints, operating point) | `experiments/runs/*/record.json`, `benchmark.json` headers | done |
| 15 | Confidence / uncertainty | trajectory band per horizon; per-horizon Platt calibration fit on validation; Brier and reliability per horizon | `risk_curve.band_low/high`; `calibration_published_label` in `benchmark.json` | done |
| 16 | Lead time | per-episode warning/latency/lead time at the frozen threshold | `per_episode` in `benchmark.json`; `ground_truth` in the CLI summary | **measured**: 0 of 15 (test) and 0 of 5 (holdout) episodes warned before onset; 3 + 3 alerted inside (holdout latency 1 / 8 / 29 min). No advance-warning claim is made |

Rows 2, 4, 12 and 16 carry the project's open scientific questions; they are compliance
in the sense of "implemented and measured", not "solved". Row 2 is the one the data
supports outright; row 4's world-model margin over persistence is positive only on the
holdout day and not resolved from zero; rows 4 (Task B) and 16 are limited by the
dataset's lack of same-host precursors. See `REAL_DATA_RESULTS.md` Run 8 and
`MODEL_CARD.md` for what the numbers do and do not support.

---

## Addendum — 2026-09-30

Two rows change in what they can claim; nothing above is edited.

**Row 11, logistic-regression benchmark.** The rows above point at `benchmark.json`,
whose per-system precision, recall, F1 and FPR are all read at the world model's
threshold (0.718). That threshold was chosen for the world model's calibrated score, so
the baselines' confusion matrices there are not a fair comparison (DECISIONS.md D148).
The PS benchmark is `reports/PS_BASELINE_BENCHMARK.md`, where each system uses its own
validation-chosen threshold:

| | world model test | LR test | world model holdout | LR holdout |
|---|---:|---:|---:|---:|
| AP | **0.058** | 0.033 | **0.438** | 0.241 |
| precision | **0.886** | 0.482 | **0.848** | 0.462 |
| recall | 0.032 | **0.038** | **0.320** | 0.279 |
| F1 | 0.061 | **0.071** | **0.464** | 0.348 |
| FPR | **0.000017** | 0.000171 | **0.000019** | 0.000110 |
| false alarms/h | **0.48** | 4.80 | **0.86** | 4.89 |

Status: **measured** — fewer false alarms and higher precision on both splits, higher F1
on holdout, lower F1 on test. The advantage is not attributable to the rollout: the same
risk head on the current state reaches comparable AP (row 4).

**Row 12, unseen-attack generalisation.** Run 9 added CTU-13 and withheld its largest
botnet family (Neris) from training entirely: AP 0.717 [0.485, 0.860], within-host ROC
0.949 across ten infected hosts — and persistence reaches 0.772. The model transfers to
an unseen family; it does not transfer better than persistence
(`reports/RUN9_FINAL_REPORT_2026-09-23.md`).

The architecture document the PS asks for (max 2 pages) is `ARCHITECTURE.md`; the
previous long-form version is `ARCHITECTURE_DETAIL.md`.

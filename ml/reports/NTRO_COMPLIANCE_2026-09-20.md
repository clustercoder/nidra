# NTRO PS 26153 — compliance matrix (2026-09-20)

Each row names the requirement, what implements it, and **how it is demonstrated** — a
command that runs and an artifact it writes. "Implemented" without a demonstrating artifact
is marked as such. Numbers are in `REAL_DATA_RESULTS.md` (Run 8) and
`experiments/runs/production/artifacts/metrics/<split>/benchmark.json`.

| # | Requirement | Implementation | Demonstrated by | Status |
|---|---|---|---|---|
| 1 | Network-state representation | 45-feature per-host state at Δ=60 s: 15 flow, 11 packet, 8 graph scalars, 11 dynamics (`nidra/data/schema.py`, `windowize.py`); feature-specific preprocessing fit on train only (`normalize.py`) | `python -m nidra.data.audit` → `artifacts/metadata/data_audit_w60.json`; `preprocessing_audit.md` next to every scaler | done |
| 2 | Temporal dynamics P(S_{t+1} \| S_t) | GRU encoder → diagonal-Gaussian delta transition (`nidra/models/transition.py`), trained with multi-step unrolled NLL and scheduled sampling, selected on free-running validation NLL | per-epoch free-running NLL/MSE/skill/coverage in `model_seed_*_metadata.json`; `state_forecast` block of every `benchmark.json` (skill vs persistence and vs ridge, per horizon) | done — see limitations: a two-lag ridge is competitive at the state level |
| 3 | K-step simulation | recursive stochastic rollout, 100 trajectories/member × 5 members (`WorldModel.rollout`) | `risk_curve` and `predicted_features` per horizon in every `forecast()`; `task_C_progression.per_horizon` in `benchmark.json` | done |
| 4 | Future attack likelihood | frozen risk head on rolled-out states, pooled and calibrated at the validation operating point | primary metric: natural-prevalence AP with episode-bootstrap CI on test/holdout (`task_published_label`); Task B onset tables (`task_B_onset_forecast`) | done — Task B is at prevalence level (limitation) |
| 5 | Attack progression | frozen stage head on rolled-out states → projected stage sequence → ATT&CK tactics (`nidra/data/attack_mapping.progression_summary`) | `progression` in every `forecast()`; `stage_top1_accuracy_on_attack_futures` per horizon in `benchmark.json` | done — six-bucket taxonomy, no exfiltration in the data |
| 6 | MITRE ATT&CK mapping | curated tables `nidra/data/attack_mapping.py`, generated `docs/ATTACK_MAPPING.md`, consistency test | `python -m nidra.data.attack_mapping --write`; `attack_mapping` per horizon in `forecast()` | done — presentation mapping, not inference |
| 7 | Explainability | SHAP on the current state (risk), SHAP on the projected stage, temporal saliency, and integrated-gradient attributions of the forecast with a deletion faithfulness check (`nidra/explain/forecast_attribution.py`) | `explain()` output (`forecast_attributions.faithfulness`); `tests/test_forecast_attribution.py` | done |
| 8 | Flow + packet telemetry | CICFlowMeter CSV + tshark packet parquet fused per (host, minute) (`windowize._fuse_flow_and_packet_windows`) | `data_audit_w60.json` packet/flow feature population rates on active rows | done |
| 9 | PCAP / CSV ingestion, offline | `python -m nidra.cli.forecast --pcap … / --csv …` → forecasts.csv, alerts.json, summary.json; `python -m nidra.cli.report_html` | `tests/test_cli_forecast.py`; the offline run in Run 8 | done |
| 10 | Offline inference, no network | predictor loads committed artifacts; no service required | `python -m nidra.scripts.demo_forecast`; the CLI above | done |
| 11 | Logistic-regression benchmark | LR on S_t and LR on the flattened history, plus GBDT, ridge two-lag, GRU sequence classifier, persistence and noised persistence (`nidra/eval/systems.py`) | every `benchmark.json`, `task_published_label.systems` | done |
| 12 | Unseen-attack generalisation | every evaluation family is unseen in training by construction (test: Bot/PortScan/DDoS; holdout: Web×3/Infiltration); leave-one-day-out retrains (`experiments/runs/lodo_*`) | holdout `benchmark.json`; LODO records | done for holdout; LODO: see Run 8 |
| 13 | Working demo | web console (`web/`, unchanged), offline CLI + HTML report, `demo_forecast` | Run 8 offline run artifacts | done |
| 14 | Reproducible configuration | `config/default.yaml` is the complete training configuration; every run writes a provenance record (git commit, config hash, dataset digests, geometry, seeds, checkpoints, operating point) | `experiments/runs/*/record.json`, `benchmark.json` headers | done |
| 15 | Confidence / uncertainty | trajectory band per horizon; per-horizon Platt calibration fit on validation; Brier and reliability per horizon | `risk_curve.band_low/high`; `calibration_published_label` in `benchmark.json` | done |
| 16 | Lead time | per-episode warning/latency/lead time at the frozen threshold | `per_episode` in `benchmark.json`; `ground_truth` in the CLI summary | done — reported honestly (see Run 8) |

Rows 2, 4 and 12 carry the project's open scientific questions; they are compliance
in the sense of "implemented and measured", not "solved". See `REAL_DATA_RESULTS.md` Run 8
and `MODEL_CARD.md` for what the numbers do and do not support.

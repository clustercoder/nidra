_val: no benchmark.json at experiments/runs/xeval_ctu2cic_state+hidden/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `mean|q=-|max`, threshold 0.108 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.004 | [0.001, 0.006] | 0.161 | 0.00 / 0.00 / 0.00 | 0.00 | 0.003 | 0.000 / 0.001 |
| NIDRA (stochastic rollout) | 0.003 | [0.001, 0.005] | 0.180 | 0.00 / 0.01 / 0.00 | 298.76 | 0.002 | 0.000 / 0.001 |
| NIDRA (deterministic rollout) | 0.002 | [0.001, 0.004] | 0.040 | 0.00 / 0.00 / 0.00 | 63.65 | 0.001 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.004 | [0.001, 0.008] | 0.373 | 0.01 / 0.03 / 0.01 | 699.82 | 0.003 | 0.000 / 0.001 |
| persistence + isotropic noise | 0.002 | — | 0.207 | 0.00 / 0.27 / 0.00 | 20711.71 | 0.002 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.002 | [0.001, 0.004] | 0.033 | 0.00 / 0.00 / 0.00 | 94.76 | 0.002 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.002 | — | 0.047 | 0.00 / 0.01 / 0.00 | 425.04 | 0.002 | 0.000 / 0.000 |
| logistic regression on S_t | 0.016 | [0.001, 0.034] | 0.238 | 0.03 / 0.14 / 0.05 | 563.72 | 0.010 | 0.002 / 0.003 |
| logistic regression on the L-window history | 0.003 | [0.001, 0.008] | 0.049 | 0.01 / 0.01 / 0.01 | 105.00 | 0.002 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.024 | [0.002, 0.053] | 0.268 | 0.03 / 0.16 / 0.05 | 712.34 | 0.015 | 0.020 / 0.022 |
| GRU sequence classifier | 0.017 | [0.001, 0.043] | 0.131 | 0.34 / 0.00 / 0.00 | 0.24 | 0.010 | 0.012 / 0.024 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.001 | [0.000, 0.001] |
| world_model - persistence_rollout | 0.001 | [0.000, 0.001] |
| world_model - noised_persistence | -0.002 | [-0.003, -0.000] |
| world_model - isotropic_noise_persistence | 0.000 | [0.000, 0.001] |
| world_model - world_model_deterministic | 0.001 | [0.000, 0.001] |
| world_model_deterministic - persistence | -0.000 | [-0.000, 0.000] |
| noised_persistence - persistence | 0.002 | [0.000, 0.004] |
| world_model - lr_current_state | -0.013 | [-0.030, -0.000] |
| world_model - lr_flattened_history | -0.000 | [-0.005, 0.001] |
| world_model - gbdt_current_state | -0.021 | [-0.049, -0.001] |
| world_model - gru_classifier | -0.014 | [-0.040, 0.000] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.430, period2_persistence 0.041, ridge_two_lag 0.426; NIDRA vs ridge 0.006.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.003 | 0.001 | 0.002 | 0.03137 | 0.00 (664) |
| 2 | 2 | 0.00292 | 0.003 | 0.001 | 0.002 | 0.03127 | 0.00 (662) |
| 3 | 3 | 0.00291 | 0.002 | 0.001 | 0.002 | 0.03122 | 0.00 (661) |
| 4 | 4 | 0.00291 | 0.002 | 0.001 | 0.002 | 0.03117 | 0.00 (660) |
| 5 | 5 | 0.00290 | 0.002 | 0.001 | 0.002 | 0.03106 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.002 | 0.001 | 0.002 | 0.03097 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.000 | 0.000 | — | 0.084 | 0.000 | — |
| 3 | 36 | 0.000 | 0.000 | — | 0.030 | 0.000 | — |
| 5 | 60 | 0.000 | 0.000 | — | 0.020 | 0.000 | — |
| 10 | 113 | 0.000 | 0.000 | — | 0.025 | 0.000 | — |
| 15 | 161 | 0.001 | 0.000 | — | 0.022 | 0.000 | — |
| 30 | 299 | 0.001 | 0.001 | — | 0.024 | 0.001 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.108), 2 consecutive windows: 15 episodes, 0 warned before onset, 0 alerted inside the episode; median lead None s, median latency None min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.002 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.003 | no (—) | no (—) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.014 | no (—) | no (—) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.005 | no (—) | no (—) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.001 | no (—) | no (—) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | no (—) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.007 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.010 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.005 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 0.003 | no (—) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.008 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.006 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.006 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `friday_morning:c2` | — | 862 | 7 | 7 | 0.002 | 0.002 | 0.002 | 0.367 |
| `friday_portscan:recon` | — | 58 | 6 | 1¹ | 0.001 | 0.001 | 0.001 | 0.405 |
| `friday_ddos:exfil` | — | 26 | 2 | 2 | 0.000 | 0.000 | 0.000 | -1.300 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.036 | 0.218 | 0.1312 | 9 | 0.3927 | 0.391 | 0.99× | 0.4721 | carries timing signal |
| world_model | 0.027 | 0.219 | 0.1944 | 9 | 0.3927 | 0.367 | 0.93× | 0.4348 | carries timing signal |
| world_model_deterministic | 0.024 | 0.114 | 0.1320 | 9 | 0.3927 | 0.352 | 0.90× | 0.3971 | carries timing signal |
| noised_persistence | 0.030 | 0.314 | 0.2673 | 9 | 0.3927 | 0.357 | 0.91× | 0.4424 | carries timing signal |
| isotropic_noise_persistence | 0.029 | 0.315 | 0.2046 | 9 | 0.3927 | 0.379 | 0.96× | 0.5002 | carries timing signal |
| persistence | 0.024 | 0.090 | 0.1424 | 9 | 0.3927 | 0.327 | 0.83× | 0.3795 | carries timing signal |
| oracle_true_future | 0.024 | 0.106 | 0.1529 | 9 | 0.3927 | 0.332 | 0.84× | 0.3595 | carries timing signal |
| lr_current_state | 0.055 | 0.452 | 0.8770 | 9 | 0.3927 | 0.364 | 0.93× | 0.4745 | carries timing signal |
| lr_flattened_history | 0.026 | 0.134 | 0.3637 | 9 | 0.3927 | 0.336 | 0.86× | 0.3964 | carries timing signal |
| gbdt_current_state | 0.065 | 0.329 | 0.8685 | 9 | 0.3927 | 0.369 | 0.94× | 0.4496 | carries timing signal |
| gru_classifier | 0.046 | 0.220 | 0.4657 | 9 | 0.3927 | 0.383 | 0.98× | 0.4294 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00486 | 0.04757 | 0.00417 | 0.00415 | 9/10 |
| calibrated | 0.00418 | 0.04583 | 0.00417 | 0.00415 | 1/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `mean|q=-|max`, threshold 0.108 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.000 | [0.000, 0.001] | 0.086 | 0.00 / 0.00 / 0.00 | 0.00 | 0.000 | 0.000 / 0.000 |
| NIDRA (stochastic rollout) | 0.000 | [0.000, 0.000] | 0.103 | 0.00 / 0.00 / 0.00 | 368.76 | 0.000 | 0.000 / 0.000 |
| NIDRA (deterministic rollout) | 0.000 | [0.000, 0.000] | 0.009 | 0.00 / 0.00 / 0.00 | 76.96 | 0.000 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.000 | [0.000, 0.001] | 0.286 | 0.00 / 0.01 / 0.00 | 899.31 | 0.000 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.000 | — | 0.081 | 0.00 / 0.11 / 0.00 | 33980.83 | 0.000 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.000 | [0.000, 0.000] | 0.005 | 0.00 / 0.00 / 0.00 | 106.58 | 0.000 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.000 | — | 0.005 | 0.00 / 0.00 / 0.00 | 491.60 | 0.000 | 0.000 / 0.000 |
| logistic regression on S_t | 0.000 | [0.000, 0.001] | 0.141 | 0.00 / 0.03 / 0.00 | 642.17 | 0.000 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.000 | [0.000, 0.000] | 0.007 | 0.00 / 0.00 / 0.00 | 169.51 | 0.000 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.001 | [0.000, 0.005] | 0.311 | 0.00 / 0.19 / 0.01 | 771.83 | 0.000 | 0.000 / 0.003 |
| GRU sequence classifier | 0.000 | [0.000, 0.000] | 0.070 | 0.00 / 0.00 / 0.00 | 0.29 | 0.000 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.00 / R 0.00 / F1 0.00, 0 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.000 | [0.000, 0.000] |
| world_model - persistence_rollout | 0.000 | [0.000, 0.000] |
| world_model - noised_persistence | -0.000 | [-0.000, -0.000] |
| world_model - isotropic_noise_persistence | 0.000 | [0.000, 0.000] |
| world_model - world_model_deterministic | 0.000 | [0.000, 0.000] |
| world_model_deterministic - persistence | 0.000 | [0.000, 0.000] |
| noised_persistence - persistence | 0.000 | [0.000, 0.000] |
| world_model - lr_current_state | -0.000 | [-0.001, 0.000] |
| world_model - lr_flattened_history | 0.000 | [0.000, 0.000] |
| world_model - gbdt_current_state | -0.001 | [-0.005, -0.000] |
| world_model - gru_classifier | 0.000 | [0.000, 0.000] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.457, period2_persistence 0.045, ridge_two_lag 0.459; NIDRA vs ridge -0.005.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.000 | 0.000 | 0.000 | 0.00341 | 0.00 (68) |
| 2 | 2 | 0.00019 | 0.000 | 0.000 | 0.000 | 0.00335 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.000 | 0.000 | 0.000 | 0.00330 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.000 | 0.000 | 0.000 | 0.00324 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.000 | 0.000 | 0.000 | 0.00319 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.000 | 0.000 | 0.000 | 0.00314 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 3 | 12 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 5 | 20 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 10 | 38 | 0.000 | 0.000 | — | 0.002 | 0.000 | — |
| 15 | 52 | 0.000 | 0.000 | — | 0.003 | 0.000 | — |
| 30 | 73 | 0.000 | 0.000 | — | 0.002 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.108), 2 consecutive windows: 5 episodes, 0 warned before onset, 0 alerted inside the episode; median lead None s, median latency None min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | no (—) |
| 172.16.0.1@1499346900 | 27 | 14 | 0.003 | no (—) | no (—) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.007 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.000 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.005 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `thursday_infiltration:lateral` | — | 73 | 3 | 1¹ | 0.000 | 0.000 | 0.000 | 0.230 |
| `thursday_web:initial_access` | — | 49 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.157 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.005 | 0.148 | 0.0724 | 2 | 0.3664 | 0.334 | 0.91× | 0.3572 | carries timing signal |
| world_model | 0.004 | 0.146 | 0.1594 | 2 | 0.3664 | 0.274 | 0.75× | 0.2668 | carries timing signal |
| world_model_deterministic | 0.003 | 0.050 | 0.1947 | 2 | 0.3664 | 0.237 | 0.65× | 0.1787 | carries timing signal |
| noised_persistence | 0.004 | 0.231 | 0.1496 | 2 | 0.3664 | 0.301 | 0.82× | 0.3574 | carries timing signal |
| isotropic_noise_persistence | 0.003 | 0.160 | 0.0551 | 2 | 0.3664 | 0.274 | 0.75× | 0.3082 | carries timing signal |
| persistence | 0.003 | 0.033 | 0.0875 | 2 | 0.3664 | 0.232 | 0.63× | 0.1643 | carries timing signal |
| oracle_true_future | 0.003 | 0.030 | 0.2651 | 2 | 0.3664 | 0.220 | 0.60× | 0.0937 | carries timing signal |
| lr_current_state | 0.004 | 0.319 | 0.6251 | 2 | 0.3664 | 0.261 | 0.71× | 0.2709 | carries timing signal |
| lr_flattened_history | 0.003 | 0.043 | 0.1923 | 2 | 0.3664 | 0.247 | 0.67× | 0.2151 | carries timing signal |
| gbdt_current_state | 0.007 | 0.353 | 0.5670 | 2 | 0.3664 | 0.339 | 0.92× | 0.4532 | carries timing signal |
| gru_classifier | 0.003 | 0.188 | 0.1539 | 2 | 0.3664 | 0.369 | 1.01× | 0.4909 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00090 | 0.05103 | 0.00034 | 0.00034 | 9/10 |
| calibrated | 0.00035 | 0.04966 | 0.00034 | 0.00034 | 1/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

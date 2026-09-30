### Systems — val
**val** — 20113 scored rows, natural prevalence 0.00012, operating point `mean|q=-|integrated`, threshold 0.502 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.781 | — | 0.904 | 0.95 / 0.75 / 0.84 | 0.38 | 0.965 | 0.000 / 0.000 |
| NIDRA (stochastic rollout) | 0.778 | [0.254, 0.969] (2 positive episodes: not informative) | 0.874 | 0.01 / 0.82 / 0.01 | 1361.21 | 0.962 | 0.000 / 0.000 |
| NIDRA (deterministic rollout) | 0.793 | [0.348, 0.979] (2 positive episodes: not informative) | 0.827 | 0.19 / 0.80 / 0.31 | 36.03 | 0.980 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.756 | [0.068, 0.969] (2 positive episodes: not informative) | 0.946 | 0.03 / 0.83 / 0.05 | 312.11 | 0.931 | 0.001 / 0.004 |
| persistence + isotropic noise | 0.741 | — | 0.889 | 0.02 / 0.83 / 0.05 | 364.43 | 0.914 | 0.001 / 0.002 |
| persistence (risk head on S_t) | 0.740 | [0.022, 0.969] (2 positive episodes: not informative) | 0.871 | 0.41 / 0.74 / 0.52 | 11.60 | 0.914 | 0.000 / 0.021 |
| ridge two-lag dynamics + risk head | 0.000 | [0.000, 0.001] (2 positive episodes: not informative) | 0.206 | 0.00 / 0.03 / 0.00 | 260.30 | 0.000 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.797 | — | 1.000 | 0.09 / 0.98 / 0.16 | 110.63 | 0.880 | 0.190 / 0.073 |
| logistic regression on S_t | 0.378 | [0.000, 0.804] (2 positive episodes: not informative) | 0.712 | 0.04 / 0.69 / 0.08 | 157.42 | 0.463 | 0.001 / 0.002 |
| logistic regression on the L-window history | 0.511 | [0.000, 0.838] (2 positive episodes: not informative) | 0.668 | 0.09 / 0.60 / 0.15 | 68.27 | 0.632 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.197 | [0.000, 0.573] (2 positive episodes: not informative) | 0.723 | 0.08 / 0.69 / 0.15 | 80.85 | 0.243 | 0.000 / 0.001 |

At the mandated 0.75 threshold on the calibrated score: P 0.95 / R 0.70 / F1 0.80, 65 alerts.

### Attribution — val
**val** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.038 | [-0.009, 0.311] |
| world_model - persistence_rollout | 0.060 | [-0.009, 0.311] |
| world_model - noised_persistence | 0.022 | [-0.007, 0.218] |
| world_model - isotropic_noise_persistence | 0.037 | [-0.007, 0.271] |
| world_model - world_model_deterministic | -0.014 | [-0.098, -0.002] |
| world_model - ridge_two_lag | 0.778 | [0.254, 0.969] |
| world_model_deterministic - persistence | 0.052 | [0.003, 0.328] |
| world_model_deterministic - ridge_two_lag | 0.793 | [0.348, 0.978] |
| noised_persistence - persistence | 0.016 | [-0.003, 0.172] |
| world_model - lr_current_state | 0.401 | [0.163, 0.518] |
| world_model - lr_flattened_history | 0.267 | [0.126, 0.349] |
| world_model - gbdt_current_state | 0.581 | [0.254, 0.704] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.609, period2_persistence 0.084, ridge_two_lag -0.279; NIDRA vs ridge 0.694.

### Horizon — val
**val** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00010 | 0.826 | 0.829 | 0.883 | 0.00080 | 0.23 (74) |
| 2 | 2 | 0.00010 | 0.925 | 0.951 | 0.891 | 0.00055 | 0.00 (73) |
| 3 | 3 | 0.00010 | 0.815 | 0.819 | 0.882 | 0.00086 | 0.00 (72) |
| 4 | 4 | 0.00010 | 0.919 | 0.930 | 0.900 | 0.00124 | 0.00 (71) |
| 5 | 5 | 0.00010 | 0.755 | 0.814 | 0.860 | 0.00254 | 0.00 (70) |
| 6 | 6 | 0.00010 | 0.213 | 0.843 | 0.861 | 0.00319 | 0.00 (69) |

### Onset forecasting — val
**val** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 3 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 5 | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 10 | 11 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 15 | 16 | 0.000 | 0.021 | 0.000 | 0.001 | 0.000 | — |
| 30 | 31 | 0.001 | 0.112 | 0.001 | 0.008 | 0.000 | — |

### Episodes — val
**val** — per episode at the selected threshold (0.502), 2 consecutive windows: 2 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499188140 | 62 | 1 | 0.000 | no (—) | yes (0.0) |
| 172.16.0.1@1499278320 | 20 | 30 | 0.069 | no (—) | no (—) |

### Per attack group — val
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `tuesday:initial_access` | — | 63 | 1 | 1¹ | 0.958 | 0.992 | 0.963 | 0.318 |
| `wednesday:initial_access` | — | 26 | 1 | 1¹ | 0.286 | 0.066 | 0.028 | 0.336 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — val
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.796 | 0.923 | 0.9993 | 1 | 0.7607 | 0.961 | 1.26× | 0.8571 | carries timing signal |
| world_model | 0.791 | 0.900 | 0.9993 | 1 | 0.7607 | 0.960 | 1.26× | 0.8499 | carries timing signal |
| world_model_deterministic | 0.800 | 0.868 | 0.9993 | 1 | 0.7607 | 0.960 | 1.26× | 0.8475 | carries timing signal |
| noised_persistence | 0.785 | 0.921 | 0.9993 | 1 | 0.7607 | 0.955 | 1.25× | 0.8371 | carries timing signal |
| isotropic_noise_persistence | 0.775 | 0.897 | 0.9918 | 1 | 0.7607 | 0.957 | 1.26× | 0.8499 | carries timing signal |
| persistence | 0.771 | 0.869 | 0.9993 | 1 | 0.7607 | 0.947 | 1.24× | 0.8162 | carries timing signal |
| ridge_two_lag | 0.003 | 0.193 | 0.9079 | 1 | 0.7607 | 0.626 | 0.82× | 0.2215 | **host identity** |
| oracle_true_future | 0.867 | 0.999 | 0.9993 | 1 | 0.7607 | 0.963 | 1.27× | 0.8860 | carries timing signal |
| lr_current_state | 0.506 | 0.806 | 0.9993 | 1 | 0.7607 | 0.880 | 1.16× | 0.6354 | carries timing signal |
| lr_flattened_history | 0.579 | 0.778 | 0.9988 | 1 | 0.7607 | 0.904 | 1.19× | 0.6932 | carries timing signal |
| gbdt_current_state | 0.374 | 0.820 | 0.9992 | 1 | 0.7607 | 0.836 | 1.10× | 0.6152 | **host identity** |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — val
**val** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.08979 | 0.28070 | 0.00012 | 0.00012 | 10/10 |
| calibrated | 0.00005 | 0.05004 | 0.00012 | 0.00012 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `mean|q=-|integrated`, threshold 0.502 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.111 | — | 0.870 | 0.74 / 0.03 / 0.06 | 1.32 | 0.102 | 0.006 / 0.011 |
| NIDRA (stochastic rollout) | 0.096 | [0.040, 0.205] | 0.827 | 0.08 / 0.49 / 0.13 | 691.49 | 0.090 | 0.006 / 0.012 |
| NIDRA (deterministic rollout) | 0.039 | [0.007, 0.174] | 0.237 | 0.09 / 0.04 / 0.06 | 47.53 | 0.052 | 0.001 / 0.002 |
| persistence + learned noise (mean disabled) | 0.127 | [0.060, 0.266] | 0.863 | 0.13 / 0.33 / 0.19 | 259.93 | 0.126 | 0.032 / 0.026 |
| persistence + isotropic noise | 0.082 | — | 0.575 | 0.08 / 0.30 / 0.13 | 380.86 | 0.090 | 0.024 / 0.016 |
| persistence (risk head on S_t) | 0.092 | [0.038, 0.225] | 0.585 | 0.33 / 0.05 / 0.08 | 11.83 | 0.101 | 0.010 / 0.007 |
| ridge two-lag dynamics + risk head | 0.093 | [0.030, 0.193] | 0.630 | 0.16 / 0.40 / 0.23 | 236.74 | 0.074 | 0.001 / 0.004 |
| oracle: risk head on the true future | 0.155 | — | 0.834 | 0.11 / 0.11 / 0.11 | 102.53 | 0.106 | 0.286 / 0.149 |
| logistic regression on S_t | 0.040 | [0.008, 0.177] | 0.142 | 0.11 / 0.09 / 0.10 | 85.50 | 0.046 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.035 | [0.006, 0.198] | 0.088 | 0.45 / 0.06 / 0.11 | 9.01 | 0.020 | 0.063 / 0.051 |
| gradient-boosted trees on S_t | 0.048 | [0.014, 0.137] | 0.222 | 0.16 / 0.11 / 0.13 | 71.10 | 0.054 | 0.000 / 0.001 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.03 / F1 0.06, 29 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.004 | [-0.042, 0.037] |
| world_model - persistence_rollout | 0.007 | [-0.033, 0.040] |
| world_model - noised_persistence | -0.031 | [-0.098, -0.005] |
| world_model - isotropic_noise_persistence | 0.013 | [-0.033, 0.046] |
| world_model - world_model_deterministic | 0.057 | [0.013, 0.122] |
| world_model - ridge_two_lag | 0.002 | [-0.084, 0.139] |
| world_model_deterministic - persistence | -0.053 | [-0.110, -0.018] |
| world_model_deterministic - ridge_two_lag | -0.054 | [-0.171, 0.119] |
| noised_persistence - persistence | 0.035 | [0.009, 0.089] |
| world_model - lr_current_state | 0.055 | [0.010, 0.118] |
| world_model - lr_flattened_history | 0.061 | [-0.016, 0.130] |
| world_model - gbdt_current_state | 0.048 | [0.016, 0.106] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.552, period2_persistence 0.045, ridge_two_lag -0.553; NIDRA vs ridge 0.711.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.122 | 0.096 | 0.100 | 0.02980 | 0.03 (664) |
| 2 | 2 | 0.00292 | 0.108 | 0.063 | 0.097 | 0.03017 | 0.03 (662) |
| 3 | 3 | 0.00291 | 0.091 | 0.038 | 0.098 | 0.03032 | 0.02 (661) |
| 4 | 4 | 0.00291 | 0.062 | 0.032 | 0.094 | 0.03048 | 0.02 (660) |
| 5 | 5 | 0.00290 | 0.031 | 0.023 | 0.098 | 0.03052 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.020 | 0.017 | 0.098 | 0.03062 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.001 | 0.003 | 0.000 | 0.000 | 0.017 | — |
| 3 | 36 | 0.005 | 0.004 | 0.000 | 0.000 | 0.046 | — |
| 5 | 60 | 0.006 | 0.010 | 0.001 | 0.000 | 0.063 | — |
| 10 | 113 | 0.011 | 0.009 | 0.002 | 0.001 | 0.063 | — |
| 15 | 161 | 0.011 | 0.007 | 0.004 | 0.001 | 0.051 | — |
| 30 | 299 | 0.013 | 0.005 | 0.009 | 0.002 | 0.038 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.502), 2 consecutive windows: 15 episodes, 0 warned before onset, 3 alerted inside the episode; median lead None s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.002 | no (—) | no (—) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.000 | no (—) | yes (0.0) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.010 | no (—) | yes (5.0) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.001 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | yes (0.0) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.117 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.025 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.054 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 0.589 | no (—) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.067 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.086 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.070 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `friday_morning:c2` | — | 862 | 7 | 7 | 0.052 | 0.071 | 0.042 | 0.498 |
| `friday_portscan:recon` | — | 58 | 6 | 1¹ | 0.163 | 0.640 | 0.204 | 0.391 |
| `friday_ddos:exfil` | — | 26 | 2 | 2 | 0.829 | 0.958 | 0.868 | -0.414 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.232 | 0.860 | 0.8960 | 9 | 0.3927 | 0.495 | 1.26× | 0.5933 | carries timing signal |
| world_model | 0.224 | 0.844 | 0.8970 | 9 | 0.3927 | 0.477 | 1.22× | 0.5750 | carries timing signal |
| world_model_deterministic | 0.079 | 0.434 | 0.6289 | 9 | 0.3927 | 0.481 | 1.23× | 0.5582 | carries timing signal |
| noised_persistence | 0.204 | 0.785 | 0.9003 | 9 | 0.3927 | 0.507 | 1.29× | 0.6010 | carries timing signal |
| isotropic_noise_persistence | 0.144 | 0.566 | 0.7824 | 9 | 0.3927 | 0.507 | 1.29× | 0.5890 | carries timing signal |
| persistence | 0.145 | 0.554 | 0.9115 | 9 | 0.3927 | 0.501 | 1.27× | 0.5748 | carries timing signal |
| ridge_two_lag | 0.170 | 0.609 | 0.9430 | 9 | 0.3927 | 0.406 | 1.04× | 0.5087 | **host identity** |
| oracle_true_future | 0.242 | 0.777 | 0.9086 | 9 | 0.3927 | 0.541 | 1.38× | 0.6385 | carries timing signal |
| lr_current_state | 0.079 | 0.258 | 0.6110 | 9 | 0.3927 | 0.471 | 1.20× | 0.5510 | carries timing signal |
| lr_flattened_history | 0.068 | 0.144 | 0.1317 | 9 | 0.3927 | 0.423 | 1.08× | 0.5174 | carries timing signal |
| gbdt_current_state | 0.114 | 0.543 | 0.7521 | 9 | 0.3927 | 0.439 | 1.12× | 0.5164 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.08719 | 0.26607 | 0.00417 | 0.00415 | 10/10 |
| calibrated | 0.00402 | 0.04620 | 0.00417 | 0.00415 | 9/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `mean|q=-|integrated`, threshold 0.502 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.332 | — | 0.830 | 0.51 / 0.28 / 0.36 | 4.02 | 0.573 | 0.002 / 0.003 |
| NIDRA (stochastic rollout) | 0.312 | [0.005, 0.736] | 0.811 | 0.01 / 0.67 / 0.02 | 962.07 | 0.537 | 0.002 / 0.003 |
| NIDRA (deterministic rollout) | 0.312 | [0.000, 0.780] | 0.509 | 0.11 / 0.37 / 0.17 | 45.90 | 0.574 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.359 | [0.007, 0.754] | 0.897 | 0.04 / 0.63 / 0.07 | 247.94 | 0.575 | 0.001 / 0.002 |
| persistence + isotropic noise | 0.330 | — | 0.699 | 0.02 / 0.54 / 0.04 | 362.14 | 0.502 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.356 | [0.002, 0.750] | 0.680 | 0.41 / 0.39 / 0.40 | 8.33 | 0.596 | 0.000 / 0.000 |
| ridge two-lag dynamics + risk head | 0.007 | [0.000, 0.034] | 0.379 | 0.02 / 0.25 / 0.03 | 234.69 | 0.001 | 0.000 / 0.001 |
| oracle: risk head on the true future | 0.489 | — | 0.961 | 0.13 / 0.60 / 0.21 | 60.88 | 0.588 | 0.214 / 0.084 |
| logistic regression on S_t | 0.124 | [0.000, 0.376] | 0.470 | 0.04 / 0.39 / 0.08 | 127.74 | 0.169 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.381 | [0.000, 0.662] | 0.485 | 0.81 / 0.38 / 0.51 | 1.32 | 0.209 | 0.094 / 0.215 |
| gradient-boosted trees on S_t | 0.141 | [0.001, 0.468] | 0.445 | 0.09 / 0.37 / 0.14 | 56.37 | 0.240 | 0.001 / 0.002 |

At the mandated 0.75 threshold on the calibrated score: P 0.75 / R 0.28 / F1 0.41, 45 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.045 | [-0.073, 0.014] |
| world_model - persistence_rollout | 0.001 | [-0.051, 0.123] |
| world_model - noised_persistence | -0.047 | [-0.097, -0.001] |
| world_model - isotropic_noise_persistence | -0.018 | [-0.073, 0.071] |
| world_model - world_model_deterministic | 0.000 | [-0.058, 0.070] |
| world_model - ridge_two_lag | 0.305 | [0.004, 0.736] |
| world_model_deterministic - persistence | -0.045 | [-0.098, 0.012] |
| world_model_deterministic - ridge_two_lag | 0.305 | [-0.009, 0.780] |
| noised_persistence - persistence | 0.002 | [-0.035, 0.070] |
| world_model - lr_current_state | 0.188 | [0.003, 0.490] |
| world_model - lr_flattened_history | -0.070 | [-0.298, 0.284] |
| world_model - gbdt_current_state | 0.171 | [0.005, 0.422] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.575, period2_persistence 0.049, ridge_two_lag -0.366; NIDRA vs ridge 0.689.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.524 | 0.534 | 0.559 | 0.00196 | 0.00 (68) |
| 2 | 2 | 0.00019 | 0.404 | 0.465 | 0.554 | 0.00222 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.362 | 0.388 | 0.545 | 0.00246 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.244 | 0.316 | 0.550 | 0.00288 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.164 | 0.309 | 0.527 | 0.00307 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.009 | 0.105 | 0.513 | 0.00312 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | — |
| 3 | 12 | 0.001 | 0.000 | 0.000 | 0.001 | 0.054 | — |
| 5 | 20 | 0.002 | 0.000 | 0.000 | 0.001 | 0.094 | — |
| 10 | 38 | 0.002 | 0.000 | 0.001 | 0.001 | 0.251 | — |
| 15 | 52 | 0.003 | 0.000 | 0.001 | 0.002 | 0.215 | — |
| 30 | 73 | 0.007 | 0.001 | 0.002 | 0.003 | 0.153 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.502), 2 consecutive windows: 5 episodes, 0 warned before onset, 2 alerted inside the episode; median lead None s, median latency 15.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | yes (29.0) |
| 172.16.0.1@1499346900 | 27 | 14 | 0.000 | no (—) | yes (1.0) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.210 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.028 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.420 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `thursday_infiltration:lateral` | — | 73 | 3 | 1¹ | 0.017 | 0.157 | 0.043 | 0.406 |
| `thursday_web:initial_access` | — | 49 | 2 | 1¹ | 0.663 | 0.836 | 0.722 | 0.246 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.386 | 0.842 | 0.9868 | 2 | 0.3664 | 0.691 | 1.88× | 0.6730 | carries timing signal |
| world_model | 0.369 | 0.841 | 0.9742 | 2 | 0.3664 | 0.682 | 1.86× | 0.6653 | carries timing signal |
| world_model_deterministic | 0.347 | 0.594 | 0.9839 | 2 | 0.3664 | 0.663 | 1.81× | 0.6447 | carries timing signal |
| noised_persistence | 0.415 | 0.844 | 0.9758 | 2 | 0.3664 | 0.720 | 1.97× | 0.7107 | carries timing signal |
| isotropic_noise_persistence | 0.387 | 0.700 | 0.9327 | 2 | 0.3664 | 0.720 | 1.97× | 0.7209 | carries timing signal |
| persistence | 0.397 | 0.669 | 0.9902 | 2 | 0.3664 | 0.706 | 1.93× | 0.7071 | carries timing signal |
| ridge_two_lag | 0.018 | 0.367 | 0.9287 | 2 | 0.3664 | 0.347 | 0.95× | 0.3584 | **host identity** |
| oracle_true_future | 0.555 | 0.934 | 0.9902 | 2 | 0.3664 | 0.845 | 2.31× | 0.8456 | carries timing signal |
| lr_current_state | 0.213 | 0.540 | 0.8759 | 2 | 0.3664 | 0.624 | 1.70× | 0.6381 | carries timing signal |
| lr_flattened_history | 0.395 | 0.585 | 0.9621 | 2 | 0.3664 | 0.746 | 2.04× | 0.7796 | carries timing signal |
| gbdt_current_state | 0.226 | 0.737 | 0.8207 | 2 | 0.3664 | 0.604 | 1.65× | 0.6761 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.08860 | 0.27732 | 0.00034 | 0.00034 | 10/10 |
| calibrated | 0.00030 | 0.04990 | 0.00034 | 0.00034 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — val
**val** — 20113 scored rows, natural prevalence 0.00012, operating point `mean|q=-|integrated`, threshold 0.882 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.832 | — | 0.999 | 0.96 / 0.79 / 0.86 | 0.38 | 0.988 | 0.068 / 0.062 |
| NIDRA (stochastic rollout) | 0.831 | [0.275, 0.984] (2 positive episodes: not informative) | 0.998 | 0.28 / 0.84 / 0.42 | 23.08 | 0.987 | 0.075 / 0.069 |
| NIDRA (deterministic rollout) | 0.770 | [0.105, 0.984] (2 positive episodes: not informative) | 0.996 | 0.18 / 0.83 / 0.29 | 41.00 | 0.921 | 0.014 / 0.054 |
| persistence + learned noise (mean disabled) | 0.808 | [0.207, 0.983] (2 positive episodes: not informative) | 0.993 | 0.63 / 0.81 / 0.71 | 5.05 | 0.962 | 0.211 / 0.086 |
| persistence + isotropic noise | 0.788 | — | 0.947 | 0.40 / 0.81 / 0.54 | 12.76 | 0.942 | 0.208 / 0.085 |
| persistence (risk head on S_t) | 0.890 | [0.240, 0.984] (2 positive episodes: not informative) | 0.997 | 0.95 / 0.70 / 0.80 | 0.38 | 0.985 | 0.023 / 0.035 |
| oracle: risk head on the true future | 0.943 | — | 1.000 | 1.00 / 0.73 / 0.84 | 0.00 | 0.935 | 0.915 / 0.455 |
| logistic regression on S_t | 0.378 | [0.000, 0.804] (2 positive episodes: not informative) | 0.712 | 0.09 / 0.65 / 0.16 | 67.67 | 0.463 | 0.001 / 0.002 |
| logistic regression on the L-window history | 0.511 | [0.000, 0.838] (2 positive episodes: not informative) | 0.668 | 0.70 / 0.57 / 0.63 | 2.66 | 0.632 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.197 | [0.000, 0.573] (2 positive episodes: not informative) | 0.723 | 0.16 / 0.69 / 0.26 | 37.23 | 0.243 | 0.000 / 0.001 |

At the mandated 0.75 threshold on the calibrated score: P 0.85 / R 0.81 / F1 0.83, 84 alerts.

### Attribution — val
**val** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.059 | [-0.175, 0.067] |
| world_model - persistence_rollout | 0.098 | [0.008, 0.265] |
| world_model - noised_persistence | 0.023 | [-0.025, 0.099] |
| world_model - isotropic_noise_persistence | 0.043 | [0.004, 0.170] |
| world_model - world_model_deterministic | 0.062 | [0.000, 0.193] |
| world_model_deterministic - persistence | -0.120 | [-0.335, 0.000] |
| noised_persistence - persistence | -0.082 | [-0.198, 0.034] |
| world_model - lr_current_state | 0.454 | [0.180, 0.556] |
| world_model - lr_flattened_history | 0.320 | [0.147, 0.395] |
| world_model - gbdt_current_state | 0.634 | [0.275, 0.726] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.609, period2_persistence 0.084, ridge_two_lag -0.279; NIDRA vs ridge 0.694.

### Horizon — val
**val** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00010 | 0.857 | 0.861 | 0.975 | 0.00071 | 0.73 (74) |
| 2 | 2 | 0.00010 | 0.968 | 0.970 | 0.976 | 0.00037 | 0.05 (73) |
| 3 | 3 | 0.00010 | 0.846 | 0.845 | 0.970 | 0.00091 | 0.00 (72) |
| 4 | 4 | 0.00010 | 0.949 | 0.937 | 0.970 | 0.00055 | 0.00 (71) |
| 5 | 5 | 0.00010 | 0.844 | 0.842 | 0.978 | 0.00093 | 0.00 (70) |
| 6 | 6 | 0.00010 | 0.885 | 0.864 | 0.971 | 0.00075 | 0.00 (69) |

### Onset forecasting — val
**val** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 0.034 | 0.021 | — | 0.000 | 0.000 | — |
| 3 | 4 | 0.101 | 0.028 | — | 0.000 | 0.000 | — |
| 5 | 6 | 0.068 | 0.023 | — | 0.000 | 0.000 | — |
| 10 | 11 | 0.041 | 0.028 | — | 0.000 | 0.000 | — |
| 15 | 16 | 0.062 | 0.035 | — | 0.001 | 0.000 | — |
| 30 | 31 | 0.074 | 0.054 | — | 0.008 | 0.000 | — |

### Episodes — val
**val** — per episode at the selected threshold (0.882), 2 consecutive windows: 2 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499188140 | 62 | 1 | 0.001 | no (—) | yes (0.0) |
| 172.16.0.1@1499278320 | 20 | 30 | 0.888 | no (—) | no (—) |

### Per attack group — val
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `tuesday:initial_access` | — | 63 | 1 | 1¹ | 0.984 | 1.000 | 0.984 | 0.318 |
| `wednesday:initial_access` | — | 26 | 1 | 1¹ | 0.287 | 0.541 | 0.278 | 0.336 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — val
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.857 | 0.994 | 0.9993 | 1 | 0.7607 | 0.964 | 1.27× | 0.8672 | carries timing signal |
| world_model | 0.857 | 0.993 | 0.9993 | 1 | 0.7607 | 0.963 | 1.27× | 0.8652 | carries timing signal |
| world_model_deterministic | 0.833 | 0.995 | 0.9988 | 1 | 0.7607 | 0.947 | 1.24× | 0.8399 | carries timing signal |
| noised_persistence | 0.833 | 0.980 | 0.9991 | 1 | 0.7607 | 0.971 | 1.28× | 0.8989 | carries timing signal |
| isotropic_noise_persistence | 0.818 | 0.956 | 0.9993 | 1 | 0.7607 | 0.966 | 1.27× | 0.8772 | carries timing signal |
| persistence | 0.937 | 0.999 | 0.9993 | 1 | 0.7607 | 0.983 | 1.29× | 0.9434 | carries timing signal |
| oracle_true_future | 0.977 | 1.000 | 0.9993 | 1 | 0.7607 | 0.997 | 1.31× | 0.9912 | carries timing signal |
| lr_current_state | 0.506 | 0.806 | 0.9993 | 1 | 0.7607 | 0.880 | 1.16× | 0.6354 | carries timing signal |
| lr_flattened_history | 0.579 | 0.778 | 0.9988 | 1 | 0.7607 | 0.904 | 1.19× | 0.6932 | carries timing signal |
| gbdt_current_state | 0.374 | 0.820 | 0.9992 | 1 | 0.7607 | 0.836 | 1.10× | 0.6152 | **host identity** |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — val
**val** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.06725 | 0.23344 | 0.00012 | 0.00012 | 10/10 |
| calibrated | 0.00007 | 0.05007 | 0.00012 | 0.00012 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `mean|q=-|integrated`, threshold 0.882 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.064 | — | 0.560 | 0.10 / 0.04 / 0.06 | 40.26 | 0.069 | 0.004 / 0.006 |
| NIDRA (stochastic rollout) | 0.049 | [0.012, 0.221] | 0.497 | 0.09 / 0.07 / 0.08 | 74.45 | 0.058 | 0.004 / 0.006 |
| NIDRA (deterministic rollout) | 0.014 | [0.003, 0.120] | 0.208 | 0.09 / 0.06 / 0.07 | 76.96 | 0.013 | 0.004 / 0.006 |
| persistence + learned noise (mean disabled) | 0.072 | [0.022, 0.226] | 0.652 | 0.10 / 0.05 / 0.06 | 48.27 | 0.075 | 0.002 / 0.003 |
| persistence + isotropic noise | 0.038 | — | 0.278 | 0.10 / 0.05 / 0.07 | 52.09 | 0.049 | 0.001 / 0.002 |
| persistence (risk head on S_t) | 0.041 | [0.007, 0.190] | 0.303 | 0.15 / 0.02 / 0.04 | 14.34 | 0.044 | 0.003 / 0.005 |
| oracle: risk head on the true future | 0.051 | — | 0.374 | 0.10 / 0.04 / 0.06 | 43.50 | 0.042 | 0.013 / 0.014 |
| logistic regression on S_t | 0.040 | [0.008, 0.177] | 0.142 | 0.17 / 0.07 / 0.10 | 41.29 | 0.046 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.035 | [0.006, 0.198] | 0.088 | 0.49 / 0.06 / 0.11 | 7.44 | 0.020 | 0.063 / 0.051 |
| gradient-boosted trees on S_t | 0.048 | [0.014, 0.137] | 0.222 | 0.36 / 0.06 / 0.11 | 13.14 | 0.054 | 0.000 / 0.001 |

At the mandated 0.75 threshold on the calibrated score: P 0.10 / R 0.04 / F1 0.06, 422 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.008 | [-0.031, 0.035] |
| world_model - persistence_rollout | 0.011 | [-0.044, 0.092] |
| world_model - noised_persistence | -0.024 | [-0.070, 0.011] |
| world_model - isotropic_noise_persistence | 0.011 | [-0.001, 0.061] |
| world_model - world_model_deterministic | 0.035 | [0.005, 0.123] |
| world_model_deterministic - persistence | -0.027 | [-0.093, 0.006] |
| noised_persistence - persistence | 0.031 | [0.008, 0.060] |
| world_model - lr_current_state | 0.008 | [-0.022, 0.064] |
| world_model - lr_flattened_history | 0.014 | [-0.046, 0.079] |
| world_model - gbdt_current_state | 0.001 | [-0.018, 0.083] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.552, period2_persistence 0.045, ridge_two_lag -0.553; NIDRA vs ridge 0.711.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.066 | 0.053 | 0.044 | 0.03504 | 0.03 (664) |
| 2 | 2 | 0.00292 | 0.058 | 0.043 | 0.043 | 0.03581 | 0.02 (662) |
| 3 | 3 | 0.00291 | 0.048 | 0.035 | 0.042 | 0.03493 | 0.00 (661) |
| 4 | 4 | 0.00291 | 0.037 | 0.030 | 0.042 | 0.03370 | 0.00 (660) |
| 5 | 5 | 0.00290 | 0.029 | 0.026 | 0.042 | 0.03233 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.023 | 0.022 | 0.042 | 0.03145 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.003 | 0.001 | — | 0.000 | 0.017 | — |
| 3 | 36 | 0.003 | 0.002 | — | 0.000 | 0.046 | — |
| 5 | 60 | 0.004 | 0.003 | — | 0.000 | 0.063 | — |
| 10 | 113 | 0.006 | 0.004 | — | 0.001 | 0.063 | — |
| 15 | 161 | 0.006 | 0.005 | — | 0.001 | 0.051 | — |
| 30 | 299 | 0.008 | 0.009 | — | 0.002 | 0.038 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.882), 2 consecutive windows: 15 episodes, 1 warned before onset, 5 alerted inside the episode; median lead 240.0 s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.022 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.004 | no (—) | yes (7.0) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.558 | no (—) | yes (0.0) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.031 | no (—) | yes (0.0) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.372 | no (—) | yes (5.0) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.039 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | yes (0.0) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.000 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.000 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.000 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 1.000 | yes (240) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.000 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.001 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.003 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `friday_morning:c2` | — | 862 | 7 | 7 | 0.011 | 0.019 | 0.010 | 0.498 |
| `friday_portscan:recon` | — | 58 | 6 | 1¹ | 0.076 | 0.049 | 0.012 | 0.391 |
| `friday_ddos:exfil` | — | 26 | 2 | 2 | 0.798 | 0.408 | 0.784 | -0.414 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.122 | 0.541 | 0.8936 | 9 | 0.3927 | 0.496 | 1.26× | 0.6035 | carries timing signal |
| world_model | 0.095 | 0.474 | 0.2746 | 9 | 0.3927 | 0.477 | 1.21× | 0.5902 | carries timing signal |
| world_model_deterministic | 0.047 | 0.261 | 0.5122 | 9 | 0.3927 | 0.453 | 1.15× | 0.5822 | carries timing signal |
| noised_persistence | 0.136 | 0.613 | 0.8429 | 9 | 0.3927 | 0.537 | 1.37× | 0.6086 | carries timing signal |
| isotropic_noise_persistence | 0.074 | 0.322 | 0.1504 | 9 | 0.3927 | 0.509 | 1.30× | 0.5808 | carries timing signal |
| persistence | 0.096 | 0.435 | 0.8185 | 9 | 0.3927 | 0.505 | 1.28× | 0.5835 | carries timing signal |
| oracle_true_future | 0.115 | 0.458 | 0.8622 | 9 | 0.3927 | 0.515 | 1.31× | 0.6031 | carries timing signal |
| lr_current_state | 0.079 | 0.258 | 0.6110 | 9 | 0.3927 | 0.471 | 1.20× | 0.5510 | carries timing signal |
| lr_flattened_history | 0.068 | 0.144 | 0.1317 | 9 | 0.3927 | 0.423 | 1.08× | 0.5174 | carries timing signal |
| gbdt_current_state | 0.114 | 0.543 | 0.7521 | 9 | 0.3927 | 0.439 | 1.12× | 0.5164 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.06332 | 0.21085 | 0.00417 | 0.00415 | 10/10 |
| calibrated | 0.00565 | 0.04791 | 0.00417 | 0.00415 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `mean|q=-|integrated`, threshold 0.882 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.380 | — | 0.734 | 0.52 / 0.35 / 0.42 | 4.88 | 0.617 | 0.004 / 0.049 |
| NIDRA (stochastic rollout) | 0.377 | [0.000, 0.915] | 0.707 | 0.19 / 0.42 / 0.26 | 27.08 | 0.619 | 0.002 / 0.046 |
| NIDRA (deterministic rollout) | 0.236 | [0.000, 0.662] | 0.526 | 0.15 / 0.39 / 0.21 | 33.44 | 0.358 | 0.001 / 0.018 |
| persistence + learned noise (mean disabled) | 0.350 | [0.001, 0.844] | 0.730 | 0.39 / 0.36 / 0.38 | 8.41 | 0.534 | 0.004 / 0.132 |
| persistence + isotropic noise | 0.339 | — | 0.483 | 0.27 / 0.38 / 0.32 | 15.06 | 0.504 | 0.005 / 0.131 |
| persistence (risk head on S_t) | 0.342 | [0.000, 0.843] | 0.479 | 0.84 / 0.34 / 0.49 | 0.99 | 0.496 | 0.019 / 0.219 |
| oracle: risk head on the true future | 0.375 | — | 0.560 | 0.65 / 0.40 / 0.50 | 3.28 | 0.481 | 0.100 / 0.242 |
| logistic regression on S_t | 0.124 | [0.000, 0.376] | 0.470 | 0.08 / 0.36 / 0.13 | 63.41 | 0.169 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.381 | [0.000, 0.662] | 0.485 | 0.82 / 0.35 / 0.49 | 1.19 | 0.209 | 0.094 / 0.215 |
| gradient-boosted trees on S_t | 0.141 | [0.001, 0.468] | 0.445 | 0.21 / 0.35 / 0.26 | 20.48 | 0.240 | 0.001 / 0.002 |

At the mandated 0.75 threshold on the calibrated score: P 0.47 / R 0.35 / F1 0.40, 90 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.035 | [0.000, 0.132] |
| world_model - persistence_rollout | 0.107 | [-0.000, 0.308] |
| world_model - noised_persistence | 0.027 | [-0.001, 0.101] |
| world_model - isotropic_noise_persistence | 0.038 | [0.000, 0.119] |
| world_model - world_model_deterministic | 0.141 | [0.000, 0.398] |
| world_model_deterministic - persistence | -0.106 | [-0.347, 0.052] |
| noised_persistence - persistence | 0.008 | [-0.001, 0.027] |
| world_model - lr_current_state | 0.253 | [-0.017, 0.650] |
| world_model - lr_flattened_history | -0.005 | [-0.357, 0.470] |
| world_model - gbdt_current_state | 0.236 | [-0.001, 0.553] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.575, period2_persistence 0.049, ridge_two_lag -0.366; NIDRA vs ridge 0.689.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.550 | 0.549 | 0.483 | 0.00217 | 0.13 (68) |
| 2 | 2 | 0.00019 | 0.510 | 0.522 | 0.465 | 0.00227 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.448 | 0.472 | 0.451 | 0.00241 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.399 | 0.437 | 0.423 | 0.00252 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.356 | 0.402 | 0.413 | 0.00259 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.341 | 0.371 | 0.396 | 0.00254 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.000 | 0.002 | — | 0.000 | 0.019 | — |
| 3 | 12 | 0.002 | 0.005 | — | 0.001 | 0.054 | — |
| 5 | 20 | 0.004 | 0.019 | — | 0.001 | 0.094 | — |
| 10 | 38 | 0.015 | 0.091 | — | 0.001 | 0.251 | — |
| 15 | 52 | 0.049 | 0.219 | — | 0.002 | 0.215 | — |
| 30 | 73 | 0.039 | 0.164 | — | 0.003 | 0.153 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.882), 2 consecutive windows: 5 episodes, 1 warned before onset, 2 alerted inside the episode; median lead 840.0 s, median latency 14.5 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | yes (29.0) |
| 172.16.0.1@1499346900 | 27 | 14 | 1.000 | yes (840) | yes (0.0) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.002 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.000 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.518 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `thursday_infiltration:lateral` | — | 73 | 3 | 1¹ | 0.004 | 0.001 | 0.000 | 0.406 |
| `thursday_web:initial_access` | — | 49 | 2 | 1¹ | 0.885 | 0.914 | 0.847 | 0.246 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.418 | 0.734 | 0.9786 | 2 | 0.3664 | 0.705 | 1.92× | 0.7503 | carries timing signal |
| world_model | 0.408 | 0.703 | 0.5415 | 2 | 0.3664 | 0.710 | 1.94× | 0.7515 | carries timing signal |
| world_model_deterministic | 0.308 | 0.570 | 0.9731 | 2 | 0.3664 | 0.671 | 1.83× | 0.7824 | carries timing signal |
| noised_persistence | 0.375 | 0.715 | 0.8963 | 2 | 0.3664 | 0.647 | 1.77× | 0.6778 | carries timing signal |
| isotropic_noise_persistence | 0.353 | 0.516 | 0.4405 | 2 | 0.3664 | 0.640 | 1.75× | 0.6867 | carries timing signal |
| persistence | 0.363 | 0.580 | 0.9852 | 2 | 0.3664 | 0.634 | 1.73× | 0.6984 | carries timing signal |
| oracle_true_future | 0.396 | 0.617 | 0.9849 | 2 | 0.3664 | 0.645 | 1.76× | 0.6809 | carries timing signal |
| lr_current_state | 0.213 | 0.540 | 0.8759 | 2 | 0.3664 | 0.624 | 1.70× | 0.6381 | carries timing signal |
| lr_flattened_history | 0.395 | 0.585 | 0.9621 | 2 | 0.3664 | 0.746 | 2.04× | 0.7796 | carries timing signal |
| gbdt_current_state | 0.226 | 0.737 | 0.8207 | 2 | 0.3664 | 0.604 | 1.65× | 0.6761 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.06534 | 0.22726 | 0.00034 | 0.00034 | 10/10 |
| calibrated | 0.00039 | 0.05007 | 0.00034 | 0.00034 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

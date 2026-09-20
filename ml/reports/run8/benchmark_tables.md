### Systems — val
**val** — 20113 scored rows, natural prevalence 0.00012, operating point `median|q=-|max`, threshold 0.718 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.726 | — | 0.995 | 1.00 / 0.66 / 0.80 | 0.00 | 0.878 | 0.002 / 0.005 |
| NIDRA (stochastic rollout) | 0.726 | [0.014, 0.969] (2 positive episodes: not informative) | 0.995 | 1.00 / 0.65 / 0.79 | 0.00 | 0.878 | 0.002 / 0.005 |
| NIDRA (deterministic rollout) | 0.748 | [0.030, 0.964] (2 positive episodes: not informative) | 0.991 | 0.91 / 0.66 / 0.77 | 0.12 | 0.904 | 0.001 / 0.014 |
| persistence + learned noise (mean disabled) | 0.756 | [0.032, 0.969] (2 positive episodes: not informative) | 0.987 | 0.94 / 0.67 / 0.78 | 0.08 | 0.926 | 0.001 / 0.016 |
| persistence + isotropic noise | 0.734 | — | 0.981 | 0.89 / 0.66 / 0.76 | 0.14 | 0.901 | 0.001 / 0.016 |
| persistence (risk head on S_t) | 0.749 | [0.033, 0.968] (2 positive episodes: not informative) | 0.971 | 0.89 / 0.67 / 0.77 | 0.14 | 0.920 | 0.001 / 0.022 |
| ridge two-lag dynamics + risk head | 0.693 | [0.002, 0.984] (2 positive episodes: not informative) | 0.978 | 0.73 / 0.69 / 0.71 | 0.45 | 0.848 | 0.003 / 0.075 |
| oracle: risk head on the true future | 0.770 | — | 1.000 | 0.57 / 0.71 / 0.63 | 0.93 | 0.870 | 0.191 / 0.076 |
| logistic regression on S_t | 0.432 | [0.000, 0.873] (2 positive episodes: not informative) | 0.725 | 0.07 / 0.69 / 0.13 | 15.76 | 0.530 | 0.001 / 0.002 |
| logistic regression on the L-window history | 0.519 | [0.000, 0.958] (2 positive episodes: not informative) | 0.737 | 0.26 / 0.67 / 0.38 | 3.29 | 0.641 | 0.000 / 0.004 |
| gradient-boosted trees on S_t | 0.688 | [0.000, 0.982] (2 positive episodes: not informative) | 0.734 | 0.25 / 0.70 / 0.36 | 3.77 | 0.849 | 0.000 / 0.007 |
| GRU sequence classifier | 0.706 | [0.007, 0.983] (2 positive episodes: not informative) | 0.979 | 0.90 / 0.67 / 0.77 | 0.13 | 0.849 | 0.030 / 0.013 |
| onset head (explicit supervision, Task B only) | — | — | — | — | — | — | 0.000 / 0.063 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.65 / F1 0.79, 58 alerts.

### Attribution — val
**val** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.024 | [-0.045, 0.052] |
| world_model - noised_persistence | -0.030 | [-0.048, 0.031] |
| world_model - isotropic_noise_persistence | -0.008 | [-0.016, 0.051] |
| world_model - world_model_deterministic | -0.023 | [-0.053, 0.013] |
| world_model - ridge_two_lag | 0.033 | [-0.022, 0.131] |
| world_model_deterministic - persistence | -0.001 | [-0.020, 0.058] |
| world_model_deterministic - ridge_two_lag | 0.055 | [-0.030, 0.152] |
| noised_persistence - persistence | 0.007 | [-0.007, 0.035] |
| world_model - lr_current_state | 0.294 | [0.014, 0.426] |
| world_model - lr_flattened_history | 0.207 | [0.006, 0.328] |
| world_model - gbdt_current_state | 0.038 | [-0.021, 0.138] |
| world_model - gru_classifier | 0.020 | [-0.023, 0.115] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.672, period2_persistence 0.081, ridge_two_lag 0.653; NIDRA vs ridge 0.056.

### Horizon — val
**val** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00010 | 0.832 | 0.828 | 0.894 | 0.00078 | 0.76 (74) |
| 2 | 2 | 0.00010 | 0.855 | 0.876 | 0.905 | 0.00076 | 0.73 (73) |
| 3 | 3 | 0.00010 | 0.834 | 0.828 | 0.893 | 0.00072 | 0.71 (72) |
| 4 | 4 | 0.00010 | 0.855 | 0.905 | 0.906 | 0.00066 | 0.69 (71) |
| 5 | 5 | 0.00010 | 0.844 | 0.854 | 0.880 | 0.00063 | 0.56 (70) |
| 6 | 6 | 0.00010 | 0.819 | 0.903 | 0.878 | 0.00069 | 0.42 (69) |

### Onset forecasting — val
**val** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 0.001 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |
| 3 | 4 | 0.003 | 0.001 | 0.004 | 0.001 | 0.000 | 0.000 |
| 5 | 6 | 0.002 | 0.001 | 0.003 | 0.000 | 0.000 | 0.000 |
| 10 | 11 | 0.002 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | 0.005 | 0.022 | 0.075 | 0.007 | 0.004 | 0.063 |
| 30 | 31 | 0.038 | 0.131 | 0.124 | 0.161 | 0.002 | 0.115 |

### Episodes — val
**val** — per episode at the selected threshold (0.718), 2 consecutive windows: 2 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499188140 | 62 | 1 | 0.000 | no (—) | yes (0.0) |
| 172.16.0.1@1499278320 | 20 | 30 | 0.386 | no (—) | no (—) |

### Systems — test
**test** — 21173 scored rows, natural prevalence 0.00417, operating point `median|q=-|max`, threshold 0.718 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.058 | — | 0.398 | 0.89 / 0.03 / 0.06 | 0.48 | 0.069 | 0.010 / 0.013 |
| NIDRA (stochastic rollout) | 0.058 | [0.018, 0.199] | 0.394 | 0.89 / 0.03 / 0.06 | 0.48 | 0.069 | 0.010 / 0.013 |
| NIDRA (deterministic rollout) | 0.081 | [0.031, 0.216] | 0.509 | 0.77 / 0.03 / 0.06 | 1.20 | 0.088 | 0.005 / 0.010 |
| persistence + learned noise (mean disabled) | 0.057 | [0.018, 0.201] | 0.338 | 0.90 / 0.03 / 0.07 | 0.48 | 0.074 | 0.012 / 0.012 |
| persistence + isotropic noise | 0.064 | — | 0.346 | 0.76 / 0.03 / 0.06 | 1.20 | 0.081 | 0.009 / 0.009 |
| persistence (risk head on S_t) | 0.065 | [0.020, 0.206] | 0.371 | 0.73 / 0.03 / 0.06 | 1.44 | 0.082 | 0.006 / 0.007 |
| ridge two-lag dynamics + risk head | 0.083 | [0.030, 0.218] | 0.460 | 0.76 / 0.03 / 0.06 | 1.20 | 0.092 | 0.001 / 0.003 |
| oracle: risk head on the true future | 0.117 | — | 0.669 | 0.49 / 0.06 / 0.11 | 7.66 | 0.081 | 0.240 / 0.107 |
| logistic regression on S_t | 0.033 | [0.005, 0.151] | 0.113 | 0.12 / 0.05 / 0.08 | 44.81 | 0.044 | 0.000 / 0.002 |
| logistic regression on the L-window history | 0.036 | [0.006, 0.165] | 0.097 | 0.48 / 0.07 / 0.12 | 8.67 | 0.021 | 0.067 / 0.054 |
| gradient-boosted trees on S_t | 0.024 | [0.006, 0.113] | 0.113 | 0.36 / 0.04 / 0.07 | 7.65 | 0.030 | 0.000 / 0.001 |
| GRU sequence classifier | 0.164 | [0.059, 0.342] | 0.976 | 0.49 / 0.03 / 0.05 | 3.11 | 0.139 | 0.013 / 0.018 |
| onset head (explicit supervision, Task B only) | — | — | — | — | — | — | 0.001 / 0.002 |

At the mandated 0.75 threshold on the calibrated score: P 0.94 / R 0.03 / F1 0.06, 31 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.007 | [-0.026, 0.011] |
| world_model - noised_persistence | 0.001 | [-0.011, 0.021] |
| world_model - isotropic_noise_persistence | -0.006 | [-0.023, 0.013] |
| world_model - world_model_deterministic | -0.023 | [-0.048, -0.003] |
| world_model - ridge_two_lag | -0.025 | [-0.074, -0.000] |
| world_model_deterministic - persistence | 0.016 | [-0.007, 0.043] |
| world_model_deterministic - ridge_two_lag | -0.002 | [-0.041, 0.015] |
| noised_persistence - persistence | -0.008 | [-0.025, 0.006] |
| world_model - lr_current_state | 0.025 | [0.004, 0.058] |
| world_model - lr_flattened_history | 0.022 | [-0.030, 0.076] |
| world_model - gbdt_current_state | 0.034 | [0.008, 0.100] |
| world_model - gru_classifier | -0.106 | [-0.293, 0.020] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.587, period2_persistence 0.039, ridge_two_lag 0.565; NIDRA vs ridge 0.052.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00293 | 0.063 | 0.076 | 0.082 | 0.03014 | 0.00 (664) |
| 2 | 2 | 0.00292 | 0.048 | 0.066 | 0.078 | 0.03041 | 0.00 (662) |
| 3 | 3 | 0.00291 | 0.042 | 0.059 | 0.080 | 0.03048 | 0.00 (661) |
| 4 | 4 | 0.00291 | 0.036 | 0.046 | 0.078 | 0.03058 | 0.00 (660) |
| 5 | 5 | 0.00290 | 0.030 | 0.041 | 0.080 | 0.03079 | 0.00 (658) |
| 6 | 6 | 0.00289 | 0.024 | 0.029 | 0.078 | 0.03097 | 0.00 (656) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.003 | 0.003 | 0.000 | 0.000 | 0.016 | 0.000 |
| 3 | 36 | 0.007 | 0.003 | 0.000 | 0.000 | 0.044 | 0.000 |
| 5 | 60 | 0.010 | 0.006 | 0.001 | 0.000 | 0.067 | 0.001 |
| 10 | 113 | 0.016 | 0.009 | 0.003 | 0.001 | 0.066 | 0.001 |
| 15 | 161 | 0.013 | 0.007 | 0.003 | 0.001 | 0.054 | 0.002 |
| 30 | 299 | 0.010 | 0.005 | 0.003 | 0.001 | 0.038 | 0.003 |

### Episodes — test
**test** — per episode at the selected threshold (0.718), 2 consecutive windows: 15 episodes, 0 warned before onset, 3 alerted inside the episode; median lead None s, median latency 0.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499443500 | 1 | 4 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499446320 | 9 | 30 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499447640 | 2 | 13 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499449860 | 5 | 30 | 0.000 | no (—) | yes (0.0) |
| 172.16.0.1@1499450580 | 10 | 6 | 0.000 | no (—) | yes (5.0) |
| 172.16.0.1@1499451660 | 2 | 7 | 0.000 | no (—) | no (—) |
| 172.16.0.1@1499453760 | 20 | 0 | — | no (—) | yes (0.0) |
| 192.168.10.14@1499433840 | 156 | 30 | 0.000 | no (—) | no (—) |
| 192.168.10.15@1499432760 | 174 | 29 | 0.000 | no (—) | no (—) |
| 192.168.10.17@1499437200 | 2 | 30 | 0.008 | no (—) | no (—) |
| 192.168.10.50@1499454360 | 1 | 30 | 0.586 | no (—) | no (—) |
| 192.168.10.5@1499434140 | 151 | 30 | 0.000 | no (—) | no (—) |
| 192.168.10.8@1499434560 | 143 | 30 | 0.000 | no (—) | no (—) |
| 192.168.10.9@1499432640 | 175 | 30 | 0.004 | no (—) | no (—) |
| 205.174.165.73@1499432640 | 29 | 0 | — | no (—) | no (—) |

### Systems — holdout
**holdout** — 20171 scored rows, natural prevalence 0.00034, operating point `median|q=-|max`, threshold 0.718 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.438 | — | 0.646 | 0.85 / 0.32 / 0.46 | 0.86 | 0.638 | 0.001 / 0.002 |
| NIDRA (stochastic rollout) | 0.439 | [0.000, 0.768] | 0.644 | 0.89 / 0.32 / 0.47 | 0.58 | 0.638 | 0.001 / 0.002 |
| NIDRA (deterministic rollout) | 0.380 | [0.000, 0.667] | 0.676 | 0.72 / 0.40 / 0.52 | 2.30 | 0.491 | 0.001 / 0.002 |
| persistence + learned noise (mean disabled) | 0.383 | [0.000, 0.709] | 0.616 | 0.78 / 0.34 / 0.47 | 1.44 | 0.516 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.293 | — | 0.588 | 0.78 / 0.34 / 0.48 | 1.44 | 0.373 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.295 | [0.000, 0.553] | 0.592 | 0.72 / 0.34 / 0.47 | 2.01 | 0.391 | 0.000 / 0.000 |
| ridge two-lag dynamics + risk head | 0.275 | [0.000, 0.725] | 0.471 | 0.61 / 0.30 / 0.41 | 2.88 | 0.515 | 0.000 / 0.001 |
| oracle: risk head on the true future | 0.400 | — | 0.843 | 0.60 / 0.55 / 0.57 | 5.47 | 0.317 | 0.098 / 0.039 |
| logistic regression on S_t | 0.241 | [0.000, 0.432] | 0.531 | 0.09 / 0.47 / 0.15 | 70.17 | 0.260 | 0.000 / 0.001 |
| logistic regression on the L-window history | 0.443 | [0.000, 0.719] | 0.590 | 0.68 / 0.53 / 0.60 | 3.74 | 0.229 | 0.029 / 0.081 |
| gradient-boosted trees on S_t | 0.294 | [0.000, 0.742] | 0.407 | 0.27 / 0.34 / 0.31 | 13.78 | 0.527 | 0.001 / 0.002 |
| GRU sequence classifier | 0.382 | [0.004, 0.731] | 0.997 | 0.84 / 0.35 / 0.50 | 0.99 | 0.421 | 0.057 / 0.271 |
| onset head (explicit supervision, Task B only) | — | — | — | — | — | — | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 0.89 / R 0.31 / F1 0.46, 42 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.143 | [-0.000, 0.283] |
| world_model - noised_persistence | 0.055 | [-0.001, 0.147] |
| world_model - isotropic_noise_persistence | 0.146 | [-0.000, 0.305] |
| world_model - world_model_deterministic | 0.058 | [-0.014, 0.169] |
| world_model - ridge_two_lag | 0.163 | [-0.003, 0.339] |
| world_model_deterministic - persistence | 0.085 | [0.000, 0.147] |
| world_model_deterministic - ridge_two_lag | 0.105 | [-0.094, 0.303] |
| noised_persistence - persistence | 0.088 | [-0.000, 0.183] |
| world_model - lr_current_state | 0.198 | [-0.000, 0.366] |
| world_model - lr_flattened_history | -0.004 | [-0.292, 0.423] |
| world_model - gbdt_current_state | 0.145 | [-0.050, 0.330] |
| world_model - gru_classifier | 0.057 | [-0.081, 0.174] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.616, period2_persistence 0.044, ridge_two_lag 0.595; NIDRA vs ridge 0.052.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00019 | 0.582 | 0.480 | 0.400 | 0.00180 | 0.04 (68) |
| 2 | 2 | 0.00019 | 0.505 | 0.367 | 0.393 | 0.00196 | 0.00 (67) |
| 3 | 3 | 0.00018 | 0.454 | 0.326 | 0.369 | 0.00209 | 0.00 (66) |
| 4 | 4 | 0.00018 | 0.404 | 0.429 | 0.436 | 0.00216 | 0.00 (65) |
| 5 | 5 | 0.00018 | 0.352 | 0.317 | 0.451 | 0.00230 | 0.00 (64) |
| 6 | 6 | 0.00017 | 0.285 | 0.355 | 0.416 | 0.00242 | 0.00 (63) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 |
| 3 | 12 | 0.000 | 0.000 | 0.000 | 0.001 | 0.018 | 0.000 |
| 5 | 20 | 0.001 | 0.000 | 0.000 | 0.001 | 0.029 | 0.000 |
| 10 | 38 | 0.001 | 0.000 | 0.001 | 0.001 | 0.070 | 0.000 |
| 15 | 52 | 0.002 | 0.000 | 0.001 | 0.002 | 0.081 | 0.000 |
| 30 | 73 | 0.003 | 0.001 | 0.002 | 0.003 | 0.058 | 0.001 |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.718), 2 consecutive windows: 5 episodes, 0 warned before onset, 3 alerted inside the episode; median lead None s, median latency 8.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 172.16.0.1@1499343300 | 16 | 0 | — | no (—) | yes (29.0) |
| 172.16.0.1@1499346900 | 27 | 14 | 0.003 | no (—) | yes (1.0) |
| 192.168.10.8@1499361540 | 0 | 30 | 0.004 | no (—) | no (—) |
| 192.168.10.8@1499362080 | 14 | 8 | 0.000 | no (—) | no (—) |
| 192.168.10.8@1499364240 | 41 | 21 | 0.044 | no (—) | yes (8.0) |

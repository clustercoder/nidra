_val: no benchmark.json at experiments/runs/xeval_ctu2ctu_state+hidden/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 25099 scored rows, natural prevalence 0.00749, operating point `mean|q=-|max`, threshold 0.108 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.322 | — | 0.680 | 0.81 / 0.26 / 0.40 | 7.67 | 0.424 | 0.003 / 0.010 |
| NIDRA (stochastic rollout) | 0.320 | [0.121, 0.582] | 0.692 | 0.18 / 0.42 / 0.25 | 234.82 | 0.422 | 0.003 / 0.010 |
| NIDRA (deterministic rollout) | 0.305 | [0.113, 0.581] | 0.633 | 0.59 / 0.39 / 0.47 | 34.76 | 0.403 | 0.002 / 0.006 |
| persistence + learned noise (mean disabled) | 0.339 | [0.165, 0.583] | 0.801 | 0.12 / 0.48 / 0.19 | 452.69 | 0.441 | 0.003 / 0.007 |
| persistence + isotropic noise | 0.319 | — | 0.737 | 0.01 / 0.74 / 0.02 | 8255.40 | 0.414 | 0.002 / 0.006 |
| persistence (risk head on S_t) | 0.321 | [0.137, 0.591] | 0.665 | 0.59 / 0.41 / 0.48 | 35.64 | 0.421 | 0.003 / 0.007 |
| oracle: risk head on the true future | 0.352 | — | 0.702 | 0.50 / 0.48 / 0.49 | 59.17 | 0.417 | 0.014 / 0.016 |
| logistic regression on S_t | 0.197 | [0.094, 0.407] | 0.808 | 0.11 / 0.60 / 0.19 | 616.56 | 0.265 | 0.002 / 0.004 |
| logistic regression on the L-window history | 0.042 | [0.013, 0.163] | 0.421 | 0.18 / 0.18 / 0.18 | 98.48 | 0.053 | 0.001 / 0.003 |
| gradient-boosted trees on S_t | 0.242 | [0.111, 0.470] | 0.632 | 0.13 / 0.43 / 0.20 | 344.05 | 0.325 | 0.002 / 0.004 |
| GRU sequence classifier | 0.291 | [0.121, 0.504] | 0.576 | 0.69 / 0.32 / 0.44 | 17.82 | 0.388 | 0.002 / 0.005 |

At the mandated 0.75 threshold on the calibrated score: P 0.58 / R 0.01 / F1 0.01, 36 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | -0.001 | [-0.028, 0.020] |
| world_model - persistence_rollout | 0.013 | [-0.023, 0.047] |
| world_model - noised_persistence | -0.019 | [-0.070, 0.021] |
| world_model - isotropic_noise_persistence | 0.001 | [-0.045, 0.034] |
| world_model - world_model_deterministic | 0.015 | [-0.006, 0.028] |
| world_model_deterministic - persistence | -0.016 | [-0.032, 0.001] |
| noised_persistence - persistence | 0.018 | [-0.010, 0.050] |
| world_model - lr_current_state | 0.123 | [-0.049, 0.274] |
| world_model - lr_flattened_history | 0.279 | [0.108, 0.445] |
| world_model - gbdt_current_state | 0.078 | [0.000, 0.149] |
| world_model - gru_classifier | 0.029 | [-0.019, 0.112] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.545, period2_persistence 0.028, ridge_two_lag 0.494; NIDRA vs ridge 0.101.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00568 | 0.420 | 0.395 | 0.420 | 0.08839 | 0.07 (2768) |
| 2 | 2 | 0.00565 | 0.416 | 0.385 | 0.414 | 0.09740 | 0.00 (2774) |
| 3 | 3 | 0.00566 | 0.399 | 0.379 | 0.418 | 0.10053 | 0.00 (2781) |
| 4 | 4 | 0.00566 | 0.386 | 0.377 | 0.415 | 0.10245 | 0.00 (2787) |
| 5 | 5 | 0.00568 | 0.363 | 0.372 | 0.404 | 0.10303 | 0.00 (2795) |
| 6 | 6 | 0.00571 | 0.358 | 0.366 | 0.400 | 0.10339 | 0.00 (2804) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 126 | 0.001 | 0.001 | — | 0.000 | 0.000 | — |
| 3 | 382 | 0.002 | 0.002 | — | 0.001 | 0.001 | — |
| 5 | 635 | 0.003 | 0.003 | — | 0.002 | 0.001 | — |
| 10 | 1188 | 0.006 | 0.005 | — | 0.003 | 0.002 | — |
| 15 | 1523 | 0.010 | 0.007 | — | 0.004 | 0.003 | — |
| 30 | 2171 | 0.011 | 0.008 | — | 0.006 | 0.005 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.108), 2 consecutive windows: 131 episodes, 0 warned before onset, 11 alerted inside the episode; median lead None s, median latency 21.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313504340 | 216 | 0 | — | no (—) | yes (71.0) |
| 147.32.84.165@1313519940 | 12 | 14 | 0.013 | no (—) | no (—) |
| 147.32.84.165@1313522640 | 863 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.165@1313586240 | 0 | 30 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313588580 | 0 | 30 | 0.005 | no (—) | no (—) |
| 147.32.84.165@1313591040 | 164 | 30 | 0.004 | no (—) | yes (15.0) |
| 147.32.84.165@1313664420 | 0 | 8 | 0.065 | no (—) | no (—) |
| 147.32.84.165@1313665440 | 2 | 16 | 0.077 | no (—) | no (—) |
| 147.32.84.165@1313668380 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313669160 | 16 | 12 | 0.005 | no (—) | no (—) |
| 147.32.84.165@1313670780 | 2 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313675220 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313675820 | 0 | 9 | 0.002 | no (—) | no (—) |
| 147.32.84.165@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313677800 | 1 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313678460 | 5 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313679420 | 2 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313586240 | 0 | 5 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313588580 | 0 | 30 | 0.009 | no (—) | no (—) |
| 147.32.84.191@1313592180 | 149 | 30 | 0.003 | no (—) | yes (17.0) |
| 147.32.84.191@1313665560 | 0 | 17 | 0.120 | no (—) | no (—) |
| 147.32.84.191@1313666580 | 0 | 16 | 0.023 | no (—) | no (—) |
| 147.32.84.191@1313668380 | 0 | 29 | 0.005 | no (—) | no (—) |
| 147.32.84.191@1313669160 | 16 | 12 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313670720 | 3 | 9 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313675280 | 0 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313675820 | 0 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.191@1313676360 | 10 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.191@1313677800 | 0 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313678460 | 6 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313679420 | 2 | 9 | 0.004 | no (—) | no (—) |
| 147.32.84.192@1313586240 | 1 | 4 | 0.003 | no (—) | no (—) |
| 147.32.84.192@1313588580 | 0 | 30 | 0.006 | no (—) | no (—) |
| 147.32.84.192@1313592480 | 139 | 30 | 0.004 | no (—) | yes (48.0) |
| 147.32.84.192@1313665500 | 0 | 9 | 0.061 | no (—) | no (—) |
| 147.32.84.192@1313666580 | 0 | 17 | 0.020 | no (—) | no (—) |
| 147.32.84.192@1313668380 | 0 | 29 | 0.005 | no (—) | no (—) |
| 147.32.84.192@1313669280 | 14 | 14 | 0.005 | no (—) | no (—) |
| 147.32.84.192@1313670720 | 3 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313675280 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.192@1313675820 | 0 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313676360 | 9 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.192@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313678460 | 7 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313679480 | 1 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313586240 | 0 | 3 | 0.005 | no (—) | no (—) |
| 147.32.84.193@1313588580 | 0 | 30 | 0.006 | no (—) | no (—) |
| 147.32.84.193@1313592600 | 140 | 30 | 0.003 | no (—) | yes (22.0) |
| 147.32.84.193@1313665620 | 0 | 10 | 0.055 | no (—) | no (—) |
| 147.32.84.193@1313666580 | 0 | 15 | 0.021 | no (—) | no (—) |
| 147.32.84.193@1313668380 | 0 | 29 | 0.007 | no (—) | no (—) |
| 147.32.84.193@1313669100 | 18 | 11 | 0.004 | no (—) | no (—) |
| 147.32.84.193@1313670720 | 3 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313675340 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313675820 | 0 | 7 | 0.004 | no (—) | no (—) |
| 147.32.84.193@1313676360 | 9 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.193@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313678460 | 7 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313679480 | 0 | 9 | 0.005 | no (—) | no (—) |
| 147.32.84.204@1313586240 | 0 | 1 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313588580 | 0 | 30 | 0.005 | no (—) | no (—) |
| 147.32.84.204@1313592600 | 139 | 30 | 0.003 | no (—) | yes (21.0) |
| 147.32.84.204@1313665500 | 0 | 7 | 0.072 | no (—) | no (—) |
| 147.32.84.204@1313666520 | 0 | 16 | 0.026 | no (—) | no (—) |
| 147.32.84.204@1313668380 | 0 | 30 | 0.018 | no (—) | no (—) |
| 147.32.84.204@1313669280 | 14 | 14 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313670720 | 3 | 9 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313675340 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313675820 | 0 | 7 | 0.001 | no (—) | no (—) |
| 147.32.84.204@1313676360 | 9 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.204@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313678460 | 8 | 10 | 0.004 | no (—) | no (—) |
| 147.32.84.204@1313679420 | 2 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313586240 | 1 | 0 | — | no (—) | no (—) |
| 147.32.84.205@1313588580 | 0 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.205@1313592600 | 139 | 30 | 0.003 | no (—) | yes (15.0) |
| 147.32.84.205@1313665620 | 1 | 9 | 0.073 | no (—) | no (—) |
| 147.32.84.205@1313666640 | 0 | 16 | 0.026 | no (—) | no (—) |
| 147.32.84.205@1313668380 | 1 | 28 | 0.004 | no (—) | no (—) |
| 147.32.84.205@1313669160 | 15 | 12 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313670780 | 2 | 11 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313675460 | 6 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313676360 | 10 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313679480 | 1 | 7 | 0.002 | no (—) | no (—) |
| 147.32.84.206@1313588580 | 0 | 30 | 0.006 | no (—) | no (—) |
| 147.32.84.206@1313592600 | 146 | 30 | 0.003 | no (—) | yes (19.0) |
| 147.32.84.206@1313666460 | 1 | 15 | 0.032 | no (—) | no (—) |
| 147.32.84.206@1313668380 | 0 | 30 | 0.030 | no (—) | no (—) |
| 147.32.84.206@1313669100 | 16 | 11 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313670720 | 3 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313675460 | 6 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.206@1313676360 | 9 | 8 | 0.005 | no (—) | no (—) |
| 147.32.84.206@1313677800 | 1 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313679480 | 0 | 6 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313588580 | 0 | 30 | 0.008 | no (—) | no (—) |
| 147.32.84.207@1313592600 | 143 | 30 | 0.003 | no (—) | yes (29.0) |
| 147.32.84.207@1313666460 | 0 | 16 | 0.045 | no (—) | no (—) |
| 147.32.84.207@1313668380 | 5 | 30 | 0.017 | no (—) | no (—) |
| 147.32.84.207@1313669220 | 15 | 9 | 0.008 | no (—) | no (—) |
| 147.32.84.207@1313670720 | 3 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313675520 | 5 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.207@1313676360 | 9 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.207@1313677800 | 0 | 14 | 0.004 | no (—) | no (—) |
| 147.32.84.207@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313679540 | 0 | 7 | 0.002 | no (—) | no (—) |
| 147.32.84.208@1313588580 | 0 | 30 | 0.008 | no (—) | no (—) |
| 147.32.84.208@1313590380 | 0 | 29 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313592600 | 6 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313593380 | 131 | 6 | 0.008 | no (—) | yes (50.0) |
| 147.32.84.208@1313666400 | 0 | 15 | 0.040 | no (—) | no (—) |
| 147.32.84.208@1313668380 | 4 | 30 | 0.020 | no (—) | no (—) |
| 147.32.84.208@1313669100 | 17 | 7 | 0.010 | no (—) | no (—) |
| 147.32.84.208@1313670720 | 2 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313675520 | 5 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313676360 | 9 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.208@1313677800 | 20 | 14 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313679540 | 0 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313588580 | 0 | 30 | 0.006 | no (—) | no (—) |
| 147.32.84.209@1313592600 | 7 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.209@1313593500 | 127 | 8 | 0.011 | no (—) | yes (18.0) |
| 147.32.84.209@1313666340 | 0 | 15 | 0.027 | no (—) | no (—) |
| 147.32.84.209@1313668380 | 0 | 30 | 0.015 | no (—) | no (—) |
| 147.32.84.209@1313669280 | 25 | 14 | 0.005 | no (—) | no (—) |
| 147.32.84.209@1313675520 | 5 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.209@1313676360 | 9 | 8 | 0.004 | no (—) | no (—) |
| 147.32.84.209@1313677800 | 0 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.209@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313679540 | 0 | 7 | 0.000 | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_9:exfil` | neris | 1373 | 11 | 10 | 0.742 | 0.692 | 0.718 | 0.444 |
| `ctu_8:c2` | murlo | 1069 | 3 | 1¹ | 0.006 | 0.029 | 0.011 | 0.300 |
| `ctu_10:recon` | rbot | 488 | 59 | 10 | 0.001 | 0.001 | 0.001 | 0.213 |
| `ctu_10:exfil` | rbot | 307 | 38 | 10 | 0.001 | 0.001 | 0.000 | 0.520 |
| `ctu_10:c2` | rbot | 232 | 45 | 10 | 0.015 | 0.015 | 0.012 | 0.082 |
| `ctu_9:recon` | neris | 151 | 22 | 10 | 0.000 | 0.011 | 0.001 | 0.083 |
| `ctu_9:c2` | neris | 35 | 8 | 8 | 0.005 | 0.025 | 0.007 | 0.120 |
| `ctu_8:recon` | murlo | 31 | 3 | 1¹ | 0.001 | 0.010 | 0.004 | 0.402 |
| `ctu_8:exfil` | murlo | 2 | 2 | 1¹ | 0.000 | 0.001 | 0.000 | 0.093 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.523 | 0.685 | 0.9532 | 10 | 0.7192 | 0.852 | 1.19× | 0.5956 | carries timing signal |
| world_model | 0.525 | 0.697 | 0.9526 | 10 | 0.7192 | 0.848 | 1.18× | 0.5981 | carries timing signal |
| world_model_deterministic | 0.527 | 0.715 | 0.9525 | 10 | 0.7192 | 0.822 | 1.14× | 0.5464 | carries timing signal |
| noised_persistence | 0.569 | 0.753 | 0.9527 | 10 | 0.7192 | 0.899 | 1.25× | 0.7321 | carries timing signal |
| isotropic_noise_persistence | 0.551 | 0.715 | 0.9527 | 10 | 0.7192 | 0.888 | 1.24× | 0.7074 | carries timing signal |
| persistence | 0.560 | 0.747 | 0.9526 | 10 | 0.7192 | 0.838 | 1.16× | 0.5797 | carries timing signal |
| oracle_true_future | 0.610 | 0.762 | 0.9526 | 10 | 0.7192 | 0.865 | 1.20× | 0.6238 | carries timing signal |
| lr_current_state | 0.567 | 0.841 | 0.9268 | 10 | 0.7192 | 0.905 | 1.26× | 0.7924 | carries timing signal |
| lr_flattened_history | 0.266 | 0.559 | 0.9407 | 10 | 0.7192 | 0.705 | 0.98× | 0.3565 | **host identity** |
| gbdt_current_state | 0.518 | 0.752 | 0.9525 | 10 | 0.7192 | 0.817 | 1.14× | 0.5630 | carries timing signal |
| gru_classifier | 0.491 | 0.680 | 0.9526 | 10 | 0.7192 | 0.809 | 1.13× | 0.5344 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00614 | 0.04795 | 0.00749 | 0.00743 | 10/10 |
| calibrated | 0.00643 | 0.04570 | 0.00749 | 0.00743 | 9/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 21045 scored rows, natural prevalence 0.00409, operating point `mean|q=-|max`, threshold 0.108 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.221 | — | 0.769 | 1.00 / 0.15 / 0.27 | 0.00 | 0.224 | 0.005 / 0.006 |
| NIDRA (stochastic rollout) | 0.224 | [0.003, 0.289] | 0.824 | 0.12 / 0.23 / 0.16 | 96.83 | 0.227 | 0.005 / 0.007 |
| NIDRA (deterministic rollout) | 0.195 | [0.000, 0.216] | 0.492 | 0.51 / 0.19 / 0.28 | 10.94 | 0.204 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.247 | [0.001, 0.353] | 0.887 | 0.09 / 0.38 / 0.15 | 215.74 | 0.253 | 0.002 / 0.003 |
| persistence + isotropic noise | 0.217 | — | 0.837 | 0.01 / 0.90 / 0.02 | 6605.55 | 0.226 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.195 | [0.000, 0.215] | 0.466 | 0.53 / 0.19 / 0.28 | 10.10 | 0.204 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.197 | — | 0.526 | 0.29 / 0.20 / 0.24 | 27.82 | 0.204 | 0.000 / 0.000 |
| logistic regression on S_t | 0.045 | [0.001, 0.135] | 0.354 | 0.03 / 0.22 / 0.06 | 363.85 | 0.046 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.074 | [0.000, 0.119] | 0.251 | 0.10 / 0.13 / 0.11 | 66.15 | 0.077 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.198 | [0.001, 0.237] | 0.533 | 0.07 / 0.23 / 0.10 | 194.45 | 0.206 | 0.000 / 0.000 |
| GRU sequence classifier | 0.189 | [0.000, 0.205] | 0.277 | 0.97 / 0.18 / 0.31 | 0.39 | 0.198 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.00 / F1 0.01, 3 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.029 | [0.003, 0.075] |
| world_model - persistence_rollout | 0.022 | [0.003, 0.043] |
| world_model - noised_persistence | -0.023 | [-0.062, 0.011] |
| world_model - isotropic_noise_persistence | 0.007 | [-0.005, 0.017] |
| world_model - world_model_deterministic | 0.029 | [0.002, 0.075] |
| world_model_deterministic - persistence | -0.000 | [-0.002, 0.003] |
| noised_persistence - persistence | 0.051 | [0.001, 0.138] |
| world_model - lr_current_state | 0.180 | [0.001, 0.199] |
| world_model - lr_flattened_history | 0.150 | [0.003, 0.183] |
| world_model - gbdt_current_state | 0.026 | [0.002, 0.055] |
| world_model - gru_classifier | 0.035 | [0.003, 0.086] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.625, period2_persistence 0.060, ridge_two_lag 0.559; NIDRA vs ridge 0.149.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00391 | 0.208 | 0.204 | 0.203 | 0.04321 | 0.15 (992) |
| 2 | 2 | 0.00391 | 0.207 | 0.203 | 0.203 | 0.04503 | 0.03 (992) |
| 3 | 3 | 0.00391 | 0.210 | 0.203 | 0.200 | 0.04551 | 0.00 (994) |
| 4 | 4 | 0.00391 | 0.212 | 0.202 | 0.200 | 0.04571 | 0.00 (994) |
| 5 | 5 | 0.00391 | 0.215 | 0.201 | 0.200 | 0.04565 | 0.00 (994) |
| 6 | 6 | 0.00392 | 0.216 | 0.198 | 0.198 | 0.04569 | 0.00 (996) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.001 | 0.000 | — | 0.000 | 0.000 | — |
| 3 | 10 | 0.004 | 0.000 | — | 0.000 | 0.000 | — |
| 5 | 18 | 0.005 | 0.000 | — | 0.000 | 0.000 | — |
| 10 | 27 | 0.006 | 0.000 | — | 0.000 | 0.000 | — |
| 15 | 27 | 0.006 | 0.000 | — | 0.000 | 0.000 | — |
| 30 | 27 | 0.006 | 0.000 | — | 0.000 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.108), 2 consecutive windows: 6 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 33.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313428560 | 977 | 0 | — | no (—) | yes (33.0) |
| 147.32.84.165@1313752680 | 5 | 8 | 0.022 | no (—) | no (—) |
| 147.32.84.165@1313753400 | 9 | 7 | 0.013 | no (—) | no (—) |
| 147.32.84.191@1313753100 | 14 | 8 | 0.018 | no (—) | no (—) |
| 147.32.84.192@1313752500 | 15 | 0 | — | no (—) | no (—) |
| 147.32.84.192@1313754000 | 0 | 4 | 0.017 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_13:exfil` | virut | 752 | 1 | 1¹ | 0.279 | 0.265 | 0.266 | 0.558 |
| `ctu_13:recon` | virut | 150 | 1 | 1¹ | 0.003 | 0.000 | 0.000 | 0.596 |
| `ctu_12:c2` | nsis_ay | 63 | 5 | 3 | 0.010 | 0.000 | 0.000 | 0.505 |
| `ctu_13:c2` | virut | 47 | 1 | 1¹ | 0.002 | 0.002 | 0.001 | 0.531 |
| `ctu_12:recon` | nsis_ay | 18 | 4 | 2 | 0.002 | 0.000 | 0.000 | 0.414 |
| `ctu_12:exfil` | nsis_ay | 10 | 2 | 1¹ | 0.002 | 0.000 | 0.000 | 0.356 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.326 | 0.778 | 0.9995 | 3 | 0.9933 | 0.991 | 1.00× | 0.2418 | **host identity** |
| world_model | 0.341 | 0.827 | 0.9992 | 3 | 0.9933 | 0.990 | 1.00× | 0.2338 | **host identity** |
| world_model_deterministic | 0.253 | 0.664 | 0.9958 | 3 | 0.9933 | 0.990 | 1.00× | 0.2387 | **host identity** |
| noised_persistence | 0.365 | 0.832 | 0.9965 | 3 | 0.9933 | 0.994 | 1.00× | 0.4415 | **host identity** |
| isotropic_noise_persistence | 0.325 | 0.808 | 0.9853 | 3 | 0.9933 | 0.996 | 1.00× | 0.6236 | **host identity** |
| persistence | 0.250 | 0.650 | 0.9855 | 3 | 0.9933 | 0.991 | 1.00× | 0.2497 | **host identity** |
| oracle_true_future | 0.260 | 0.688 | 0.9889 | 3 | 0.9933 | 0.991 | 1.00× | 0.2783 | **host identity** |
| lr_current_state | 0.144 | 0.567 | 0.9535 | 3 | 0.9933 | 0.991 | 1.00× | 0.2525 | **host identity** |
| lr_flattened_history | 0.141 | 0.438 | 0.9795 | 3 | 0.9933 | 0.995 | 1.00× | 0.5328 | **host identity** |
| gbdt_current_state | 0.309 | 0.786 | 0.9995 | 3 | 0.9933 | 0.992 | 1.00× | 0.3234 | **host identity** |
| gru_classifier | 0.218 | 0.383 | 0.9776 | 3 | 0.9933 | 0.998 | 1.01× | 0.8015 | **host identity** |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00359 | 0.04774 | 0.00409 | 0.00408 | 10/10 |
| calibrated | 0.00375 | 0.04701 | 0.00409 | 0.00408 | 9/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

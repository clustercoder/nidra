_val: no benchmark.json at experiments/runs/xeval_comb2ctu_state+hidden/artifacts/metrics/val/benchmark.json_
### Systems — test
**test** — 25099 scored rows, natural prevalence 0.00749, operating point `mean|q=-|max`, threshold 0.088 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.373 | [0.204, 0.586] | 0.795 | 0.69 / 0.35 / 0.47 | 20.04 | 0.483 | 0.003 / 0.009 |
| NIDRA (stochastic rollout) | 0.370 | [0.200, 0.581] | 0.795 | 0.11 / 0.49 / 0.19 | 471.61 | 0.480 | 0.003 / 0.008 |
| NIDRA (deterministic rollout) | 0.321 | [0.136, 0.587] | 0.711 | 0.50 / 0.41 / 0.45 | 49.96 | 0.421 | 0.003 / 0.008 |
| persistence + learned noise (mean disabled) | 0.361 | [0.214, 0.549] | 0.798 | 0.07 / 0.58 / 0.13 | 921.31 | 0.474 | 0.002 / 0.005 |
| persistence + isotropic noise | 0.343 | — | 0.779 | 0.01 / 0.83 / 0.02 | 10631.37 | 0.456 | 0.001 / 0.003 |
| persistence (risk head on S_t) | 0.362 | [0.195, 0.565] | 0.732 | 0.68 / 0.38 / 0.49 | 22.10 | 0.466 | 0.004 / 0.012 |
| oracle: risk head on the true future | 0.400 | — | 0.778 | 0.52 / 0.45 / 0.49 | 52.36 | 0.488 | 0.011 / 0.015 |
| logistic regression on S_t | 0.114 | [0.044, 0.231] | 0.810 | 0.04 / 0.68 / 0.08 | 1906.45 | 0.151 | 0.002 / 0.003 |
| logistic regression on the L-window history | 0.022 | [0.009, 0.083] | 0.423 | 0.08 / 0.18 / 0.11 | 275.58 | 0.026 | 0.001 / 0.003 |
| gradient-boosted trees on S_t | 0.244 | [0.123, 0.486] | 0.792 | 0.07 / 0.63 / 0.12 | 1102.74 | 0.328 | 0.002 / 0.004 |

At the mandated 0.75 threshold on the calibrated score: P 0.88 / R 0.14 / F1 0.24, 583 alerts.

### Attribution — test
**test** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.007 | [-0.014, 0.030] |
| world_model - persistence_rollout | 0.061 | [0.023, 0.103] |
| world_model - noised_persistence | 0.009 | [-0.027, 0.046] |
| world_model - isotropic_noise_persistence | 0.027 | [-0.010, 0.070] |
| world_model - world_model_deterministic | 0.048 | [-0.023, 0.083] |
| world_model_deterministic - persistence | -0.041 | [-0.071, 0.032] |
| noised_persistence - persistence | -0.001 | [-0.037, 0.035] |
| world_model - lr_current_state | 0.256 | [0.064, 0.440] |
| world_model - lr_flattened_history | 0.347 | [0.185, 0.503] |
| world_model - gbdt_current_state | 0.126 | [-0.022, 0.223] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.549, period2_persistence 0.025, ridge_two_lag 0.366; NIDRA vs ridge 0.289.

### Horizon — test
**test** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00568 | 0.458 | 0.407 | 0.479 | 0.07444 | 0.14 (2768) |
| 2 | 2 | 0.00565 | 0.457 | 0.412 | 0.456 | 0.07968 | 0.01 (2774) |
| 3 | 3 | 0.00566 | 0.441 | 0.397 | 0.447 | 0.08766 | 0.00 (2781) |
| 4 | 4 | 0.00566 | 0.414 | 0.372 | 0.452 | 0.09369 | 0.00 (2787) |
| 5 | 5 | 0.00568 | 0.391 | 0.341 | 0.462 | 0.09885 | 0.00 (2795) |
| 6 | 6 | 0.00571 | 0.370 | 0.311 | 0.453 | 0.09861 | 0.00 (2804) |

### Onset forecasting — test
**test** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 126 | 0.000 | 0.001 | — | 0.000 | 0.000 | — |
| 3 | 382 | 0.002 | 0.002 | — | 0.001 | 0.001 | — |
| 5 | 635 | 0.003 | 0.004 | — | 0.002 | 0.001 | — |
| 10 | 1188 | 0.007 | 0.009 | — | 0.003 | 0.002 | — |
| 15 | 1523 | 0.009 | 0.012 | — | 0.004 | 0.003 | — |
| 30 | 2171 | 0.009 | 0.013 | — | 0.006 | 0.005 | — |

### Episodes — test
**test** — per episode at the selected threshold (0.088), 2 consecutive windows: 131 episodes, 7 warned before onset, 13 alerted inside the episode; median lead 600.0 s, median latency 18.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313504340 | 216 | 0 | — | no (—) | yes (70.0) |
| 147.32.84.165@1313519940 | 12 | 14 | 0.110 | no (—) | yes (9.0) |
| 147.32.84.165@1313522640 | 863 | 30 | 0.069 | no (—) | yes (0.0) |
| 147.32.84.165@1313586240 | 0 | 30 | 0.007 | no (—) | no (—) |
| 147.32.84.165@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313591040 | 164 | 30 | 0.004 | no (—) | yes (15.0) |
| 147.32.84.165@1313664420 | 0 | 8 | 0.118 | no (—) | no (—) |
| 147.32.84.165@1313665440 | 2 | 16 | 0.129 | yes (780) | no (—) |
| 147.32.84.165@1313668380 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313669160 | 16 | 12 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313670780 | 2 | 10 | 0.013 | no (—) | no (—) |
| 147.32.84.165@1313675220 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313675820 | 0 | 9 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.165@1313677800 | 1 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313678460 | 5 | 10 | 0.001 | no (—) | no (—) |
| 147.32.84.165@1313679420 | 2 | 10 | 0.006 | no (—) | no (—) |
| 147.32.84.191@1313586240 | 0 | 5 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313588580 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313592180 | 149 | 30 | 0.004 | no (—) | yes (15.0) |
| 147.32.84.191@1313665560 | 0 | 17 | 0.629 | yes (1020) | no (—) |
| 147.32.84.191@1313666580 | 0 | 16 | 0.010 | no (—) | no (—) |
| 147.32.84.191@1313668380 | 0 | 29 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313669160 | 16 | 12 | 0.001 | no (—) | no (—) |
| 147.32.84.191@1313670720 | 3 | 9 | 0.012 | no (—) | no (—) |
| 147.32.84.191@1313675280 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.191@1313675820 | 0 | 8 | 0.005 | no (—) | no (—) |
| 147.32.84.191@1313676360 | 10 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313678460 | 6 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.191@1313679420 | 2 | 9 | 0.009 | no (—) | no (—) |
| 147.32.84.192@1313586240 | 1 | 4 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313592480 | 139 | 30 | 0.002 | no (—) | yes (18.0) |
| 147.32.84.192@1313665500 | 0 | 9 | 0.420 | yes (540) | no (—) |
| 147.32.84.192@1313666580 | 0 | 17 | 0.006 | no (—) | no (—) |
| 147.32.84.192@1313668380 | 0 | 29 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313669280 | 14 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313670720 | 3 | 9 | 0.008 | no (—) | no (—) |
| 147.32.84.192@1313675280 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313675820 | 0 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313676360 | 9 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313677800 | 0 | 14 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313678460 | 7 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.192@1313679480 | 1 | 10 | 0.012 | no (—) | no (—) |
| 147.32.84.193@1313586240 | 0 | 3 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313592600 | 140 | 30 | 0.003 | no (—) | yes (19.0) |
| 147.32.84.193@1313665620 | 0 | 10 | 0.132 | yes (600) | no (—) |
| 147.32.84.193@1313666580 | 0 | 15 | 0.007 | no (—) | no (—) |
| 147.32.84.193@1313668380 | 0 | 29 | 0.001 | no (—) | no (—) |
| 147.32.84.193@1313669100 | 18 | 11 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313670720 | 3 | 8 | 0.006 | no (—) | no (—) |
| 147.32.84.193@1313675340 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.193@1313675820 | 0 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313678460 | 7 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.193@1313679480 | 0 | 9 | 0.035 | no (—) | no (—) |
| 147.32.84.204@1313586240 | 0 | 1 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.204@1313592600 | 139 | 30 | 0.006 | no (—) | yes (25.0) |
| 147.32.84.204@1313665500 | 0 | 7 | 0.225 | yes (420) | no (—) |
| 147.32.84.204@1313666520 | 0 | 16 | 0.060 | no (—) | no (—) |
| 147.32.84.204@1313668380 | 0 | 30 | 0.007 | no (—) | no (—) |
| 147.32.84.204@1313669280 | 14 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313670720 | 3 | 9 | 0.024 | no (—) | no (—) |
| 147.32.84.204@1313675340 | 0 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313675820 | 0 | 7 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313678460 | 8 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.204@1313679420 | 2 | 7 | 0.011 | no (—) | no (—) |
| 147.32.84.205@1313586240 | 1 | 0 | — | no (—) | no (—) |
| 147.32.84.205@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313592600 | 139 | 30 | 0.004 | no (—) | yes (10.0) |
| 147.32.84.205@1313665620 | 1 | 9 | 0.332 | yes (540) | no (—) |
| 147.32.84.205@1313666640 | 0 | 16 | 0.011 | no (—) | no (—) |
| 147.32.84.205@1313668380 | 1 | 28 | 0.003 | no (—) | no (—) |
| 147.32.84.205@1313669160 | 15 | 12 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313670780 | 2 | 11 | 0.011 | no (—) | no (—) |
| 147.32.84.205@1313675460 | 6 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.205@1313676360 | 10 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.205@1313679480 | 1 | 7 | 0.016 | no (—) | no (—) |
| 147.32.84.206@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313592600 | 146 | 30 | 0.003 | no (—) | yes (19.0) |
| 147.32.84.206@1313666460 | 1 | 15 | 0.024 | no (—) | no (—) |
| 147.32.84.206@1313668380 | 0 | 30 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313669100 | 16 | 11 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313670720 | 3 | 10 | 0.013 | no (—) | no (—) |
| 147.32.84.206@1313675460 | 6 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313676360 | 9 | 8 | 0.001 | no (—) | no (—) |
| 147.32.84.206@1313677800 | 1 | 14 | 0.003 | no (—) | no (—) |
| 147.32.84.206@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.206@1313679480 | 0 | 6 | 0.024 | no (—) | no (—) |
| 147.32.84.207@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.207@1313592600 | 143 | 30 | 0.002 | no (—) | yes (20.0) |
| 147.32.84.207@1313666460 | 0 | 16 | 0.030 | no (—) | no (—) |
| 147.32.84.207@1313668380 | 5 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.207@1313669220 | 15 | 9 | 0.001 | no (—) | no (—) |
| 147.32.84.207@1313670720 | 3 | 9 | 0.014 | no (—) | no (—) |
| 147.32.84.207@1313675520 | 5 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.207@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313677800 | 0 | 14 | 0.003 | no (—) | no (—) |
| 147.32.84.207@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.207@1313679540 | 0 | 7 | 0.017 | no (—) | no (—) |
| 147.32.84.208@1313588580 | 0 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.208@1313590380 | 0 | 29 | 0.004 | no (—) | no (—) |
| 147.32.84.208@1313592600 | 6 | 30 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313593380 | 131 | 6 | 0.005 | no (—) | yes (18.0) |
| 147.32.84.208@1313666400 | 0 | 15 | 0.070 | no (—) | no (—) |
| 147.32.84.208@1313668380 | 4 | 30 | 0.010 | no (—) | no (—) |
| 147.32.84.208@1313669100 | 17 | 7 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313670720 | 2 | 9 | 0.004 | no (—) | no (—) |
| 147.32.84.208@1313675520 | 5 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.208@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.208@1313677800 | 20 | 14 | 0.001 | no (—) | no (—) |
| 147.32.84.208@1313679540 | 0 | 8 | 0.010 | no (—) | no (—) |
| 147.32.84.209@1313588580 | 0 | 30 | 0.003 | no (—) | no (—) |
| 147.32.84.209@1313592600 | 7 | 30 | 0.004 | no (—) | no (—) |
| 147.32.84.209@1313593500 | 127 | 8 | 0.003 | no (—) | yes (18.0) |
| 147.32.84.209@1313666340 | 0 | 15 | 0.255 | yes (900) | no (—) |
| 147.32.84.209@1313668380 | 0 | 30 | 0.012 | no (—) | no (—) |
| 147.32.84.209@1313669280 | 25 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313675520 | 5 | 30 | 0.002 | no (—) | no (—) |
| 147.32.84.209@1313676360 | 9 | 8 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313677800 | 0 | 14 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313678460 | 9 | 10 | 0.000 | no (—) | no (—) |
| 147.32.84.209@1313679540 | 0 | 7 | 0.027 | no (—) | no (—) |

### Per attack group — test
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_9:exfil` | neris | 1373 | 11 | 10 | 0.795 | 0.768 | 0.764 | 0.465 |
| `ctu_8:c2` | murlo | 1069 | 3 | 1¹ | 0.024 | 0.052 | 0.030 | 0.319 |
| `ctu_10:recon` | rbot | 488 | 59 | 10 | 0.001 | 0.001 | 0.001 | 0.222 |
| `ctu_10:exfil` | rbot | 307 | 38 | 10 | 0.004 | 0.001 | 0.000 | 0.539 |
| `ctu_10:c2` | rbot | 232 | 45 | 10 | 0.016 | 0.009 | 0.020 | 0.076 |
| `ctu_9:recon` | neris | 151 | 22 | 10 | 0.000 | 0.018 | 0.002 | 0.097 |
| `ctu_9:c2` | neris | 35 | 8 | 8 | 0.002 | 0.013 | 0.004 | 0.161 |
| `ctu_8:recon` | murlo | 31 | 3 | 1¹ | 0.004 | 0.016 | 0.002 | 0.359 |
| `ctu_8:exfil` | murlo | 2 | 2 | 1¹ | 0.000 | 0.000 | 0.000 | 0.148 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — test
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.584 | 0.751 | 0.9552 | 10 | 0.7192 | 0.911 | 1.27× | 0.7786 | carries timing signal |
| world_model | 0.580 | 0.745 | 0.9527 | 10 | 0.7192 | 0.915 | 1.27× | 0.7858 | carries timing signal |
| world_model_deterministic | 0.574 | 0.766 | 0.9526 | 10 | 0.7192 | 0.853 | 1.19× | 0.6262 | carries timing signal |
| noised_persistence | 0.571 | 0.735 | 0.9544 | 10 | 0.7192 | 0.924 | 1.29× | 0.7995 | carries timing signal |
| isotropic_noise_persistence | 0.557 | 0.733 | 0.9369 | 10 | 0.7192 | 0.916 | 1.27× | 0.7743 | carries timing signal |
| persistence | 0.620 | 0.810 | 0.9531 | 10 | 0.7192 | 0.853 | 1.19× | 0.6297 | carries timing signal |
| oracle_true_future | 0.669 | 0.819 | 0.9526 | 10 | 0.7192 | 0.890 | 1.24× | 0.7036 | carries timing signal |
| lr_current_state | 0.473 | 0.809 | 0.9100 | 10 | 0.7192 | 0.912 | 1.27× | 0.8094 | carries timing signal |
| lr_flattened_history | 0.224 | 0.541 | 0.8931 | 10 | 0.7192 | 0.708 | 0.98× | 0.3537 | carries timing signal |
| gbdt_current_state | 0.601 | 0.834 | 0.9544 | 10 | 0.7192 | 0.894 | 1.24× | 0.7644 | carries timing signal |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — test
**test** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00605 | 0.04818 | 0.00749 | 0.00743 | 10/10 |
| calibrated | 0.00573 | 0.04574 | 0.00749 | 0.00743 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

### Systems — holdout
**holdout** — 21045 scored rows, natural prevalence 0.00409, operating point `mean|q=-|max`, threshold 0.088 (selected on val).

| system | AP (natural) | 95% CI (episode bootstrap) | ROC-AUC | P / R / F1 @thr | false alarms / h (whole split) | det. AP | onset AP 5 / 15 min |
|---|---|---|---|---|---|---|---|
| NIDRA (stochastic rollout, calibrated) | 0.247 | [0.000, 0.344] | 0.929 | 0.68 / 0.20 / 0.31 | 5.60 | 0.255 | 0.000 / 0.000 |
| NIDRA (stochastic rollout) | 0.258 | [0.000, 0.368] | 0.940 | 0.09 / 0.40 / 0.14 | 249.46 | 0.266 | 0.000 / 0.000 |
| NIDRA (deterministic rollout) | 0.225 | [0.000, 0.264] | 0.698 | 0.42 / 0.22 / 0.29 | 17.65 | 0.231 | 0.000 / 0.000 |
| persistence + learned noise (mean disabled) | 0.238 | [0.000, 0.332] | 0.910 | 0.05 / 0.51 / 0.09 | 572.55 | 0.248 | 0.000 / 0.000 |
| persistence + isotropic noise | 0.221 | — | 0.821 | 0.01 / 0.91 / 0.01 | 8607.28 | 0.231 | 0.000 / 0.000 |
| persistence (risk head on S_t) | 0.203 | [0.000, 0.222] | 0.456 | 0.64 / 0.20 / 0.30 | 6.71 | 0.211 | 0.000 / 0.000 |
| oracle: risk head on the true future | 0.206 | — | 0.558 | 0.32 / 0.20 / 0.25 | 25.22 | 0.214 | 0.000 / 0.000 |
| logistic regression on S_t | 0.045 | [0.000, 0.164] | 0.835 | 0.04 / 0.73 / 0.07 | 1162.56 | 0.046 | 0.000 / 0.000 |
| logistic regression on the L-window history | 0.016 | [0.000, 0.112] | 0.247 | 0.04 / 0.14 / 0.06 | 189.09 | 0.016 | 0.000 / 0.000 |
| gradient-boosted trees on S_t | 0.299 | [0.001, 0.533] | 0.957 | 0.08 / 0.93 / 0.14 | 657.33 | 0.312 | 0.000 / 0.000 |

At the mandated 0.75 threshold on the calibrated score: P 1.00 / R 0.15 / F1 0.26, 155 alerts.

### Attribution — holdout
**holdout** — paired episode-bootstrap difference in natural-prevalence AP (published label).

| comparison | ΔAP | 95% CI |
|---|---|---|
| world_model - persistence | 0.056 | [0.000, 0.149] |
| world_model - persistence_rollout | 0.086 | [0.000, 0.153] |
| world_model - noised_persistence | 0.021 | [0.000, 0.055] |
| world_model - isotropic_noise_persistence | 0.037 | [0.000, 0.096] |
| world_model - world_model_deterministic | 0.034 | [0.000, 0.105] |
| world_model_deterministic - persistence | 0.022 | [0.000, 0.043] |
| noised_persistence - persistence | 0.035 | [0.000, 0.110] |
| world_model - lr_current_state | 0.213 | [-0.001, 0.257] |
| world_model - lr_flattened_history | 0.243 | [0.000, 0.323] |
| world_model - gbdt_current_state | -0.041 | [-0.177, 0.000] |

State forecast skill vs persistence (MSE, kept features, all origins / active origins): world_model_deterministic 0.612, period2_persistence 0.052, ridge_two_lag 0.468; NIDRA vs ridge 0.270.

### Horizon — holdout
**holdout** — per-horizon: is t+k an attack window? (natural prevalence)

| k | minutes ahead | base rate | AP NIDRA | AP deterministic | AP oracle (true future) | Brier | stage top-1 on attack futures (n) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.00391 | 0.219 | 0.212 | 0.210 | 0.03814 | 0.17 (992) |
| 2 | 2 | 0.00391 | 0.231 | 0.217 | 0.209 | 0.03883 | 0.09 (992) |
| 3 | 3 | 0.00391 | 0.244 | 0.223 | 0.207 | 0.03998 | 0.02 (994) |
| 4 | 4 | 0.00391 | 0.253 | 0.229 | 0.207 | 0.04105 | 0.00 (994) |
| 5 | 5 | 0.00391 | 0.261 | 0.239 | 0.207 | 0.04228 | 0.00 (994) |
| 6 | 6 | 0.00392 | 0.266 | 0.245 | 0.205 | 0.04209 | 0.00 (996) |

### Onset forecasting — holdout
**holdout** — Task B: an episode begins within h minutes (origins outside any episode).

| h (min) | positives | NIDRA | persistence (head on S_t) | ridge | GBDT S_t | LR history | onset head |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 3 | 10 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 5 | 18 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 10 | 27 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 15 | 27 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 30 | 27 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |

### Episodes — holdout
**holdout** — per episode at the selected threshold (0.088), 2 consecutive windows: 6 episodes, 0 warned before onset, 1 alerted inside the episode; median lead None s, median latency 29.0 min.

| episode | length (windows) | pre-onset rows | max score pre-onset | warned before onset (lead s) | alerted inside (latency min) |
|---|---|---|---|---|---|
| 147.32.84.165@1313428560 | 977 | 0 | — | no (—) | yes (29.0) |
| 147.32.84.165@1313752680 | 5 | 8 | 0.003 | no (—) | no (—) |
| 147.32.84.165@1313753400 | 9 | 7 | 0.014 | no (—) | no (—) |
| 147.32.84.191@1313753100 | 14 | 8 | 0.002 | no (—) | no (—) |
| 147.32.84.192@1313752500 | 15 | 0 | — | no (—) | no (—) |
| 147.32.84.192@1313754000 | 0 | 4 | 0.006 | no (—) | no (—) |

### Per attack group — holdout
| group | family | positives | episodes | hosts | world model AP | oracle AP | persistence AP | state skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| `ctu_13:exfil` | virut | 752 | 1 | 1¹ | 0.322 | 0.275 | 0.273 | 0.566 |
| `ctu_13:recon` | virut | 150 | 1 | 1¹ | 0.005 | 0.000 | 0.000 | 0.579 |
| `ctu_12:c2` | nsis_ay | 63 | 5 | 3 | 0.001 | 0.001 | 0.000 | 0.516 |
| `ctu_13:c2` | virut | 47 | 1 | 1¹ | 0.047 | 0.065 | 0.053 | 0.532 |
| `ctu_12:recon` | nsis_ay | 18 | 4 | 2 | 0.001 | 0.000 | 0.000 | 0.386 |
| `ctu_12:exfil` | nsis_ay | 10 | 2 | 1¹ | 0.001 | 0.000 | 0.000 | 0.372 |

¹ All of this group's positives are on a single host, so its AP does not separate the attack's behaviour from that host's identity. Treat it as a within-host result until a capture with two infected hosts in the same stage says otherwise.

### Host identity vs timing — holdout
| system | AP | ROC | host-mean ROC | positive hosts | within-host base rate | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| world_model_calibrated | 0.381 | 0.872 | 0.9916 | 3 | 0.9933 | 0.997 | 1.00× | 0.7262 | **host identity** |
| world_model | 0.412 | 0.885 | 0.9924 | 3 | 0.9933 | 0.998 | 1.00× | 0.7685 | **host identity** |
| world_model_deterministic | 0.339 | 0.820 | 0.9845 | 3 | 0.9933 | 0.996 | 1.00× | 0.5430 | **host identity** |
| noised_persistence | 0.334 | 0.816 | 0.9738 | 3 | 0.9933 | 0.999 | 1.01× | 0.8775 | **host identity** |
| isotropic_noise_persistence | 0.298 | 0.738 | 0.9490 | 3 | 0.9933 | 1.000 | 1.01× | 0.9434 | **host identity** |
| persistence | 0.255 | 0.603 | 0.9818 | 3 | 0.9933 | 0.992 | 1.00× | 0.3390 | **host identity** |
| oracle_true_future | 0.269 | 0.671 | 0.9802 | 3 | 0.9933 | 0.992 | 1.00× | 0.3223 | **host identity** |
| lr_current_state | 0.162 | 0.827 | 0.9238 | 3 | 0.9933 | 0.998 | 1.00× | 0.7556 | **host identity** |
| lr_flattened_history | 0.072 | 0.389 | 0.9765 | 3 | 0.9933 | 0.990 | 1.00× | 0.2411 | **host identity** |
| gbdt_current_state | 0.581 | 0.958 | 0.9988 | 3 | 0.9933 | 0.998 | 1.00× | 0.8283 | **host identity** |

Within-host ROC is the column that survives the single-host confound: it asks, on the infected host alone, whether the system orders the attack windows above that host's own benign ones.

### Calibration — holdout
**holdout** — calibration of the published score (natural prevalence).

| arm | Brier | ECE | constant base rate | Brier of that constant | occupied bins |
|---|---|---|---|---|---|
| raw | 0.00363 | 0.04858 | 0.00409 | 0.00408 | 10/10 |
| calibrated | 0.00332 | 0.04678 | 0.00409 | 0.00408 | 10/10 |

ECE is weighted by each bin's population weight, not its sampled row count — the evaluation subsamples negatives, so the two are not the same number. A Brier below the constant-base-rate column is the minimum bar, not a result.

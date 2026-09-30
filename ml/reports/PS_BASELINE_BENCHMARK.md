Each system is read at **its own validation-selected threshold** — the weighted-F1-optimal point on the validation split, frozen before test or holdout was scored. That is the rule the world model's operating point already used; `benchmark.json` instead reads every baseline at the world model's threshold, which was chosen for nothing but the world model.

Natural prevalence throughout. *AP margin* is the world model's AP minus the baseline's, paired episode-cluster bootstrap, 95% interval.

| split | system | threshold | AP | AP margin of the world model | precision | recall | F1 | FPR | false alarms/h |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| test | world_model_calibrated | 0.7178 | 0.058 | — | 0.886 | 0.032 | 0.061 | 0.000017 | 0.48 |
| test | lr_current_state | 1.0000 | 0.033 | +0.025 [+0.005, +0.058] | 0.482 | 0.038 | 0.071 | 0.000171 | 4.80 |
| test | lr_flattened_history | 1.0000 | 0.036 | +0.022 [-0.030, +0.076] | 0.489 | 0.050 | 0.090 | 0.000218 | 6.11 |
| test | gru_classifier | 0.5560 | 0.164 | -0.106 [-0.292, +0.019] | 0.474 | 0.027 | 0.052 | 0.000128 | 3.59 |
| test | gbdt_current_state | 0.9989 | 0.024 | +0.034 [+0.008, +0.100] | 0.722 | 0.005 | 0.010 | 0.000009 | 0.24 |
| holdout | world_model_calibrated | 0.7178 | 0.438 | — | 0.848 | 0.320 | 0.464 | 0.000019 | 0.86 |
| holdout | lr_current_state | 1.0000 | 0.241 | +0.197 [-0.000, +0.366] | 0.462 | 0.279 | 0.348 | 0.000110 | 4.89 |
| holdout | lr_flattened_history | 1.0000 | 0.443 | -0.004 [-0.292, +0.423] | 0.821 | 0.262 | 0.398 | 0.000019 | 0.86 |
| holdout | gru_classifier | 0.5560 | 0.382 | +0.056 [-0.081, +0.173] | 0.852 | 0.377 | 0.523 | 0.000022 | 0.99 |
| holdout | gbdt_current_state | 0.9989 | 0.294 | +0.145 [-0.050, +0.329] | 0.000 | 0.000 | 0.000 | 0.000006 | 0.29 |

Span of each split in hours (the denominator of false alarms/h): test 8.05, holdout 8.08.

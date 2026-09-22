### Variant comparison — `world_model` on **val**

Reference system: `persistence`. AP is at natural prevalence; intervals are episode-cluster bootstrap. ΔAP is the paired difference, and is only evidence of an improvement when its interval excludes zero. Each variant selected its **own operating point** on this split, so the P / R / F1 and FA/h columns are each at a different threshold and rank nothing; AP and ROC-AUC are threshold-free and are what the rows are ordered by.

| variant | AP | 95% CI | ΔAP vs reference | 95% CI | sig. | calibrated AP | ROC-AUC | P / R / F1 | FA/h | oracle AP | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| state+hidden | 0.473 | [0.001, 0.814] | -0.008 | [-0.021, 0.004] | no | 0.478 | 0.814 | 0.06 / 0.54 / 0.11 | 383.85 | 0.506 | 0.034 |
| hidden | 0.459 | [0.001, 0.812] | -0.031 | [-0.080, 0.007] | no | 0.463 | 0.757 | 0.54 / 0.53 / 0.54 | 20.43 | 0.508 | 0.049 |
| state+hidden+logvar | 0.436 | [0.001, 0.806] | -0.025 | [-0.054, 0.002] | no | 0.452 | 0.780 | 0.09 / 0.54 / 0.15 | 245.42 | 0.485 | 0.049 |
| state+hidden+delta+logvar | 0.430 | [0.001, 0.804] | -0.036 | [-0.073, 0.000] | no | 0.435 | 0.788 | 0.17 / 0.54 / 0.26 | 112.77 | 0.482 | 0.052 |
| state | 0.412 | [0.001, 0.737] | 0.063 | [-0.000, 0.119] | no | 0.408 | 0.760 | 0.41 / 0.49 / 0.45 | 30.97 | 0.381 | -0.031 |
| state+logvar | 0.396 | [0.001, 0.762] | 0.019 | [-0.011, 0.032] | no | 0.393 | 0.821 | 0.05 / 0.52 / 0.10 | 403.73 | 0.416 | 0.020 |

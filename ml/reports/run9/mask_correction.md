**28 of 28 cells re-scored: 21 up, 7 down.** 5 moved the published arm by at least 0.05 AP. The deltas are not averaged: the fix removed an untrained signal that helped some cells and hurt others, and the split is the finding.

| cell | split | AP before | AP after | Δ cal | Δ raw | beats oracle (before → after) | note |
|---|---|---:|---:|---:|---:|:--:|---|
| cic2cic_state | val | 0.7809 | 0.7491 | -0.0319 | -0.0371 | no → no |  |
| cic2cic_state | test | 0.1110 | 0.1633 | +0.0523 | +0.0666 | no → yes |  |
| cic2cic_state | holdout | 0.3323 | 0.3370 | +0.0046 | +0.0174 | no → no |  |
| cic2cic_state+hidden | val | 0.8316 | 0.8367 | +0.0051 | +0.0139 | no → no |  |
| cic2cic_state+hidden | test | 0.0642 | 0.1173 | +0.0531 | +0.0854 | no → yes |  |
| cic2cic_state+hidden | holdout | 0.3797 | 0.3869 | +0.0072 | +0.0094 | yes → yes |  |
| cic2ctu_state | test | 0.0098 | 0.0127 | +0.0029 | +0.0037 | no → yes |  |
| cic2ctu_state | holdout | 0.0354 | 0.0373 | +0.0020 | +0.0049 | no → yes |  |
| cic2ctu_state+hidden | test | 0.0087 | 0.0105 | +0.0018 | +0.0048 | no → yes |  |
| cic2ctu_state+hidden | holdout | 0.0142 | 0.0159 | +0.0017 | +0.0038 | no → yes |  |
| comb2cic_state | val | 0.3914 | 0.4015 | +0.0101 | +0.0099 | no → no |  |
| comb2cic_state | test | 0.1645 | 0.1759 | +0.0114 | +0.0090 | yes → yes |  |
| comb2cic_state | holdout | 0.2243 | 0.3229 | +0.0986 | +0.0984 | no → no |  |
| comb2cic_state+hidden | val | 0.5381 | 0.5156 | -0.0225 | -0.0362 | no → no |  |
| comb2cic_state+hidden | test | 0.1584 | 0.1990 | +0.0406 | +0.0548 | yes → yes |  |
| comb2cic_state+hidden | holdout | 0.3282 | 0.3392 | +0.0111 | +0.0130 | no → no |  |
| comb2ctu_state | test | 0.1568 | 0.1848 | +0.0280 | +0.0410 | no → no |  |
| comb2ctu_state | holdout | 0.3070 | 0.2826 | -0.0244 | -0.0251 | yes → yes |  |
| comb2ctu_state+hidden | test | 0.3734 | 0.3537 | -0.0197 | -0.0147 | no → no |  |
| comb2ctu_state+hidden | holdout | 0.2469 | 0.2400 | -0.0070 | -0.0093 | yes → yes |  |
| ctu2cic_state | test | 0.0448 | 0.1147 | +0.0699 | +0.0574 | no → yes |  |
| ctu2cic_state | holdout | 0.0012 | 0.0048 | +0.0036 | +0.0033 | no → yes |  |
| ctu2cic_state+hidden | test | 0.0035 | 0.0040 | +0.0005 | +0.0019 | yes → yes |  |
| ctu2cic_state+hidden | holdout | 0.0003 | 0.0003 | +0.0000 | +0.0001 | yes → yes |  |
| ctu2ctu_state | test | 0.1964 | 0.1916 | -0.0047 | -0.0157 | yes → no |  |
| ctu2ctu_state | holdout | 0.1884 | 0.2410 | +0.0526 | +0.0366 | yes → yes |  |
| ctu2ctu_state+hidden | test | 0.3221 | 0.3186 | -0.0035 | +0.0050 | no → no |  |
| ctu2ctu_state+hidden | holdout | 0.2206 | 0.2273 | +0.0067 | +0.0156 | yes → yes |  |

### Cells whose oracle verdict changed

- **cic2cic_state/test** — no → yes
- **cic2cic_state+hidden/test** — no → yes
- **cic2ctu_state/test** — no → yes
- **cic2ctu_state/holdout** — no → yes
- **cic2ctu_state+hidden/test** — no → yes
- **cic2ctu_state+hidden/holdout** — no → yes
- **ctu2cic_state/test** — no → yes
- **ctu2cic_state/holdout** — no → yes
- **ctu2ctu_state/test** — yes → no

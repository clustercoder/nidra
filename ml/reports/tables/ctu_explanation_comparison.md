### Explanation comparison — 24 highest-risk **val** origins

Integrated gradients through the rollout back to the input history, on the same origins for every head. `share recent` is the fraction of absolute attribution in the last 5 windows of the 30-window history; `mean age` is the attribution-weighted mean age of a cell. The deletion test zeroes the top-8 attributed cells and compares the drop against 8 random ones.

| head | forecast score | share recent (5 of 30) | mean age (min) | drop top-8 | drop random | beats random |
|---|---|---|---|---|---|---|
| ctu_heads__state | 0.584 ± 0.492 | 0.631 ± 0.341 | 6.1 ± 5.9 | 0.579 ± 0.490 | 0.006 ± 0.020 | 0.75 ± 0.43 |
| ctu_heads__hidden | 0.610 ± 0.478 | 0.342 ± 0.252 | 11.2 ± 4.9 | 0.100 ± 0.270 | 0.005 ± 0.023 | 0.45 ± 0.49 |
| ctu_heads__state+hidden | 0.625 ± 0.484 | 0.368 ± 0.244 | 11.4 ± 4.5 | 0.055 ± 0.199 | 0.000 ± 0.000 | 0.67 ± 0.47 |

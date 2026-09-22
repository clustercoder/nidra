### Positives that carry no information in the state

`risk_label` marks the windows BEFORE an attack starts, and a host that is silent in such a window produces the scaler's silent state — the same 45 numbers every other silent window has. Those rows cannot be ordered above the negatives they are identical to by any function of the state; only the host's history distinguishes them.

| split | rows | positives | pre-onset | of those, silent | at the exact floor | floor stratum | state-only AP ceiling there | share of all positives |
|---|---|---|---|---|---|---|---|---|
| train | 2274548 | 179 | 45 | 24 | 20 | 2075362 | 0.000010 | 11.2% |
| val | 893701 | 94 | 22 | 9 | 8 | 808496 | 0.000010 | 8.5% |

The ceiling is the stratum's own prevalence. A state-only head that beats it on a split is reading something the state does not contain, so the number is also a leak check.

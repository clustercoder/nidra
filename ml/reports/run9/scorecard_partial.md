> **SUPERSEDED — kept, not cited.** Rendered 06:05 on 2026-09-23, before three
> corrections to the scorecard itself. Its `FA/h` column reports a per-row
> fraction rather than a rate (§3.18); its `oracle AP` column has no
> `deterministic AP` beside it, which invites the comparison §3.24 shows is
> unsound; and it predates the calibration table, the gap listing and the head
> variant appearing in the row label, so each regime is one row rather than two.
> The current artifact is `scorecard.md`. This is kept because a superseded
> render is still a record of what was believed when, and deleting results is
> not something this phase does — but no number here should be quoted.

| Training | Evaluated on | split | rows | prevalence | AP [95% CI] | ROC | P | R | F1 | FA/h | alerts/h | FA rate on active benign | best baseline (AP) | oracle AP | state skill vs persistence / ridge | onset AP 5/15 | episodes warned | positive hosts | within-host ROC |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---:|---:|
| CIC | CIC | val | 20,113 | 0.00012 | 0.781 | 0.904 | 0.955 | 0.753 | 0.842 | 0.38 | 8.40 | 0.00007 | world_model_deterministic (0.793) | 0.797 | — / 0.694 | 0.000 / 0.000 | 0 / 2 | 1¹ | 0.8571 |
| CIC | CIC | test | 21,173 | 0.00417 | 0.111 | 0.870 | 0.745 | 0.033 | 0.063 | 1.32 | 5.16 | 0.00037 | noised_persistence (0.127) | 0.155 | — / 0.711 | 0.006 / 0.011 | 0 / 15 | 9 | 0.5933 |
| CIC | CIC | holdout | 20,171 | 0.00034 | 0.332 | 0.830 | 0.511 | 0.279 | 0.361 | 4.02 | 8.22 | 0.00093 | lr_flattened_history (0.381) | 0.489 | — / 0.689 | 0.002 / 0.003 | 0 / 5 | 2 | 0.6730 |
| CIC | CTU | val | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| CIC | CTU | test | 25,099 | 0.00749 | 0.010 | 0.604 | 0.011 | 0.019 | 0.014 | 207.97 | 210.32 | 0.05204 | noised_persistence (0.014) | 0.011 | — / 0.873 | 0.001 / 0.003 | 0 / 131 | 10 | 0.6487 |
| CIC | CTU | holdout | 21,045 | 0.00409 | 0.035 | 0.863 | 0.002 | 0.005 | 0.002 | 186.18 | 186.46 | 0.06353 | noised_persistence (0.033) | 0.034 | — / 0.874 | 0.000 / 0.001 | 0 / 6 | 3 | 0.7154 |

¹ Every positive in that evaluation sits on a single host, so its AP does not separate the attack's behaviour from that host's identity. **Within-host ROC** is the column that does: on the infected host alone, does the system order the attack windows above that host's own benign ones? It is prevalence-independent and is the only column here that compares fairly across datasets.

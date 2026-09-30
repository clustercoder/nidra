### Frozen head diagnostic per stage — observed states, **test**

The stage head's own class probability against that stage one-vs-rest, beside the risk head's score on the same rows. Observed states only: both heads are scored where they were trained, so a gap is about the objective rather than about rollout error.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| recon | 27 | 0.00009 | 0.000 | 1× | 0.016 | 0.021 | 0.678 |
| c2 | 648 | 0.00216 | 0.006 | 3× | 0.213 | 0.006 | 0.188 |
| exfil | 22 | 0.00007 | 0.931 | 12667× | 0.955 | 0.926 | 0.955 |

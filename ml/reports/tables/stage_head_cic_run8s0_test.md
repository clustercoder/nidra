### Frozen head diagnostic per stage — observed states, **test**

The stage head's own class probability against that stage one-vs-rest, beside the risk head's score on the same rows. Observed states only: both heads are scored where they were trained, so a gap is about the objective rather than about rollout error.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| recon | 27 | 0.00009 | 0.211 | 2345× | 0.883 | 0.192 | 0.893 |
| c2 | 648 | 0.00216 | 0.031 | 14× | 0.438 | 0.016 | 0.253 |
| exfil | 22 | 0.00007 | 0.049 | 662× | 0.999 | 0.874 | 1.000 |

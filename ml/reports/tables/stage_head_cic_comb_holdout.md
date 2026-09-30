### Frozen head diagnostic per stage — observed states, **holdout**

The stage head's own class probability against that stage one-vs-rest, beside the risk head's score on the same rows. Observed states only: both heads are scored where they were trained, so a gap is about the objective rather than about rollout error.

| stage | windows | base rate | stage-head AP | lift | stage-head ROC | risk-head AP | risk-head ROC |
|---|---|---|---|---|---|---|---|
| initial_access | 68 | 0.00015 | 0.995 | 6594× | 1.000 | 0.910 | 1.000 |
| lateral | 28 | 0.00006 | 0.000 | 1× | 0.008 | 0.000 | 0.014 |

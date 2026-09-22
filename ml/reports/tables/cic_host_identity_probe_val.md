### What a head's scores encode over the whole split — **val**

The same decomposition applied to every row, not only the uninformative ones. Here a state-only head has real features to work with, so its columns are informative too and the comparison between the two heads is the point. **host-mean** replaces each score by its host's mean, keeping only host identity. **within-host** restricts to the infected host(s), where identity is constant and only the timing question remains — which is the question advance warning asks.

| head | rows | positives | AP | lift over ceiling | ROC | host-mean AP | host-mean ROC | within-host prevalence | within-host AP | within-host lift | within-host ROC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| state-only | 893701 | 94 | 0.678077 | 6446.79× | 0.8507 | 0.4196 | 0.9999 | 0.4196 | 0.8552 | 2.04× | 0.8256 | carries timing signal |
| state+hidden | 893701 | 94 | 0.782544 | 7440.01× | 0.9797 | 0.4196 | 0.9999 | 0.4196 | 0.9224 | 2.20× | 0.9245 | carries timing signal |

A lift that disappears within the host is the head recognising *who*, not *when*.

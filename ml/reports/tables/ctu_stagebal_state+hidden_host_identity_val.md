### What a head's scores encode over the whole split — **val**

The same decomposition applied to every row, not only the uninformative ones. Here a state-only head has real features to work with, so its columns are informative too and the comparison between the two heads is the point. **host-mean** replaces each score by its host's mean, keeping only host identity. **within-host** restricts to the infected host(s), where identity is constant and only the timing question remains — which is the question advance warning asks.

| head | rows | positives | positive hosts | AP | lift over ceiling | ROC | host-mean AP | host-mean ROC | within-host prevalence | within-host AP | within-host lift | within-host ROC | per-host ROC (macro) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage-balanced state+hidden | 98876 | 288 | 1 | 0.492716 | 169.16× | 0.7523 | 0.7579 | 0.9995 | 0.7579 | 0.8952 | 1.18× | 0.6642 | 0.6642 | carries timing signal |

A lift that disappears within the host is the head recognising *who*, not *when*. With more than one infected host the pooled **within-host** columns still carry a between-host component — host identity again, one level down — so **per-host ROC (macro)**, computed inside each host and averaged, is the figure to read there. With a single infected host the two are identical.

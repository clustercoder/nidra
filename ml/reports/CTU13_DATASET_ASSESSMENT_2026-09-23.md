# CTU-13 dataset assessment

**Date:** 2026-09-23 · **Branch:** `research/run9-ctu13` · **Baseline:** `baseline-run8`
**Machine-readable companion:** [`artifacts/metrics/ctu13_audit.json`](../artifacts/metrics/ctu13_audit.json)
(one record per scenario: file digest, loader drop report, label inventory, temporal
coverage, host population, windowed geometry, episode structure, per-feature coverage).

Produced by `python -m nidra.scripts.ctu13_audit --config config/ctu13.yaml`, which runs the
same per-scenario pipeline training runs — so every number here is a number a model would
see, not an approximation of one.

---

## 1. What is on disk

`/Users/manteksinghburn/nidra/CTU-13-Dataset`, 74 GB, thirteen numbered directories. Each
holds four things: a `.binetflow` file, a `.pcap`, a README, and the malware binary the
scenario ran. Totals: **2.73 GB of Argus bidirectional NetFlow, 19,976,700 flow records**,
of which 444,699 (2.23%) are labelled Botnet, 356,433 (1.78%) verified Normal, and
19,175,568 (95.99%) Background.

### The PCAPs cannot be used, and this is the single most consequential finding

Each scenario's README states it plainly: the complete capture "is not made public because
it contains too much private information about the users of the network." What ships is
`botnet-capture-<date>-<family>.pcap` — **only the infected machine's own traffic**.

Deriving NIDRA's eleven packet-level features from those files would hand the model a
perfect label: every host with packet data would be the botnet. They are not merely
incomplete, they are unusable for supervised work, and the 66 GB of scenario 10 is 66 GB of
one host's UDP flood. CTU-13 is therefore a **flow-only dataset** for this project, and the
malware binaries are inventoried and never executed.

### Scenarios

| Sc | family | behaviour | start (capture clock) | hours | flows (all src) | botnet | normal | background |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | neris | IRC, spam, click-fraud | 2011-08-10 09:46 | 6.1 | 2,824,636 | 40,961 | 30,387 | 2,753,288 |
| 2 | neris | IRC, spam, click-fraud | 2011-08-11 09:49 | 4.2 | 1,808,122 | 20,941 | 9,120 | 1,778,061 |
| 3 | rbot | IRC, port scan | 2011-08-12 15:24 | 66.8 | 4,710,638 | 26,822 | 116,887 | 4,566,929 |
| 4 | rbot | IRC, UDP DDoS | 2011-08-15 10:42 | 4.5 | 1,121,076 | 2,580 | 25,268 | 1,093,228 |
| 5 | virut | spam, port scan, HTTP | 2011-08-15 16:43 | 0.5 | 129,832 | 901 | 4,679 | 124,252 |
| 6 | menti | port scan | 2011-08-16 10:01 | 2.2 | 558,919 | 4,630 | 7,494 | 546,795 |
| 7 | sogou | HTTP | 2011-08-16 13:51 | 0.4 | 114,077 | 63 | 1,677 | 112,337 |
| 8 | murlo | port scan | 2011-08-16 14:18 | 19.5 | 2,954,230 | 6,127 | 72,822 | 2,875,281 |
| 9 | neris | IRC, spam, click-fraud, port scan (10 bots) | 2011-08-17 11:34 | 5.6 | 2,087,508 | 184,987 | 29,967 | 1,872,554 |
| 10 | rbot | IRC, UDP DDoS (10 bots) | 2011-08-18 09:56 | 5.1 | 1,309,791 | 106,352 | 15,847 | 1,187,592 |
| 11 | rbot | IRC, ICMP DDoS (3 bots) | 2011-08-18 15:39 | 0.3 | 107,251 | 8,164 | 2,718 | 96,369 |
| 12 | nsis_ay | P2P (3 bots) | 2011-08-19 10:02 | 1.7 | 325,471 | 2,168 | 7,628 | 315,675 |
| 13 | virut | spam, port scan, HTTP | 2011-08-15 17:13 | 16.4 | 1,925,149 | 40,003 | 31,939 | 1,853,207 |

Families: Neris (1, 2, 9), Rbot (3, 4, 10, 11), Virut (5, 13), Menti (6), Sogou (7),
Murlo (8), NSIS.ay (12). Scenario 3's capture spans 66.8 hours across three calendar days;
scenario 11's spans 15 minutes.

---

## 2. Label semantics

CTU-13's `Label` column is a sentence, not a class name:
`flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted`. Three families of string appear:

* `flow=Background-*` and `flow=To-Background-*` — unlabelled traffic. **Not verified
  benign.** The authors assigned Background first and then overrode it where they could.
* `flow=From-Normal-*` / `flow=To-Normal-*` / `flow=Normal-*` — traffic from machines the
  authors checked by hand.
* `flow=From-Botnet-*` — the infected machine's traffic.

Two facts were verified across all thirteen scenarios rather than assumed:

1. **There are no `To-Botnet` flows.** Every botnet-labelled flow names the bot as the
   source, so NIDRA's existing source-attributed labelling is correct for this dataset
   with no special handling.
2. **A CIC-IDS2017 rule set silently mislabels this dataset.** `flow=Background-TCP-Attempt`
   contains the word that marks botnet scanning; under the CIC rules every CTU attack scores
   benign. The `ctu` dialect in `nidra/data/labels.py` is therefore gated on the
   `From-Botnet` prefix before any behaviour rule is consulted, and selecting it is explicit
   — there is no auto-detection.

Behaviour → stage, curated the same way the CIC mapping is (and with the same caveat: this
is presentation, not ATT&CK ground truth): `CC<n>`/`IRC` → c2 · `Binary-Download` →
initial_access · `SPAM`/`HTTP-Ad`/`DDoS`/`ICMP` → exfil (the impact bucket) · `Attempt`/`PS`
→ recon · anything else labelled Botnet → c2, never benign. No label in any scenario failed
to match a rule.

### The negative class is weaker than CIC-IDS2017's

96% of CTU-13 is Background — unlabelled, not cleared. A false positive on a Background host
may be a genuine false alarm or may be an unlabelled compromise; the dataset cannot tell
them apart. Every precision and false-alarm number on CTU-13 is therefore a **lower bound on
precision**, and this is a property of the dataset, not of the model.

---

## 3. Temporal resolution, and whether Δ=60 still holds

Argus timestamps are microsecond-resolution wall clock (`2011/08/10 09:46:59.607825`). Every
scenario shows sub-second precision and tens of thousands of distinct sub-minute offsets —
the defect that forced Δ=60 on CIC-IDS2017 (minute-resolution CSV timestamps, so every :30
window was an artificial empty state) **does not exist here**. CTU-13 would support Δ=30 or
Δ=15.

Δ=60 is kept anyway. Changing it invalidates every trained artifact and every recorded
metric (CLAUDE.md), and a cross-dataset comparison in which the two datasets are windowed
differently measures the windowing. That CTU-13 *could* carry a shorter window is recorded
here as an option for a later phase, not taken now.

The printed clock is read literally; no timezone offset is applied. Nothing in the CTU path
joins against a second clock, so the only thing an offset would move is which calendar day
a window is assigned to, and reading local time literally keeps a capture's own midnight at
midnight.

---

## 4. Host population

CTU-13 captures a whole university network, so 23,000–90,000 distinct IPs appear as flow
sources per scenario, almost all of them Internet peers seen in one window. NIDRA forecasts
the risk of hosts you monitor, so the host population is restricted to the monitored network
**147.32.0.0/16** — the same scoping CIC-IDS2017 has by construction.

The filter keeps 13,402,192 of 19,976,700 flows (67.1%) and **88 of 444,699 botnet flows are
dropped** (0.02%): seven external addresses that carry a Botnet label because they are the
other end of a C2 or P2P conversation (`38.229.70.20` in scenario 3; six single-digit-flow
P2P peers in scenario 12). Ten internal machines are infected across the dataset —
`147.32.84.165` in every scenario, plus `.191 .192 .193 .204 .205 .206 .207 .208 .209` in
scenarios 9, 10, 11 and 12.

**`147.32.84.165` is the infected host in all thirteen scenarios.** Host identity is not a
feature — the state vector is traffic statistics only — but the same physical Windows XP
machine carries the infection everywhere, so its benign baseline is shared across every
split. A scenario holdout is not a host holdout. The family holdout (§7) and the
multi-bot host split available inside scenarios 9/10 are the honest generalisation tests.

---

## 5. Feature compatibility with CIC-IDS2017

Thirty-two of the 45 canonical features survive the crossing; the mapping and each
deviation are stated in `nidra/data/ctu_load.py` and declared once in
`schema.FEATURE_REGIMES`.

| Feature group | Availability in CTU-13 |
|---|---|
| bytes, packets-per-flow, flow duration mean/var, active flow count (5) | exact |
| the 8 graph scalars | exact — computed from the 5-tuple and time by the same code |
| the dynamics deltas and slopes of available bases (5) | exact |
| `is_active` | exact |
| `iat_mean`, `iat_var` | exact estimator: Argus `Dur/(TotPkts-1)` is the same quantity CICFlowMeter's Flow IAT Mean reports, in the same microsecond unit |
| the 6 TCP flag ratios, `d_syn_ratio`, `slope3_syn_ratio` (8) | **different estimator.** CICFlowMeter counts packets carrying each flag; Argus records which flags were SEEN per direction. Close for SYN/FIN/RST, badly under-counted for ACK and PSH |
| `iat_max` | **absent** — Argus reports no per-packet timing |
| the 11 packet aggregates, `d_retrans_rate` (12) | **absent** — see §1 |

Two unit traps were pinned by tests rather than discovered later: Argus `Dur` is seconds
while CICFlowMeter's Flow Duration is microseconds, and Argus's non-TCP state words (`CON`,
`URP`, `RED`, `ECR`) contain the very letters a naive reader scores as RST/PSH/URG flags —
flags are parsed only for TCP states carrying the `_` direction separator. `TotPkts` has no
directional split; all of it is assigned to the forward column, which is lossless because
every feature in FEATURE_ORDER uses only the sum.

Measured consequence: fourteen features are zero in every window of every CTU scenario
(the twelve absent ones, plus `urg_ratio`, which Argus never reports for these captures, and
`iat_max`). With `reciprocity` also constant on active rows, a CTU-trained model reads **30
informative features**.

### The two named regimes

* **`cross_core` (32 features)** — everything both datasets compute from the same underlying
  quantity, plus the flag ratios. The default for every cross-dataset experiment.
* **`cross_strict` (24 features)** — also removes the eight flag-derived columns whose
  estimator differs. The sensitivity check.
* **`full` (45)** — CIC-only. What Run 8 used.

FEATURE_ORDER stays 45 wide (it is a cross-service contract); a regime's excluded columns are
declared `drop` in the scaler, which is the mechanism already used for a feature the model
must not read, and the regime is recorded in the scaler artifact so training and serving
cannot disagree.

---

## 6. Windowed geometry

At Δ=60 with the monitored-host filter:

| Sc | monitored hosts | state rows | active rows | attack windows | prevalence | episodes | episode hosts | median episode (min) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 425 | 107,276 | 33,649 | 284 | 0.0026 | 2 | 1 | 142 |
| 2 | 392 | 70,725 | 22,902 | 204 | 0.0029 | 3 | 1 | 1 |
| 3 | 504 | 1,242,263 | 188,180 | 870 | 0.0007 | 13 | 1 | 1 |
| 4 | 374 | 65,982 | 22,888 | 90 | 0.0014 | 22 | 1 | 1 |
| 5 | 254 | 4,471 | 2,587 | 21 | 0.0047 | 1 | 1 | 21 |
| 6 | 354 | 33,484 | 12,417 | 123 | 0.0037 | 2 | 1 | 62 |
| 7 | 294 | 3,733 | 2,412 | 5 | 0.0013 | 3 | 1 | 1 |
| 8 | 478 | 335,311 | 67,082 | 1,094 | 0.0033 | 25 | 1 | 35 |
| 9 | 398 | 89,808 | 29,949 | 1,403 | 0.0156 | 55 | 10 | 1 |
| 10 | 413 | 84,153 | 27,298 | 357 | 0.0042 | 191 | 10 | 1 |
| 11 | 272 | 2,562 | 1,703 | 11 | 0.0043 | 5 | 3 | 2 |
| 12 | 303 | 16,751 | 6,962 | 81 | 0.0048 | 32 | 3 | 2 |
| 13 | 433 | 251,774 | 50,422 | 978 | 0.0039 | 2 | 1 | 489 |

**2,308,293 state rows, 5,521 attack windows (0.24%), 356 episodes across 13 scenarios.**
For comparison, CIC-IDS2017 at the same geometry yields roughly 2.3M rows and about twenty
episodes. CTU-13 has an order of magnitude more independent episodes, which is the first
thing this phase needed: Run 8's confidence intervals were wide because the episode
bootstrap had almost nothing to resample.

Three scenarios yield no usable samples at L=30, K=6: **5 (30 min), 7 (21 min) and 11
(15 min)** are shorter than the L+K = 36 windows one sample requires. They are declared in
the config and produce no rows. At a shorter geometry they would return.

---

## 7. Pre-onset inventory — the advance-warning question

Rows that sit *h* minutes before an episode onset **on the same host**, and the number of
independent onsets contributing at each *h*. Episodes are merged across gaps of five windows
first, so a flapping C2 channel counts once.

| Sc | 1 min | 3 min | 5 min | 10 min | 15 min | 30 min | 60 min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 3 | 10 / 10 | 30 / 10 | 50 / 10 | 100 / 10 | 143 / 10 | 263 / 10 | 489 / 10 |
| 4 | 7 / 7 | 21 / 7 | 35 / 7 | 69 / 7 | 84 / 7 | 99 / 7 | 125 / 7 |
| 5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 6 | 1 / 1 | 3 / 1 | 5 / 1 | 6 / 1 | 6 / 1 | 6 / 1 | 6 / 1 |
| 7 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 8 | 2 / 2 | 6 / 2 | 10 / 2 | 20 / 2 | 29 / 2 | 44 / 2 | 46 / 2 |
| 9 | 34 / 34 | 102 / 34 | 170 / 34 | 332 / 34 | 485 / 34 | 911 / 34 | 1257 / 34 |
| 10 | 99 / 99 | 297 / 99 | 495 / 99 | 924 / 99 | 1140 / 99 | 1444 / 99 | 1766 / 99 |
| 11 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 12 | 7 / 7 | 21 / 7 | 35 / 7 | 52 / 7 | 52 / 7 | 52 / 7 | 52 / 7 |
| 13 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| **all** | **160 / 160** | **480 / 160** | **800 / 160** | **1503 / 160** | **1939 / 160** | **2819 / 160** | **3741 / 160** |


**160 independent pre-onset episodes at every horizon from 1 to 60 minutes**, against
**3 / 9 / 15 at 1 / 3 / 5 minutes in CIC-IDS2017's 2.27M training origins** (Run 8 §8.7).
This is roughly a fifty-fold increase in the evidence available for Task B, and it is the
main scientific reason to add this dataset.

Two qualifications, both material:

* The 160 onsets are concentrated in scenarios 9 and 10 (34 and 99), which are the two
  multi-bot captures. Ten hosts contribute most of them, so the effective independence for a
  bootstrap is closer to "ten hosts across four captures" than to 160 free draws. The
  benchmark's episode-cluster bootstrap should be read with that in mind and a host-level
  cluster reported alongside.
* Five scenarios (1, 2, 5, 7, 13) contribute **zero** pre-onset rows: their infected host is
  already emitting botnet traffic in its first observed window, so there is no run-up to
  forecast from. That is the same failure mode CIC-IDS2017 has throughout.

---

## 8. Proposed preprocessing

No change to the pipeline: CTU-13 enters through an adapter
(`nidra/data/ctu_load.py`) that emits the internal flow columns, and windowing, graph
scalars, dynamics, labelling, splits, training and the benchmark are shared with
CIC-IDS2017 unchanged. Concretely:

1. stream each `.binetflow`, drop unparseable rows with a reason, restrict sources to
   147.32.0.0/16;
2. map to internal columns with the unit conversions of §5;
3. window at Δ=60 with the existing `windowize_day`, flow-only mode, packet features zero;
4. label with the `ctu` dialect;
5. cache one parquet per scenario under a key that carries the format and the host scope,
   so a CIC table and a CTU table can never collide.

Normalisation is fit on the training captures only, under the declared regime, and the
scaler records both the regime and the training population.

---

## 9. Recommended split, and why

**Primary (CTU-only):** temporal first, family holdout second.

| role | captures | families | note |
|---|---|---|---|
| train | 1, 2, 3 | Neris, Rbot | ends 2011-08-15 10:13 |
| validation | 4, 6 | Rbot, **Menti** | Menti is in no training capture, so selection is scored on a family transfer |
| test | 8, 9, 10 | **Murlo**, Neris, Rbot | one unseen family, two seen — the same mixture CIC's Friday has |
| holdout | 13, 12 | **Virut**, **NSIS.ay** | both families entirely unseen |

Every evaluation capture begins after the last training capture ends. This mirrors the
CIC-IDS2017 protocol exactly — early days train, later days test, an unseen-family holdout —
which is what makes a CIC number and a CTU number comparable at all.

Validation is **held-out captures rather than a trailing time block**. The trailing carve is
right for CIC, where each training day holds one attack in its middle; on CTU it is
pathological. Scenario 1's episode runs from 1.3 h to past the 70% mark of a 6.1 h capture,
the no-straddle nudge moves the cut to 0.8 h, and 88% of the capture — with every one of its
attack windows — lands in validation. Measured: under the trailing carve, train held 161
attack windows and validation held 1,197. Under held-out captures it is 1,358 and 213.

**Alternatives considered and rejected as the primary.**

* *Random or within-scenario row split* — rejected outright: host, scenario and episode all
  leak.
* *Pure family holdout* (train Neris+Menti+Sogou+NSIS.ay, test Rbot, holdout Virut) —
  stronger on family separation but violates chronology, training on August 19 to predict
  August 12. Kept as a **secondary generalisation experiment** (leave-one-family-out), where
  family transfer is the question being asked and the chronology violation is stated.
* *Host holdout* — only scenarios 9, 10, 11 and 12 have more than one bot, and
  `147.32.84.165` is in all of them. A partial host holdout inside 9/10 (train on five bots,
  evaluate on the other five) is a real test and is kept as a secondary experiment.
* *Scenario holdout by size* (put the big captures in test) — rejected: it chooses the split
  by what it contains rather than by a rule.

**Cross-dataset regimes** are separate configs over the same captures: CIC→CTU and CTU→CIC
keep validation in the SOURCE dataset (an operating point selected on the target is not a
transfer result), and the combined run trains on both and is evaluated separately on each
dataset's held-out captures.

---

## 10. Limitations

1. **Flow-only.** Eleven packet features and `iat_max` do not exist here. A CTU number and a
   Run 8 number are not comparable without the `cross_core` mask on both sides.
2. **The negative class is unverified.** 96% Background; precision is a lower bound (§2).
3. **One physical infected host across all thirteen captures** (§4). A scenario holdout is
   not a host holdout.
4. **Pre-onset evidence is concentrated.** 160 onsets, but ten hosts and four captures
   produce nearly all of them (§7).
5. **Two flag features are not semantically comparable** across datasets (`ack_ratio`,
   `psh_ratio`); `cross_strict` exists to measure what they are worth.
6. **Three captures are unusable at L=30, K=6** (§6).
7. **2011 traffic.** Protocol mix, TLS adoption and port usage are of their time; nothing
   here says a 2011-trained model would transfer to a modern network.
8. **Background labelling is heuristic.** The authors' own README describes assigning
   Background first and overriding it with filters, so the label boundary between Normal and
   Background is soft.

# NIDRA — Complete Project Deep Dive

**Purpose of this document:** you should be able to read this file start to
finish with *zero* prior machine learning or cybersecurity background and
come out the other side able to (a) explain what this project does and why
it's built this way, (b) defend every non-obvious design decision if
someone pushes back on it, and (c) rebuild the system yourself from a blank
directory. It goes deep into theory, then walks the actual code, then
explains why each choice was made instead of the obvious alternative.

This is a companion to, not a replacement for:
- `README.md` — quick orientation and command reference.
- `MODEL_CARD.md` — compact "what is this model" reference card.
- `TRAINING.md` / `EVALUATION.md` — exact commands and flag meanings.
- `REAL_DATA_RESULTS.md` — the actual numbers from real runs, with honest caveats.
- `PRODUCTION_RUN_GUIDE.md` — a beginner's walkthrough of running the full-scale training job.

Read this one when you want to understand *why*, not just *how*.

---

## Part 1 — The problem, in plain English

### 1.1 What is network intrusion detection, normally?

A computer network is a bunch of machines ("hosts") sending each other
packets of data. Somewhere on that network, security software watches the
traffic and tries to answer one question: **is someone attacking us right
now?**

The traditional approach is a **classifier**: look at a chunk of traffic
(usually one "flow" — one connection between two machines), extract some
numbers from it (how many bytes, how many packets, how long did it last),
and feed those numbers into a model that outputs "attack" or "not attack."
This is what most antivirus/IDS (Intrusion Detection System) products do,
and it's what almost every published CIC-IDS2017 paper does too — the
dataset was originally built for exactly this kind of per-flow
classification.

The problem: a classifier tells you about an attack **after** it has
already happened, or at best, the instant it's happening. It never tells
you "this host is 3 minutes away from being compromised." It has no
concept of time passing, no concept of "where is this heading."

### 1.2 What NIDRA does instead: forecasting, not classifying

NIDRA (originally called HORIZON in the design spec — see §11.1 for why the
name changed) doesn't ask "is this flow an attack?" It asks a different
question entirely:

> "Given everything this host has done in the last 15 minutes, what is it
> likely to be doing 30 seconds, 1 minute, ... all the way up to 3 minutes
> from now — and how risky does that future look?"

This is a forecasting problem, not a classification problem. The
distinction matters enormously for what the model has to be capable of:
a classifier maps "now" to "a label." A forecaster maps "the recent past"
to "a *simulated future*," and only then asks "how risky is that simulated
future." This is what's called a **world model** in machine learning (see
Part 2) — and it's the reason this project can report things a classifier
structurally cannot, like "lead time" (how many minutes of warning did we
get before the attack fully materialized) and "state forecast error" (how
wrong was our simulated future, measured against what actually happened
later).

### 1.3 Why per-host, not per-flow

CIC-IDS2017 (the dataset — see §3.1) is normally consumed one flow at a
time: each row of the published CSV is one network connection, with ~80
statistics CICFlowMeter computed about it. NIDRA does not use rows this
way. Instead, it aggregates **all the traffic for one host, in one 30-second
window**, into a single 45-number "state vector" (§3.3). A "host" here is
one IP address; its state vector describes what that machine was doing
network-wise in that half-minute.

Why per-host instead of per-flow? Because a compromise is a process that
plays out on a *machine* over *time* — first it gets scanned, then
probed, then a foothold is established, then it starts talking to new
peers, then it starts moving data out. Any single flow only shows you one
brief connection; the host's *trajectory* across many windows is what
reveals the escalation. A per-flow classifier cannot see a trajectory; a
per-host, per-window sequence model can.

### 1.4 What "stage" means here

NIDRA labels each window with one of six **stages**, a simplified,
CIC-IDS2017-specific stand-in for the kind of attack-lifecycle taxonomy
security people call "tactics" (as in MITRE ATT&CK, the industry-standard
catalogue of attacker techniques):

```
benign → recon → initial_access → lateral → c2 → exfil
```

**Important claims-discipline point, stated once here and repeated in
`MODEL_CARD.md`/`README.md`:** this six-stage mapping is a *curated
presentation label* built by pattern-matching CIC-IDS2017's own `Label`
column (see `nidra/data/labels.py`), not a technique-level ATT&CK mapping
produced by any authoritative process. The dataset itself doesn't carry
enough information to do real ATT&CK-technique attribution. Two
simplifications are made explicitly and documented in code:
`Infiltration` (a two-phase attack — initial access then lateral movement
in the dataset's own description) is mapped wholesale to `lateral`, and
DoS/DDoS/Heartbleed (which the original design spec mapped to "Impact," a
category NIDRA's 6-class head doesn't have) are mapped to `exfil` as the
closest available terminal-impact bucket. If someone asks "is this real
ATT&CK mapping," the honest answer is no — it's a scoped, documented
simplification, and saying so up front is a sign of rigor, not a weakness.

---

## Part 2 — The core ML idea: world models

### 2.1 What a "world model" is, conceptually

The term comes from a lineage of ML research (most famously Ha & Schmidhuber's
2018 "World Models" paper, and its intellectual descendants in model-based
reinforcement learning) built around one idea: **instead of learning a
direct mapping from an observation to an action or a label, learn a
compressed internal model of how the environment evolves over time, then
use that model to imagine forward.**

Concretely, a world model has three pieces:

1. **An encoder** that compresses a sequence of raw observations into a
   compact internal summary (a "hidden state" or "latent state").
2. **A transition model** (sometimes called a "dynamics model") that
   predicts how that hidden state changes one step into the future.
3. Optionally, **decoder/head networks** that turn a hidden or predicted
   state back into something interpretable — a prediction, a risk score, a
   classification.

The defining behavior that separates a world model from a plain sequence
predictor is **recursive rollout**: you don't just predict one step ahead.
You predict step 1, then **feed your own prediction back in as if it were a
real observation**, and predict step 2 from that, and so on. This is
"imagining forward" — simulating a plausible future trajectory, K steps
deep, using nothing but the model's own beliefs about how the world
evolves. A plain classifier or a direct multi-horizon regression head
cannot do this: it has no notion of an internal state to feed forward.

### 2.2 Why NIDRA is built this way, concretely

Map the abstract world-model idea onto this codebase:

| World-model concept | NIDRA implementation | File |
|---|---|---|
| Encoder | 2-layer GRU over 30 windows of 45-dim state | `nidra/models/encoder.py` |
| Transition model | MLP emitting a Gaussian (mean + variance) over the *change* in state | `nidra/models/transition.py` |
| Recursive rollout | Feed the transition's own sampled output back into the GRU, K=6 times | `nidra/models/world_model.py::rollout` |
| Heads (decoder-ish) | Small MLPs mapping a (real or imagined) 45-dim state to risk probability / attack-stage distribution | `nidra/models/heads.py` |

This is why the project can produce metrics no classifier could: **state
forecast error** (`state_nrmse` — how far off was the *entire imagined
45-dimensional future* from what actually happened, not just "was the
label right"), and **lead time** (since the model produces a risk curve
continuously into the simulated future, you can ask "how many seconds
before the real attack did the simulated risk first cross the alert
threshold").

### 2.3 Why recursive rollout, not direct multi-horizon regression

A tempting simpler design: instead of rolling out step-by-step, why not
just train a network with 6 separate output heads, one per horizon
(`k=1..6`), each directly regressing the state at that horizon from the
input history? This is called **direct multi-horizon regression**, and
it's simpler to implement and often trains more stably (no compounding
error across steps).

NIDRA deliberately does **not** do this, for two reasons:

1. **It cannot express "trajectories."** A direct multi-horizon model
   produces 6 independent point predictions with no guarantee they're even
   *consistent with each other* as one coherent future — there's nothing
   stopping horizon-3's prediction from implying a completely different
   trend than horizon-4's. A recursive rollout, by construction, always
   produces one continuous, self-consistent simulated trajectory, because
   each step is generated *from* the previous one.
2. **It's the only design that supports genuine uncertainty quantification
   over whole futures.** Because the transition model is stochastic (it
   samples noise at every step — see §4.2), running the rollout multiple
   times produces a *distribution* over entire future trajectories, not
   just a per-horizon confidence interval computed independently at each k.
   This is what backs the "~1000 sampled futures" claim and the confidence
   bands in the forecast output.

The tradeoff, paid honestly: recursive rollout **compounds error** — a
mistake at step 2 propagates into step 3's input. This is real and
measured (see the horizon-curve ablation in §6.4/§9). Direct regression
doesn't have this problem but gives up the two capabilities above. This
project's problem statement explicitly asks for a forecasting/simulation
capability, so the trade was made deliberately in that direction — see the
design-decision log in Part 11 for the full "why not the simpler thing"
reasoning.

### 2.4 Why predict a *delta*, not the absolute next state

`nidra/models/transition.py` predicts `mu`, and the next state is computed
as `S_hat[t+1] = S[t] + mu` — a *change*, added to the current state —
rather than predicting the absolute next state directly. Two reasons,
both load-bearing:

1. **It's a strong, correct prior.** Network state at 30-second resolution
   is highly autocorrelated — most of the time, "what happens next" looks
   a lot like "what's happening now," plus a small change. Handing the
   model this structure for free means its learning capacity goes toward
   modeling the *interesting part* (the change) instead of re-deriving
   "things don't change much" from scratch every time.
2. **It makes the most important baseline free.** The **persistence
   baseline** — "assume nothing changes, `S_hat[t+k] = S[t]` for every
   k" — is *exactly* the `mu = 0` case of this same model. This isn't a
   coincidence; it's why the persistence ablation in `eval/ablations.py`
   can cleanly isolate "how much did the *learned dynamics* add," by
   running the identical frozen risk head against the identical frozen
   encoder, with only the transition's contribution zeroed out. If the
   model predicted absolute states instead, there'd be no clean way to ask
   "is the transition model doing anything useful at all," which is one of
   this project's central falsifiable claims (Rule 3, §11).

### 2.5 Why the transition is a *distribution* (Gaussian), not a point estimate

`Transition.forward` returns `(mu, logvar)` — a predicted mean *and* a
predicted (log) variance, per feature. The next state at rollout time is
`mu + noise`, where `noise ~ Normal(0, exp(logvar))`. This means the
model doesn't just guess "the network will look like this" — it also says
"...and I'm this uncertain about it."

Why this matters:

- **Training signal.** The loss function (Gaussian negative log-likelihood,
  §5.1) rewards the model for being *appropriately* uncertain — confidently
  wrong is penalized harder than admitting uncertainty on a genuinely hard
  window. A model trained with plain mean-squared-error has no way to
  express "I don't know" at all.
- **It's what makes the ~1000-trajectory ensemble meaningful.** Every one
  of those simulated futures is a different *sample* from this Gaussian at
  every step. If the transition were deterministic, running the rollout
  1000 times would produce 1000 identical trajectories — the whole
  uncertainty-quantification story would collapse.

The variance is **clamped** to `[logvar_min=-6, logvar_max=3]`
(`nidra/models/transition.py`). This is not a stylistic choice — the code
comment states plainly why: *"Without this the model discovers that
predicting infinite variance on hard features minimises NLL, variance
explodes, and rollout produces NaN by k~3."* This is a real, specific
failure mode of Gaussian NLL training (an unconstrained model can always
reduce its loss on a genuinely unpredictable feature by inflating its
predicted variance toward infinity, which technically minimizes the loss
function but produces a useless, numerically unstable model) — the clamp
is a hard guardrail against it, discovered and documented in the original
build spec (`docs/IMPLEMENTATION-ML.md` §3.2/§9) before this codebase was
even written, and confirmed necessary again during this session's
calibration investigation (§10).

### 2.6 Why two training stages, strictly separated

This is the single most important architectural rule in the whole project
(**Rule 1** in the original spec, `docs/IMPLEMENTATION-ML.md` §0):

> *"The risk head and stage head are trained on observed states and never
> see a predicted state during training... If you ever find yourself
> fine-tuning a head on rollout output because the numbers improve, you
> have destroyed the central argument of the project."*

Concretely:

- **Stage 1** trains only the encoder + transition (the "dynamics"), on a
  self-supervised objective — predict the next K real states from the
  input history. This stage never uses attack labels at all.
- **Stage 2** *freezes* the encoder and transition (`model.freeze_dynamics()`),
  then trains only the risk head and stage head, and — critically — **only
  on real, observed states** (`S_t`, never a state the model imagined
  during rollout). Once done, the heads are frozen too
  (`model.freeze_heads()`), and nothing in the codebase ever unfreezes
  anything again.

**Why this separation matters so much:** if you let a classifier head see
and train on the *model's own rolled-out predictions*, you create a
feedback loop where the head can learn to compensate for the dynamics
model's specific quirks and mistakes — the head effectively "cheats" by
adapting to whatever the rollout happens to produce, rather than learning
a genuine mapping from network state to risk. The evaluation numbers would
look better, but the entire architecture's *forecasting* claim becomes
unfalsifiable: you'd no longer be able to tell whether good performance
came from "the dynamics model correctly simulates the future" or "the head
learned to rationalize whatever garbage the dynamics model produces."
Keeping the head blind to rollout output during training is what makes
"the dynamics model's simulated future, scored by an honest, unbiased
classifier" a meaningful, falsifiable claim rather than a circular one.

---

## Part 3 — The data pipeline, step by step

### 3.1 The dataset: CIC-IDS2017

CIC-IDS2017 is a public network-traffic dataset from the Canadian Institute
for Cybersecurity, built by running real attack tools against a small lab
network over 5 working days (Monday–Friday) and capturing everything: raw
packet captures (PCAP files, ~10-14GB each) and CICFlowMeter-derived flow
summaries (CSV, one row per network flow with ~80 statistics).

| Day | What happened | Role in NIDRA |
|---|---|---|
| Monday | Nothing — pure benign baseline traffic | Train (benign) |
| Tuesday | FTP-Patator, SSH-Patator (credential brute-forcing) | Train |
| Wednesday | DoS variants (Hulk, GoldenEye, Slowloris, Slowhttptest), Heartbleed | Train |
| Thursday | Web attacks (Brute Force, XSS, SQL Injection) in the morning; **Infiltration** in the afternoon | **Holdout — never trained on** |
| Friday | Botnet, PortScan, DDoS (LOIT) | Test |

**Why hold out Thursday/Infiltration entirely** rather than splitting it
across train/test: this is the project's **generalization claim**. Every
other split contains attack types the model has at least seen *some*
version of during training. Infiltration is different in kind (a two-stage
credential-compromise-then-lateral-movement attack, not a single flooding
or scanning pattern) and is never shown to the model in any form during
training. Evaluating on it answers a genuinely different, harder question
than the test split does: "does this generalize to an attack family it has
never seen," not just "does this work on more examples of attack families
it has seen." (The honest answer this project found: no, not yet — see
Part 9.)

**Why use the raw PCAPs at all, when CICFlowMeter's CSVs already exist?**
Two reasons. First, some published CIC-IDS2017 CSV releases
("MachineLearningCVE") strip out Source/Destination IP and Timestamp
entirely — the "TrafficLabelling" release keeps them, and this project
uses that one specifically (`nidra/data/join.py`'s docstring calls this
out). Second, and more fundamentally, CICFlowMeter's flow statistics don't
include several signal types this project's 45-feature schema needs — TTL,
TCP window size, fragmentation, retransmissions — which only exist at the
raw packet level. Getting those requires re-parsing the actual capture
files with `tshark`.

### 3.2 PCAP extraction: why tshark, not PyShark/Scapy

`nidra/data/pcap_extract.py`'s docstring states the reasoning directly:
PyShark (a popular Python wrapper) shells out to the underlying `tshark`
binary **once per packet**, which is roughly two orders of magnitude too
slow for a 10-14GB capture with tens of millions of packets. Scapy's
pure-Python packet parser has the same bulk-throughput problem. The fix:
call `tshark` directly as a single subprocess with `-T fields`, asking it
to emit only the exact columns needed (`frame.time_epoch`, `ip.src`,
`ip.ttl`, `tcp.window_size_value`, etc.) as CSV-like text, and parse that
text stream in Python. This is roughly two orders of magnitude faster
because tshark does all the actual packet parsing in its own optimized C++
code, and Python only has to split lines and build a DataFrame.

**Why streamed, never loaded fully into memory**: a 10-14GB PCAP can carry
tens of millions of packets. `stream_tshark_chunks` reads tshark's stdout
line by line, batches 200,000 lines at a time, and writes each batch to a
growing Parquet file immediately (`pq.ParquetWriter`) before reading more.
At no point does the full capture, or even the full parsed text output,
exist in memory at once. This is the difference between "runs on a laptop"
and "needs a server with 64GB of RAM."

**A real bug this design caught**: `-E occurrence=f` in the tshark
invocation. Some packets (ICMP errors) embed a *second* copy of an IP
header (the original packet that triggered the error) inside them, so a
field like `ip.src` can legitimately have two values in one packet. Without
`occurrence=f` (take only the first), tshark joins both values with a comma
— the same character used as the field separator — silently producing rows
with more commas than expected. This was caught against the real Monday
capture, where it was corrupting about 0.25% of rows, and is now guarded
against both by the tshark flag and by a runtime check that drops (and
logs) any row whose column count doesn't match expectations.

### 3.3 Windowing: turning a firehose of packets/flows into a clean timeline

Every flow and every packet gets assigned to a `(host_id, window_ts)` pair,
where `host_id` is the source IP and `window_ts` is the packet/flow's
timestamp **floored to the nearest 30-second boundary** (`align_window` in
`nidra/data/windowize.py`) — not relative to the first observed packet.
This matters: if window boundaries were relative to "the first thing this
particular file happened to record," two different capture files (or two
different hosts starting activity at different times) would have
misaligned windows and could never be joined or compared consistently.
Aligning to absolute epoch multiples of 30 means "window starting at Unix
time 1499437020" means the same thing everywhere, always.

**Why 30 seconds specifically?** This is one of the project's protected
design constants (`Δ = 30s` in the original spec). It's a balance: short
enough that an attack's escalation across windows is visible at reasonably
fine granularity, long enough that a single window contains enough packets
for aggregate statistics (ratios, entropies, variances) to be meaningful
rather than dominated by noise from 1-2 packets.

**Why 30 windows of context (15 minutes) and 6 windows of forecast horizon
(3 minutes)?** Also protected constants (`L=30`, `K=6`). 15 minutes of
history is enough to see a slow ramp-up (recon → probing → escalation)
without asking the GRU to remember an impractically long sequence. 3
minutes of forecast horizon is short enough that the recursive rollout's
compounding error (§2.3, §9) stays somewhat bounded, while still being long
enough to be operationally useful — a security analyst getting a 3-minute
warning genuinely has time to act, versus a 3-second warning which does
not.

**Empty windows are never dropped.** If a host does nothing in some
30-second window, it still gets a row: all zero-valued features, plus an
`is_active=0` flag. `_fill_empty_windows` in `windowize.py` explicitly
reindexes every host's timeline to be gap-free across its entire active
range. Why this matters: if silent windows were simply omitted, the
sequence "window 10, window 11, window 14" would look — to a model that
only sees index position, not real elapsed time — identical to "window 10,
window 11, window 12." Every downstream time-based calculation (lead time
in seconds, the GRU's implicit assumption of uniform time steps, the
delta/slope features described below) silently breaks if windows are
allowed to have gaps. This is exactly the kind of bug that produces
results that look fine in aggregate metrics but are subtly wrong in a way
that's very hard to catch later — so it's enforced structurally, not left
as a "remember to check this" note.

### 3.4 The 45-feature state vector

Every `(host, window)` pair is described by exactly 45 numbers, defined
once, in one fixed order, in `nidra/data/schema.py::FEATURE_ORDER`. Every
other module in the codebase — windowing, normalization, the model, SHAP,
serving — imports this list rather than ever reconstructing feature order
from a dictionary (dict key order is easy to accidentally shuffle across a
refactor; a single canonical ordered list, asserted to have exactly 45
unique entries at import time, cannot silently drift).

The 45 features break into five groups:

**Flow aggregates (15)** — computed from CICFlowMeter's per-flow numbers,
summed/averaged per host per window: TCP flag ratios (`syn_ratio`,
`ack_ratio`, `rst_ratio`, `fin_ratio`, `psh_ratio`, `urg_ratio` — each is
"count of flags of this type / total packets this window," a scan or flood
tends to have an unusual flag-ratio fingerprint), byte/packet volume
(`bytes_total`, `bytes_up_down_ratio`, `pkts_per_flow_mean`), flow timing
(`flow_duration_mean`, `flow_duration_var`, inter-arrival-time `iat_mean`/
`iat_var`/`iat_max`), and `active_flow_count` (how many distinct
connections this host had open this window).

**Packet aggregates (11)** — only available where a PCAP has been
extracted for that day (§3.2); zero-filled and logged loudly if not
(`build_state_rows`'s warning). TTL statistics (`ttl_mean`, `ttl_var` — an
attacker's traffic sometimes has an unusually consistent or unusual TTL,
since it often originates from a different OS/hop-count than the rest of
the network), TCP window size (`tcp_window_mean`, `tcp_window_entropy`),
fragmentation rate (`frag_flag_rate` — unusual fragmentation is a classic
evasion/scanning signal), payload size statistics (`payload_size_mean`/
`_var`/`_p95`/`_entropy`), and retransmission behavior (`retrans_count`,
`retrans_rate` — floods and unstable/malicious connections retransmit at
different rates than healthy traffic).

**Graph scalars (8)** — computed by treating each 30-second window's flows
as a transient directed graph over hosts (`nidra/data/graph_features.py`),
discarded immediately after its 8 numbers are extracted (never persisted —
it's a feature extractor, not a stored object): `out_degree`/`in_degree`
(how many distinct peers this host talked to/from), `dst_ip_entropy`/
`dst_port_entropy` (Shannon entropy over which IPs/ports were contacted —
rising entropy means "spreading out," the fingerprint of scanning),
`new_peer_count` (peers never contacted before, tracked via a running
per-host memory — the *one* genuinely stateful feature in the whole
pipeline, computed in strict chronological order, never vectorized, because
"have I seen this peer before" is inherently a function of everything that
came before it), `neighbour_risk_fraction` (a **heuristic**, not
model-derived, prior — see the important note below), `local_clustering_coeff`
and `reciprocity` (standard graph-theory measures of how tightly-knit a
host's neighborhood is and how often its outgoing contacts talk back).

> **Why `neighbour_risk_fraction` is heuristic, not model-based — a subtle
> but important design decision.** It would seem natural to define "what
> fraction of my peers are risky" using the *model's own* risk predictions
> for those peers. This project deliberately does not do that. Doing so
> would create a feedback loop with no ground truth to debug against: host
> A's risk score would depend on host B's risk score, which depends on
> host A's, compounding through the network with every training step, and
> any resulting garbage would be nearly impossible to diagnose. Instead,
> `neighbour_risk_fraction` uses a fixed, non-circular heuristic: the
> fraction of this host's peers whose *out-degree* (not risk score) in the
> *previous* window exceeded that window's median out-degree — "was I
   talking to peers who look unusually chatty," a real but simple, debuggable
> signal.

**Dynamics features (10)** — first differences (`d_syn_ratio`,
`d_dst_port_entropy`, `d_out_degree`, `d_new_peer_count`, `d_iat_var`,
`d_retrans_rate` — literally `x[t] - x[t-1]`) and 3-point rolling OLS
slopes (`slope3_syn_ratio`, `slope3_dst_port_entropy`, `slope3_out_degree`,
`slope3_iat_var`) over the ~6 most discriminative base features. **Every
one of these is strictly backward-looking** — computed only from `t-2..t`,
padded with zero at the start of a host's sequence, never from a future
value. This is a deliberate, explicit design choice: even though these are
*input features*, not labels, it would be trivially easy to accidentally
compute a "smoothed" or "centered" version of a rolling statistic that
peeks slightly into the future — and because these are exactly the
features an escalating attack would move the most, that kind of leak would
inflate every headline metric while being very hard to spot just by
looking at final numbers. Rule 2 in the original spec exists specifically
because of this risk: *"Nothing after time t may touch the input... Every
leak makes the results better and the project worthless."*

**Activity (1)** — `is_active`, the empty-window flag from §3.3.

**Two normalization details that matter:**

1. Five heavy-tailed, strictly-non-negative "count-like" features
   (`bytes_total`, `active_flow_count`, `out_degree`, `new_peer_count`,
   `retrans_count`) get `log1p`-transformed *before* scaling
   (`nidra/data/normalize.py::LOG1P_FEATURES`). A single DDoS window can
   send `bytes_total` orders of magnitude higher than a typical window;
   without log-compression, that one huge value would dominate a linear
   scaler's fitted range and squash every normal-traffic value into a tiny
   sliver near zero.
2. **`RobustScaler`, not `StandardScaler`**, fit **only on the training
   split**. `RobustScaler` centers/scales using the median and interquartile
   range instead of the mean and standard deviation — a choice that matters
   a lot here because network traffic features are heavy-tailed, and a
   single extreme DDoS window can otherwise drag a mean/std-based scaler's
   fit far enough that it distorts the whole normal-traffic range. Fitting
   only on the training split (never test/holdout, never refit at serving
   time — enforced structurally: `FeatureScaler.transform` raises if
   called before `fit()`/`load()`) is the single most common way this class
   of project accidentally leaks information: if the scaler "sees" test
   statistics during fitting, test-split evaluation numbers become
   contaminated.

### 3.5 Labels: risk_label and stage_label

Two separate targets are built, both in `nidra/data/labels.py`, and
**kept structurally separate from the 45 input features** (they live in a
different module and are never merged into `FEATURE_ORDER`):

- **`stage_label[t]`**: the most severe attack stage present *at* window
  `t` itself (benign if none). This is the *observed* state.
- **`risk_label[t]`**: `1` if *any* non-benign window occurs in `(t, t+K]`
  — i.e., "does an attack happen somewhere in the next 3 minutes" — else
  `0`. This is the *forward-looking* supervision target for Stage 2's risk
  head.

`risk_label` looks at the future — deliberately, and this is legitimate.
The critical distinction, stated explicitly in the original spec (§2.9) and
worth being able to explain confidently: **using future information to
construct a label is completely different from using it to construct an
input feature.** Every supervised learning problem does the former — you
need to know the true answer to train against it. What would be illegitimate
(and is what Rule 2 exists to prevent) is if that same future information
leaked into the *45-dimensional state vector itself*, because then the
model would effectively be looking at the answer while making its
prediction. `attach_risk_label`'s own docstring makes this boundary
explicit: *"Uses future windows ONLY to construct these two target
columns — this function must never be called anywhere near model-input
construction."*

### 3.6 Splitting: temporal, episode-aware, and day-based

Three layers of protection against a subtle and very common bug class in
this kind of project — **leakage across the train/test boundary**:

1. **Day-based first**: train = Monday+Tuesday+Wednesday, test = Friday,
   holdout = Thursday. Since CIC-IDS2017's attacks are each confined to a
   single day, this trivially guarantees no single attack episode spans
   two different splits.
2. **Within train, a contiguous trailing time block** (not a random
   sample!) is carved out for validation. A **random** row-level split
   would let the model implicitly memorize windows that sit *immediately
   next to* windows it trained on — since consecutive windows for the same
   host are highly correlated, "test" performance under a random split can
   look great for reasons that have nothing to do with genuine forecasting
   ability. A contiguous trailing block avoids this: everything in
   validation happened strictly after everything remaining in train.
3. **Episode-boundary nudging**: if the naive time-based cutoff would fall
   in the *middle* of a live attack episode (splitting one continuous
   attack across train and validation), the cutoff is automatically moved
   earlier, to that episode's start (`temporal_train_val_split` in
   `nidra/data/splits.py`). Two structural test-facing checks
   (`assert_no_episode_leakage`, `assert_no_temporal_overlap`) verify this
   holds, programmatically, every time splits are built — not just
   asserted in a comment.

---

## Part 4 — The model, file by file

### 4.1 Encoder (`nidra/models/encoder.py`)

A 2-layer GRU (Gated Recurrent Unit — a type of recurrent neural network
designed to carry information forward through a sequence while mitigating
the "vanishing gradient" problem plain RNNs suffer from over long
sequences). Input: 30 windows × 45 features. Output: a 128-dimensional
hidden vector `h_t` summarizing "everything relevant about this host's
last 15 minutes," plus the full recurrent state needed to continue the
sequence one step at a time later (this is what lets rollout re-enter the
GRU incrementally instead of re-processing the whole history at every
step).

**Why a GRU and not, say, a Transformer?** A single raw observation window
is not "Markov" — you cannot tell from one window alone whether a ratio is
rising or falling. You need `h_t = f(S_1..S_t)`, some function of the whole
history. A GRU is a well-understood, computationally cheap way to build
that function incrementally, and — importantly for this project's
recursive-rollout design — a GRU's hidden state is naturally *the exact
object you feed the next step's input alongside*, one token at a time.
(A Transformer *could* do this too, but would need either a full
re-encoding of the growing sequence at every rollout step, which is far
more expensive, or a more complex incremental-attention scheme; the GRU's
simplicity was the right trade for a CPU-target hackathon deliverable.)

### 4.2 Transition (`nidra/models/transition.py`)

Already covered in depth in §2.4/§2.5. Structurally: `h_t → MLP(256,256)
→ (mu: 45, logvar: 45)`, `logvar` clamped to `[-6, 3]`.

### 4.3 Heads (`nidra/models/heads.py`)

Two small MLPs: `RiskHead` (45 → 64 → 1, sigmoid) and `StageHead` (45 → 64
→ 6, softmax). **Both operate on the raw 45-dimensional feature vector,
never on the GRU's 128-dim hidden state.** This is deliberate, for two
reasons stated directly in the module docstring: (1) a head that only
understands named features can be applied identically to a real observed
state *or* a rolled-out predicted state — there's no representational gap
to bridge, since the transition model already decodes all the way down to
the same 45 named features the head was trained on; (2) small models
operating on a modest number of *named* features are what makes KernelSHAP
(§7) computationally tractable and its output actually interpretable
("`dst_port_entropy` contributed +0.3 to this risk score" is meaningful in
a way "hidden unit 47 contributed +0.3" is not).

### 4.4 World model assembly and rollout (`nidra/models/world_model.py`)

Ties encoder + transition + heads together and implements `rollout()`,
already discussed at length in §2.1-2.3. One more detail worth calling out:
`rollout()` supports `n_samples > 1` by **tiling the batch** (`repeat_interleave`)
rather than looping in Python — i.e., "run this rollout 500 times" is
implemented as one big vectorized batch of size `B × 500`, not 500 sequential
Python-level calls. This is a real performance-relevant design choice: it's
what makes generating ~1000 sampled trajectories per forecast fast enough
to hit the sub-300ms serving latency target on CPU.

`score_states()` is the method that applies the (frozen, at inference time)
heads to any batch of 45-dim states, real or imagined — the single
function that both "the observed state's current risk" and "every rolled-
out future state's risk" both go through.

---

## Part 5 — Training, precisely

### 5.1 Stage 1: the dynamics loss

`nidra/train/losses.py::dynamics_loss`. For each of the K=6 steps:

1. The transition predicts `(mu, logvar)` from the current hidden state.
2. `pred = cur + mu` (the delta-prediction design, §2.4).
3. Loss for this step = **Gaussian negative log-likelihood** between
   `pred` and the true next state, weighted by `0.85^k` (a **horizon
   discount** — errors at distant, inherently harder-to-predict horizons
   contribute less to the gradient than errors at horizon 1, so training
   doesn't get dominated by trying to perfect an inherently noisy
   6-steps-ahead prediction at the expense of getting the easier near-term
   steps right).
4. **Scheduled sampling** decides what gets fed back into the encoder for
   the next step: with probability `teacher_forcing_p`, the *true* next
   state (helps training stay stable early on); otherwise, the model's own
   (detached — no gradient flows back through this path) prediction. This
   probability is annealed from `1.0` down to `0.3` over the first 60% of
   training.

**Why scheduled sampling matters — this is not a minor detail.** If you
always feed the true next state back in during training ("teacher
forcing," full stop), the model never practices recovering from its own
small errors — every step of training, it always sees a perfect,
error-free history. But at *inference* time (real rollout), there is no
true next state to feed back — the model must consume its own,
imperfect predictions, and small errors compound. A model trained purely
with teacher forcing can look great at k=1 and fall apart by k=3-6,
because it's never been exposed to the accumulating-error regime it will
actually face. Annealing the teacher-forcing probability down during
training forces the model to gradually get used to consuming its own
imperfect output — exactly what it'll have to do for real.

### 5.2 Stage 2: head training, the freezing discipline, and class imbalance

Already covered architecturally in §2.6. Practically:
`model.freeze_dynamics()` sets `requires_grad=False` on every encoder and
transition parameter before Stage 2 starts, so the optimizer physically
cannot update them even by accident. The risk head trains with
`BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` — since attacked windows are a
small minority of all windows, `pos_weight` up-weights the loss
contribution of the rare positive class so the model isn't rewarded for
just always predicting "benign." The stage head uses inverse-frequency
class weights in its cross-entropy loss for the same reason across 6
classes instead of 2. After Stage 2 completes, `model.freeze_heads()` is
called — from that point on, in this process and every artifact saved from
it, **nothing in the model ever trains again.**

### 5.3 The ensemble

Five independent training runs (`seeds = [0,1,2,3,4]`), identical
architecture and data, different random initialization/shuffling. At
inference time, all 5 members' rollout trajectories are pooled together
before scoring (`NidraPredictor._ensemble_rollout`) — not just averaged
after independently scoring each member. Why an ensemble at all: a single
neural network's uncertainty estimate (the Gaussian `logvar` it emits) only
captures *aleatoric* uncertainty — "how noisy is this specific situation."
It cannot capture *epistemic* uncertainty — "how much does the model
itself vary depending on which random initialization/training run you
happened to get." Training 5 independent models and pooling their outputs
is the standard, simple way to approximate that second kind of uncertainty
without a more exotic Bayesian architecture — a well-established, low-risk
technique for a project with a hard deadline.

---

## Part 6 — Evaluation framework

### 6.1 The four mandated baselines — and why each one exists

From `nidra/eval/baselines.py`, in order of "how much of the interesting
machinery does this baseline include":

1. **LR on `S_t`** — plain logistic regression on just the current
   45-dim state, no history at all. Mandated by the original problem
   statement; the simplest possible sanity floor.
2. **LR on flattened history `S[t-L..t]`** (1,350 features for L=30) —
   **"the one that matters."** This baseline has access to the exact same
   raw information the world model does (the full 15-minute history) but
   *no learned transition/dynamics model* — just a flat, linear function
   of all 1,350 numbers. If the world model cannot beat this, the entire
   argument for building a recursive dynamics model (rather than just
   throwing a classifier at the flattened history) falls apart. This is
   the real test.
3. **Persistence** (`S_hat[t+k] = S_t` for every k) — scored using the
   **exact same frozen risk head** the world model uses. This isolates one
   specific question: does the *learned transition* contribute anything
   beyond "assume nothing changes"? Any gap between this and the world
   model's score is attributable to the transition model alone, not to a
   difference in classifier quality.
4. **Oracle** (frozen heads applied to the *true* future state, which a
   real deployment could never have access to) — the theoretical upper
   bound. If your model beat this, something is leaking.

### 6.2 State forecast nRMSE — the metric only a world model can report

`state_nrmse` measures, per feature and per horizon, how far the *entire
simulated 45-dimensional future* diverged from what actually happened,
normalized by that feature's typical spread (so a feature that's naturally
"loud," like `bytes_total`, doesn't automatically dominate the error just
because its raw numbers are bigger). **A classifier cannot report this at
all** — it never emits a predicted state, only a label — which is exactly
why this metric is emphasized as proof that a world model, not just a
better classifier, was actually built.

**A real, instructive bug lives in this project's history here**: an
earlier version normalized by the standard deviation of whatever small
evaluation batch happened to be sampled, rather than a stable,
population-level statistic. Since many of the 45 features are structurally
near-constant across large slices of real traffic, a small stratified
batch's own std collapsed toward zero for those features, and dividing by
a near-zero number inflated the reported nRMSE by orders of magnitude
(154,775 in one early run) — a metric-computation bug that looked, at a
glance, like a catastrophic modeling failure. It was fixed by normalizing
against `FeatureScaler.reference_std_`, a per-feature standard deviation
computed once over the entire training population. Post-fix, values became
legible single digits. **The lesson generalizes**: any time a per-feature
or per-unit normalization constant is recomputed from a small sample
rather than a large, stable reference population, near-constant features
are a landmine.

### 6.3 Lead time — the headline metric

Defined exactly once (`nidra/eval/metrics.py::lead_time_for_episode`): for
each real attack episode, walk the model's continuously-produced risk
curve backward from the attack's true onset, and find the **earliest
sustained crossing** of the risk threshold (0.75) — sustained meaning `m=2`
consecutive windows above threshold, not a single noisy spike. The gap
between that crossing and the true onset, in seconds, is the lead time for
that episode.

Two disciplined reporting rules, both explicit in code comments and both
worth being able to defend: **always report the median and the full
distribution, never the maximum** (one lucky episode with a huge lead time
is not a result — it's cherry-picking), and **always report the fraction
of episodes with no warning at all** (`fraction_no_warning`) alongside any
lead-time number that did get computed — because a median computed only
over the episodes that *did* get warned silently hides the recall problem
if most episodes got no warning whatsoever.

### 6.4 Ablations — the falsification suite

Each of these (`nidra/eval/ablations.py`) is a real experiment with a real,
pre-specified possible failure mode, spelled out in the original spec's
Rule 3: *"If persistence matches the model, we report that. A clearly
reported negative result survives judging. A positive result that
collapses under one question does not."*

- **Persistence ablation**: replace the transition with `mu=0`
  (copy-forward) and compare AUC-PR against the full world model, same
  frozen head. Expected: substantial collapse. If it doesn't collapse, the
  learned dynamics aren't adding value — a real possible outcome, reported
  as-is either way.
- **Time-shuffle ablation**: randomly permute the order of the 30 input
  windows before encoding. Expected: performance collapses, because a
  shuffled sequence destroys any real trend the encoder could read.
  If it does *not* collapse, that's evidence the model may be using
  per-window snapshot features only, not genuine temporal structure — and
  the project's "this is a sequence model that reads trends" claim would
  be false.
- **Horizon curve**: AUC-PR and state-nRMSE plotted against `k=1..6`.
  Smooth degradation as `k` increases is expected (predicting further
  ahead is harder). A perfectly **flat** curve is explicitly flagged as a
  *leakage red flag*, not a good result — if forecasting 3 minutes out is
  exactly as easy as forecasting 30 seconds out, something is probably
  leaking future information into the input.
- **Surprise signal**: compares one-step forecast error between benign
  windows and "pre-attack" windows (the origin window is still benign, but
  an attack occurs somewhere in its forecast horizon). A rise in forecast
  error *before* the attack is visible in labels is a positive secondary
  signal — a kind of "the model is surprised by what's coming" — that a
  plain classifier, which never produces a forecast error at all, cannot
  produce even in principle.

### 6.5 Calibration — Brier score and reliability diagrams

`nidra/eval/calibration.py` measures, per horizon, whether a predicted
probability of (say) 0.7 actually corresponds to attacks happening about
70% of the time among windows the model scored around 0.7 (a "reliability
diagram" bins predictions and compares predicted vs. observed frequency;
the Brier score is a single summary number for the same idea — mean
squared error between predicted probability and the true 0/1 outcome).
This is a fundamentally different question from AUC-PR: AUC-PR only asks
"does the model rank true positives above true negatives" (a purely
*relative* property); calibration asks "are the *absolute magnitudes* of
the predicted probabilities trustworthy." A model can have excellent AUC-PR
and terrible calibration at the same time — which is exactly what this
project found (see Part 9/10).

---

## Part 7 — Explainability: three distinct mechanisms, on purpose

The original spec is emphatic that these three questions must use three
*different* mechanisms and never be conflated — "the most common weakness
in submissions of this type." NIDRA implements all three, plus two more:

**(a) "Why is the current state risky?"** — `nidra/explain/shap_runner.py::explain_current_risk`.
**KernelSHAP** (a model-agnostic technique that estimates each input
feature's contribution to a specific prediction by observing how the
prediction changes as features are masked/replaced, drawing on cooperative
game theory — specifically Shapley values) applied to the frozen risk head,
scored against the real *observed* state. The background/reference set
used for masking is **100 k-means centroids of benign training states**,
never a random sample of raw rows — the module docstring explains why:
random backgrounds make KernelSHAP both slow (needs more perturbation
samples to get a stable estimate) and noisy (each random background row
adds its own idiosyncratic variance).

**(b) "Why is *this future* predicted?"** — `nidra/explain/saliency.py::temporal_saliency`.
**Input-gradient saliency** (via Captum): take the gradient of a specific
predicted-future-feature with respect to every window in the input
history, and sum the absolute gradient magnitude per window. This answers
"which of the last 30 windows most influenced this particular forecast" —
e.g., "the forecast is driven by the change between windows t-3 and t-1."
This is a mechanistically different question from (a): SHAP explains a
*classification decision* over feature *values*; saliency explains a
*forecast* over input *time steps*. Conflating them (e.g., running SHAP
over time windows instead of features, or vice versa) would answer neither
question correctly.

**(c) "Why *this stage*?"** — SHAP again, but now applied to the **stage
head**, scored against a *predicted* (rolled-out) future state rather than
an observed one. This is only expressible at all because the transition
model decodes all the way down to the same 45 named features the head
understands (§4.3) — there's no opaque intermediate representation in the
way.

**(d) Model-internal counterfactual** — `nidra/explain/counterfactual.py`.
Clamp one named feature to a fixed value at every step of the rollout, and
re-simulate. This answers "what would *this model* predict if this feature
were held at this value" — explicitly and repeatedly labeled
`"model-internal what-if"` everywhere it surfaces in code, output, or
documentation, **never** "causal," "intervention," or any language implying
a claim about the real network. See Part 12 for why this distinction is
non-negotiable.

**(e) Flow bridge** — `nidra/explain/flow_bridge.py`. A per-host state
model, by construction, cannot point to individual network connections —
it only ever sees aggregated 30-second summaries. But "flagged flows" was
a stated deliverable. The bridge is a **lookup table, never a second
model**: a registry (`FEATURE_TO_FLOW_PREDICATE`) mapping specific feature
names (e.g. `dst_port_entropy`, `new_peer_count`, `retrans_rate`) to a
predicate function that filters the *raw* flow table for the flows that
plausibly drove that feature's value in that window (e.g., "`new_peer_count`
rising" → the specific flows to peers never contacted before). This
resolves a SHAP-flagged feature back into something an analyst can actually
click into.

**A sixth, unprompted addition — behavioral regimes**
(`nidra/explain/regimes.py`): unsupervised k-means clustering over the
encoder's latent hidden state `h_t`, structurally prevented from ever
seeing attack labels (`discover_regimes`'s function signature has no label
parameter at all — enforced by a dedicated test,
`test_discover_regimes_never_sees_labels`). Labels are used only
*afterward*, to describe what a discovered cluster tends to correspond to
("regime 3 turned out to be 91% pre-attack windows") — interpretation, not
training signal. This produced a genuine, real, unprompted positive
finding during this project's evaluation (see Part 9): the latent space
organizes itself around risk-relevant structure on its own, without ever
being told to.

---

## Part 8 — Serving

`nidra/serve/predictor.py::NidraPredictor` is, by design, **the only
class the backend is allowed to import.** Everything else in `nidra/` —
data pipeline, model internals, explainability machinery — is invisible to
the backend. This is a standard and important architectural boundary: the
backend team can build against three methods (`forecast`, `counterfactual`,
`explain`) and a fixed schema, without ever needing to know anything about
GRUs, rollout, or SHAP internals. If the ML implementation changes
completely internally, as long as `NidraPredictor`'s public contract stays
the same, nothing on the backend side has to change.

Loaded once, at process start (`__init__` loads the scaler and every
ensemble member's weights) — never per-request. `forecast()` takes the
caller's own `[L, F]` window buffer as an argument rather than holding one
internally: the predictor is stateless with respect to per-host history,
so any inference worker can serve a request for any host without needing
sticky sessions or shared state — sequence management belongs to the
backend/Redis layer, not here.

Every input is validated before it touches the model
(`_validate_and_scale`): wrong shape, wrong feature count, or any NaN/Inf
value raises immediately rather than silently producing a wrong (or
NaN-poisoned) forecast three steps into the rollout, where it would be far
harder to trace back to the actual input problem.

---

## Part 9 — What was actually measured (see `REAL_DATA_RESULTS.md` for full detail and provenance)

This section summarizes; treat `REAL_DATA_RESULTS.md` as the source of
truth for exact numbers, since they get updated as new runs happen. **This
section reflects the current best-supported results (Run 4 and the Run 3
pooled-ensemble addendum, full-scale `config/default.yaml`/
`config/default_logvar15.yaml`, 5-seed ensembles)** — the paragraphs below
were originally written against the earlier MVP-scale run (§Run 2) and
have been updated in place where the full-scale run changed the finding;
see Part 10's addendum for the calibration/rollout-noise story
specifically.

**Supported by real measurement**: the full pipeline runs end to end on
the complete real CIC-IDS2017 dataset (all 8 day-files, all 5 PCAPs
extracted); the world model beats both the "the one that matters"
flattened-history LR baseline and the persistence baseline on AUC-PR on
**both** the test split (0.92 pooled-ensemble AUC-PR vs. 0.67 persistence)
**and the Infiltration holdout split** (0.73 vs. 0.59) — at MVP scale the
holdout edge was not measurable at all, and it appearing cleanly at full
scale is the single strongest piece of evidence in the project, since
Infiltration is never trained on; state-forecast nRMSE is computable,
legible, and now favors the world model over persistence on both splits
(reversed from MVP scale); the 5-seed ensemble trains, converges more
stably with more data, and loads/serves correctly from a clean process
within the 300ms latency budget; unsupervised latent-space clustering
separates cleanly into high-risk/low-risk regimes without ever seeing
labels — a genuine positive, unprompted finding.

**Reported honestly as unresolved, not hidden or tuned away**: the
time-shuffle ablation is still inconsistent between splits at full scale
(large collapse on holdout, smaller on test) — the MVP-scale
"no collapse at all on test" finding resolved, but the split-asymmetry
itself did not. An odd/even horizon-parity oscillation is still present
in the per-horizon AUC-PR curve — it has **relocated** between runs
(MVP-scale test → full-scale holdout, with its phase flipped), which is
itself evidence against "just needs more data" as the explanation, since
an undertraining artifact should shrink rather than move. Head-training
validation loss still does not converge cleanly across any of the 5 seeds,
at either scale. A new item surfaced at full scale: the world model's
pooled-ensemble AUC-PR on the test split (0.92) slightly *exceeds* the
theoretical oracle ceiling (0.90) — flagged rather than hidden, most
likely small-sample AUC-PR estimation noise since it does not appear on
the holdout split, where the oracle correctly stays on top.

**The calibration gap — real, but now measurably narrower than first
diagnosed, and the model's rollout-noise mechanism has since been
independently validated and improved by retraining**: despite good AUC-PR
ranking, the model's probability outputs still cross the mandated 0.75
decision threshold less often than would be ideal — this remains the
clearest next-step item — but it is no longer purely a diagnosed-but-
unaddressed problem: see Part 10's addendum below for what changed.

---

## Part 10 — The calibration investigation (this session)

### 10.1 The symptom

Test-split evaluation showed AUC-PR = 0.803 (beating persistence's 0.694 —
a real win) but recall at threshold=0.75 of only 0.043, and 0 of 10 test-split
attack episodes ever received a sustained lead-time warning. A model that
ranks risk well but almost never crosses its own decision threshold is a
specific, diagnosable pattern, not a vague "the model is bad."

### 10.2 Root-causing it, with real instrumentation against the trained model

Two mechanisms were found and measured directly, by instrumenting
`WorldModel.rollout`/`score_states` against the actual trained seed-0
checkpoint on real test-split data — not guessed at:

**Mechanism 1 — individual rollout-trajectory scores are near-binary, and
averaging them is a real, honest measurement of disagreement, not an
artifact.** Each of the ~100-500 pooled trajectories per forecast gets its
own `sigmoid(risk_head(state))` score. Measured directly: these
individual scores are overwhelmingly close to 0 or 1 (only 12-18% land in
the ambiguous 0.25-0.75 band; the standard deviation across trajectories
for a given sample averages ~0.40-0.47, close to the theoretical maximum
of 0.5 for a bounded probability). Because of this near-binary behavior,
the reported `risk_mean_k` (the arithmetic mean of these trajectory scores)
behaves almost exactly like "what fraction of imagined futures does the
head call risky" (measured correlation with that literal statistic:
0.94-0.97). For genuine future-attack windows, that fraction averaged only
about 43-48% — even though 98-100% of those same windows had *at least
one* of the ~100 sampled trajectories individually cross 0.75. In plain
language: the model correctly recognizes that a meaningful subset of
plausible futures look dangerous, but a comparable-sized subset don't, and
honestly averaging that real disagreement lands well under the 0.75 bar.

**Mechanism 2 — the stochastic rollout noise itself measurably erodes
separation as the horizon grows.** Comparing the real (noisy) stochastic
rollout against a noise-free, deterministic (mu-only) version of the exact
same rollout on the same inputs: the negative-class (benign) mean score
nearly *tripled* by horizon 5 under the noisy rollout (0.119 → 0.356)
while staying essentially flat without noise, and the positive-class mean
dropped by roughly 0.08-0.10. In other words, the transition model's
own injected process noise — the same `exp(logvar)`-scaled Gaussian noise
discussed in §2.5/§4.2 — pushes some benign trajectories' simulated futures
into risk-head territory, and this effect compounds with horizon depth.

**A third, smaller contributing factor**: even the noise-free ceiling for
true positives only reached about 0.51-0.58, still under 0.75 — indicating
part of the gap is a genuine class-separability limit of the risk head on
this particular feature representation, which the rollout noise then
compounds rather than solely causes.

### 10.3 The attempted fix: post-hoc Platt-scaling calibration — implemented, tested, and found NOT to help

**What it is**: `nidra/eval/calibrate.py` fits a 2-parameter logistic
remap, `sigmoid(a * logit(p) + b)`, on the validation split's pooled-
ensemble forecast probabilities against real future-attack labels — the
standard technique known as **Platt scaling**, applied per forecast
horizon (since the calibration behavior differs materially across k, per
the horizon-parity finding above). It is saved once, to
`risk_calibration.json` next to the trained weights
(`nidra/scripts/fit_calibration.py`), and can be consumed by both the
evaluation harness (`run_eval.py`, always, for comparison) and the serving
predictor (`NidraPredictor`, only if `apply_calibration=True` is passed
explicitly — see why below).

**Why this does not violate Rule 1 ("heads trained only on observed states,
then frozen")**: the frozen risk head's *weights* are never touched —
`requires_grad=False` remains set, nothing calls `.backward()` anywhere
near it after Stage 2 completes. Platt scaling operates entirely *outside*
that boundary, as a fixed post-processing function applied to the head's
already-computed output probability, fit using labeled validation data the
same way a decision threshold itself would be chosen.

**A precise, corrected note on the ranking-invariance claim**: within one
fixed horizon k, this transform is strictly monotonic (any `a > 0`) and
therefore provably cannot change the ranking, so per-horizon AUC-PR
(`ablations.horizon_curve`) and per-horizon Brier/reliability
(`calibration.py`) are computed on a like-for-like footing before and
after. **This guarantee does NOT extend to `risk_over_horizon` (`max` over
k of the per-horizon-calibrated scores)** — since each horizon gets its own
(a_k, b_k), two samples whose scores peak at different horizons can have
their relative order after the max-reduction changed. Measured directly:
`world_model` vs `world_model_calibrated` AUC-PR on the test split differed
by about 0.012 (0.7935 → 0.7815) — small, but real, and this project's own
docs originally overstated this as an exact invariance before the
measurement was actually run. Corrected here and in `nidra/eval/calibrate.py`'s
docstring.

**The actual empirical result — a genuine negative finding, not hidden**:
fit against the real 5-seed ensemble (validation split, n=4,000, all 6
horizons fit non-degenerately, `a` in the 1.8–3.6 range — i.e. the fit did
find it should *sharpen*, not flatten, the score), then measured two ways:

1. A single-seed spot-check via `run_eval.py`'s `world_model_calibrated`
   baseline row (test split, n=4,000): recall at threshold=0.75 fell from
   0.043 (raw) to 0.010 (calibrated) — **worse, not better.**
2. A direct check against the actual serving path — the pooled 5-seed
   ensemble, exactly as `NidraPredictor` computes it — on 500 stratified
   test-split windows: recall at threshold=0.75 dropped to **0.000 at
   every one of the 6 horizons** (from a raw baseline of 0.124, 0.036,
   0.017, 0.000, 0.005, 0.000 across k=0..5). This rules out "it's just a
   single-seed artifact" — the same degradation appears in the exact path
   real forecasts would take.

**Root cause of why the "fix" backfired, diagnosed rather than left as a
mystery**: `fit_platt` uses a plain (unweighted) logistic regression. On
this dataset, true-positive windows are a small minority of the validation
set. An unweighted fit's optimum for overall log-loss is dominated by the
huge volume of easy true negatives — the fit correctly learns that, in
aggregate, **even the highest raw scores in this dataset rarely correspond
to a genuine ≥75% true-positive rate** (the per-bin reliability numbers
back this up directly: the 0.75–0.85 raw-score bin's observed frequency
was 0.583, the 0.85–0.95 bin's was 0.308 — both *below* their own bin
center, i.e. mildly overconfident already at the very top, on small,
noisy sample counts of 13 and 10). A correctly-weighted, base-rate-
respecting calibration doesn't invent confidence that isn't there — it
reports the true confidence honestly, and the honest answer for this model
on this data is that a genuine 75%-plus true-positive rate essentially
never occurs. **This is a more informative, more rigorous result than "we
fixed it" would have been**: it shows the 0.75 threshold, for this specific
risk head's actual discriminative ceiling on this dataset, is not merely
hitting a fixable scaling artifact — it may be asking for a confidence
level this classifier's real signal rarely reaches. Reweighting the Platt
fit (`class_weight="balanced"`, exactly as `nidra/eval/baselines.py`'s own
`fit_logistic_regression` already does for the LR baselines) would very
likely flip these numbers to "look better" — and was deliberately NOT done,
because it would silently manufacture inflated confidence for the specific
purpose of clearing a fixed decision threshold, which is exactly the kind
of "tuning to make the numbers look better" this project's own rules (and
`EVALUATION.md`'s explicit statement about the 0.75/m=2 constants)
prohibit.

**What was done about it, concretely**: `NidraPredictor.__init__` gained an
`apply_calibration: bool = False` parameter — **off by default**. The
calibration file is still generated and still loaded/inspectable, and
`run_eval.py` still always computes and reports the calibrated comparison
(that comparison is precisely the evidence above) — but a real deployment
of this predictor, using the default constructor call, is **not** silently
handed a version of the model with measurably worse recall. This is a
"prefer minimal, reversible changes" response to a finding discovered
mid-project, not a redesign: the code path, tests, and documentation for
calibration all still exist and are correct on their own terms — they're
just not switched on by default, because switching them on by default
would make the shipped system worse at its actual job, and that fact was
only discoverable by actually running the experiment rather than trusting
the a-priori theory.

**What this means for the calibration gap going forward**: it's still the
single most consequential open item (§10.1 stands), but this session's
work narrows *what kind* of fix is needed. It rules out "the raw score is
just uniformly compressed and a monotonic remap fixes it" as too simple a
story. It's consistent with a genuine capacity/separability limitation at
the very top of the risk head's confidence range (few samples, already
close to its true ceiling there) compounded by the rollout-noise erosion
described in Mechanism 2 (§10.2) — meaning the more promising next step is
likely the `logvar_max` reduction + retrain (documented as a recommendation,
not yet executed, in `MODEL_CARD.md`), not a better post-hoc remap of an
already-frozen head's output.

### 10.4 Addendum (later session, full production scale): both open threads from §10.3 were followed up — one reversed, one confirmed

Everything above (§10.1-10.3) was diagnosed at MVP scale against a
single-epoch-capped checkpoint. Two follow-up sessions, working against
the full-scale `config/default.yaml` production checkpoint (5-seed
ensemble, 500k/50k samples), closed out both open threads §10.3 left
hanging. Neither required touching the frozen risk head's weights — the
Rule 1 boundary from §10.3 still holds throughout.

**Thread 1 — calibration's direction reversed.** §10.3's MVP-scale finding
was unambiguous: Platt-scaling calibration made recall *worse*, at every
horizon, on both the single-seed spot-check and the real pooled-ensemble
serving path. Re-running that same pooled-ensemble verification (500
stratified samples, `n_samples_per_member=50`, exactly the rigor §10.3
originally used) against the full-scale checkpoint found the opposite
direction: recall at threshold=0.75 improved on both splits (test
~1.0%→2.4%, holdout ~0.7%→9.1%), with precision staying at 1.000
throughout — zero new false positives introduced. **The magnitude is far
more modest than a single-seed approximation initially suggested** (that
approximation showed "9 of 10 test episodes warned," which did not survive
pooled-ensemble verification and was caught before being reported as a
headline number — see `REAL_DATA_RESULTS.md`'s Run 3 calibration section
for the full two-look comparison). The most likely explanation for the
reversal itself: §10.3's root-cause diagnosis (an unweighted Platt fit
correctly learning that even high raw scores rarely correspond to a
genuine ≥75% true-positive rate on this dataset) was a property of *that*
checkpoint's actual score distribution, and more training data changed
that distribution enough to shift the fit's effect from harmful to mildly
helpful — consistent with, not contradicting, §10.3's root-cause reasoning.
The underlying miscalibration problem the 0.75 threshold surfaces did not
go away; its direction did.

**Thread 2 — the `logvar_max` reduction §10.3 recommended but had not yet
run was executed, and it worked.** A full 5-seed retrain with
`model.transition.logvar_max=1.5` (down from 3.0, `config/
default_logvar15.yaml`, otherwise identical to `config/default.yaml`)
directly tests Mechanism 2 from §10.2 (rollout noise eroding separation
with horizon depth). At matched evaluation settings against the
`logvar_max=3.0` checkpoint: world-model AUC-PR improved on both splits
(test 0.878→0.918, holdout 0.700→0.719), and the horizon-curve ablation
improved at **all 6 of 6 horizon steps on both splits**, with the single
largest gain at the deepest horizon tested (test k=5: 0.266→0.376) —
exactly where §10.2's Mechanism 2 predicted the biggest effect, since
injected process noise compounds with each additional rollout step. Every
baseline that does not depend on rollout (oracle, persistence, both LR
baselines) was byte-identical between the two checkpoints, isolating the
improvement specifically to the variance-clamp change rather than some
other difference between the two training runs. No NaN instability was
observed at the lower clamp. This is now a validated result, not a
diagnosed-but-untested hypothesis — see `REAL_DATA_RESULTS.md`'s "Run 4"
section and `MODEL_CARD.md` limitation 7 for the full numbers.

**Net effect on the calibration gap (§10.1)**: still open, still the
clearest next-step item, but measurably less severe on two independent
fronts than it looked after §10.3 — calibration no longer actively hurts
(and mildly helps), and the rollout-noise mechanism §10.2 identified as a
likely contributor has been directly targeted and confirmed improved by
retraining, not just theorized about.

---

## Part 11 — Design decision log (Q&A format)

Use this section when someone asks "why did you do X and not Y."

**Q: Why forecast at all — why not just build a better classifier?**
A: A classifier can only ever tell you about *now*. It cannot report a
lead time, cannot simulate a future trajectory, and cannot produce a state-
forecast error. The problem statement explicitly asks for forecasting; a
world model is the standard, principled architecture for that class of
problem.

**Q: Why per-host aggregation instead of per-flow, when CIC-IDS2017 is
normally used per-flow?**
A: Compromise is a process that plays out on a machine over time. A single
flow shows one brief connection; only a host's trajectory across many
windows reveals escalation. See §1.3.

**Q: Why 45 features specifically, and why these five groups?**
A: Chosen to cover five distinct signal families attackers' traffic
patterns tend to disturb: raw volume/flag statistics (flow aggregates),
low-level transport fingerprints (packet aggregates), who's-talking-to-
whom structure (graph scalars), rate-of-change (dynamics), and whether the
host was even active (activity). No single group would catch every attack
type in the dataset; a portscan shows up mainly in graph/entropy features,
a DDoS mainly in volume/flag features, credential brute-forcing mainly in
flow-timing features.

**Q: Why RobustScaler and not StandardScaler or min-max scaling?**
A: Network traffic is heavy-tailed; a single extreme window (e.g. a DDoS
burst) would badly distort a mean/std-based fit. RobustScaler's use of
median/IQR is far less sensitive to that kind of outlier. See §3.4.

**Q: Why recursive rollout instead of direct multi-horizon regression?**
A: Direct regression can't guarantee the 6 horizon predictions form one
coherent trajectory, and can't support genuine trajectory-level uncertainty
quantification (running the same deterministic head 6 times gives you 6
independent confidence intervals, not a distribution over *whole futures*).
See §2.3 for the full trade-off, paid honestly (compounding error).

**Q: Why not let the risk head see rolled-out states during training, if
it would improve the numbers?**
A: This is Rule 1, the single most protected design decision in the
project. Doing so creates an unfalsifiable feedback loop where a head can
learn to compensate for the dynamics model's specific quirks rather than
learning a genuine state→risk mapping, destroying the ability to
distinguish "the simulated future is realistic" from "the classifier
learned to rationalize whatever the simulator produces." See §2.6.

**Q: Why hold out Thursday/Infiltration entirely instead of using it for
more training data?**
A: It's the project's generalization claim — the one test that asks "does
this work on an attack type never seen in any form during training,"
rather than "does this work on more examples of attack types already
seen." Diluting it into training would remove the only experiment that
answers that specific, harder question. See §3.1.

**Q: Why does the reported latent size (128) not match the spec's ~32-dim
design target?**
A: The spec's "~32-dim latent" implies a dimensionality-reduction
bottleneck layer between the GRU and the transition/heads. No such layer
exists in the actual, tested, working implementation — the GRU's 128-dim
hidden state is used directly. Retrofitting a real 32-dim bottleneck now
would require retraining the entire pipeline and risks destabilizing
already-validated rollout behavior. This is documented as an honest
design-target-vs-implementation gap in `MODEL_CARD.md`, not silently
resolved by writing "32" into metadata that corresponds to no real tensor.
Never claim the implemented latent size is 32-dimensional if asked — say
exactly this.

**Q: Why 5 seeds and not more/fewer?**
A: The original spec fixes this at 5 as a protected constant, chosen as a
practical balance for a CPU-only, time-boxed hackathon project — enough to
meaningfully approximate epistemic uncertainty (§5.3) without making
training time impractical.

**Q: Why ~500-1000 trajectories and not more?**
A: This is a measured, documented latency trade-off, not an arbitrary
choice. 200 samples/member (1000 total, matching the design target)
measured p95=616ms on the reference CPU — over the 300ms serving target.
Per the project's own documented policy ("cut samples toward 100 before
cutting ensemble members"), 100/member (500 total) was chosen instead:
p95=108ms. On faster hardware, this reverts to 200/member — it's a config
value, not a code change. See `TRAINING.md`.

**Q: Why label the counterfactual "model-internal what-if" instead of
"causal" or "simulation of an intervention"?**
A: Because it isn't one, and claiming otherwise is the single most likely
way to lose credibility with a technically literate reviewer. See Part 12.

**Q: Why does `neighbour_risk_fraction` use a heuristic instead of the
model's own risk score?**
A: To avoid an undebuggable circular feedback loop across the host graph.
See the callout in §3.4.

**Q: Why fit Platt-scaling calibration instead of just lowering the
decision threshold below 0.75?**
A: 0.75 is a protected constant from the original problem specification —
tuning it to make numbers look better is explicitly prohibited by the
project's claims-discipline rules (`EVALUATION.md` states this directly:
*"do not tune them to make lead-time numbers look better"*). A correctly
(base-rate-respecting) fit Platt scaling doesn't have that problem — but
see §10.3: fitting one and actually measuring the result found it makes
recall at 0.75 *worse*, not better, which is itself an important, honestly
reported finding, not the fix it was hypothesized to be. It's kept in the
codebase (tested, off by default in serving) as correct, useful
infrastructure and as documented evidence for what doesn't work — not as a
shipped improvement.

**Q: If the calibration fix didn't work, why not just make the Platt fit
class-weighted (`class_weight="balanced"`) so it clears the 0.75 bar,
matching how the LR baselines already handle imbalance?**
A: Because that would manufacture confidence that isn't really there,
specifically to clear a fixed decision threshold — which is the same
"tuning to make the numbers look better" the previous answer just ruled
out, just moved one level down into the calibration fit instead of the
threshold itself. An honestly-weighted calibration reporting "this
classifier's real ceiling on this data rarely reaches 75% confidence" is a
true, useful finding. An artificially reweighted one reporting "75%
confidence achieved" when the underlying signal doesn't support it would
not be — even though the resulting numbers would look better in a demo.
See §10.3 for the full reasoning.

---

## Part 12 — Claims discipline (read this before describing the project to anyone)

Directly from the original build spec (`docs/IMPLEMENTATION-ML.md` §10),
and enforced throughout this codebase's documentation and code comments:

**What this project legitimately has**: correlation, temporal prediction,
and learned dynamics. All three are real and non-trivial.

**What this project does NOT have, and must never claim**: causal
inference. Establishing causality requires intervention, randomization, or
identification under an explicit causal graph — none of which apply here.
CIC-IDS2017 is observational data, and it's confounded by its own
collection protocol (attacks were run at scheduled times, so
"time-of-day" correlates with "attack present" as an artifact of how the
dataset was built, not a real property of networks). Predicting `S_{t+1}`
from `S_t` is not causal — at most it's "Granger-causal" in a narrow
statistical sense, and even that assumes a stationarity the data doesn't
strictly satisfy.

**Words that are fine to use**: learned dynamics, forecast, simulated
trajectory, projected state, model-internal counterfactual/what-if.

**Words to never use**: "causal simulator," "the model understands why,"
"predicts attacker intent," "guaranteed," or any phrasing implying the
counterfactual module (§7d) is a real intervention on the actual network
rather than a question about the model's own internal beliefs.

One more scoping statement worth being able to say confidently if asked
"is this AGI-style world model / a full simulator": no. This project sits
at "a latent-state transition model with (attempted) calibrated multi-step
rollout" — architecturally a world model, operationally a forecaster. It
does not claim to be a full generative, action-conditioned world model
(the kind used in some game-playing/robotics research) — that would
require an action space and interventional data that a public, passively-
collected dataset like CIC-IDS2017 cannot provide. Claiming more than this
invites a question the project cannot answer, and being caught overclaiming
costs far more credibility than scoping honestly ever loses.

---

## Part 13 — Glossary

- **Flow**: one network connection (typically identified by source IP,
  destination IP, source port, destination port, protocol) and the
  packets exchanged during it.
- **Host**: one machine on the network, identified here by its IP address.
- **Window**: a fixed 30-second time slice.
- **State vector**: the 45 numbers summarizing one host's activity in one
  window.
- **GRU (Gated Recurrent Unit)**: a type of recurrent neural network layer
  that processes a sequence one element at a time while carrying a "memory"
  (hidden state) forward, designed to handle longer sequences better than
  a plain RNN.
- **Latent state / hidden state**: a neural network's internal, compressed
  numeric summary of everything it has seen so far — not human-readable on
  its own, but the basis every downstream prediction is computed from.
- **World model**: an architecture that learns to simulate how an
  environment evolves over time, by encoding observations into a hidden
  state and learning a transition function over that state, then
  predicting forward by feeding its own outputs back in.
- **Rollout**: the act of running a model's transition function repeatedly,
  feeding each step's output back in as the next step's input, to simulate
  a multi-step future.
- **Recursive vs. direct (multi-horizon) forecasting**: recursive =
  predict one step, feed it back, repeat; direct = predict every horizon
  independently from the same original input in one shot.
- **Teacher forcing / scheduled sampling**: during training a sequence
  model, "teacher forcing" means always feeding the true next value back
  in; "scheduled sampling" anneals from doing that toward feeding the
  model's own prediction back in instead, to prepare it for inference-time
  conditions.
- **Gaussian NLL (negative log-likelihood)**: a loss function used when a
  model predicts a full probability distribution (here, a mean and
  variance) rather than a single point estimate — it rewards both
  accuracy and appropriately-calibrated uncertainty.
- **Ensemble**: multiple independently-trained models whose outputs are
  combined (here, by pooling their sampled rollout trajectories) to
  produce a more robust prediction and a better uncertainty estimate.
- **Aleatoric vs. epistemic uncertainty**: aleatoric = irreducible
  noise/randomness in the situation itself; epistemic = uncertainty from
  the model itself not being sure (e.g. because different training runs
  disagree) — the two need different techniques to estimate (a predicted
  variance for the former, an ensemble for the latter).
- **AUC-PR (Area Under the Precision-Recall Curve)**: a threshold-free
  ranking metric — how well a model ranks true positives above true
  negatives, independent of any specific decision threshold. Preferred
  over plain AUC-ROC when the positive class (here, risky windows) is
  rare, which is the case here.
- **Calibration**: whether a model's predicted probabilities correspond to
  real-world frequencies (a prediction of 0.7 should come true about 70%
  of the time). Distinct from ranking quality (AUC-PR) — a model can rank
  perfectly while being badly miscalibrated.
- **Platt scaling**: a standard post-hoc calibration technique — fitting a
  1-D logistic regression (2 parameters) on a model's own output score
  against true labels, to recalibrate its probability magnitude without
  retraining the underlying model.
- **Persistence baseline**: "assume nothing changes" — the simplest
  possible forecast, used as a sanity floor any learned dynamics model
  must beat to be worth its complexity.
- **SHAP / Shapley values**: a model-agnostic explainability technique,
  based on cooperative game theory, that attributes a prediction to its
  input features by measuring how the prediction changes as features are
  masked out, in a way that fairly distributes "credit" across features
  that interact with each other.
- **Saliency (input-gradient)**: an explainability technique that uses the
  gradient of an output with respect to the input to estimate which input
  elements most influenced that output.
- **k-means clustering**: an unsupervised algorithm that groups data points
  into a fixed number of clusters based on similarity, with no access to
  labels.
- **Leakage**: any situation where information that wouldn't be available
  at real prediction time (most commonly, future information, or
  statistics computed across a train/test boundary) accidentally
  influences a model's input, training, or evaluation — the single most
  common way ML security research projects produce results that look great
  but are meaningless.
- **Lead time**: how far in advance (in seconds/minutes) a forecasting
  system warned about an event before it actually happened.
- **Episode**: one continuous run of non-benign activity for one host, from
  its first attack-labeled window to its last, before returning to benign.

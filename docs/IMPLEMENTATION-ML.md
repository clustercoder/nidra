# IMPLEMENTATION-ML.md

**HORIZON — Predictive Network World Model**
Problem Statement 26153 · NTRO · SIH

This document is the build spec for the machine learning half of HORIZON. It assumes you have read the PRD. Where the PRD says *what* and *why*, this says *how*, in the order you should build it.

---

## 0. Read this first

Three rules that govern every decision below. If you find yourself violating one, stop and reconsider rather than working around it.

**Rule 1 — The heads are frozen.** The risk head and stage head are trained on *observed* states and never see a predicted state during training. This is not a stylistic choice. It is the reason our forecasting claim is falsifiable. If you ever find yourself fine-tuning a head on rollout output "because the numbers improve," you have destroyed the central argument of the project.

**Rule 2 — Nothing after time `t` may touch the input.** Not a normalisation constant, not a centred rolling mean, not a label-derived feature. Every leak makes the results better and the project worthless.

**Rule 3 — The ablation is a real experiment with a real possible failure.** If persistence matches the model, we report that. A clearly reported negative result survives judging. A positive result that collapses under one question does not.

---

## 1. Environment

```bash
python 3.11
torch==2.4.0          # CPU is sufficient; CUDA if available
numpy pandas pyarrow
scikit-learn          # baselines, metrics, calibration
shap                  # KernelSHAP on the heads
captum                # temporal saliency
pyyaml
matplotlib            # eval plots only
tqdm
```

System dependency: `tshark` (from `wireshark-common`). Verify before anything else:

```bash
tshark -v
```

If this fails, the packet-level half of the pipeline is dead and the project does not meet the stated data requirement. Fix it on hour one.

### Repository layout

```
horizon/
  config/
    default.yaml              # single source of truth for all hyperparameters
  horizon/
    data/
      pcap_extract.py         # tshark → parquet
      flow_load.py            # CIC CSV → parquet
      join.py                 # five-tuple + timestamp join
      windowize.py            # → [host, window, features]
      graph_features.py       # per-window communication graph scalars
      schema.py               # FEATURE_ORDER — THE canonical feature list
      splits.py               # temporal / episode / holdout
    models/
      encoder.py              # GRU
      transition.py           # Gaussian transition head
      heads.py                # risk + stage
      world_model.py          # assembly + rollout
    train/
      train_dynamics.py
      train_heads.py
      losses.py
    eval/
      metrics.py              # lead time, early-warning precision, nRMSE
      baselines.py            # 4 baselines
      ablations.py            # persistence, shuffle, horizon curve
      calibration.py
    explain/
      shap_runner.py
      saliency.py
      counterfactual.py
      flow_bridge.py          # SHAP feature → contributing flows
    serve/
      predictor.py            # the only thing the backend imports
  artifacts/
    weights/  scaler/  metrics/
```

`horizon/serve/predictor.py` is the entire ML surface the backend sees. Keep it that way — one class, one `forecast()` method.

---

## 2. Data pipeline

This phase is **60% of the ML work**. Budget accordingly. The model is the easy part.

### 2.1 Dataset

**CIC-IDS2017**, Tuesday / Wednesday / Friday.

| Day | Attacks | Role |
|---|---|---|
| Monday | none | Benign baseline; normalisation stats |
| Tuesday | FTP-Patator, SSH-Patator | Train |
| Wednesday | DoS variants, Heartbleed | Train |
| Thursday | Web attacks, **Infiltration** | **Held out entirely** — generalisation claim |
| Friday | Botnet, PortScan, DDoS | Test |

Do not use CSE-CIC-IDS2018 raw PCAPs. They are hundreds of gigabytes and will consume the build window. Support 2018 CSVs as a flow-only input path and say so in the README.

### 2.2 Packet extraction — start here

Do not use PyShark or Scapy for bulk extraction. PyShark shells out to tshark per packet and is roughly two orders of magnitude too slow. Use tshark's field output directly:

```bash
tshark -r Wednesday-WorkingHours.pcap \
  -T fields -E separator=, -E quote=n \
  -e frame.time_epoch \
  -e ip.src -e ip.dst \
  -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport \
  -e ip.proto \
  -e ip.ttl \
  -e tcp.window_size_value \
  -e ip.flags.mf -e ip.frag_offset \
  -e frame.len -e tcp.len \
  -e tcp.flags \
  -e tcp.analysis.retransmission \
  -Y "ip" \
  > wed_packets.csv
```

Notes that will save you an hour each:

- `tcp.analysis.retransmission` is empty for normal packets and `1` for retransmissions. Treat empty as 0.
- `ip.flags.mf` is the more-fragments bit; combine with a non-zero `ip.frag_offset` to get a fragmentation indicator.
- UDP and TCP ports are separate fields; coalesce them in pandas.
- Output is large. Write to parquet immediately and delete the CSV.

**Bound the extraction.** You do not need the whole day. For each labelled attack episode take `[onset − 30 min, end + 10 min]`, plus two or three hours of contiguous benign traffic for baseline. Use `-Y "frame.time >= ..."` or slice with `editcap` beforehand.

### 2.3 Flow loading

The published CICFlowMeter CSVs give you the flow half directly. Two known issues:

- Column names have leading spaces (` Flow Duration`). Strip them on load.
- There are duplicate header rows mid-file in some releases. Drop rows where a numeric column fails to parse.
- Labels are in the `Label` column. Keep them; they are used only for supervision targets and evaluation.

### 2.4 Join

Join packet aggregates to flows on `(src_ip, dst_ip, src_port, dst_port, protocol)` with a timestamp tolerance. In practice you do not need a perfect join: you are aggregating both sides to `(host, window)` anyway. The simpler and more robust path is:

1. Assign every packet to `(host, window)` independently.
2. Assign every flow to `(host, window)` independently.
3. Aggregate each separately.
4. Join the two aggregate tables on `(host_id, window_ts)`.

This avoids five-tuple matching entirely and is far less brittle. Do this.

Note the `host` assignment: a flow contributes to the state of its **source** host. If you want bidirectional context, emit a second row for the destination host with a `direction` marker — but v1 should be source-only for simplicity.

### 2.5 Windowing

```
Δ (window)     = 30 s
L (context)    = 30 windows  (15 min)
K (horizon)    = 6 windows   (3 min)
```

Window boundaries are aligned to absolute epoch multiples of 30, not to the first packet — otherwise windows differ between files and nothing joins.

**Empty windows are real data.** A host that goes silent for a window has state; do not drop it. Emit a zero-valued state row with an `is_active=0` flag feature. Dropping empty windows silently breaks the time axis and corrupts every lead-time measurement downstream.

Filter hosts with fewer than `L + K` total windows.

### 2.6 Feature set

Defined once in `schema.py` as an ordered list. Every service imports it. Never construct feature order from a dict iteration.

```python
FEATURE_ORDER = [
    # --- flow aggregates (15) ---
    "syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "psh_ratio", "urg_ratio",
    "bytes_total", "bytes_up_down_ratio", "pkts_per_flow_mean",
    "flow_duration_mean", "flow_duration_var",
    "iat_mean", "iat_var", "iat_max", "active_flow_count",
    # --- packet aggregates (11) ---
    "ttl_mean", "ttl_var", "tcp_window_mean", "tcp_window_entropy",
    "frag_flag_rate", "payload_size_mean", "payload_size_var", "payload_size_p95",
    "payload_size_entropy", "retrans_count", "retrans_rate",
    # --- graph scalars (8) ---
    "out_degree", "in_degree", "dst_ip_entropy", "dst_port_entropy",
    "new_peer_count", "neighbour_risk_fraction", "local_clustering_coeff", "reciprocity",
    # --- dynamics (10) ---
    "d_syn_ratio", "d_dst_port_entropy", "d_out_degree", "d_new_peer_count",
    "d_iat_var", "d_retrans_rate",
    "slope3_syn_ratio", "slope3_dst_port_entropy", "slope3_out_degree", "slope3_iat_var",
    # --- activity (1) ---
    "is_active",
]
assert len(FEATURE_ORDER) == 45
```

**Ratio features** are computed over flows in the window: `syn_ratio = SYN-flagged packets / total packets`. Guard against divide-by-zero by returning 0 when the denominator is 0, not NaN.

**Entropy features** use Shannon entropy over the empirical distribution within the window, natural log, unnormalised. `dst_port_entropy` near 0 means one port; rising values mean spreading.

**`new_peer_count`** requires a per-host memory of previously contacted peers. Maintain a running set, per host, in window order. This is the one stateful feature — it must be computed in a single forward pass over sorted windows, never vectorised out of order.

**`neighbour_risk_fraction`** has a bootstrapping problem: it needs risk scores that the model produces. For v1, define it against a *heuristic* prior (fraction of this host's peers that had elevated fan-out in the previous window), not against model output. Using model output creates a feedback loop you cannot debug in five days.

**Delta and slope features** are strictly backward-looking. `d_x[t] = x[t] - x[t-1]`; `slope3_x[t]` is the OLS slope over `x[t-2..t]`. At sequence start, pad with zeros, never with future values.

### 2.7 Graph scalars

Per window, build a directed multigraph over hosts from the flows in that window. Compute per-node metrics. Do not persist the graph — it is a transient object used to produce eight scalars, then discarded.

`local_clustering_coeff` on a directed graph is expensive on dense windows. Cap it: if a node's degree exceeds 200, sample 200 neighbours. The value is a signal, not an exact quantity.

### 2.8 Normalisation

Fit on the **training period only**. Serialise the fitted scaler next to the weights. Load it at serving time. This is the single most common source of silent leakage in this class of project.

Use `RobustScaler` rather than `StandardScaler` — traffic features are heavy-tailed and a single DDoS window will destroy a mean/std fit. Then clip to `[-10, 10]` post-scaling to keep rollout numerically stable.

Log-transform count-like features before scaling: `bytes_total`, `active_flow_count`, `out_degree`, `new_peer_count`, `retrans_count`. Use `log1p`.

### 2.9 Labels

Two supervision targets, derived from the dataset's attack timeline.

```python
# per (host, window)
stage_label[t]  in {benign, recon, initial_access, lateral, c2, exfil}
risk_label[t]   = 1 if any attack-stage window occurs in (t, t+K], else 0
```

`risk_label` is the forward-looking target. It looks at the future *only to build the label*, never to build the input. That is legitimate supervised learning; the distinction matters and you should be able to state it clearly if asked.

**Stage mapping** from CIC-IDS2017 labels to tactics — this is a curated mapping, documented as such:

| Dataset label | Mapped tactic |
|---|---|
| PortScan | Reconnaissance |
| FTP-Patator, SSH-Patator | Initial Access (credential attack) |
| Web Attack – Brute Force / XSS / SQL Injection | Initial Access |
| Infiltration | Initial Access → Lateral Movement |
| Bot | Command & Control |
| DoS / DDoS variants | Impact |

Say in the README that this is a presentation mapping over dataset labels, not ATT&CK technique ground truth. The dataset does not support technique-level resolution and a security-literate judge will notice if you claim otherwise.

### 2.10 Splits

```python
train  = Tuesday, Wednesday  (+ Monday benign)
test   = Friday
holdout= Thursday            # Infiltration — never trained on
```

Within `train`, hold out a contiguous **time block** for validation, not a random sample. Episode-level integrity: no windows from one attack episode may appear on both sides of any split.

---

## 3. Model

### 3.1 Encoder

```python
class Encoder(nn.Module):
    def __init__(self, F=45, H=128, layers=2, dropout=0.1):
        self.gru = nn.GRU(F, H, layers, batch_first=True, dropout=dropout)

    def forward(self, x, h=None):        # x: [B, L, F]
        out, h = self.gru(x, h)
        return out[:, -1, :], h          # h_t: [B, H]
```

Returning `h` matters: rollout re-enters the GRU one step at a time and must carry hidden state forward.

### 3.2 Transition model

Emits a diagonal Gaussian over the next state.

```python
class Transition(nn.Module):
    def __init__(self, H=128, F=45):
        self.net = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.mu = nn.Linear(256, F)
        self.logvar = nn.Linear(256, F)

    def forward(self, h):
        z = self.net(h)
        return self.mu(z), self.logvar(z).clamp(-6, 3)
```

The `logvar` clamp is not cosmetic. Without it, the model discovers that predicting infinite variance on hard features minimises NLL, variance explodes, and rollout produces NaN by step 3. Clamp it.

**Predict the delta, not the absolute state:** `S_hat[t+1] = S[t] + mu`. This is a meaningful improvement — it hands the model a strong prior (states are autocorrelated) so capacity goes into modelling *change*, which is what you actually care about. It also makes the persistence baseline exactly the `mu = 0` case, which is a clean conceptual story.

### 3.3 Heads

```python
class RiskHead(nn.Module):    # F → 64 → 1, sigmoid
class StageHead(nn.Module):   # F → 64 → 6, softmax
```

Deliberately small and operating on the **feature vector**, not the hidden state. Two reasons: they must be applicable to a decoded predicted state, and small models make KernelSHAP tractable.

### 3.4 Rollout

```python
def rollout(self, x, K=6, n_samples=200, stochastic=True):
    """x: [B, L, F] observed. Returns predicted states + head outputs.
       Consumes NO data after the last window of x."""
    h_t, h = self.encoder(x)
    cur = x[:, -1, :]
    traj = []
    for k in range(K):
        mu, logvar = self.transition(h_t)
        nxt = cur + mu
        if stochastic:
            nxt = nxt + torch.randn_like(mu) * (0.5 * logvar).exp()
        nxt = nxt.clamp(-10, 10)
        traj.append(nxt)
        h_t, h = self.encoder(nxt.unsqueeze(1), h)   # feed prediction back
        cur = nxt
    return torch.stack(traj, dim=1)                  # [B, K, F]
```

The line `h_t, h = self.encoder(nxt.unsqueeze(1), h)` is the world model. Everything else is plumbing. It feeds the model's own output back as if it were an observation — which is precisely what "forward simulation" means and precisely what a classifier cannot do.

For a full forecast: run this `n_samples` times per ensemble member, apply the frozen heads to every sampled trajectory, and take quantiles across samples for the confidence band.

---

## 4. Training

Two stages, in this order. Do not merge them.

### Stage 1 — dynamics

Train encoder + transition on next-state prediction with a **multi-step unrolled loss**.

```python
def dynamics_loss(model, x, y_future, K=6):
    """y_future: [B, K, F] ground-truth next K states."""
    h_t, h = model.encoder(x)
    cur, total = x[:, -1, :], 0.0
    for k in range(K):
        mu, logvar = model.transition(h_t)
        pred = cur + mu
        total += gaussian_nll(pred, logvar, y_future[:, k]) * (0.85 ** k)
        # scheduled sampling: feed truth early in training, own prediction later
        nxt = y_future[:, k] if random.random() < teacher_forcing_p else pred.detach()
        h_t, h = model.encoder(nxt.unsqueeze(1), h)
        cur = nxt
    return total / K
```

The `0.85 ** k` discount stops distant, inherently-uncertain horizons from dominating the gradient.

**Scheduled sampling schedule:** `teacher_forcing_p` from 1.0 → 0.3 linearly over the first 60% of epochs. This is the difference between a model that looks fine at k=1 and diverges at k=3, and one that holds to k=6.

```yaml
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
batch_size: 256
epochs: 60
grad_clip: 1.0
scheduler: cosine
early_stopping: val multi-step NLL, patience 8
```

**Class imbalance:** attack windows are a small minority. For dynamics training this is fine — you *want* the model to learn benign dynamics well, since that is what makes anomalous transitions surprising. Do not rebalance stage 1.

### Stage 2 — heads

Freeze encoder and transition. Train the risk and stage heads on **observed states only**.

```python
for p in model.encoder.parameters():    p.requires_grad = False
for p in model.transition.parameters(): p.requires_grad = False
```

Here you *do* rebalance: `pos_weight` in `BCEWithLogitsLoss` set to `n_neg / n_pos`, and class weights in the stage cross-entropy.

Then freeze the heads too. Nothing is trained after this point.

### Ensemble

Five seeds, identical config, `[0, 1, 2, 3, 4]`. Train all five. Trains in minutes on CPU; there is no reason not to. Report the ensemble everywhere.

### Config

Everything above lives in `config/default.yaml`. Nothing is hardcoded in a script. A named deliverable is "reproducible training configuration" — you get that for free if you are disciplined from the start, and it is painful to retrofit.

---

## 5. Evaluation

### 5.1 Baselines — all four

| # | Baseline | Purpose |
|---|---|---|
| 1 | `LogisticRegression` on `S_t` → risk in `(t, t+K]` | **Mandated by the problem statement.** Headline table row. |
| 2 | `LogisticRegression` on flattened `S[t-L..t]` (45×30 = 1350 features) | **The one that matters.** History, no dynamics. Beat this or the project has no result. |
| 3 | Persistence: `S_hat[t+k] = S[t]` | State-forecast baseline and the core ablation. |
| 4 | Oracle: risk head on the *true* `S[t+k]` | Upper bound. You should land between #2 and #4. |

If you cannot beat #2, do not hide it. Diagnose: usually window size too large, context too short, or the state is missing the discriminative features. Report the honest gap and what you tried.

### 5.2 Forecast lead time

The headline metric. Define it precisely and implement it exactly once.

```python
def lead_time(risk_curve, onset_ts, threshold=0.75, m=2, delta=30):
    """Earliest sustained threshold crossing before onset.
       m consecutive windows above threshold — this is what stops a single
       noisy spike from counting as a prediction."""
    run = 0
    for i, (ts, p) in enumerate(risk_curve):
        if ts >= onset_ts: break
        run = run + 1 if p >= threshold else 0
        if run >= m:
            return (onset_ts - risk_curve[i - m + 1][0]).total_seconds()
    return None   # no warning
```

Report **median and the full distribution**. Never the maximum — a single lucky episode is not a result. Report the fraction of episodes with no warning at all; that is the recall side of the same coin.

### 5.3 Other metrics

```
Standard:  F1, precision, recall, AUC-PR, FPR
           FPR also as alerts/hour/host — the number a SOC actually cares about

Forecast:  early_warning_precision(k)        curve over k=1..6
           detection_before_stage_completion  fraction of episodes
           stage_accuracy(k)                  top-1 and top-2
           state_nrmse(k, feature)            per-feature, per-horizon
           brier(k) + reliability diagram     calibration
```

`state_nrmse` is the metric that proves you built a world model. A classifier cannot report it because it never emits a state. Put it in the deck.

### 5.4 Ablations — the falsification suite

| Ablation | Implementation | Expected |
|---|---|---|
| **Persistence** | Replace transition with `mu = 0` | Lead time and AUC collapse |
| **Time-shuffle** | `x = x[:, torch.randperm(L), :]` | Collapse. If not, the model uses per-window features only and the temporal claim is false |
| **Horizon curve** | AUC and nRMSE vs k | Smooth degradation. **Flat is a red flag for leakage, not a good result** |
| **Surprise signal** | nRMSE in benign vs pre-attack windows | Rises before onset — regime change. A good secondary detector and a strong plot |

Run these before you build the UI. If persistence wins, you need the remaining days to fix it.

---

## 6. Explainability

Three separate questions, three separate mechanisms. Do not conflate them.

**(a) Why is the current state risky?** KernelSHAP on the risk head over observed `S_t`. Background set: 100 k-means centroids of benign training states, not random samples — random background makes SHAP slow and noisy.

**(b) Why is *this future* predicted?** Input-gradient saliency over past windows via Captum:

```python
saliency = torch.autograd.grad(pred_state[:, target_feat].sum(), x)[0]  # [B, L, F]
window_importance = saliency.abs().sum(dim=2)                          # [B, L]
```
Output: "the forecast is driven by the change between windows t−3 and t−1."

**(c) Why *this stage*?** SHAP on the stage head applied to the **predicted** state. Only expressible because the transition decodes to named features.

**(d) Counterfactual rollout.** The capability no classifier has.

```python
def counterfactual(model, x, feature_idx, clamp_value, K=6):
    """Clamp one feature at every rollout step, re-simulate."""
    # identical to rollout() but with:  nxt[:, feature_idx] = clamp_value
```

Label it **"model-internal what-if"** in the API response and the UI. It answers what this model would predict given a different input — a question about the model, not about the network. Drawing that distinction unprompted is the single strongest signal of rigour you can send.

**(e) Flow bridge.** A stated deliverable ("flagged flows") that the per-host state model does not produce natively.

```python
def flagged_flows(host, window_ts, top_shap_features, flow_table, n=20):
    """Resolve SHAP features back to the flows that produced them."""
    # e.g. dst_port_entropy ↑  → flows in this window to distinct dst ports
    #      new_peer_count   ↑  → flows to never-before-seen peers
    #      retrans_rate     ↑  → flows containing retransmissions
```

Maintain a `FEATURE_TO_FLOW_PREDICATE` dict mapping each feature name to a filter over the flow table. It is a lookup, not a second model — but it must be built, and it is easy to forget until the night before.

---

## 7. Serving interface

The only thing the backend imports.

```python
class HorizonPredictor:
    def __init__(self, weights_dir, scaler_path, config_path): ...

    def forecast(self, states: np.ndarray, host_id: str, origin_ts: datetime) -> dict:
        """states: [L, F] raw (unscaled) observed windows, oldest first.
           Returns the Forecast schema from the PRD."""

    def counterfactual(self, states, feature_name, clamp_value) -> dict: ...
    def explain(self, states, horizon_k) -> dict: ...
```

Load weights once at process start. Scaler comes from the artifact directory, never refitted. Assert `len(FEATURE_ORDER) == states.shape[1]` on every call — this catches schema drift immediately rather than three hours later in a SHAP plot.

Target latency: under 300 ms for K=6 with 5 models × 200 samples on CPU. If slower, cut samples to 100 before cutting ensemble members.

---

## 8. Build order

| Day | Work | Exit criterion |
|---|---|---|
| **1** | tshark extraction, flow load, join, windowing | A parquet file `[host, window, 45 features]` for three days, packet columns confirmed non-null, scaler fitted on train only |
| **2** | Graph scalars, deltas, labels, splits, dataset class | `train/val/test/holdout` tensors materialised; sanity plots of a known PortScan host showing the escalation |
| **3** | Encoder, transition, rollout, stage-1 and stage-2 training | Training converges; rollout produces K=6 states; `state_nrmse` measurable at every horizon |
| **4** | Baselines, ablations, lead time, calibration | All four baselines run; persistence collapse confirmed; median lead time computed; Infiltration holdout evaluated |
| **5** | SHAP, saliency, counterfactual, flow bridge, `HorizonPredictor` | Backend can import and call `forecast()`; explanation payload complete |

Days 1–2 are the risk. A perfect model on incomplete features fails the stated data requirement; a merely good model on complete features does not.

---

## 9. Failure modes and what to do

| Symptom | Cause | Fix |
|---|---|---|
| Rollout → NaN by k=3 | logvar unclamped, or no state clamp | Clamp both. Non-negotiable. |
| k=1 good, k=6 useless | Single-step training | Multi-step unrolled loss + scheduled sampling |
| Persistence matches model | Window too large or context too short | Try Δ=15 s, L=40. If still flat, report honestly |
| AUC flat across all horizons | **Leakage** | Audit normalisation fit and label window boundaries. Flat is not a good result |
| F1 ≈ 0.99 | Random split, or episode split across train/test | Temporal + episode-level split |
| SHAP takes minutes | Random background set | k-means centroids, 100 samples |
| Features all-zero for some hosts | Empty windows dropped upstream | Emit zero rows with `is_active=0` |

---

## 10. Claims discipline

What we have: **correlation**, **temporal prediction**, **learned dynamics**. All three are legitimate and non-trivial.

What we do not have: **causal inference**. It requires intervention, randomisation, or identification under an explicit causal graph. Our data is observational and confounded by collection protocol — in CIC-IDS2017 attacks were scripted at scheduled times, so time-of-day correlates with attack presence as an artefact of the experiment, not a property of networks.

Predicting `S_{t+1}` from `S_t` is **not causal**. At most it is Granger-causal in a narrow statistical sense, and even that assumes a stationarity the data violates.

**Use:** learned dynamics, forecast, simulated trajectory, projected state, model-internal counterfactual.
**Avoid:** causal simulator, "the model understands why", "predicts attacker intent".

Where we sit on the world-model ladder: a **latent-state transition model with calibrated multi-step rollout**. Architecturally a world model, operationally a forecaster. We do not claim a full generative, action-conditioned world model — that needs an action space and interventional data the public datasets cannot provide. Claiming it invites a question you cannot answer, and being caught overclaiming costs far more than scoping honestly.

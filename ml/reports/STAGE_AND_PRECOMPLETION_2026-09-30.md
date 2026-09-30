# ATT&CK stage on unseen stages, and "predict before compromise completes" (2026-09-30)

Two of the four PS items the project falls short on, each tested with the cheapest
experiment that could change the plan. Neither retrains anything or re-scores a split:
the stage test scores existing frozen heads on observed states, and the lead-time test
reads the Run 8 benchmark's score dumps back. Decisions: `DECISIONS.md` D149, D150.

## 1. Does training the stage head on CIC + CTU-13 let it name Friday's and Thursday's stages?

**No. It makes the ranking worse, and the cause is the CTU data, not the reduced feature set.**

The idea: CIC-IDS2017's training days hold only `initial_access` and `exfil`, so the
stage head cannot name `recon` (PortScan), `c2` (Bot) or `lateral` (Infiltration).
CTU-13 has `recon` and `c2`, and on CTU the frozen stage head ranks them (ROC 0.638 /
0.854, `CTU13_MODEL_IMPROVEMENT` §3.9). Run 9's `combined_heads__state` was trained on
CIC + CTU and had never had its stage output scored on CIC.

Top-1 accuracy on attack futures, from Run 9's existing `mask_comb2cic_state`
benchmarks: 0.029 → 0.000 on test (k = 1 → 6), 0.147 → 0.000 on holdout. The head
predicts `benign` or `exfil` for every Friday attack future and never `recon` or `c2`.

Top-1 hides ranking, so `nidra.scripts.stage_head_diagnostic` scored each stage
one-vs-rest on observed CIC states. Seed 0 in every column. The middle column drops the
same 13 packet features the combined run drops (CTU has no packets), which separates
"added CTU" from "lost features":

| split | stage | windows | Run 8 (45 features, CIC) | CIC only, 32 features | CIC + CTU, 32 features |
|---|---|---:|---:|---:|---:|
| test | recon | 27 | 0.883 | 0.743 | **0.016** |
| test | c2 | 648 | 0.438 | 0.596 | **0.213** |
| test | exfil | 22 | 0.999 | 1.000 | 0.955 |
| holdout | initial_access | 68 | 1.000 | 1.000 | 1.000 |
| holdout | lateral | 28 | 0.313 | 0.222 | **0.008** |

Stage-head ROC for its own class, one-vs-rest. Tables with AP and the risk head beside
them: `tables/stage_head_cic_{run8s0,ciccore,comb}_{test,holdout}.md`; the five-seed
Run 8 ensemble is in `tables/stage_head_cic_run8_*.md` (recon 0.952, c2 0.514,
lateral 0.308).

- Dropping the packet features costs a little: recon falls from 0.883 to 0.743.
- Adding CTU's labelled `recon` and `c2` then pushes CIC's recon, c2 and lateral windows
  **below benign** (0.016, 0.213, 0.008). The head learned CTU's versions of those stages
  (Rbot and Neris botnet traffic, flow-only), and CIC's PortScan, Ares C2 and internal
  scan look like the opposite of them.
- The CIC-only heads' recon ROC (0.74–0.88) comes from a class with **no** training
  positives. It measures "attack-like" leaking into every non-benign column. It is not
  stage identification, and top-1 stays at zero.

**What this rules out:** a shared stage label across these two corpora. The label
`recon` does not mean the same traffic in both. One seed per column; the direction is
the same for all three unseen stages and large compared with the 0.047 head-training
sd §3.29 measured.

**What it leaves:** the stage head can name only stages it trained on, from the same
kind of network. The honest output for Thursday's and Friday's stages is "attack-like,
stage not seen in training", not a nearest-bucket tactic. Real stage coverage needs a
corpus whose multi-stage labels come from the same collection process as the data it is
scored on.

## 2. Predict before compromise completes

`per_episode_report` asks whether a host is warned before its first attack minute:
0 of 15 (test) and 0 of 5 (holdout), and no system can do better on this dataset. The PS
item is narrower. `nidra.eval.campaign_metrics` scores it for every multi-phase campaign
on one host: was the host flagged (2 consecutive minutes at the frozen 0.718) before the
phase that completes the campaign began? Definitions are in the module docstring and
D149. `nidra.scripts.campaign_lead_time` reads the Run 8 score dumps; output is in
`tables/campaign_lead_time_run8.md`.

| split | host | phases (local time, stage) | completing phase | flagged before | lead |
|---|---|---|---|---|---|
| test | 172.16.0.1 | 6 PortScan episodes 13:05–15:21 (`recon`), DDoS 15:56 (`exfil`) | 15:56 DDoS | yes, 14:51 | 65 min |
| holdout | 172.16.0.1 | Web brute force 09:15, XSS + SQLi 10:15 (`initial_access`) | 10:15 | yes, 09:44 | 31 min |
| holdout | 192.168.10.8 | infiltration 14:19, 14:28, internal scan 15:04 (`lateral`) | 15:04 scan | **no** (peak 0.044) | — |

Read with three caveats:

- **172.16.0.1 is the attacker's address**, not a victim being compromised. Its two flags
  are the attacker's earlier scripted attack recognised before its later one. That is
  useful operationally, but it is not a host flagged before its own compromise completed.
- **The one real compromise in the dataset is 192.168.10.8, and it is not flagged.** The
  labelled exploit phase (14:28–14:41) scores 0.000 on every system, including the oracle
  (the frozen head on the true future). The states the head reads do not show it. No
  baseline flags the host before 15:04 at its own validation threshold either (LR spikes
  to 0.954 and 0.929 in single minutes; its threshold is 1.000, D148).
- **The oracle reaches 0.947 from 14:59**, five minutes before the scan, because its
  input is the true future. That is the ceiling a perfect transition model would reach.
  The world model's rollout from 14:59–15:03 does not project the scan (calibrated 0.000;
  the deterministic rollout peaks at 0.038). This is
  the onset problem again, on the one host where it matters.
- A flag during an earlier phase recognises that phase. It is not a forecast that names
  the completing one. On 172.16.0.1, 80 of 201 and 60 of 90 windows before completion
  are absent from the evaluation set (capped benign strata). Runs must be consecutive in
  time, so missing windows can hide a flag but never create one.

**Answer to the PS item, stated plainly:** on CIC-IDS2017 the system flags an attacker's
earlier phase 31–65 minutes before its later phase on the two multi-phase attacker
campaigns. It does not flag the one compromised host before its compromise completes.
Changing that needs a dataset in which a victim's early phase is visible in its own
traffic.

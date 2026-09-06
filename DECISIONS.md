# DECISIONS.md — NIDRA decision log

Non-obvious choices, doc deviations, and tie-breaks. Append-only: dated entry with
**decision · why · what it overrides**. Never rewrite past entries. Any PROMPTBOOK step
(or human session) that makes a non-obvious call adds one here.

---

## 2026-09-07

**D1 — Repository layout follows CLAUDE.md, not IMPLEMENTATION-Backend.md §10.**
Top-level `api/`, `services/`, `nidra_common/`, with `nidra/` reserved for the ML package.
Compose commands are `uvicorn api.main:app` and `python -m services.<name>`.
Overrides: the backend doc's `nidra.api` / `nidra.services.*` paths.

**D2 — WebSocket fan-out via one bus consumer per API process + in-process broadcast.**
Redis Streams consumers *within one group* split messages; the backend doc's "one consumer
name per socket" would deliver ~half the forecasts to each of two browser tabs. One
consumer (group `api`) pushes to per-connection `asyncio.Queue`s; sockets filter by
tenant/host. Overrides: IMPLEMENTATION-Backend.md §8 WebSocket sample (incl. its missing
`nonlocal` on the filter).

**D3 — Bus gets explicit `ack()` + `XAUTOCLAIM` reclaim (60 s min-idle).**
The doc states ack-after-processing but its wrapper never acks; without it the
at-least-once guarantee is prose. Overrides: IMPLEMENTATION-Backend.md §4 sample.

**D4 — Inference sequence buffer dedupes on `window_ts` before RPUSH.**
At-least-once redelivery would otherwise silently duplicate a window inside the L-window
context — wrong forecast, no exception. Extends: IMPLEMENTATION-Backend.md §7.

**D5 — `StubPredictor` behind config `predictor.impl: stub|nidra`.**
Backend pipeline builds, tests, and demos end-to-end before the ML model exists; swapping
in the real `NidraPredictor` is one config line. Backend imports only
`nidra.serve.predictor`, never `nidra.models`.

**D6 — `FEATURE_ORDER` created early, verbatim from IMPLEMENTATION-ML.md §2.6.**
15 flow + 11 packet + 8 graph + 10 dynamics + `is_active` = 45, in
`nidra/data/schema.py` as the canonical file the ML pipeline will extend. Settles the
PRD §4.3 ambiguity (its groups don't sum to 45).

**D7 — Charting is Visx, not Recharts.**
CLAUDE.md overrides PRD §8: Recharts cannot draw the solid-then-widening-band cleanly.

**D8 — Dataset split per CLAUDE.md.**
Monday = normalisation stats, Tue+Wed = train, **Thursday = held out entirely**
(Infiltration generalisation claim), Friday = test. Overrides the PRD's terse
"Tuesday / Wednesday / Friday".

**D9 — `neighbour_risk_fraction` uses the heuristic prior, never model output.**
Fraction of peers with elevated fan-out in the previous window (IMPLEMENTATION-ML.md
§2.6); model-output feedback loops are undebuggable in this build window.

**D10 — Git workflow: single `backend` branch, single PR, human authorship only.**
All backend work on `backend`, pushed to `origin backend` after independent verification,
tracked in exactly one PR into `main`. No AI co-authoring trail: no `Co-Authored-By`,
`Claude-Session`, or "Generated with Claude Code" lines in commits or the PR body —
commits and pushes are authored by MuaazSM only. The orchestrator scrubs stray trailers
from unpushed commits before pushing.

**D11 — `docs/` and `orchestration/` are gitignored.**
The reference docs (PRD, IMPLEMENTATION-*, PROMPTBOOK) and the overnight runner are build
inputs held locally, not repository deliverables; only `CLAUDE.md` and `DECISIONS.md` ship
in-tree. CLAUDE.md's "reference docs live in `docs/`" therefore describes the working
checkout, not the pushed branch. Recorded so the absence reads as deliberate.

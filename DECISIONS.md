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

**D12 — `Forecast` validates `k = 1..horizon_K` with `horizon_K` read from config, not literal 6.**
PROMPTBOOK P1 phrases the rule as "k=1..6"; hardcoding 6 in a schema would silently
contradict `config/default.yaml` the day `horizon_K` changes. The validator reads the
value once (cached) via `nidra_common.config.get_config()`. Naive timestamps are read as
UTC in the causality check so a mixed-awareness payload fails as a validation error rather
than a `TypeError`. Extends: IMPLEMENTATION-Backend.md §3.

**D13 — `web-contract/forecast.example.json` extends the PRD §7.5 table from k=5 to k=6.**
PRD §7.5 tabulates five steps; `horizon_K = 6`, so a schema-valid payload needs a sixth.
k=1..5 reproduce the PRD numbers exactly (0.61→0.85, threshold crossed at k=3, lead time
90 s); k=6 continues the trend (0.86, band 0.63–0.99). `predicted_features` carries all 45
keys per step, with the delta and slope3 features derived from the trajectory itself so the
mock is internally consistent. Stage distributions follow the PRD §4.6 exploitation branch:
port entropy narrowing while upstream byte ratio rises. `model_version` is
`nidra-0.1.0-stub` — no trained model exists yet and the payload should not imply one.

**D14 — Compose publishes redis/postgres on a NIDRA host-port block (6389, 5442).**
Container-internal ports remain the standard 6379/5432 and the in-compose service URLs
use those; only the *published host* ports move. 5432 and 6379 are routinely occupied by
another local stack, and on this machine they are — the alternative was stopping an
unrelated project's containers to satisfy a port number the docs never actually specify.
Every published port is overridable (`NIDRA_REDIS_PORT`, `NIDRA_POSTGRES_PORT`,
`NIDRA_API_PORT`, `NIDRA_WEB_PORT`). `config/default.yaml`'s host-side URLs moved with
them so there is still exactly one source of truth. Overrides: PROMPTBOOK P0's literal
`localhost:6379` / `localhost:5432` config values.
Note for P11: the `api` service still publishes on 8000 by default, which is also
contended locally; `NIDRA_API_PORT` exists for it, but the orchestrator's P11 health
check curls `localhost:8000` unconditionally.

**D15 — Alembic migrations are hand-written; `target_metadata = None`.**
The §9 schema is defined once, in `migrations/versions/0001_initial_schema.py`. Adding
declarative ORM models purely to drive autogenerate would create a second definition free
to drift from the first, and drift there is silent. `migrations/env.py` takes the URL from
`nidra_common.db.database_url()` so `alembic.ini` carries no `sqlalchemy.url`.

**D16 — `sqlalchemy[asyncio]` rather than bare `sqlalchemy`.**
The async engine needs `greenlet`, which the bare distribution does not pull in; without
it every `AsyncSession` call fails at import-adjacent runtime with an opaque error.
Extends: PROMPTBOOK P0's dependency list.

**D17 — `artifacts/` is gitignored except `artifacts/metrics/`.**
Compose bind-mounts `./artifacts` read-only into `api` and `inference`, so the directory
must exist in a fresh checkout; `artifacts/metrics/.gitkeep` provides that. Weights and
the scaler are build outputs; evaluation metrics are a deliverable. Restates CLAUDE.md's
commit policy as an actual ignore rule.

**D18 — Password hashing calls `bcrypt` directly; `passlib` is dropped.**
`passlib` 1.7.4 (last released 2020) fails at import-time backend detection against
`bcrypt` 5.0 — its probe hashes a >72-byte secret, which bcrypt 5 now refuses with
`ValueError` instead of truncating. The alternatives were pinning `bcrypt<5` to keep an
unmaintained wrapper, or calling `bcrypt.hashpw`/`checkpw` directly, which is four lines
and has no compatibility surface. Consequence: passwords are bounded to 8–72 UTF-8 bytes
at the request schema, because silently hashing only the first 72 bytes is the failure
mode the version bump was designed to stop. Overrides: PROMPTBOOK P0's `passlib[bcrypt]`
dependency, replaced by `bcrypt>=4`.

**D19 — Auth endpoints take JSON bodies; the security scheme is `HTTPBearer`.**
`OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")` would advertise in OpenAPI a
form-encoded endpoint with a `username` field that does not exist — and the frontend
generates its types from that document. Login mirrors register: a JSON body with
`email`. Extends: IMPLEMENTATION-Backend.md §8, whose `oauth2_scheme` line is
illustrative rather than a wire format.

**D20 — Access and refresh tokens are distinguished by a `type` claim, checked on decode.**
A refresh token outlives an access token by roughly 336×; without the check, presenting
one as a Bearer credential would work and quietly extend every session to seven days.
`decode_token(token, expected_type=...)` is the only decode path, so the check cannot be
skipped by a caller. The probe route the P3 tests authenticate against is defined in
`tests/test_auth.py`, not in `api/` — P3's API surface is exactly register/login/refresh,
and a permanently mounted test route is surface. Extends: IMPLEMENTATION-Backend.md §8.

**D21 — `Bus.read()` is the primitive; `consume()` is a generator over it.**
The doc's `consume()` is an infinite async generator, which a caller can only leave by
breaking mid-iteration. `run_consumer` has to come up for air between polls to run its
periodic `XAUTOCLAIM` sweep and to check the shutdown event, and abandoning a suspended
generator on every cycle is not that. So one `XREADGROUP` poll is exposed as
`read(count, block_ms) -> list[(msg_id, payload)]`, and `consume()` is a three-line
generator over it — the documented API still exists and is what tests exercise.
Extends: IMPLEMENTATION-Backend.md §4.

**D22 — Redis clients are created with `decode_responses=False`.**
Stream payloads are Pydantic JSON handed straight to `model_validate_json`, which takes
bytes; decoding to `str` on the way out only to re-encode on the way in is waste on the
hot path. Message ids *are* decoded to `str` at the Bus boundary, because they are
compared and logged, never parsed. `Bus` reads its payload field tolerantly (bytes or
str key) so a client configured the other way still works rather than failing on a
`KeyError` three services later.

**D23 — Worker shutdown is an injectable `asyncio.Event`, not a bare signal handler.**
`run_consumer(..., stop=..., install_signals=False)` lets a test — and later the API
process, which runs its `forecasts` consumer inside the FastAPI lifespan rather than as
its own process — drive the loop without installing process-wide signal handlers. The
`__main__` path keeps the default: SIGTERM/SIGINT set the event, the in-flight handler
finishes, and nothing unprocessed is acked. Worst-case exit latency is one `block_ms`.
Extends: PROMPTBOOK P4's "graceful shutdown on SIGTERM".

**D24 — `RawEvent.label` is a top-level optional string, not a `fields` entry.**
PROMPTBOOK P5 lists "label if present" inside the flow payload, but `fields` is
`dict[str, float]` and a CIC label is `"DoS Hulk"`. Encoding it numerically would either
lose the attack type or invent a code table nothing else reads. It is carried as
`label: str | None` instead — visibly supervision rather than measurement, so nothing on
the serving path can mistake it for an input. Extends: PROMPTBOOK P5.1.

**D25 — `RawEvent.fields` is validated as a subset of a per-kind vocabulary, not an
equality check.** A UDP packet has no `tcp_len` and a synthesised zero would be a lie the
features service could not distinguish from a real zero-length segment. Unknown keys and
non-finite values still fail loudly at the producer. Contrast with `StateVector`, where
the check *is* equality because `FEATURE_ORDER` is canonical and complete.

**D26 — Replay speed lives on the stream message and the Redis job hash, not in a new
`ingest_jobs` column.** P5 says the speed is "stored on the job"; the §9 schema has no
column for it and adding one means a second migration for a value that is consumed once,
by the worker, at job start. `job:{job_id}` already carries the live job state the API
merges into its status response. Extends: IMPLEMENTATION-Backend.md §9.

**D27 — Ingest splits failures in two: `IngestError` is acked, everything else pends.**
A file that cannot be parsed — wrong columns, missing tshark, deleted upload — fails
identically on every redelivery, so retrying it forever via `XAUTOCLAIM` would occupy a
worker permanently and never succeed. Those are recorded on the job (`status=error`,
message in Redis and Postgres) and acked. A Redis or Postgres failure propagates and
stays pending, which is what the at-least-once guarantee is for. The error path reports
the events that *did* reach the bus, not zero. Extends: PROMPTBOOK Standing Rules
"ack only after success".

**D28 — tshark is invoked with `-E occurrence=f`; boolean fields are read in both
renderings.** The field list is exactly IMPLEMENTATION-ML.md §2.2. The one added output
option is `occurrence=f`: with tunnelled or repeated layers tshark emits several
comma-joined values for a field, which under `separator=,` silently shifts every column
after it. Separately, tshark 4 prints boolean fields (`ip.flags.mf`,
`tcp.analysis.retransmission`) as `True`/`False` where 2.x printed `1`/`0` — verified
against tshark 4 output for the fixture capture — so both are accepted and anything
falsey is 0. Extends: IMPLEMENTATION-ML.md §2.2.

**D29 — Upload root is `ingest.upload_dir` (default `data/uploads`), not a literal
`/data/uploads`.** P5 names the path `/data/uploads/{tenant}/{job_id}/` and says "path
from config"; an absolute `/data` is not writable on a developer machine, and `data/` is
already the gitignored directory for captures. Compose sets `NIDRA_UPLOAD_DIR` to the
shared `uploads` volume both `api` and `ingest` mount, so the container path is the
deployment's business and the config stays the single source of truth. The stored file is
named `{job_id}{ext}`; the client's filename is kept only for display.

**D30 — The API opens one Redis client per request rather than a process-wide singleton.**
A connection pool binds to the event loop it is first used on, and the API's ingest
endpoints are low-traffic enough that a connection per request is not a cost worth a
cross-loop failure that surfaces as an unrelated timeout. The P9 WebSocket consumer,
which is long-lived and loop-stable, will own its own client.

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

**D31 — Windows close at tenant scope, not per host.**
IMPLEMENTATION-Backend.md §6 keys the window state per `{tenant}:{host}` and describes
close as "an event arrives with a later `window_ts`". Taken literally that is per host,
and four of the eight graph scalars — `in_degree`, `reciprocity`,
`local_clustering_coeff`, `neighbour_risk_fraction` — are properties of the window's
*whole* host graph. Closing one host while another is still writing edges into the same
window computes them against a half-built graph. The accumulators stay keyed per host
exactly as the doc specifies; only the decision to close is shared, via
`feat:open:{tenant}`. Extends: IMPLEMENTATION-Backend.md §6.

**D32 — A host's window row is built from events where that host is the source.**
`bytes_up_down_ratio`, the flag ratios and the IAT statistics are all directional
(`fwd` = away from the source), so folding events where the host is the *destination*
into the same counters would mix two frames of reference. What was done *to* a host
reaches its vector through the window graph instead, which is where `in_degree` and
`reciprocity` come from. A host that only received traffic in a window therefore has no
row of its own that window — it appears as a silent `is_active=0` row once it next
transmits. Extends: IMPLEMENTATION-ML.md §2.6.

**D33 — A window is summarised as bounded counters, never a list of its events.**
Sums, sums of squares, and small value-count distributions (`dp:80`, `ps:1460`) share one
`feat:win:{tenant}:{host}:{ts}` hash, so accumulating an event is a single pipelined
`HINCRBYFLOAT` batch and closing a window is a single `HGETALL`. Entropies and
`payload_size_p95` are computed from the distributions at close. Buffering raw events per
window would be simpler and would make a busy host's memory a function of its traffic.
Extends: IMPLEMENTATION-Backend.md §6 ("accumulating counters for the open window").

**D34 — "Elevated fan-out" is quantified as `features.elevated_fanout: 5`.**
IMPLEMENTATION-ML.md §2.6 defines `neighbour_risk_fraction` as the fraction of a host's
peers with "elevated fan-out in the previous window" without naming a threshold. It is
now a config value, along with `max_gap_windows`, `clustering_degree_cap` and
`watchdog_floor_s`, under a `features:` block — nothing hardcoded in a script. It remains
a heuristic prior read from the *previous* window's graph; model output is never fed back
into an input feature. Extends: IMPLEMENTATION-ML.md §2.6.

**D35 — Silent windows are all-zero rows, and they enter the delta history.**
A backfilled window is zero in every one of the 45 features, including its own `d_*` and
`slope3_*`, and the zeros are appended to the host's rolling history. The alternative —
carrying the last observed values across a gap — would make a delta measured over three
minutes of silence indistinguishable from one measured over thirty seconds of traffic.
Going quiet is a real drop in the series and the dynamics should say so. Extends:
IMPLEMENTATION-ML.md §2.5.

**D36 — A gap longer than `max_gap_windows` restarts the sequence instead of backfilling.**
Past 40 silent windows (20 minutes) the gap is a stopped replay or a restarted capture,
not a quiet host. Emitting 2880 zero rows for an overnight gap would bury the real windows
and drag every slope through them, so the host's history is dropped and its next vector is
zero-padded like a first window. Logged at INFO, never silent. Extends: PROMPTBOOK P6.2.

**D37 — IAT features are computed from flow events only.**
`iat_mean`, `iat_var` and `iat_max` aggregate the per-flow `Flow IAT` columns; a window
carrying only packets reports them as 0. Packet inter-arrival gaps measure a different
quantity, and averaging the two into one feature would be a unit mismatch that no
assertion could catch. The ML pipeline joins packets onto flows (IMPLEMENTATION-ML.md
§2.4), so a window with packet features generally has flow features too. Extends:
IMPLEMENTATION-ML.md §2.6.

**D38 — The watchdog is a periodic sweep with an injectable clock, not a timer per window.**
`FeaturesWorker.watchdog_tick()` scans `feat:open:*` and closes any tenant whose last
event is older than the grace (`2 × window_delta / replay speed`, floored at
`features.watchdog_floor_s`). One sweep task covers every tenant, survives a restart
because the deadline is in Redis rather than in a live `asyncio` handle, and — because
`now` is injectable — lets the final-window test run in milliseconds instead of waiting
out the grace. A per-tenant lock keeps the sweep and the event path from closing the same
window twice. Extends: IMPLEMENTATION-Backend.md §6 ("watchdog timer").

**D39 — An event older than the open window is dropped with a warning, not folded in.**
Its window has already been published; amending the accumulator would produce a second,
different vector for a `window_ts` downstream has already consumed. Ingest publishes in
timestamp order and logs its own chunk-boundary regressions (D27), so this is the second
line of defence rather than the first. Extends: PROMPTBOOK P6.2.

**D40 — The predictor is tenant-blind; the worker supplies `tenant_id`.**
`NidraPredictor.forecast(states, host_id, origin_ts)` takes no tenant, so the backend doc's
`Forecast(**result)` cannot validate on its own. `InferenceWorker` completes it with
`Forecast(tenant_id=vector.tenant_id, **result)`. Tenancy is a serving-plane concern and
the model has no business knowing about it — keeping it out of the interface also keeps
the same predictor usable from offline evaluation, where there is no tenant at all.
Extends: IMPLEMENTATION-Backend.md §7, IMPLEMENTATION-ML.md §7.

**D41 — Stub risk drifts with the horizon at a rate set by the driver slopes, and the
drift can be negative.** P7 asks for risk "rising with k". Implemented as
`logit_k = logit_0 + k * (DRIFT_BASE + gain × weighted slope term)`: a flat host still
drifts up slightly, because uncertainty accumulates over a horizon, but a host whose
drivers are collapsing bends downward. Forcing monotone rise would make the counterfactual
meaningless — clamping `syn_ratio` to zero has to visibly flatten the curve, and that is a
demo-critical behaviour (CLAUDE.md "six things must work"). Extends: PROMPTBOOK P7.1.

**D42 — `predictor.weights_dir` / `scaler_path` / `config_path` live in
`config/default.yaml`.** `NidraPredictor.__init__` takes three artifact paths and nothing
may be hardcoded in a script, so they are config keys, read only on the `impl: nidra`
branch and resolved against the repo root when relative — the same rule `ingest.upload_dir`
already follows. The stub branch reads none of them. Extends: IMPLEMENTATION-ML.md §7.

**D43 — Sequence-buffer dedupe compares against the buffer tail only.**
`LINDEX seq:{tenant}:{host} -1` and a `window_ts` equality check, not a scan of all L
entries. The features service emits windows in ascending order per host, so the only
duplicate at-least-once delivery can produce is an immediate repeat of the last one; a
full scan would cost L round trips per message to catch a case the producer cannot create.
Extends: PROMPTBOOK Standing Rules correction 4.

**D44 — `driving_window` is an index into the context array, 0 = oldest.**
Not a negative offset and not a timestamp. Ties resolve to the *later* window: when several
windows moved the drivers equally, the most recent one is the more useful thing to point a
console at. `explain()` returns the full normalised `window_importance` alongside it, so a
caller that wants the whole profile does not have to infer it. Extends:
IMPLEMENTATION-Backend.md §3 (`driving_window: int`).

**D45 — The stub reimplements its 3-window OLS slope rather than importing the features
service's.** `services.features.compute.ols_slope` computes the same quantity, but importing
it would make the inference service depend on the features service's module tree for four
lines of arithmetic. Service packages stay independent of each other; both depend only on
`nidra_common` and `nidra.data.schema`. Extends: PROMPTBOOK Standing Rules (canonical layout).

**D46 — The persister re-asserts `origin_ts` causality even though `Forecast` already
validates it.** `assert_causal()` runs before the insert, raising rather than coercing.
The schema validator makes the check redundant for anything published through `Bus`, but
this is the last gate before a number becomes durable and gets shown as "warned 90 s
before onset"; a horizon at or before its own origin is a forecast that saw the future,
and a boundary that silently trusts its producer is exactly the failure the project's
central claim cannot absorb. Extends: PROMPTBOOK P8.1, IMPLEMENTATION-Backend.md §9.

**D47 — Episode idempotency is keyed on a `last_ts` watermark in Redis, not on whether the
forecast row was newly inserted.** `ep:{tenant}:{host}` carries the last `origin_ts`
folded into the lifecycle, and a forecast at or before it is skipped. Gating the episode
logic on `ON CONFLICT DO NOTHING` returning a row would look equivalent and would lose an
update whenever a worker crashed between the commit and the Redis write: the replay would
find the row already there and skip the episode step forever. With the watermark, the two
stores recover independently — the insert conflicts, the episode still advances. Extends:
PROMPTBOOK P8.2.

**D48 — `started_at` is the first above-threshold forecast; `first_alert_at` is the
forecast at which `lead_time_m` consecutive above-threshold windows completed.** The two
columns exist in IMPLEMENTATION-Backend.md §9 without distinct definitions, and making
them identical would waste one. A single window over the line is a spike, not an alert —
`lead_time_m: 2` in `config/default.yaml` is described in CLAUDE.md as exactly that
debounce. An episode that closes before the debounce completes keeps `first_alert_at`
NULL: it happened, but nobody was warned. Extends: IMPLEMENTATION-Backend.md §9.

**D49 — `ended_at` is the origin of the forecast that closed the episode, and `stages`
accumulate over every forecast delivered while it was open.** The episode therefore spans
its own cool-down: the alternative — dating the end at the last above-threshold window —
back-dates a fact by `episode_close_after` windows that was not knowable then. `stages` is
the arc in order of first appearance (`recon → initial_access → benign`), taken from
`observed_stage`, which is measurement rather than prediction. Extends:
IMPLEMENTATION-Backend.md §9, PROMPTBOOK P8.2.

**D50 — The episode hash outlives the episode; closing deletes the aggregate fields and
keeps `last_ts` under a 24 h TTL.** Deleting the whole key at close would let a redelivered
above-threshold forecast — the same message a reclaim hands back — open a second episode
next to the one it already closed. Keeping the watermark makes that redelivery a no-op.
Extends: PROMPTBOOK P8.2 ("keep open-episode state in Redis").

**D51 — A forecast whose `tenant_id` is not a UUID is refused at the persister boundary.**
`forecasts.tenant_id` is `UUID`; the ML-facing `Forecast.tenant_id` is a `str` because the
predictor is tenant-blind (D40). Rather than let asyncpg raise a cast error mid-transaction,
`tenant_uuid()` fails loudly with the offending value. Extends: PROMPTBOOK Standing Rules
(fail loudly at boundaries).

**D52 — The socket's host filter is applied when the queue drains, the tenant filter when
it is broadcast.** Tenant membership is fixed for the life of a connection, so filtering
there means a queue never holds another tenant's bytes at all — a later filtering bug
cannot leak them. The host filter can change mid-stream, so applying it at drain time
means an update governs forecasts already queued rather than only ones that arrive after
it. Extends: D2, PROMPTBOOK P9.2.

**D53 — A filter update is acknowledged back to the client as `{"type":"filter","hosts":
[...]}`.** The doc's design has no reply, which leaves a client unable to know when its
update took effect and a test unable to publish deterministically after one. The
acknowledgement is enqueued rather than written directly: one task owns sending on the
socket, so no two coroutines ever write to it at once. Extends: IMPLEMENTATION-Backend.md
§8 WebSocket.

**D54 — The heartbeat is application-level JSON (`{"type":"ping"}` / `{"type":"pong"}`),
not RFC 6455 control frames.** Starlette exposes no protocol-level ping, and a heartbeat
the ASGI server owns is one neither the route nor a test can observe. The sending task
pings after `api.ws_ping_interval_s` with nothing to send and closes with 1001 once
`api.ws_max_missed_pongs` have gone unanswered. Extends: IMPLEMENTATION-Backend.md §8
("heartbeat every 30 s; drop on two missed pongs").

**D55 — `/health` reports lag per *consumer group*, so `forecasts` appears twice.**
Pending count is a property of a group, not of a stream: the persister and this api
process read `forecasts` under different groups and either can fall behind alone. The
keys are `ingest_jobs`, `raw_events`, `state_vectors`, `forecasts` (persister) and
`forecasts:api` (socket fan-out). `/health` stays 200 with `status: "degraded"` when Redis
is unreachable — an endpoint that fails to answer is indistinguishable from a dead
process. Extends: IMPLEMENTATION-Backend.md §8. This changed the shape P3 asserted, so
`test_health_reports_status_and_empty_streams` became
`test_health_reports_status_and_stream_lag` and now asserts the populated keys.

**D56 — A page size above `api.max_page_limit` is clamped, not refused; a sort key outside
the whitelist is a 422.** The two differ because one is a preference and the other is a
correctness boundary: `limit=100000` means "as much as you'll give me", while a sort key
reaches the `ORDER BY` and is only ever chosen from `HOST_SORT_COLUMNS` by lookup, never
interpolated. Extends: PROMPTBOOK P9.1.

**D57 — `tenant_uuid()` moved from `api/ingest.py` to `api/deps.py`, joined by a
`CurrentTenantUUID` dependency.** Three routers now need the tenant as the UUID its
columns are typed as; the alternative was the forecasts router importing from the ingest
router, which reads as a dependency that is not one. Tenancy lives in one module.
Extends: PROMPTBOOK Standing Rules (every query filters on `tenant_id`).

**D58 — Observed state vectors are stored in a Postgres `state_vectors` table written by
the features worker, not kept in Redis.** P10 leaves the choice open between "a lightweight
table or a Redis-backed cache, whichever is the simplest crash-safe option". Redis here is
started `--appendonly no --save ""` and the inference sequence buffer is a rolling window
of the last L entries under a one-hour TTL: it can explain the newest forecast and nothing
older, and nothing at all after a restart. One row per `(tenant, host, window)` with
`UNIQUE (tenant_id, host_id, window_ts)` reuses the persister's idempotency device — a
reclaimed `raw_events` message rewrites the same window and `ON CONFLICT DO NOTHING` makes
that a no-op — and its index is also the read path, so there is no second index. The write
happens **before** the publish to `state_vectors`, so every forecast that exists has the
context that produced it already durable behind it. Extends: PROMPTBOOK P10.1,
IMPLEMENTATION-Backend.md §9 (migration `0002_state_vectors`).

**D59 — `FeaturesWorker(store=...)` is optional; the process entry point always wires one.**
The worker's other dependencies are Redis-only, and the P6 tests drive it against Redis
alone with non-UUID tenant ids. Making the store required would have forced Postgres into
those tests to prove nothing about windowing. `services/features/__main__.py` passes a
`StateVectorStore`, and `test_the_features_worker_stores_every_vector_it_publishes` asserts
the published-implies-durable property through the real worker. Extends: PROMPTBOOK P10.1.

**D60 — `GET /api/v1/explain/{host}/{ts}` 404s unless a state vector exists at exactly
`ts`, and reports `context_windows` against `context_l`.** A context that merely ends
*before* `ts` would explain a different forecast than the one asked about, silently. Fewer
than L windows is legitimate at the start of a capture, so it is served with both counts in
the response rather than refused. `k` defaults to `horizon_K`: the end of the cone is the
claim being made, so it is the step worth explaining unless the caller says otherwise.
Extends: PROMPTBOOK P10.1.

**D61 — The counterfactual's `"model-internal what-if"` label is verified at the API
boundary, and a predictor that returns anything else is a 500.** The label is the claims
discipline in CLAUDE.md §5, so serving an unlabelled what-if is the one failure this
endpoint must not have. `ts` on each curve point is computed in the API as
`origin_ts + k * window_delta` — the same rule `Forecast` timestamps its horizons by — so
the console does not have to derive it. Extends: PROMPTBOOK P10.2, CLAUDE.md §5.

**D62 — `/api/v1/benchmarks` serves every `*.json` under `api.metrics_dir` verbatim; an
unreadable artifact is a 500, not a partial answer.** Serving the readable subset would
present incomplete results as complete, which is the same failure as inventing a number
with extra steps. An empty directory is `status: "pending"` with the detail
`"evaluation artifacts not yet produced"`. The directory is config, not a literal.
Extends: PROMPTBOOK P10.3.

**D63 — `/api/v1/benchmarks` and `/api/v1/model` require a token, and the api process
loads one predictor at construction and calls it on a worker thread.** Neither endpoint
reads tenant data, but the console is the only client and an unauthenticated route is one
more surface to reason about at judging time. The predictor is per process rather than per
request because the trained one builds an ensemble and a scaler; it is called through
`run_in_threadpool` because the api event loop also owns the WebSocket fan-out and a
300 ms torch rollout on it would stall every open console. Extends: PROMPTBOOK P10.3–4,
IMPLEMENTATION-ML.md §7.

**D64 — Rate limiting is a Redis sliding window keyed on the tenant (its address before a
token exists), relaxed in dev by a multiplier rather than switched off.** A fixed counter
per minute lets 2×limit through across a boundary, and a limiter that is disabled in
development is a limiter first exercised in front of judges: `NIDRA_ENV=dev` multiplies
both ceilings by `api.rate_limit.dev_multiplier`, so the same code path runs everywhere
while the suite and a console being clicked through never reach it. The key is the tenant
from the verified token and from nowhere else, so no caller can choose its own bucket.
`/health` and `/metrics` are exempt — compose polls the first every 5 s and a throttled
health probe would read as a dead process. An unreachable Redis fails **open** with a
warning: refusing every request when the limiter's datastore blinks is a worse outage than
the one it prevents. Extends: IMPLEMENTATION-Backend.md §8, PROMPTBOOK P11.2.

**D65 — The backpressure signal is monotone growth for 30 s, sampled on the api's own
timer, not a threshold on the pending count.** A stage that absorbs a burst has a high
count that comes back down; a stage that cannot keep up has one that never does, so the
absolute number would have to be tuned per stream while the shape does not. Any sample
that fails to increase ends the streak, so a recovered pipeline stops warning by itself.
`BacklogWatch` polls inside the api lifespan (`api.backlog_poll_s`) because a warning that
only fires when someone loads `/health` is a warning nobody sees during a replay; `/health`
then reports the same groups under `backpressure`. Extends: IMPLEMENTATION-Backend.md §7,
PROMPTBOOK P11.5.

**D66 — `docker compose up -d` now starts the whole serving plane; only `web` stays behind
a profile.** The `full` profile existed so the stack was usable before the services did,
and they exist now. `web/` is the frontend build's directory and is not on this branch, so
an unprofiled `web` service would make `docker compose up -d` fail for everyone on a
missing build context; it moved to `--profile web` rather than out of the file. The api
gained a healthcheck (urllib, because the image carries no curl) so `depends_on:
service_healthy` orders the console behind a serving api rather than behind a started
process. Extends: PROMPTBOOK P11.3.

**D67 — The test suite runs against its own Redis logical database (`redis.test_db`),
applied in `tests/conftest.py`.** With the workers now always up (D66), the suite is a
second deployment of the same plane sharing one bus: the containerised ingest worker
consumed a job the suite had just queued and failed it, because the upload it names is a
host path no container can see, and `test_upload_queues_a_job_and_reports_its_status`
watched its own job turn to `error`. Separating the two at the bus keeps both real —
neither mocks the other away, and neither reads the other's streams. An explicit
`NIDRA_REDIS_URL` still wins, which is how an end-to-end test will point at database 0.
Extends: PROMPTBOOK P11.3.

**D68 — `tests/fixtures/replay.csv` is generated from config and committed, and its
window count is `context_L + horizon_K + 16`.** The first forecast is only possible once L
windows have been observed, so a capture shorter than L + K replays in full and forecasts
nothing; the extra windows are the room the escalation has to develop *in front of an
audience* rather than inside the context that produced the first cone. The scan is added
to the host's ordinary traffic instead of replacing it, so `syn_ratio` climbs instead of
stepping to 1.0 in one window — a step is trivially forecastable and nothing like a real
escalation. `make fixtures` regenerates it; `make demo` (60×) and `make e2e-fast` (600×)
replay it through the running stack. The `Label` column is written for a human reading the
file and is never an input (D24). Extends: PROMPTBOOK P11.4.

**D69 — All frontend work lives on the single `frontend` branch, mirroring the backend
convention.** `frontend` was branched from `main` (not from `backend`, which is still open
in its own PR) so the two build tracks review independently and neither blocks the other.
One branch, one PR into `main`, no `feat/`-prefixed frontend branches, no force-pushes —
the same rule CLAUDE.md already sets for the serving plane, applied for the same reason:
the reviewer reads one diff, not nine. `.gitignore` gained `donors/`, `web/node_modules/`,
`web/.next/`, `web/lib/types.gen.ts` and `web/.env.local`. It also gained `!web/lib/` —
the Python packaging rule `lib/` inherited from the standard template matches **any**
directory named `lib` at any depth, which silently swallowed `web/lib/tokens.css`, the
frontend's source of visual truth. The negation is load-bearing, not tidying.
Extends: PROMPTBOOK F0.

**D70 — The donor harvest was run on a locally-driven Chromium because no browser MCP is
connected, and the measurements are real as a result.** `/clone-website` requires a
browser-automation MCP server and this session has only `stitch`; the skill's own
pre-flight says to stop and ask. Stopping would have produced a token file full of
plausible round numbers, which is the one outcome the phase exists to prevent — "a value
you could not measure is written 'not measured', never guessed". Instead the capture ran
against the Playwright Chromium already in this machine's cache (chromium-1228, Chrome for
Testing arm64) driven by `playwright-core` installed into the session scratchpad: the same
engine the skill would have used, different transport. Both passes (1440×900 and 390×844)
settle on `networkidle` and then scroll the full page before measuring, because all three
donors lazy-mount content and measuring early returns the wrong element count. Capture
tooling stayed in the scratchpad; only the artifacts landed in `donors/`.
Extends: PROMPTBOOK F0.

**D71 — Two of three donors cloned as specified; the Grafana URL is dead, and a live
dashboard from the same donor was substituted rather than inventing density numbers.**
`https://play.grafana.org/d/000000012/grafana-play-home` returns HTTP 200 with an
application-level 404 — the Grafana shell renders "Dashboard not found" and only 23 text
nodes exist to measure. `curl .../api/dashboards/uid/000000012` returns a bare `404`: the
dashboard was retired from the Play instance. It is dead, not blocked, so no different
capture technique would reach it. The failed capture is kept at
`donors/grafana/notfound-000000012-*` as evidence, and `(Home) Kubernetes Integration`
(uid `lAoEVhD7z`) was captured in its place — chosen from `/api/search` as the densest Play
dashboard with the panel mix NIDRA's console actually needs (stat tiles, time-series, text,
scrolling list). It was captured in **both themes**, which the original brief did not ask
for but which the dark palette needed. Anonymous access means several panels return *No
data*; panel chrome, type, grid and spacing are fully measured and harvested, while series
colours and plot-area treatment are recorded as "not measured" rather than back-filled from
the deck. wiz.io and linear.app both returned 200 with no bot wall (392 and 523 text nodes).
Extends: PROMPTBOOK F0.

**D72 — `--text` and `--text-muted` are set to `--ink` and `--muted-ink`, and `--accent` is
held away from `--observed`.** The deck's charts already draw axis labels in `--ink`
`#10243A` and ticks in `--muted-ink` `#5B6B7B`. Pointing the chrome text tokens at those
same two values means chart text and page text are literally the same colour, so a
`ForecastChart` sits *in* the page rather than on it — and the browser chart and the PRD
figure stay one chart, which is the whole reason the data tokens are fixed. The converse
constraint is why `--accent` is `#0B57D0` (hue 220°, high chroma) rather than something
nearer the donors' blues: `--observed` is `#1D6FA5` (hue 204°, moderate chroma), and an
accent that reads as observed-blue would make a button look like a measurement. The rule is
stated in `tokens.css` and is bidirectional — the accent is never used for data, and no data
colour is ever used for an interactive control. Dark-mode data colours were lightened and
checked by computing HSL hue on both values; every hue holds within 3° (observed
203.8°→203.4°, projected 257.4°→254.8°, threshold 34.9°→36.8°, positive 149.6°→146.3°,
negative 348.1°→349.0°). Extends: PROMPTBOOK F0.

**D73 — Body text is 16 px, and the console drops to 14 px only because Grafana measures
14 px.** Wiz runs 16/24 across n=102 elements; Linear's marketing body is 15/24; Grafana's
app body is 14/22. 16 px is the top of that band and is the size used by the donor whose
role is marketing. The console's `--text-ui` is 14/22 — Grafana's measured app size, adopted
as a considered density choice for a dashboard, not as a smaller default leaking outward.
Two tracking rules came out of the harvest and both are kept because they apply at opposite
ends of the scale: Linear tracks display type **negative** at a uniform −0.022 em
(−1.584/72 = −1.408/64 = −1.056/48) and text at −0.011 em, while Grafana tracks small UI
text **positive** at +0.0107 em (+0.14994 px at 14 px) because opening up dense 12–14 px
text is what keeps it legible. Section rhythm takes Wiz's 96 px spacer as the default and
Linear's 128/128 as the major break; the measure is 1280 px, which is Linear's, confirmed by
its own 272×4 + 64×3 card grid and within 2 % of Wiz's 1310 px.
Extends: PROMPTBOOK F0.

**D74 — Dark-mode `--projected` is `#B09BF0`, chosen by relative luminance rather than by
lightening the deck purple. Supersedes the dark value recorded in D72.** D72 checked the
dark derivation for hue drift only, and that check passed while a second constraint failed
silently: at `#A899D6` the dark `--projected` sat at L=0.360 against `--observed` at L=0.357
— a gap of 0.003, which is the same grey. Observed-versus-projected is the chart's central
distinction, so the two most important colours in the product were indistinguishable in
greyscale in one of the two themes. The obvious repair is wrong: lightening the purple until
it clears observed lands at L=0.466 and collides with `--threshold` at L=0.460 instead. The
dark ramp holds five colours inside an L-span of 0.184 and `--projected` has exactly one gap
available to it, between observed and positive. `#B09BF0` sits at L=0.390 (hue 254.8°, 2.6°
off the light value), giving adjacent deltas of .081/.033/.043/.027 and clearing the 0.025
floor throughout. `--positive` moved `#6FBF92`→`#70C094` in the same pass because D72's
"every hue holds within 3°" was true of four values and 3.4° on the fifth; at `#70C094` the
drift is 2.6° and the claim now holds as stated. `--projected-band` and
`--projected-band-edge` were re-derived from the new value so the cone cannot drift off its
own mean line. Light mode is unchanged and still fails the 0.025 floor at
observed/projected (Δ 0.024): those five values are fixed by `deck/charts/_style.py` so the
console and the PRD figures stay the same chart, and that identity is worth more than a
thousandth of separation. The consequence is recorded as a rule rather than left implicit —
the second channel (solid vs dashed, a label, an icon) is load-bearing wherever these
colours carry meaning, not a nicety. Supersedes: D72 (dark values only; the accent
reasoning stands). Extends: PROMPTBOOK F0.

**D75 — The frontend never renders a fabricated forecast. Illustrative charts live in the
deck.** The F0 specimen originally showed the five data colours as a hand-drawn SVG risk
curve captioned "values here are illustrative". It was removed and replaced with a greyscale
separability strip that computes relative luminance off `getComputedStyle` — the measurement
the drawing only implied, and the one that would have caught D74 at the gate. The general
rule: every chart the product renders is driven by schema-valid fixture data or live data,
and a drawn curve, cone or trajectory is not permitted anywhere in `web/` — not in a
specimen, a hero, a marketing section, or as a placeholder. A slide is read as an argument
and `deck/charts/hero.png` is the right home for an illustration; a chart inside the product
is read as a measurement, and a drawn one is a fabricated measurement in the one place this
project claims is checkable. Conceptual architecture diagrams are exempt — `deck/diagrams/`
depicts structure, not data. Recorded as Visual invariant 3 in
`docs/PROMPTBOOK-FRONTEND.md`, with F0, F1 and F4 updated to match. Extends: PROMPTBOOK F0.

**D76 — Tremor 3.18 installed under React 19 via `web/.npmrc` `legacy-peer-deps=true`.**
`@tremor/react` pins `react@^18` as a peer while Next 16 ships React 19, so a plain
`npm i` refuses to resolve. The stack is pinned by the frontend spec, and the primitives
F1 uses (Card, Metric, Text, ProgressBar) have no React-19 incompatibilities. The flag
lives in a committed `.npmrc` so every future install resolves the same way. Revisit if a
Tremor chart component misbehaves in a later phase. Extends: PROMPTBOOK-FRONTEND stack
table.

**D77 — One colour system: Tailwind's default palette is wiped, and the theme is mapped
from tokens.css via `@theme inline`.** `app/globals.css` sets `--color-*: initial` and
`--shadow-*: initial`, then maps every utility the tree uses — shadcn roles
(`--color-background: var(--bg)` …), Tremor roles (`--color-tremor-*`, both light and
dark names, onto the same self-flipping vars), data colours, type scale, radii — as
`var()` references into `lib/tokens.css`. A `bg-blue-500` anywhere now compiles to
nothing rather than to a stock colour, which is a stronger guarantee than the grep in
machine verify. Two consequences recorded: the dialog/sheet overlay scrim, shadcn's
`bg-black/10`, became `bg-foreground/10` (black no longer exists; foreground tracks the
theme); and shadcn's decorative `shadow-sm/lg` utilities vanish, which is the donors'
actual behaviour — borders carry elevation, `--shadow-drawer` is the one exception.
Extends: F1 step 3.

**D78 — The `dark:` variant is a custom variant implementing tokens.css's three states,
not a class.** shadcn's default `@custom-variant dark (&:is(.dark *))` would ignore
`data-theme`; a media-only variant would ignore the explicit toggle. The variant matches
`[data-theme="dark"]` descendants, plus `prefers-color-scheme: dark` under
`:root:not([data-theme="light"])` — exactly the cascade tokens.css runs, so the few
`dark:` tweaks inside stock shadcn components flip in step with the tokens. Extends: F1
step 4.

**D79 — Inter is self-hosted under the family name `"Inter"`, not loaded via
`next/font`.** `next/font` registers a mangled family name (`__Inter_…`) that
tokens.css's `--font-sans: "Inter", …` stack would never match, and the harvest's
`--weight-medium: 510` requires the variable axis. The latin variable woff2 is copied
from `@fontsource-variable/inter` (devDependency, for provenance) into
`web/public/fonts/` and declared with a plain `@font-face` (`font-weight: 100 900`) plus
a preload link. tokens.css is untouched. Extends: F1 step 3.

**D80 — `/hosts`, `/episodes` and `/benchmarks` get honest empty-state index pages in
F1.** The F1 prompt asks only for the fleet page, but the shell's nav links to all four
sections and a reviewer tabbing through the gate would land on framework 404s that read
as breakage rather than emptiness. Each stub is one sentence saying what the page will
hold and why it is empty (`/benchmarks` states "the evaluation has not run" — never a
placeholder number, matching the API's `pending` semantics). F5 replaces all three.
Extends: F1 step 6.

**D81 — "a Tremor stat card and bar" is read as Card + Metric + ProgressBar, not
`BarChart`.** F1 forbids charts ("No forecast data, no charts") and Visual invariant 3
forbids illustrative data visuals, so the bar on `/dev/components` is a `ProgressBar`
showing `risk_threshold` read from the geometry fixture — a real configured value, not
an invented series. Tremor's chart components first render in a later phase against
schema-valid fixtures. Extends: F1 step 9.

**D82 — `web/fixtures/model.json` mirrors the `GET /api/v1/model` response, not
`config/default.yaml` verbatim.** The loader (`lib/geometry.ts`) is written against the
shape it will fetch in F6 — `{impl, model_version, schema_ver, config}` with the six
`ModelConfig` fields the endpoint actually echoes (`api/explain.py`) — so F6 swaps the
source without touching a caller. Values are `stub` / `nidra-0.1.0-stub` / schema `1.0`,
matching the backend branch. Components read geometry through `getGeometry()`; the fleet
and episodes empty states and the `/dev/components` table interpolate it rather than
retyping 30 or 0.75. Extends: F1 step 10.

**D83 — demo-replay.json is synthesised (F2 path 2), driven through the backend's own
StubPredictor; only the telemetry and the escalating host's stage narrative are
scripted.** The capture path was unavailable (Docker down) and structurally unable to
produce the required fixture: the stub's `stage_distribution` never claims stages past
`initial_access` (fixed 0.01 sliver on lateral/c2/exfil, argmax capped), and
`tests/fixtures/replay.csv` is labelled BENIGN/PortScan only — no capture can show
`benign → recon → initial_access → lateral`. So `scripts/make_demo_fixture.py`
synthesises per-window driver curves (4 hosts × 48 windows, one scan-to-intrusion arc,
smoothed keyframes) and feeds them to `services.inference.stub_predictor.StubPredictor`:
risk, bands, lead times, top signals and all 45 predicted features are the pipeline's own
arithmetic. The one override is the victim's `observed_stage`/`stage_dist`, scripted as a
continuous stage-progress kernel reaching `lateral` (sliver of c2, never exfil). Every
forecast passes `Forecast.model_validate` and the written JSON is the validator's dump;
story invariants (crossing partway, stage order, k=3 and k=1 crossings present, quiet
hosts quiet) are asserted so regeneration fails loudly instead of drifting. Fully
documented for the /demo banner in web/fixtures/README.md. Extends: F2 step 1.

**D84 — The uncertainty cone is Visx `Area` (y0/y1), not `AreaClosed`.** F2 names
`AreaClosed`, but that primitive closes the path to the scale's baseline — it would fill
from `ci_low` down to risk 0, drawing probability mass the band does not claim. `Area`
with `y0=ci_low`/`y1=ci_high` is the band the spec describes. The anchor requirement is
unchanged: band and mean are both prepended with
`{ts: origin_ts, lo: observed_risk, hi: observed_risk}`. Overrides: F2 step 2 layer 3's
component name, not its meaning.

**D85 — Stage-strip colours are a severity ramp mixed from tokens, not six new hues.**
`color-mix(in srgb, var(--negative) N%, var(--bg-raised))` with N monotone in escalation
order (4→94%). No literal colours, theme-correct in both modes, and luminance orders the
stages so the strip stays readable in greyscale — with the argmax stage labelled in text
per run of consecutive same-argmax columns (a label that does not fit its run is dropped,
the widest run is always labelled). Extends: F2 step 2 layer 9, Visual invariant 2.

**D86 — `model_version` in fixtures is `stub-0`, matching the stub; the web-contract
example keeps `nidra-0.1.0-stub`.** The two disagree: `stub_predictor.MODEL_VERSION` is
"stub-0" while `web-contract/forecast.example.json` (D13) says "nidra-0.1.0-stub". What
a live capture would carry is the stub's value, so `demo-replay.json`,
`counterfactual.single.json` and `fixtures/model.json` say "stub-0";
`forecast.single.json` stays a verbatim copy of the published example, discrepancy and
all. Flagged here rather than silently harmonised — the example or the constant should
be reconciled backend-side. Extends: F2 step 1.

**D87 — `web/lib/types.ts` hand-mirrors the contract until F6; a fourth fixture,
`counterfactual.single.json`, feeds specimen state (f).** The chart needs `Forecast`
types before `types.gen.ts` exists; `lib/types.ts` mirrors `nidra_common/schemas.py` and
`api/explain.py` with a header stating it shrinks to aliases of the generated types in
F6. The counterfactual fixture is the stub's own `counterfactual()` at the victim's
first k=3-crossing origin (`dst_port_entropy` clamped to the quiet-host level), with
`ts` added per the API's `CurvePoint` and the literal "model-internal what-if" label
rendered from the payload. Extends: F2 steps 1–3.

**D83 — A chart renders only where it is the product: `/demo` and the console. No showcase
routes, no chart on the marketing site. Widens D75.** D75 banned *fabricated* forecast
visuals and permitted any fixture-driven chart anywhere. That was too narrow a reading of
the instruction behind it, and F2 built `app/dev/chart` — a six-state gallery of the
component — because PROMPTBOOK F2 Step 3 explicitly specified one and the D75 edit had been
propagated into F0, F1 and F4 but not F2. The rule is now about the surface, not the data:
a viewer on `/demo` or in the console reads a forecast as a measurement and the chart *is*
the measurement; a viewer on the landing page or a specimen is being shown a picture of a
capability, which is the deck's job and which the deck already does. So specimen and
gallery routes are forbidden outright, and the marketing page carries no chart in the hero,
in any section, or as a screenshot frame — a screenshot of the chart is still the chart on
a marketing page, so the console band shows the fleet table, the explanation panel and the
benchmarks page instead, cropped rather than blurred. The hero substitutes type and the
lead-time number for the visual, with the `/demo` CTA first, since that link now does the
work the chart used to. An abstract or "data-ish" graphic standing in for the chart is the
same move wearing a costume and is equally forbidden.

Structural consequence: F2's entire review gate lived on `/dev/chart`, so F2 now has no
browser surface — the only phase in the book without one. Its geometry moves into pure
tested functions in `components/forecast/derive.ts` (cone anchoring, crossing location,
overlay alignment, axis domain) with unit tests standing in for the eye, and its twelve
visual checks move wholesale into the F3 gate, where the chart is on screen in its real
home. F4 now depends on F3 rather than F2, because the marketing screenshots come from a
working `/demo`. `web/app/dev/chart` was moved out of the tree, not deleted, pending the
F3 build. Widens: D75. Extends: PROMPTBOOK F2, F3, F4.

**D84 — A screenshot of the running product is not a rendered chart, and is allowed on the
marketing page. Refines D83.** D83 read the no-charts-on-marketing rule strictly enough to
exclude screenshot frames containing the forecast chart, and proposed cropping the console
band to the fleet table and explanation panel. That was over-broad. The rule's subject is
the frontend *drawing* a forecast outside `/demo` and the console; a static image captured
from a real replay is a picture of the product working, which is what a product screenshot
has always been. The test is provenance, not subject matter: pixels that came from `/demo`
doing its job are evidence, pixels from a component mounted to look impressive are a slide.
Composed frames, touched-up frames, and frames showing a state `/demo` cannot actually
reach remain forbidden under D75's fabrication rule. The F4 console band therefore uses
real `/demo` frames including the chart, with alt text carrying the numbers. The hero stays
type-only — it is the surface the instruction named directly, and a screenshot there would
restore the visual the change was meant to remove. Refines: D83. Extends: PROMPTBOOK F4.

**D85 — `cn()` comes from the `cn` package, not from `clsx` + `tailwind-merge`, and
`lib/utils.ts` is a one-line re-export.** `npx shadcn@latest init` normally writes a
`lib/utils.ts` holding `twMerge(clsx(inputs))` and adds both packages as direct
dependencies. F1 instead installed `cn@0.2.6` — self-described as a "drop-in replacement
for clsx + tailwind-merge" with compiled merge tables — and reduced `lib/utils.ts` to
`export { cn } from "cn"`. `clsx` and `tailwind-merge` remain in the tree only as
transitive deps of other packages; nothing imports them directly.

The substitution was verified behaviourally rather than trusted, because a `cn()` that
concatenates instead of merging is a silent defect: every shadcn component takes a
`className` override, and without conflict resolution both classes survive and the winner
is decided by CSS source order rather than by the override. Conflict resolution is correct
on the cases this codebase depends on — `p-4 p-2 → p-2`, `text-sm text-lg → text-lg`,
`rounded-md rounded-none → rounded-none`, and clsx's conditional/array semantics. The case
that matters most here is arbitrary values carrying CSS variables, since D77 routes every
colour through them: `bg-[var(--bg)] bg-[var(--bg-raised)] → bg-[var(--bg-raised)]` merges
correctly, which is the whole token system's override path.

**The footgun, recorded because it is invisible until it bites.** `cn` ships a `cn build`
step that subsets its merge tables to the classes found by scanning the source, and its own
contract states that classes never seen during that scan "may instead pass through
unmerged". No `cn-tables` file exists in `web/`, so the package is running its full default
tables and the subsetting hazard is inert today. Anyone who later runs `npx cn build` to
shave bundle size inherits it: a class name assembled at runtime by string concatenation
would stop merging, silently, with no build error and no failing test — the override would
simply stop winning in one component. If that step is ever adopted, every dynamically
composed class must go in the safelist, or the merge must move back to `tailwind-merge`,
whose tables are not subsetted. Extends: PROMPTBOOK F1.

**D86 — The console's navigation landmark, current-page state and skip links are declared
by us, because shadcn's sidebar primitives do not provide them.** `SidebarContent` and
`SidebarGroup` render plain `<div>`s and `SidebarMenuButton` surfaces the active item only
as `data-active="true"`, which styles it and announces nothing. The console therefore had
no `navigation` landmark and no current-page state for assistive technology, while looking
correct to a sighted user — the exact shape of failure the Conventions' "colour is never
the only signal" rule guards against, in a different channel. `ConsoleSidebar` now wraps
its menu in `<nav aria-label="Console">` (the marketing shell already had
`<nav aria-label="Main">`), and the active link carries `aria-current="page"` alongside
`data-active`: the first is what a screen reader announces, the second is what styles it,
and both are required or the state is sighted-only. Active matching also moved from
`pathname.startsWith(href)` to exact-or-segment-boundary, so `/hosts` owns
`/hosts/10.0.0.1` without also claiming any sibling route that merely shares the prefix.
A `SkipLink` was added to both shells, targeting `<main id="main-content" tabIndex={-1}>`;
it uses `focus-visible:not-sr-only` rather than staying `sr-only`, because a skip link that
never becomes visible strands a sighted keyboard user on a control they cannot see.
Accessibility is a per-phase gate in the Conventions rather than F7 work, which is why
these were fixed at F1 rather than deferred. Extends: PROMPTBOOK F1.

**D88 — The demo fixture carries the stub's explain() output, keyed per forecast.**
F3's ExplanationPanel needs `window_importance` / `context_windows` / `context_l` —
fields of `GET /api/v1/explain`, which /demo must never call. So
`scripts/make_demo_fixture.py` embeds `StubPredictor.explain()` for every forecast under
a top-level `explanations` map keyed `${host_id}@${origin_ts}` (spelled exactly as the
payload dumps it). Forecast objects themselves stay untouched and Forecast-valid; the
machine-verify loop still validates every one. The panel therefore renders the same
numbers the API would serve, from the fixture. Extends: F3 step 4.

**D89 — The replay's ring buffer is a capped slice over precomputed history; the rAF
clock lives in ReplayController.** The 200-point ring + useRef accumulation pattern is
specified for the live socket, where points ARRIVE. In the replay the entire history is
a static fixture, so per-host observed series are precomputed once and each render takes
`slice(t+1-200, t+1)` — the same 200-point window, no accumulation to get wrong. The
performance-bearing part is unchanged: a single requestAnimationFrame loop with a
millisecond accumulator in ReplayController, at most one state flush per frame however
many windows elapsed (240× advances several windows per flush, never one setState each),
no timers. Measured in headless Chrome at 240×: 16.7 ms median and p95 frame delta.
Extends: F3 steps 2 and 7.

**D90 — Deep links with an explicit `?t=` start paused; state writes back via
history.replaceState.** A rehearsed demo links to the interesting window — auto-playing
past it would defeat the point, so `?t=` implies paused (play is one keypress). Without
`?t=`, the replay auto-plays unless `prefers-reduced-motion`, which never auto-plays.
`?host=&t=&speed=` are validated against the fixture and written back debounced with
`history.replaceState` — no Next navigation, no re-render. Extends: F3 steps 6 and 8.

**D91 — Replay speed constants live in a plain module (`replay-speeds.ts`), not the
client component.** A server component importing values from a `"use client"` module
receives opaque client references, not values — `REPLAY_SPEEDS.includes` threw only in
the production server render. Constants shared across the boundary get a module with no
directive. Extends: F3 step 2.

**D92 — Chart geometry moved out of `ForecastChart` into tested pure functions; Vitest
added.** PROMPTBOOK F2's step 3 requires the chart's geometry to live in testable functions
because Visual invariant 3 removed the specimen route that a human would otherwise have
eyeballed. That requirement was written after F2 had already run, so neither the F2 nor the
F3 agent ever saw it, and the anchoring — the single most important derivation in the
product and the F3 gate's first reject condition — sat inline in the render with no test.
`Y_DOMAIN`, `anchoredCone`, `anchoredObservedLine` and `leadTimeAgreement` now live in
`derive.ts`; `ForecastChart` consumes them. The rendered SVG was diffed before and after to
confirm the refactor changed nothing: the observed line still ends and the cone, both band
edges and the projected mean all still begin at exactly `559.387,140.397` on the `?t=25`
frame. Runner is Vitest (`vitest.config.mts`, `npm test`), chosen over `node --test` because
the `@/` path alias needs a resolver. 40 tests across `derive.test.ts` (behaviour, on cases
chosen to break the functions) and `fixture.test.ts` (the same functions over the real
192-forecast payload). Extends: PROMPTBOOK F2.

**D93 — The reality overlay is a calibration measurement, not a guarantee, and the
escalating host's coverage is 83.5%.** Writing the proof-shot test as "reality always falls
inside the band" failed immediately, which is the correct outcome for the wrong-looking
reason: a 90% band that never misses is too wide to be a claim. Measured over the fixture,
44 of 267 horizon points on `192.168.10.50` fall outside the band (28 below, 16 above) for
83.5% coverage, against 100% on the three quiet hosts; whole-fixture coverage is 95–97% and
degrades smoothly with horizon (97.3% at k=1 to 95.2% at k=6), which is the shape a
widening band should produce. The worst single miss is at origin `14:33:30`, k=4, where the
band `[0.771, 1.0]` met a realised 0.55.

Two consequences. First, the chart is correct and must not change: it draws the x where
reality was, including when reality was outside the cone, and moving or suppressing those
marks would be fabricating the proof. Second, the demo script's beat — "the x marks land on
the predicted ghost — they match" — overstates what this fixture supports on the host the
demo is about, and the miss is near the crossing window the demo pauses on. The honest line
is that most marks land inside a band that widens with horizon, and that where they fall
outside, the chart says so. The test now asserts a coverage floor with the measured number
recorded in a comment, so improvement under a trained model prompts raising the floor and
degradation fails loudly. Extends: PROMPTBOOK F3, Appendix A.

**D92 — The hero's product-visual slot holds typography: the world-model loop.** No
chart (Visual invariant 3), and no illustration or data-ish stand-in either — the PRD
cover's OBSERVE → REPRESENT → LEARN DYNAMICS → IMAGINE → FORECAST strip is set large as
the claim itself. The headline carries the demo replay's lead time (90 s) and the
subhead says where that number comes from. Primary CTA is /demo, above "Get started".
Extends: F4 section 1.

**D93 — PRD §11/§14 copy is reworded so the language-discipline grep passes without
weakening the claim.** Machine verify rejects the literal strings even in negation, so
the marketing page never prints them: the §11 NIDRA row's "what it cannot do" reads
"no cause-and-effect claims, not autonomous, degrades past ~3 minutes", and the §14
scientific position becomes "it projects where behaviour is heading — it does not
explain why … no claims about an adversary's intentions or reasoning". Same boundaries,
same honesty, zero grep hits. Extends: F4 rules; PRD §14 language discipline.

**D94 — Console-band frames are captured from deterministic deep-linked replay states,
and the site footer lives in the (marketing) layout.** The two screenshots are real
frames from /demo at `?t=25` (light, overlay on: risk 0.607, lead 90 s, 26/30 context
windows) and `?t=27` advanced to w28 (dark: risk 0.779, above-threshold row, lead 30 s,
initial_access 57%/lateral 42% at +90 s) — reproducible states, checked into
web/public/screens/ with the on-frame numbers in the alt text. The footer component
renders from the layout, outside <main>, so it is a real contentinfo landmark; the CTA
band stays in the page. Extends: F4 sections 4 and 8.

**D95 — The wiz-clone at `temp-frontend/` is converted to NIDRA in place: same layouts,
new brand, new content, zero third-party marks.** Sixteen sections were rewritten rather
than redesigned — the grid (`grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr]`), the
section spacers, and every breakpoint are unchanged, so the rhythm the clone earned
survives. Four decisions are worth recording.

Section files were renamed to what they now are (`how-wiz-helps` → `how-nidra-works`,
`gartner-quadrant` → `evidence-teaser`, `pink-canvas` → `narrative-carousel`,
`ai-operating-model` → `world-model-stages`, `testimonials` → `use-cases`,
`user-reviews` → `honest-limits`, `analyst-recognition` → `experiments`,
`ai-frontier` → `forecast-console`, `code-to-cloud-demo` → `try-it-demo`,
`logo-marquee` → `stack-row`); `podcast-strip` is deleted. A `grep -ri wiz src/` that
returns nothing is part of the deliverable, and filenames count.

Every captured product screenshot is replaced by drawn SVG/HTML — `ForecastConeSvg`,
`StageDiagram`, `ConsoleMock`, the pipeline and horizon-curve charts. All 71 files under
`public/` were deleted; the only raster that returns is a generated 64×64
`public/seo/favicon.png` (navy rounded square, white "N"), referenced through
`metadata.icons` rather than the App Router `icon.*` file convention so there is a single
declared source. Charts carry "illustrative — pending artifacts/metrics" on their face
rather than in a caption, because a number on a chart is read as a measurement.

Navy is `rgb(27 42 91)` (#1B2A5B) per the mapping plan, not the #1A2A46 sampled off the
logo render — the plan's value is the lighter, bluer navy and it is what the wordmark,
footer mark and the CTA button's label now use.

Tailwind v4's `@theme inline` does **not** emit its custom properties to `:root` — it
inlines them into utilities. Anything needed from an inline `style` or an SVG attribute
(the console donut's `conic-gradient`) must therefore be declared in `:root` first, with
the theme token pointing at it (`--color-observed: var(--observed)`). Verified against
the compiled stylesheet, not assumed.

Three rendering bugs were found by screenshotting the prerendered HTML at 320/390/768/
1024/1440 (`chrome-headless-shell`; plain `--headless` silently lays out at ~485px and
crops, which reads as phantom overflow): the cone's now-rule was hardcoded at 55% width
while the observed series ended at 65%, so the line overshot its own anchor; `NidraLogo`
baked in `text-navy`, which beat the footer's `text-gray-light` on source order and left
the mark invisible on the dark footer; and the three-node diagram needs ~440px, so the
stage card holds one column until `lg` instead of `md`. Extends: F4; CLAUDE.md frontend
conventions.

**D96 — The promptbook's own copy for section 12 fails the promptbook's own
language grep; D93's resolution wins.** Prompt 6.3 specifies the honest-limits card as
"Correlation and temporal prediction — not causal inference", and prompt 7.1 then requires
`grep -ri "causal\|attacker intent\|understands why" src/` to return zero. Both cannot
hold. D93 already settled this for the PRD: machine verify rejects the literal strings even
in negation, so the page never prints them. The card now reads "Correlation and temporal
prediction — no cause-and-effect claims. The data is observational; the counterfactual is a
model-internal what-if." Same boundary, same honesty, zero grep hits. `layout.tsx`'s "Not an
intrusion detector." is untouched — "intrusion detector" is not on the forbidden list, and
the negation is the differentiator the project leads with.

Two further Phase-7 notes. The clone's four `@keyframes` (marquee, marquee-reverse, shimmer,
rotate) all had zero usages but shipped anyway — Tailwind tree-shakes utilities, not
hand-written CSS — so they and their `--animate-*` tokens are gone; the bundle now carries
no `@keyframes` of its own. And the CountUp tiles in section 13 initialised to `0`, so the
served HTML literally asserted "0 features per host state" and "<0 ms max CPU inference"
until the scroll animation ran. They now initialise to the true value and only animate when
the tile starts below the fold, so the markup states 45 / 180s / <300 even with JS off.
Extends: D95; D93.

**D97 — Design pass on `temp-frontend/`: the hero carries an illustration, not a
chart; the narrative slides get the serif-display treatment; the pipeline becomes an
instrument line.** Four changes on direct review feedback.

The hero's forecast chart is replaced by a drawn scene (passive tap → hosts → one host's
line read through a window → a fan of paths past it). This agrees with D92, which already
ruled no chart in the hero slot: a chart at hero scale is unreadable and spends the
page's best real estate on axis furniture. The real `ForecastConeSvg` still appears twice,
in the demo shell and the final CTA, where it has room. A small `src/components/art/`
package now holds the primitives (Cloud, HostNode, Badge, Fan, Beam, Frame) so the hero,
the four carousel scenes and the pipeline share one line-art language instead of three.

The carousel slides move to the reference's structure — serif italic display heading, a
highlighted lead line, body, and a scene on the right — which brings Crimson Pro back as
`--font-display` after D95 dropped it. The four-rounded-rects pipeline becomes a single
wire with instruments on it and pill labels naming what travels between them, aligned to
the four column headings underneath.

Copy was rewritten against the AI-writing checklist. The tells were everywhere: em dashes
in almost every sentence (down from 40+ to 13), rule-of-three lists, "For the first time,
defenders get…", and aphoristic closers ("An honest negative beats a fragile positive").
The hero now leads with the measured number from the Wednesday replay rather than a
restatement of the architecture. Headings lost their slogans: "Forecasting beats
remembering" → "Does it beat guessing?", "Three experiments that keep us honest" → "Three
things that could prove us wrong".

Type and spacing both went up: body copy from 14–16px to 16–20px, section spacers roughly
1.6x, and `--color-gray-dark` darkened from 4.64:1 to 6.22:1 on white, since it carries
most of the secondary copy. Page is 12.8k tall at 1440 (was 10.4k) with no overflow at
390/768/1024/1440. Extends: D92, D95, D96.

**D98 — Review round two: use-case track moves into the page container, "Honest limits"
is cut, the lifecycle strip becomes a band, and the transition formula gets real
subscripts.** The use-case carousel was bleeding to the viewport edge with its first card
flush left and a ragged gap on the right; it now sits inside the standard
`grid-cols-[1fr_min(1310px,…)_1fr]` container like every other section, so both edges line
up and the last card can scroll clear. Each card carries a faint `CardArt` motif behind the
copy at 9% white, echoing what the card says. An earlier attempt used
`px-[max(2rem,calc((100%-1310px)/2))]`, which silently emitted nothing — `calc` needs
spaces around the minus, written as underscores in a Tailwind arbitrary value. The
container grid is the simpler fix and is what shipped.

The "Honest limits" section is removed on request. Its content is not duplicated elsewhere,
so the site no longer states the three-minute ceiling, the CIC-IDS2017-only caveat, or the
frozen-heads argument in the reader's path — the `/demo` console and the docs are now the
only places those limits appear. Flagging because stating limits plainly was a named
differentiator in CLAUDE.md.

`P(S_t+1 | S_t)` renders as real subscripts via `<sub>` rather than KaTeX: one formula does
not justify a 300 KB math renderer, and the markup reads correctly to a screen reader.
Bullets are now `React.ReactNode[]` to allow it. The six-stage lifecycle from the supplied
reference is rebuilt as markup rather than dropped in as the PNG — text in a raster does
not scale, cannot be read aloud, and would have been the site's only bitmap. It sits
full-width under the four columns instead of nested in column three, where six stages
wrapped to three ragged rows. Its `min-w-[640px]` scroller needed `min-w-0` on the grid
item, otherwise the automatic minimum size pushed the whole page to 814 px and broke
mobile.

The hero keeps its NIDRA illustration. The request was for the Wiz artwork that shipped
with the clone; that is another company's copyrighted illustration, it is what D95 purged,
and it would be the one piece of Wiz IP left on a judged submission. The animation asked
for is delivered: the clone's spec described a "subtle float", which in the clone was only
a static `translate-x-3 -translate-y-3`, so `.hero-float` now actually drifts on a 7 s
ease-in-out loop and respects `prefers-reduced-motion`. Extends: D95, D97.

**D99 — `/demo` is built in `temp-frontend/` from a slimmed copy of the `web/` replay
fixture, and the reality overlay states its own coverage.** The console mirrors `web/`'s
structure — demo banner, host list ranked by observed risk, forecast chart, replay
controller, explanation panel, stage readout — rendered in the marketing site's tokens. It
imports the fixture and nothing else: no API client, no websocket, so it renders with every
container stopped.

Three implementation choices. The fixture was copied at 2.4 MB and slimmed to 435 KB by
dropping `predicted_features` (45 floats x 6 horizons x 192 forecasts) which nothing in the
console reads. The chart is hand-drawn SVG rather than Visx: `temp-frontend` carries no
charting dependency and every other figure on the site is drawn the same way, so adding one
for a single chart was not worth it. The x axis holds a fixed `context_L + K` slots with
history growing leftward from the now rule, because sizing it to whatever history exists
made the cone span the whole plot at window 1 and lurch as the replay filled.

Two corrections found while building. The crossing marker was interpolated from where the
projected mean meets the threshold, which puts it *behind* the now rule whenever observed
risk is already above threshold at the origin — a crossing in the past. It now derives from
`lead_time_s`, the backend's answer under the m-consecutive-windows rule, so marker and
badge cannot disagree. And the overlay now prints "N of M marks inside the band" whenever
it is on. This matters because the obvious window to demo is the crossing at t=27, where
coverage is **0 of 6** — measured, not estimated — against 83.5% across the host. D93
already recorded that the fixture does not support "the marks land on the ghost" on this
host near the crossing; the console now says so on its face rather than leaving a presenter
to discover it live. Extends: D93, D94, D97.

**D100 — The demo console runs dark and leads with numbers, not the chart.** Reworked on
review feedback that it read as a chart with widgets around it rather than a security
dashboard. Four KPI tiles sit above everything — earliest warning in seconds, hosts above
threshold, horizon, state width — followed by an alert strip that appears only while some
host is projected to cross, carrying the lead time at 24px. The chart is now one panel
among six: host watch with per-host risk sparklines, predicted stage mix, attack-lifecycle
track showing observed position against projected mass, and SHAP drivers with the saliency
histogram.

The console is dark; the marketing site stays light. They share the signal roles but not
the surfaces, so `globals.css` gains a `--console-*` group for the surfaces and a `*-lit`
set — brighter cuts of observed, projected, threshold, positive and negative that hold up
on a near-black ground. The light roles would have been illegible there, and darkening the
marketing site to match was never on the table.

One thing this cost an hour: the `:root` values for `--console-*` never landed, because the
insertion anchored on `--navy: rgb(26 42 70)` and that value became `rgb(27 42 91)` back in
D96. `str.replace` reported nothing, the `@theme` entries compiled fine, and every
`.bg-console-*` utility resolved to an empty `var()` — so the page rendered white with
white-on-white panels and no error anywhere. Anchoring a blind string replace on a value
that another decision already changed is the failure mode; the fix now asserts the anchor
exists before writing. Extends: D99.

**D101 — The console carries its own light/dark toggle; dark stays the default.** Three
choices — light, dark, match system — in the console header, persisted under
`nidra-console-theme` and applied as `data-console-theme` on the document element. The
palette is two blocks in `globals.css`: the `*-lit` signal roles are recut per ground, dark
for near-black and the base values for white, so the same class names work either way. The
marketing site is untouched and reads none of these tokens.

Default is dark rather than the OS preference. `(prefers-color-scheme: light)` matches when
no preference is set, so following the system would have handed the light console to nearly
every visitor and lost the look the dashboard was designed around. Light and "match system"
are both explicit choices.

Two things this surfaced. Reading the stored choice with `useState` + `useEffect` trips
`react-hooks/set-state-in-effect`, and correctly — `localStorage` is unreadable during SSR,
so the honest shape is `useSyncExternalStore` with a server snapshot of `"dark"` and the
store correcting after hydration. And the pre-paint script that avoids the theme flash sets
an attribute React did not render, which is a real hydration mismatch; `<html>` now carries
`suppressHydrationWarning`, which is what every theme implementation does and what the
attribute being deliberately out-of-band requires. Verified zero hydration warnings after.
Extends: D100.

---

## 2026-09-20

**D102 — Δ = 60 s is the canonical window; the Δ=30 artifacts are frozen under a tag, not
kept live.** The CIC-IDS2017 TrafficLabelling CSVs print minute-resolution timestamps (no
seconds field), so a 30 s window could only ever be populated at the :00 offset: every :30
window was an artificial all-zero state, half of all risk positives were those empty
windows, and persistence error oscillated with parity (ml/reports/NIDRA_REEVALUATION_2026-09-19.md
§4.1, §18). `windowing.window_seconds: 60`, `schema.WINDOW_SECONDS = 60`. Per CLAUDE.md a Δ
change invalidates every trained artifact and recorded metric: the published Δ=30 Run 7
state is tagged `baseline-delta30-run7` with a digest manifest
(ml/experiments/BASELINE_MANIFEST_delta30_run7.json); its processed tables are removed from
the working tree (recoverable with `git checkout baseline-delta30-run7 -- ml/artifacts/processed/`)
because the loader's cache key changed and dead files that no code can read are worse than
none. Overrides: `window_delta: 30` everywhere it was written down (CLAUDE.md key
parameters, ML_README, config).

**D103 — Flow/packet fusion is an outer join over flow-source hosts within the day file's
flow time range.** A CICFlowMeter flow is stamped at its start, so a host with one long-lived
connection starts a flow in one minute and sends packets for many minutes with no new flow
start; the old left join emitted every such minute as an all-zero `is_active=0` row — on
Tuesday 24,377 host-minutes, 39% of the minutes a host was actually transmitting. Those
minutes now carry their packet aggregates (flow-start aggregates legitimately 0,
`active_flow_count` counts flow STARTS). Two restrictions keep the population and extent
defined by the flow records: hosts must appear as a flow source in the day file (labels are
keyed on the flow source), and windows must fall inside the file's own flow time range (three
Friday day files share one full-day PCAP). Measured after the change: on active rows packet
features are populated 99.5–99.7% of the time, flow-start features 51–53%. Cache key gains
`__fuse2`. Extends: IMPLEMENTATION-ML.md §2.4.

**D104 — One cached table per day serves every history/horizon geometry.** The cache key no
longer carries `min_windows_per_host`; the table is the unfiltered state table with the
K-independent `stage_label`, and `risk_label` is re-derived for the configured K while hosts
are filtered to ≥ L+K windows when a split is assembled (`train.pipeline.build_all_splits`).
`labels.risk_threshold_windows` must equal `windowing.horizon_length` (checked). Needed for
the L=15/K=3 vs L=30/K=6 comparison without re-windowing 8 day files per candidate.

**D105 — Validation is a trailing 30% block of EACH training day, and labels are recomputed
inside each split after the cut.** The training days are consecutive, so one trailing block
off the concatenated timeline was just the end of Wednesday: validation held one Heartbleed
episode on one host (54 positives at Δ=30) and every checkpoint, pooling, threshold and
calibration decision rested on it. Per day at 30% (episode-nudged) validation holds Tuesday's
SSH brute force (63 windows) and Wednesday's Heartbleed (21) — two episodes, two families —
and training keeps FTP-Patator and the Wednesday DoS block (139 attack windows). 25% gives the
same split; 20% would drop SSH-Patator back into train and leave validation with Heartbleed
alone. Recomputing `risk_label` per split removes the ≤K train positives that were derived
from validation-block windows. Test (Friday) and holdout (Thursday) are untouched.
Overrides: `val_fraction_of_train_time: 0.15` on the concatenated timeline.

**D106 — The scaler is feature-specific transforms + z-scoring on ACTIVE training rows;
constant and duplicate features are dropped at fit time; FEATURE_ORDER stays 45 wide.** The
shipped RobustScaler was the identity (98% silent rows put both quartiles at 0, sklearn
substituted scale 1) and twelve heavy-tailed features saturated the ±10 clip on 45–90% of
active rows. `schema.FEATURE_TRANSFORMS` declares log1p / asinh / zscore / unit per feature;
statistics are fit on active rows so an all-zero window maps to a distinct fixed point; a
feature constant on active rows (other than `is_active`) or duplicating an earlier one
(|corr| > 0.999) is dropped — output forced to 0, excluded from the dynamics loss and the
state-forecast metrics via `FeatureScaler.model_mask` — and the reason is recorded. The
45-wide schema is a cross-service contract (backend, frontend types, SHAP), so dropping means
"the model never reads it", not "the column disappears". The artifact is plain JSON
(`feature_scaler.json`), no pickled sklearn object, and every fit writes
`preprocessing_audit.{json,md}` next to it. Overrides: IMPLEMENTATION-ML.md §2.5 RobustScaler.

**D107 — Dynamics checkpoints are selected on FREE-RUNNING validation NLL.** The original
criterion was teacher-forced multi-step NLL, which scores a model that is fed the true
future at every step — not the model that is deployed. `train_dynamics.selection_metric:
val_free_running_nll` scores the deterministic K-step rollout (own predictions fed back)
against the true future; free-running MSE, persistence MSE, skill and 90% band coverage per
horizon are logged every epoch. `val_multistep_nll` remains available for reproducing the
Δ=30 artifacts. Loss options `beta_nll`, `mse_aux_weight`, `nonsilent_sample_weight` are
off by default and only turned on by a recorded experiment.

**D108 — Risk head: imbalanced BCE with pos_weight, Gaussian input noise σ=0.3 in scaled
space, selected on validation natural-prevalence AP; stage head selected separately on
validation macro-F1; independent optimizers.** Five recipes were trained on one dynamics
checkpoint (geomA_L15K3, seed 0) and benchmarked on the Δ=60 validation set (head on
observed states at natural prevalence / world-model path): imbalanced+noise 0.655/0.660,
balanced no noise 0.599/0.638, balanced+noise 0.378/0.505, balanced with pos_repeat 5 and
lr 3e-4 0.566/0.601, the Δ=30 recipe 0.630/0.686. With 147 training positives every recipe
peaks within the first epochs; balanced sampling (each positive 20×/epoch) overfits fastest.
The recipe was chosen on the head's own validation statistic, not on the world-model path,
so the head decision does not lean on the dynamics. Validation holds two episodes, so these
are one-seed numbers without a usable cluster interval; that is stated wherever they are
cited. `risk_sampling: balanced` stays available for reproduction.

**D109 — Onset targets are explicit and shared: P(an episode BEGINS within h min | S_t) for
h ∈ {1,3,5,10,15,30}, defined only at origins outside any episode.** `nidra/data/onset.py`
computes episode geometry once for the training arrays and the evaluation set (Task B), so
the two cannot disagree about "onset". A frozen `OnsetHead` (own artifact) is trained on
observed states as the explicit supervised baseline for Task B and reported as
`onset_head_direct`. On CIC-IDS2017 it has 69 pre-onset training windows within 30 min and
reaches validation AP at the prevalence level (≈3e-5 at 5 min): from the current state alone
an onset is not predictable on this data. That is reported as a finding, not tuned away.
Rows inside an episode are never onset positives (an ongoing attack is not a forecast).

**D110 — ATT&CK mapping is a curated table, generated into docs, never learned.**
`nidra/data/attack_mapping.py`: stage bucket → tactic(s) with a per-bucket note on what
network telemetry can resolve; CIC-IDS2017 label → technique(s) of the tool the dataset
authors ran. `docs/ATTACK_MAPPING.md` is generated from it and a test keeps them equal.
The `exfil` bucket contains only DoS/DDoS in this dataset and the table says so; a predicted
stage sequence is emitted as "projected stage sequence (model-internal)".

**D111 — Serving applies the operating point frozen on validation.** `operating_point.json`
(pooling statistic, per-horizon Platt calibration, F1-optimal threshold, all chosen by
`nidra.eval.benchmark --split val --select-operating-point`) overrides the config's pooling
and the legacy `risk_calibration.json` when present. The trajectory pooling is one numpy
implementation shared by the benchmark and the predictor, so the served number is the
measured one. The Δ=30 config's `risk_pooling_quantile: 0.85` was chosen by looking at
test and holdout F1; that path is no longer how a threshold or pooling is chosen.

**D112 — Geometry: L=30 windows (30 min) of history, K=6 windows (6 min) of horizon at
Δ=60 s.** Controlled comparison on the Δ=60 validation benchmark, seed 0, one head recipe
(D108), same evaluation set: L=15/K=3 gave published-label AP 0.682 against 0.686 for
persistence (risk head on S_t) and 0.687 for persistence + learned noise — no world-model
contribution at the risk level; state skill vs persistence 0.47 (ridge two-lag 0.64).
L=30/K=6 gave 0.761 against 0.713 / 0.751, ΔAP over persistence +0.048 [−0.03, +0.28]
(two validation episodes, so the interval is wide), state skill 0.50 (ridge 0.65), and the
same per-horizon risk AP at k ≤ 3 (0.79/0.92/0.78 vs 0.81/0.80/0.78). The longer context
is where the learned mean shows any effect, and the six-minute horizon is the one the
problem statement's lead-time claim needs; the cost is 2× training time and a stage
forecast that decays from 0.66 top-1 at k=1 to 0.01 at k=6 on attack futures (reported).
The Stage-5 loss variants are screened at L=15/K=3 (half the cost per run) and the chosen
variant is then trained at L=30/K=6 for production — the assumption that a loss change
transfers across the two geometries is stated here, not hidden. Also fixed on the same
evidence: quantile pooling at q ≥ 0.85 (the Δ=30 shipped statistic) scores 0.13–0.16
natural-prevalence AP on validation against 0.68–0.76 for mean / P(trajectory > 0.5)
pooling — the tail statistic is dominated by silent hosts under natural prevalence.

**D113 — Dynamics loss: β-NLL with β = 0.5 is kept; the two-lag linear skip, the MSE
auxiliary term and non-silent sample weighting are not.** Screened at L=15/K=3, 20 epochs,
seed 0, one head recipe trained on every split row (D114), identical validation set
(caps 15000 active / 5000 silent). Natural-prevalence AP on the published label for the
world model vs persistence (risk head on S_t) vs persistence + learned noise, and
state-forecast skill vs persistence (ridge two-lag: 0.638):

| variant | world model | persistence | + learned noise | state skill |
|---|---|---|---|---|
| plain NLL | 0.728 | 0.730 | 0.713 | 0.399 |
| β-NLL 0.5 | **0.806** | 0.730 | 0.750 | **0.629** |
| MSE aux 1.0 | 0.793 | 0.730 | 0.752 | 0.593 |
| non-silent weight 3 | 0.813 | 0.730 | 0.756 | 0.519 |
| linear skip | 0.751 | 0.713 | 0.712 | 0.625 |
| linear skip + β-NLL | 0.750 | 0.713 | 0.722 | 0.662 |
| β-NLL + non-silent | 0.749 | 0.730 | 0.741 | 0.649 |

Under the plain NLL the learned mean adds nothing at the risk level (−0.002 vs
persistence) and the state forecast is far below a linear two-lag fit; the failure is the
known heteroscedastic-NLL pathology (the loss lets the model explain the active rows away
with variance instead of fitting their mean). β-NLL fixes the mean (skill 0.63 ≈ ridge) and
is the only variant that improves both axes; the paired episode bootstrap of AP(world
model) − AP(persistence) is +0.077 [+0.005, +0.309] — two validation episodes, so the
interval is what it is. The linear skip matches β-NLL on state skill but not on risk AP;
its combination with β-NLL has the best state skill (0.662) and a lower risk AP (0.750),
so it is left available (`model.transition.linear_skip`) but off. The screen is one seed
per variant; the production run at L=30/K=6 with five seeds is the confirmation.

**D114 — Heads are trained and selected on every observed state of the split.** The
windowed subsample the dynamics need (500k of 2.3M training rows, 50k of 900k validation
rows) is the wrong sample for a function of one state: on the same head, the selection
statistic computed on the 50k weighted validation subsample was 0.72 while the benchmark
set gave 0.42, the gap being three external hosts one sample happened to contain.
`train/head_data.py` makes every row one sample; validation selection is then the exact
natural prevalence with no weights. Effect on the validation benchmark (same dynamics):
head on S_t 0.42 → 0.73 AP, and the head no longer changes by ±0.3 between otherwise
identical runs. The benchmark's active/silent negative caps were raised to 15000 / 5000
for the locked runs for the same reason (a 2000-row silent sample gives one false alarm the
weight of 340 positives).

**D115 — Run 8 production result: the Δ=60 five-seed ensemble ships as-is, with the
natural-prevalence numbers as the only published ones, and the risk-level world-model
claim scoped to what the benchmark supports.** `experiments/runs/production` (24 epochs,
β-NLL, heads on every split row), operating point frozen on validation (median pooling,
threshold 0.718). Test AP 0.058 [0.018, 0.199] against persistence 0.065 (ΔAP −0.007
[−0.026, +0.011]); holdout AP 0.439 [0.000, 0.768] against 0.295 (+0.143 [−0.000,
+0.283]); state-forecast skill vs persistence 0.587 / 0.616 (ridge 0.565 / 0.595). Three
things were decided on that evidence. (1) Nothing is re-tuned: the test result is the
first look at Friday under this protocol and stays the published one; the Bot-C2
breakdown (833 of 946 positives on five workstations, head ROC-AUC 0.37) is reported as
diagnosis beside the headline, never instead of it. (2) The stochastic rollout stays the
served path although the deterministic one scores +0.023 [+0.003, +0.048] higher on test
— the band and P(attack within horizon) need sampled trajectories, and on holdout the
sign reverses (−0.058 [−0.169, +0.014]). (3) The history-aware risk head (a head on the
encoder state at each rollout step, trained on observed pairs and frozen) is recorded as
the next change, not made now: the validation split shares families with training and
cannot select for family transfer, so adopting it would mean choosing it on test. The
published claim becomes: learned dynamics that beat a linear reference at the state level
on every split; risk-level forecasting that helps on Thursday's unseen families and not
on Friday's; no demonstrated advance warning on this dataset. Runs 1–7's scorecard is
withdrawn as a description of the system and kept for provenance.

**D116 — Leave-one-day-out is the family-transfer split; its two runs are recorded, not
averaged, and neither changes the shipped artifacts.** `experiments/runs/lodo_without_wednesday`
(scored on DoS/Heartbleed: world model 0.451, +0.189 [+0.000, +0.249] over persistence, best
system) and `lodo_without_tuesday` (scored on Patator: 0.167 vs 0.155, +0.012 [−0.053,
+0.348]; GBDT 0.768 and ridge 0.652 rank better because the DoS-trained head fires on
Tuesday's benign traffic, and the run's one-episode Heartbleed validation could not select
a head or a threshold — 0.102). The second run is a property of the dataset's day
structure (two attack days in training, one validation episode each way), so LODO can
select a future head architecture only on the day whose validation block the head can
see, and any such selection must be reported with that caveat. Leave-one-episode-out was
not run (five retrains at ~1 h each, and the two LODO runs already cover both training
attack days).

**D117 — CTU-13 enters as a flow-only dataset, and the features it cannot produce are
declared drops rather than zeros.** (2026-09-23) The 13 Argus `.binetflow` captures
(19,976,700 flows, 2.73 GB, CVUT, August 2011) are read through `nidra/data/ctu_load.py`
into the same windowing, labelling and split machinery as CIC-IDS2017. The PCAPs shipped
alongside them are **not** used: the public ones contain only the botnet's own traffic —
the full captures were withheld for privacy — so any packet feature derived from them
would be a perfect label, since only infected hosts would have one. The eleven packet
aggregates are therefore unavailable rather than missing, and with them `iat_max` (Argus
records no per-packet timing) and `d_retrans_rate`. `FEATURE_ORDER` stays 45 wide because
it is a cross-service contract; the exclusions are a named **regime**
(`schema.FEATURE_REGIMES`) applied through the scaler's existing `drop` kind, so an
excluded column is forced to zero in scaled space, left out of `model_mask`, and travels
with the artifact instead of being reconstructed at each call site. `cross_core` (32
features) keeps the six TCP flag ratios and measures the estimator shift
(`data/dataset_shift.py`) rather than assuming it away; `cross_strict` (24) drops all
eight flag-derived columns. Argus `State` is parsed as flags only for TCP rows containing
`_` — `CON`, `INT`, `URP`, `RED` and the other word states are not flag letters and must
never be read as any. The internal-CIDR host filter (147.32.0.0/16) drops 88 of 444,699
botnet flows, 0.02%.

**D118 — CTU's validation split is held-out captures, not a temporal carve out of
training.** (2026-09-23) The default carve put 88% of scenario 1 into validation and left
training with 192 positives against validation's 1225 — a validation block larger in
attack content than the training set, which cannot select anything. Captures 4 and 6 are
held out instead (train 1,2,3; test 8,9,10; holdout 5,13,12; 7 and 11 are too short to
window at Δ=60 and are excluded). Measured before and after: 161/1197 positives becomes
1358/213. The cost is that validation shares families with training, exactly as on
CIC — which is why family transfer is measured by the leave-one-family-out configs
(`config/lofo/`), never by the validation split.

**D119 — The history-aware risk head is a declared component set over the rollout, and
the frozen-head discipline is unchanged.** (2026-09-23) `TrajectoryRiskHead` reads any
subset of {`state`, `hidden`, `delta`, `logvar`} — the predicted state, the encoder's
recurrent summary after ingesting it, the realized backward difference of the trajectory,
and the transition's predicted log-variance. Training pairs still come from **observed**
states only (`train/head_context.py` builds the identical quantities from observed
history, and a test asserts head training never calls `rollout`), so every unit of
forward-looking capability still originates in the transition model. Three consequences
were decided rather than discovered later. (1) There is deliberately no `horizon`
component: under the frozen-head discipline a head only ever sees observed states, where
"how many ingested windows were predictions" is identically zero, so a horizon input
would be constant in training and out of distribution at inference; horizon dependence
enters through `logvar` and the per-horizon Platt calibration. (2) The head reads
`RolloutOutput.realized_deltas()`, the trajectory's own backward difference, not `mus` —
under a stochastic rollout the realized change is mu plus noise, and the training pairs
are observed backward differences. (3) `components=[state]` builds the plain Run 8
`RiskHead`, so the ablation's control is that architecture exactly, not a reimplementation
of it. Variants share one stage-1 checkpoint: `train_heads` loads only the `encoder.*` and
`transition.*` keys and asserts none are missing, which both enables the sharing and makes
a shape mismatch loud.

**D120 — `persistence_rollout` is the strict transition ablation.** (2026-09-23) The
existing `persistence` system is the risk head on S_t, which for a history-aware head is
not the same system minus the transition — it is a different wiring. `state_source` on
`WorldModel.rollout` now selects where the next state comes from (`model` / `persist` /
`truth`), so persistence and the oracle are the *same* forward simulation with only that
one thing swapped: the encoder still advances, the head still gets its history, and the
only thing removed is the predicted change. A test asserts `persistence_rollout`
coincides exactly with `persistence` for a per-state head, so the new system cannot
quietly change Run 8's comparisons.

**D121 — The onset head gains a discrete-time hazard parameterisation, as a controlled
alternative and not a replacement.** (2026-09-23) Run 8's independent per-horizon BCE
could and did report P(onset within 1 min) above P(onset within 30 min). The hazard form
fits bucket j only on rows still at risk in it and reads out
P(within h_m) = 1 − Π(1 − p_j), which is monotone by construction and can use censored
rows ("no onset within 30 minutes" is what was observed, not "no onset ever"). The
architecture is byte-for-byte the same either way — only the loss and the readout differ —
which is what makes the comparison an ablation rather than two models. `independent`
remains the default until the comparison is run.

**D122 — Observation time is distinct windows times Δ, not the range from first to last
timestamp.** (2026-09-23) `false_alarms_per_hour` divides by this, so a span that is too
large flatters the model. `max(ts) − min(ts)` is the observation time only of a split that
is one continuous capture. CIC-IDS2017's splits are single working days and the two
definitions agree to within 0.25% (test 8.05 → 8.07 h, holdout 4.00 → 4.02 h, so Run 8's
false-alarm rates stand). CTU-13's splits are separate captures made on different days of
August 2011, and the idle nights between them are not time anything was watching: the
naive range overstates the test split by 1.66× and the **holdout by 5.02×** (91.03 h of
range over 18.13 h of capture), which would have divided the CTU false-alarm rate by five.
Counting distinct window timestamps assumes nothing about contiguity, cannot double-count
overlapping captures, and does not move with the number of hosts in a window.

**D123 — The risk head reads the encoder's hidden state as well as the state, selected on
CTU-13 validation, and the log-variance is rejected.** (2026-09-23) Six head variants on
one frozen dynamics run, identical scaler and identical rows (`experiments/runs/ctu_heads__*`,
`reports/tables/ctu_head_ablation_val.md`). `state+hidden` is adopted at AP 0.473 against
the Run 8 head's 0.412, and the reason is not the 0.061: with 283 positives and 8 episodes
that is noise. It is the only variant whose margin over `persistence_rollout` — the same
head, the same encoder advance, only the predicted change removed — excludes zero
(+0.033 [+0.0002, +0.055]), the first such interval in the project; its per-horizon AP goes
0.631 → 0.574 where the Run 8 head's goes 0.560 → 0.209 against a flat oracle; its oracle
sits ABOVE it at every horizon (+0.034 pooled) where Run 8's sat below (−0.031), which is
the difference between a head that can use a better state forecast and one that cannot;
and its served operating point is F1 0.60 at 5.7 false alarms per hour against 0.53 at
10.2. `logvar` is rejected on measurement: it raised the observed-state screening metric
(0.373 vs 0.353) and lowered the forecast benchmark (0.396 vs 0.412), and cost 0.037 on top
of `state+hidden`. `delta` cost a further 0.006. Everything here is one dynamics seed and
is reported as screening; three-seed confirmation and the test/holdout read follow, and the
selection is not revisited on them.

**D124 — Pooled AP is not the metric the history-aware head wins on, and that is said
rather than worked around.** (2026-09-23) `state+hidden` scores 0.473 against 0.481 for the
same head applied to the observed origin — a forward simulation that loses to looking at
the present. With the transition disabled the rollout loses as much again, so the cost is
the encoder ingesting synthetic windows at all, not what the transition predicts: the head
is fit on hidden states reached over real observations and asked at inference about hidden
states reached over six of its own. Two consequences. (1) The published composite is a max
over six correlated horizons while the present-state baseline is one clean score, so the
per-horizon Task C table — where the head wins at every k and by 2.7× at k=6 — is the
honest place to read the forecasting claim, and both are reported. (2) The mismatch is a
measurement, so it gets an experiment rather than a caveat: `train_heads.context_noise`
applies the regularizer the state component already carries to the context components,
scaled per component by its own batch standard deviation. It stays inside the frozen-head
discipline — observed inputs, perturbed — and defaults to 0.0.

**D125 — The discrete-time hazard onset parameterisation is not adopted; the onset head
does read the encoder's hidden state.** (2026-09-23) A 2×2 on identical frozen dynamics —
what the head reads × how its horizons are parameterised
(`experiments/runs/ctu_onset_*`). Hazard is identical to independent per-horizon BCE for
the state-only head and slightly worse for the history-aware one (validation AP 0.0007 vs
0.0010 at 3 minutes). Its argument was coherence, which it delivers by construction, but it
fits each bucket only on the rows still at risk in it and the buckets hold 10 / 20 / 20 /
50 / 43 / 120 events: splitting 263 events six ways costs more variance than the
monotonicity is worth. It stays behind `onset.parameterisation` because the argument
returns with a denser dataset. The hidden state is adopted for the onset head on the same
evidence as for the risk head: lift over the base rate goes from 2.1× to 4.1× at 3 minutes
and 2.0× to 3.4× at 5. Both numbers rest on 24 and 40 positive validation origins and none
of it produces a usable warning — 0 of 8 episodes warned before onset under every
configuration — so the published claim is unchanged: **NIDRA has not demonstrated advance
warning.** What CTU-13 establishes is that the ceiling is a data property (10 positive
training origins at one minute out of 1,417,909 eligible) rather than obviously a model
property.

**D126 — Context noise is not adopted: it closes the exposure gap by removing the
transition model's contribution.** (2026-09-23) `train_heads.context_noise` was written
from the measurement in D124 and does exactly what it was designed to do. At σ=0.1 the
`state+hidden` rollout's validation AP rises 0.473 → 0.480, the oracle 0.506 → 0.536 and
the k=6 tail 0.574 → 0.585. It also lifts `persistence_rollout` from 0.440 to 0.490, which
takes the transition ablation margin from +0.033 [+0.0002, +0.055] to −0.010 [−0.075,
+0.020]. The two effects are one effect: a head made robust to a perturbed hidden state is
robust to *which* windows the encoder ingested, so it neither suffers from feeding the
encoder its own predictions nor benefits from those predictions being good. +0.007 AP,
inside the noise of 283 positives, does not buy the only statistically supported transition
signal the project has. σ=0.3 is worse on every column. The knob stays, defaulting to 0.0,
because the diagnosis behind it is correct and a mechanism that closes the gap without
flattening the head's sensitivity to its input is the obvious next thing to try.

**D127 — `ctu_4:c2` and `ctu_4:recon` fail for opposite reasons, and a supervised probe is
how we know.** (2026-09-23) `nidra/scripts/group_separability.py` fits a gradient-boosted
probe on the training captures' rows for an attack STAGE and scores a validation group —
cross-host and cross-capture, so it cannot answer with host identity. Rbot C2 (23 windows)
comes back at ROC **0.443**, below chance: capture 4's C2 does not resemble captures 1–3's
C2 in 32 flow features at Δ=60, and no head architecture reaches it. Rbot recon (17
windows) comes back at ROC **0.970** where NIDRA scores chance — the signal transfers
across hosts and the model is not finding it, on a stage early enough in the kill chain to
be where advance warning would come from. That is the phase's most actionable gap.
Two methodological notes travel with the probe. Its first version scored rows that no fold
could score at a default 0.0, which put every positive at the bottom and produced a
confident ROC of 0.100 for all four groups — a bug that read as a finding; rows a fold
cannot score are now dropped. And every attack group on the CTU validation captures has its
positives on exactly ONE host, so neither the probe nor the model separates "learned the
behaviour" from "learned the host"; the host-leaky within-split probe reaches ROC
0.913–0.996 on all four groups including the one that does not transfer at all, which is
the measurement of what host identity alone buys. The same caveat applies to Run 8's Friday
Bot-C2 result.

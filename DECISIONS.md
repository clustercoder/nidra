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

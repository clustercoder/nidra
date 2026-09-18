# IMPLEMENTATION-Backend.md

**HORIZON — Backend & Platform**
Problem Statement 26153 · NTRO · SIH

This document covers the serving plane: ingest, streaming, inference orchestration, API, persistence, auth, and deployment. It assumes `IMPLEMENTATION-ML.md` produces a working `HorizonPredictor`.

---

## 0. Scope discipline

You are building a SaaS-shaped product to demonstrate seriousness. That is a good decision for judging, and a dangerous one for the build window. The rule that keeps it safe:

**Build the pipeline for real. Build the SaaS shell thin.**

The event-driven pipeline is genuinely load-bearing — it is what makes the architecture diagram true and the scalability claim honest. Multi-tenancy, billing, org management, RBAC matrices, and audit trails are *shell*. Implement enough that they exist and demo cleanly; do not implement enough that they consume a day.

Concretely: one `tenants` table, one `users` table, JWT with a `tenant_id` claim, and every query filtered by it. That is a defensible multi-tenant story in about ninety minutes. Anything more is scope you will regret on day 5.

---

## 1. Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Pydantic schemas shared across services, native WebSocket, OpenAPI generates the frontend types |
| Bus | **Redis Streams** | Consumer groups, replay, backpressure, one container. Kafka costs half a day of broker config for a prototype that processes one PCAP — real cost, zero rubric credit |
| DB | PostgreSQL 16 | Forecasts, episodes, metrics, tenants |
| Cache | Redis (same instance) | Sequence buffers, job status, rate limits |
| Queue workers | Plain Python processes | Not Celery. Redis Streams consumer groups already are the queue; adding Celery is a second, redundant broker |
| Auth | JWT via `python-jose` + `passlib[bcrypt]` | Self-contained, offline |
| Migrations | Alembic | |
| Packaging | Docker Compose | Single `docker compose up`. Fully offline — satisfies the stated air-gap requirement |

**On Kafka:** the architecture slide should say Redis Streams in the prototype, Kafka partitioned by host hash in production, and note that consumer-group semantics are equivalent so it is a broker swap rather than a rewrite. That is a credible claim. A Kafka cluster you never stress-tested is not.

---

## 2. Service topology

Five processes plus two datastores.

```
api          FastAPI — HTTP + WebSocket, the only public surface
ingest       PCAP/CSV parse → publishes raw_events
features     raw_events → windowing + graph scalars → state_vectors
inference    state_vectors → HorizonPredictor → forecasts    [scale-out unit]
persister    forecasts → PostgreSQL
redis        streams + cache
postgres     durable state
```

`api`, `ingest`, `features`, `persister` are single-instance. **`inference` is the scale-out unit** and must be stateless: sequence context arrives on the stream, forecasts leave on the stream, nothing held in process memory between messages. This is what makes the horizontal-scaling claim in the PRD true rather than aspirational.

### Streams

| Stream | Producer | Consumer group | Partition key |
|---|---|---|---|
| `raw_events` | ingest | `features` | — |
| `state_vectors` | features | `inference` | `host_id` |
| `forecasts` | inference | `api`, `persister` | — |

Two consumer groups on `forecasts`: `api` fans out to WebSocket clients, `persister` writes to Postgres. They consume independently — this is the whole reason for using a bus rather than a function call, and it is worth pointing at in the architecture walkthrough.

---

## 3. Shared schema

One package, imported by every service. This prevents the failure mode that eats an afternoon on day 4: feature-name drift between the training pipeline and the SHAP display.

`horizon_common/schemas.py`:

```python
from pydantic import BaseModel, Field
from datetime import datetime

SCHEMA_VERSION = "1.0"

class StateVector(BaseModel):
    tenant_id: str
    host_id: str
    window_ts: datetime           # window start, UTC, aligned to 30s
    features: dict[str, float]    # keys == FEATURE_ORDER, validated
    schema_ver: str = SCHEMA_VERSION

class SignalAttribution(BaseModel):
    name: str
    shap_value: float
    direction: str                # "up" | "down"
    display: str                  # "SYN ratio rising sharply"

class HorizonPoint(BaseModel):
    k: int
    ts: datetime
    p_compromise: float
    ci_low: float
    ci_high: float
    stage_dist: dict[str, float]
    predicted_features: dict[str, float]

class Forecast(BaseModel):
    tenant_id: str
    host_id: str
    origin_ts: datetime           # time t — nothing after this was observed
    horizons: list[HorizonPoint]
    lead_time_s: float | None
    observed_stage: str
    observed_risk: float
    top_signals: list[SignalAttribution]
    driving_window: int
    model_version: str
    schema_ver: str = SCHEMA_VERSION
```

`origin_ts` is the contract that guarantees causality. Every consumer can verify that a forecast for `t+3` carries `origin_ts = t`. Assert it in the persister.

---

## 4. Redis Streams

A thin wrapper, written once.

```python
class Bus:
    def __init__(self, redis, stream, group, consumer):
        self.r, self.stream, self.group, self.consumer = redis, stream, group, consumer

    async def ensure_group(self):
        try:
            await self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e): raise

    async def publish(self, model: BaseModel):
        await self.r.xadd(self.stream, {"payload": model.model_dump_json()},
                          maxlen=100_000, approximate=True)

    async def consume(self, count=32, block_ms=2000):
        while True:
            resp = await self.r.xreadgroup(self.group, self.consumer,
                                           {self.stream: ">"}, count=count, block=block_ms)
            for _, messages in resp or []:
                for msg_id, fields in messages:
                    yield msg_id, fields[b"payload"]
```

**Acknowledge only after successful processing.** On exception, do not ack — the message stays pending and is recoverable via `XAUTOCLAIM`. This is a genuine at-least-once guarantee and worth one sentence in the architecture walkthrough.

`maxlen` with `approximate=True` caps memory. For a demo replaying one PCAP, 100k entries is generous.

---

## 5. Ingest service

Handles upload, validation, parsing, and progress reporting.

```
POST /api/v1/ingest  (multipart)
  → validate extension + size
  → write to /data/uploads/{tenant}/{job_id}/
  → create ingest_jobs row (status=queued)
  → XADD ingest_jobs stream
  → return {job_id}
```

The worker:

1. **PCAP path** — `tshark -T fields` subprocess (fields exactly as in `IMPLEMENTATION-ML.md` §2.2), streamed line-by-line into parquet chunks. Never load the whole capture into memory.
2. **CSV path** — chunked pandas read, strip leading spaces from column names, drop mid-file duplicate header rows.
3. Publish per-flow and per-packet records to `raw_events` in **timestamp order**. Out-of-order publication silently corrupts windowing.

**Progress:** write `{processed, total, status}` to a Redis key `job:{job_id}` on every chunk. The API polls this. Do not stream progress through the bus — it pollutes the event log with non-domain events.

**Replay pacing.** The demo needs traffic to arrive at a controllable rate, not instantly.

```python
class ReplayClock:
    """Maps capture time to wall-clock at a speed multiplier."""
    def __init__(self, first_ts, speed=60.0):   # 60x: 30s window every 0.5s
        self.first_ts, self.speed = first_ts, speed
        self.start = time.monotonic()

    async def wait_until(self, event_ts):
        target = (event_ts - self.first_ts).total_seconds() / self.speed
        delay = target - (time.monotonic() - self.start)
        if delay > 0: await asyncio.sleep(delay)
```

Expose `speed` as a request parameter. This is what makes the demo watchable, and it is a five-line class — build it early rather than faking pacing in the frontend.

**Limits:** 2 GB upload cap, extension allowlist `{.pcap, .pcapng, .csv}`, and reject on magic-byte mismatch. Never pass a user-supplied string into the tshark command line; build `argv` as a list and pass only the validated path.

---

## 6. Features service

Consumes `raw_events`, emits `StateVector`.

The stateful parts — window accumulation, the per-host previously-seen-peer set for `new_peer_count`, the previous window's values for deltas and slopes — live in Redis, keyed by `{tenant}:{host}`, with a TTL. Keeping them in Redis rather than process memory is what lets this service restart without corrupting the feature stream.

```
key                              type    contents
feat:win:{tenant}:{host}:{ts}    hash    accumulating counters for the open window
feat:peers:{tenant}:{host}       set     previously contacted peers   TTL 24h
feat:prev:{tenant}:{host}        hash    last 3 windows, for deltas and slopes
```

**Window closing.** A window closes when an event arrives with `window_ts` greater than the open one, or when a watchdog timer expires. The watchdog matters: without it, the last window of a capture never closes and the final forecast never fires — which is exactly the window your demo needs.

**Emit empty windows.** A host that goes silent still has state. Emit a zero-valued vector with `is_active=0`. Dropping silent windows breaks the time axis and corrupts every lead-time measurement downstream.

Validate `set(features.keys()) == set(FEATURE_ORDER)` before publishing. Fail loudly. A missing feature discovered here costs a second; discovered in a SHAP plot on day 5 it costs an evening.

---

## 7. Inference service

The scale-out unit. Stateless by construction.

```python
class InferenceWorker:
    def __init__(self):
        self.predictor = HorizonPredictor(...)   # loaded once at startup
        self.redis = ...

    async def handle(self, sv: StateVector):
        key = f"seq:{sv.tenant_id}:{sv.host_id}"
        await self.redis.rpush(key, sv.model_dump_json())
        await self.redis.ltrim(key, -L, -1)          # keep exactly L windows
        await self.redis.expire(key, 3600)

        raw = await self.redis.lrange(key, 0, -1)
        if len(raw) < L:
            return                                    # not enough context yet

        states = np.array([...])                      # [L, 45], FEATURE_ORDER order
        result = self.predictor.forecast(states, sv.host_id, sv.window_ts)
        await self.bus.publish(Forecast(**result))
```

The sequence buffer lives in Redis, not in the worker. That is the entire reason any worker can process any host, and therefore the entire basis of the scale-out claim. Say it that way when someone asks.

**Load the model once**, at process start, not per message. Set `torch.set_num_threads(2)` — PyTorch defaults to all cores and four workers on one machine will thrash.

**Backpressure:** if forecast latency exceeds the window arrival rate, `XPENDING` grows. Log it and expose it on `/health`. Under demo load it should stay near zero; showing that number during a technical Q&A is more convincing than any claim about scalability.

Target: under 300 ms for K=6 with 5 models × 200 samples on CPU.

---

## 8. API

### Surface

```
POST   /api/v1/auth/register              email, password, org name → tenant + user
POST   /api/v1/auth/login                 → access + refresh JWT
POST   /api/v1/auth/refresh

POST   /api/v1/ingest                     multipart upload → job_id
GET    /api/v1/ingest/{job_id}            status + progress
GET    /api/v1/ingest                     list jobs for tenant

WS     /api/v1/stream/forecast            live forecasts, filterable by host
GET    /api/v1/forecasts                  paginated history
GET    /api/v1/forecasts/{host}/{ts}      single forecast, full detail
GET    /api/v1/hosts                      hosts with current risk, sortable

GET    /api/v1/explain/{host}/{ts}        SHAP + saliency + flagged flows
POST   /api/v1/counterfactual             clamp feature, re-roll
GET    /api/v1/benchmarks                 baselines, ablations, horizon curves
GET    /api/v1/model                      version, config, training provenance

GET    /health                            liveness + stream lag
GET    /metrics                           Prometheus
```

`/api/v1/benchmarks` and `/api/v1/model` are not filler. They let the frontend render the evaluation results as a real product page rather than a slide, and they make the reproducibility deliverable visible in the product itself.

### WebSocket

```python
@app.websocket("/api/v1/stream/forecast")
async def stream(ws: WebSocket, token: str = Query(...)):
    tenant = verify_jwt(token).tenant_id      # browsers can't set WS headers → query param
    await ws.accept()
    hosts: set[str] | None = None             # None = all
    async def recv():                         # client can update its filter mid-stream
        while True:
            msg = await ws.receive_json()
            if msg["type"] == "filter": hosts = set(msg["hosts"])
    asyncio.create_task(recv())
    async for _, payload in bus.consume():
        f = Forecast.model_validate_json(payload)
        if f.tenant_id == tenant and (hosts is None or f.host_id in hosts):
            await ws.send_text(payload)
```

Give each socket its own consumer name so two browser tabs both receive everything. Heartbeat every 30 s; drop on two missed pongs.

### Auth

JWT, HS256, 30-minute access token, 7-day refresh. `tenant_id` in the claims. One dependency:

```python
async def current_tenant(token = Depends(oauth2_scheme)) -> str: ...
```

**Every** query filters on it. Not "most" — every. A tenant-leak found live during judging is unrecoverable; write one integration test that authenticates as tenant B and asserts 404 on tenant A's forecast.

Rate limit with a Redis sliding window: 100 req/min general, 5/hour on ingest.

---

## 9. Database

```sql
CREATE TABLE tenants (
  id UUID PRIMARY KEY, name TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE users (
  id UUID PRIMARY KEY, tenant_id UUID REFERENCES tenants(id),
  email CITEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  role TEXT DEFAULT 'analyst'
);

CREATE TABLE ingest_jobs (
  id UUID PRIMARY KEY, tenant_id UUID REFERENCES tenants(id),
  filename TEXT, kind TEXT, status TEXT, progress REAL DEFAULT 0,
  error TEXT, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE forecasts (
  id BIGSERIAL PRIMARY KEY,
  tenant_id UUID NOT NULL, host_id TEXT NOT NULL,
  origin_ts TIMESTAMPTZ NOT NULL,
  observed_risk REAL, observed_stage TEXT,
  max_p_compromise REAL, lead_time_s REAL,
  payload JSONB NOT NULL,              -- full Forecast
  model_version TEXT,
  UNIQUE (tenant_id, host_id, origin_ts)
);
CREATE INDEX ON forecasts (tenant_id, origin_ts DESC);
CREATE INDEX ON forecasts (tenant_id, host_id, origin_ts DESC);
CREATE INDEX ON forecasts (tenant_id, max_p_compromise DESC)
  WHERE max_p_compromise > 0.5;        -- partial index: the alert query

CREATE TABLE episodes (
  id UUID PRIMARY KEY, tenant_id UUID, host_id TEXT,
  started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ,
  peak_risk REAL, stages TEXT[], first_alert_at TIMESTAMPTZ,
  ground_truth_onset TIMESTAMPTZ       -- when replaying a labelled dataset
);

CREATE TABLE benchmark_runs (
  id UUID PRIMARY KEY, model_version TEXT, dataset TEXT,
  split TEXT, metrics JSONB, created_at TIMESTAMPTZ DEFAULT now()
);
```

Storing the full `Forecast` as JSONB with hot fields promoted to columns is the right trade here: the promoted columns carry every query the UI makes, and the JSONB means schema evolution costs nothing.

`episodes` is what makes lead time visible as a product feature rather than a metric in a notebook. An episode opens when risk crosses threshold and stays open until it drops for `n` windows. With `ground_truth_onset` populated during labelled replay, the UI can render "warned 90 s before onset" as a fact about that episode.

The unique constraint on `(tenant_id, host_id, origin_ts)` makes the persister idempotent — `ON CONFLICT DO NOTHING` — which matters because at-least-once delivery means duplicates will happen.

---

## 10. Deployment

`docker-compose.yml`, six services. Everything offline, no external network calls.

```yaml
services:
  redis:      { image: redis:7-alpine }
  postgres:   { image: postgres:16-alpine }
  api:        { build: ., command: uvicorn horizon.api:app --host 0.0.0.0 --port 8000 }
  ingest:     { build: ., command: python -m horizon.services.ingest }
  features:   { build: ., command: python -m horizon.services.features }
  inference:  { build: ., command: python -m horizon.services.inference }   # scale here
  persister:  { build: ., command: python -m horizon.services.persister }
  web:        { build: ./web, command: node server.js }
```

Model weights and the scaler mount read-only from `./artifacts`. Uploaded captures go to a named volume. Healthchecks on redis and postgres with `depends_on: condition: service_healthy` — without them the workers race the datastores on cold start and you spend twenty minutes debugging a nonexistent bug.

Scale-out demo: `docker compose up -d --scale inference=3`. Show `XPENDING` staying flat. That is the scalability claim, demonstrated in ten seconds.

**Air-gap check** — worth running before the demo:
```bash
docker compose up -d
docker network disconnect bridge horizon-api-1
# everything still works
```

Add a `make demo` target that seeds a tenant, uploads the prepared Wednesday slice, and starts replay at 60×. On demo day you want one command, not a sequence you might fumble.

---

## 11. Build order

| Day | Work | Exit criterion |
|---|---|---|
| **1** | Repo skeleton, `horizon_common` schemas, Docker Compose, Postgres + Alembic, auth | `docker compose up` healthy; register/login returns a JWT |
| **2** | Bus wrapper, ingest service, tshark subprocess, replay clock | Upload a PCAP → `raw_events` populates at controlled rate |
| **3** | Features service, Redis window state, `StateVector` validation | `state_vectors` carries valid 45-feature vectors for real hosts |
| **4** | Inference service, sequence buffer, persister, WebSocket | Live forecasts reach a `wscat` client end-to-end |
| **5** | Explain, counterfactual, benchmarks, flagged flows, hardening | Every endpoint returns real data; tenant-isolation test passes |

The frontend team can develop against a mock WebSocket from day 1. Publish a `forecast.example.json` on day 1 so they are never blocked — this is worth doing before anything else.

---

## 12. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Workers idle, stream has entries | Consumer group created after messages | `XGROUP CREATE ... id="0"` on startup, idempotent |
| Duplicate forecasts | At-least-once redelivery | Unique constraint + `ON CONFLICT DO NOTHING` |
| Last window never forecasts | No watchdog on window close | Timer-based close; the demo's final window depends on it |
| WebSocket drops at ~60 s | Idle proxy timeout | 30 s heartbeat |
| Inference latency climbing | PyTorch grabbing all cores per worker | `torch.set_num_threads(2)` |
| Features all zero | Empty windows dropped upstream | Emit zero rows with `is_active=0` |
| Tenant sees another's data | A query missing the filter | Integration test; treat as release-blocking |
| Cold start crash loop | Workers race datastores | Healthchecks + `service_healthy` |

---

## 13. What "scalable" means in the pitch

Be precise, because a vague claim invites a question you cannot answer and a precise one closes the topic.

**True today:** stateless inference workers, sequence state externalised to Redis, consumer groups partitioned by `host_id`, at-least-once delivery with idempotent writes, a measured throughput figure from our own pipeline, horizontal scale demonstrated live with `--scale inference=3`.

**Stated as the production path, not as built:** Kafka partitioned by host hash (broker swap, equivalent consumer-group semantics), TimescaleDB hypertables on `origin_ts`, SPAN-port or IPFIX collector replacing file replay, per-tenant model versions.

**Do not claim:** a tested Kafka cluster, benchmarked multi-node throughput, or production hardening you have not done. An honest measured number with a stated scale-out path is credible. An unverified architecture claim, probed once, is not.

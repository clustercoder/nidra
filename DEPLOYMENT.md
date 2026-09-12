# Deploying NIDRA — backend on Render, frontend on Vercel

A real, publicly reachable deployment: the backend (API + all four workers + Postgres +
Redis) on [Render](https://render.com), the frontend (`web/`) on
[Vercel](https://vercel.com). Both give you HTTPS on a public URL with no server to
patch, no nginx config, and no certbot renewal cron — the thing the previous version of
this file asked you to hand-roll on a bare Linux VM.

## 0. Prerequisites

- A Render account (free to create; the `starter` plans this Blueprint uses are Render's
  cheapest paid tier — Render's free tier does not support background workers, which
  this backend needs four of).
- A Vercel account (free tier is enough for the frontend).
- This repo pushed to GitHub, with Render and Vercel both given access to it (each asks
  you to authorize a GitHub App on first use — via their dashboards, not the CLI).
- Nothing to install locally to *deploy*: both platforms build from your GitHub repo on
  their own infrastructure. The `render` and `vercel` CLIs are optional, only useful for
  managing an existing deployment from a terminal afterwards.

The trained model weights are no longer something you copy onto a server by hand: the
5-seed ensemble + scaler (~6MB) now live in this repo under `deploy/model/` (see
`deploy/model/README.md`) and are baked into the Docker image at build time. Skip
straight to step 1.

## 1. Deploy the backend to Render via the Blueprint

This repo's `render.yaml` defines the whole backend as one Render Blueprint: a managed
Postgres database, a managed Redis (Key Value) instance, the API service, and three
background workers (`features`, `inference`, `persister`).

1. In the Render dashboard: **New → Blueprint**, pick this repo, branch `main`.
2. Render parses `render.yaml` and shows every resource it's about to create
   (`nidra-postgres`, `nidra-redis`, `nidra-api`, `nidra-features`, `nidra-inference`,
   `nidra-persister`). Click **Apply**.
3. Render builds the Docker image once and reuses it for `nidra-api` and all three
   workers (same `Dockerfile`, different `dockerCommand` — the same pattern
   `docker-compose.yml`'s `x-app` anchor uses locally). The first build takes several
   minutes (torch, tshark, scikit-learn, shap).

`predictor.impl` is switched on for you: `render.yaml` sets `NIDRA_PREDICTOR_IMPL=nidra`
on `nidra-api` and `nidra-inference` (the two services that call the predictor), so the
trained ensemble baked into the image is what actually serves forecasts from the first
deploy — not the stub.

### Why `ingest` isn't its own Render service

`docker-compose.yml` runs `ingest` as a fifth container sharing a Docker volume with
`api`: `api/ingest.py` writes an uploaded capture to disk, `services/ingest/worker.py`
reads it back from that same path. Render web services and background workers are
separate containers with no shared filesystem — a Render Disk attaches to exactly one
service. Rather than route uploads through an external object store for what is, today,
a single-instance deployment, `render.yaml` keeps `ingest` co-located inside the
`nidra-api` service: `scripts/render_api_start.sh` starts `python -m services.ingest` in
the background and runs `uvicorn` in the foreground (so Render's health check and
`$PORT` binding land on the process it expects). A 1GB Render Disk is mounted on
`nidra-api` at `/var/lib/nidra/uploads` so an in-progress upload survives a restart.

`features`, `inference`, and `persister` don't touch the upload directory — only
Redis, Postgres, and the model weights already baked into the image — so they stay
separate, independently scalable Render background workers.

## 2. Finish the two secrets Render can't fill in for you

Two env vars on `render.yaml` are marked `sync: false` — Render creates the field but
leaves it blank, because their value isn't knowable until other things exist:

**`NIDRA_POSTGRES_URL`** (on `nidra-api`, `nidra-features`, `nidra-inference`,
`nidra-persister`): open `nidra-postgres` in the Render dashboard, copy the **Internal
Database URL**, and change its scheme from `postgresql://` to `postgresql+asyncpg://`
(the app's async engine needs the driver named explicitly; Render's own URL doesn't
know your app is async). Paste the result into each of the four services' environment
tab. Internal URLs only work between services in the same Render region — that's why
every service in `render.yaml` pins `region: oregon`.

**`NIDRA_CORS_ORIGINS`** (on `nidra-api` only): leave this blank until step 4, once you
know the Vercel URL, then come back and set it — see step 4.

Redis needs no manual step: `render.yaml` wires `NIDRA_REDIS_URL` via `fromService`
automatically, and Render's Redis connection string already carries the right scheme.

`NIDRA_SECRET_KEY` also needs no manual step (`generateValue: true` — Render generates
a random one on first deploy and keeps it stable across redeploys).

Each service **Manual Deploy** after you save its env vars.

## 3. Apply migrations

Fresh database, so the schema doesn't exist yet. Open `nidra-api`'s **Shell** tab in the
Render dashboard (a terminal inside the running container) and run:

```bash
alembic upgrade head
```

## 4. Deploy the frontend to Vercel

From the `web/` directory (or point Vercel's dashboard import at this repo with `web/`
as the root directory):

```bash
cd web
npm install -g vercel   # if not already installed
vercel login
vercel link             # links this directory to a Vercel project
vercel env add NEXT_PUBLIC_API_URL production
# paste nidra-api's Render URL, e.g. https://nidra-api.onrender.com
vercel --prod
```

If importing via the Vercel dashboard instead: set **Root Directory** to `web`,
**Framework Preset** to Next.js (auto-detected), and add the same
`NEXT_PUBLIC_API_URL` environment variable under Project Settings → Environment
Variables before the first production deploy.

Once Vercel gives you the production URL (`https://<project>.vercel.app`, or a custom
domain if you attach one under Project Settings → Domains), go back to `nidra-api` in
Render and set the `NIDRA_CORS_ORIGINS` env var left blank in step 2 to that exact
origin (scheme + host, no trailing slash — e.g. `https://nidra.vercel.app`; comma-
separate more than one, such as a preview-deployment origin alongside production).
Manual Deploy `nidra-api` again to pick it up. Skipping this step doesn't break the
API — it means every browser request from the frontend is silently blocked by CORS
before it reaches a route, while `curl`/Postman keep working, which is a confusing
failure mode to debug blind.

## 5. Smoke-test end to end

- `https://<nidra-api>.onrender.com/health` returns `{"status": "ok", ...}`.
- `https://<your-vercel-domain>` loads the frontend.
- The frontend's forecast/demo views successfully call the Render API (check the
  browser network tab for CORS errors if step 4's origin doesn't match exactly).
- The frontend's live console view receives forecasts over the WebSocket route
  (`wss://<nidra-api>.onrender.com/api/v1/ws/<host>`) — Render terminates TLS and
  proxies WebSocket upgrades on its own, no nginx config needed.

## Notes on Render's free/starter tier

- A `starter`-plan web service and background workers do not spin down when idle (that
  behavior is specific to Render's *free* web service tier, not used here — the free
  tier also doesn't support background workers at all, which this backend needs four
  of). If cost matters more than always-on latency, some workers (e.g. `persister`) can
  be scaled to `plan: free` in `render.yaml` individually, at the cost of a cold-start
  delay after idle periods.
- Scale `inference` horizontally the same way `docker-compose.yml`'s comment describes
  locally: increase its instance count in the Render dashboard (Settings → Scaling).
  It's stateless — sequence buffers live in Redis — so this is safe with no code change.

## Rollback

- Render keeps every previous deploy; **Manual Deploy → Rollback to this deploy** on any
  service reverts it without touching Postgres/Redis data.
- To roll back to the stub predictor without a full rollback, set `NIDRA_PREDICTOR_IMPL`
  back to `stub` on `nidra-api` and `nidra-inference` and redeploy both.
- Nothing here is destructive to data: Render's managed Postgres and Redis persist
  independently of the app services' deploy history.

## Local development is unaffected

`docker compose up -d` still runs the whole stack against `config/default.yaml`'s
`predictor.impl: stub` default exactly as before — none of the above changes what a
local `make demo` / `make e2e` does. `NIDRA_PREDICTOR_IMPL`, `NIDRA_CORS_ORIGINS`, and
`deploy/model/` are additive: unset, they're a no-op locally.

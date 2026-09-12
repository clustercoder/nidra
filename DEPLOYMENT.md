# Deploying NIDRA — free: self-hosted backend via Cloudflare Tunnel, frontend on Vercel

**If Render (or Fly.io, Railway, ...) just asked you for a card:** that's expected. This
backend is five long-running processes plus Postgres and Redis — nothing about that is
free on a managed PaaS in 2026; even Render's Blueprint provisions billed resources
(managed Postgres, managed Redis, background workers) the moment it applies, which is
why it asks up front. `render.yaml` and the Render path are still in this repo and still
documented below, but as an *optional, paid* alternative — not the primary path.

The primary path costs nothing and asks for no card anywhere: run the existing
`docker-compose.yml` stack on a machine you already control (this Mac, or the Kali VM
under UTM — anywhere Docker runs, for as long as you want it up), and expose it to the
internet with [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/),
which Cloudflare made free with no usage limits and no credit card in July 2026. It also
solves the actual problem with the old "Linux VM + nginx + certbot" approach without a
tunnel: a home machine (or a VM on one) usually has no public IP and sits behind NAT —
`cloudflared` opens an *outbound* connection to Cloudflare's edge, so nothing needs to
be port-forwarded and no static IP or domain is required to get a real `https://` URL.

## 0. Prerequisites

- Docker + Docker Compose, already working in this repo (`docker compose ps` should show
  `postgres`/`redis` healthy if you've been following along).
- `cloudflared` — installed already in this environment (`brew install cloudflared`
  elsewhere: no account, no card, just a binary).
- A Vercel account for the frontend (Hobby tier, free, no card).
- Nothing else. No cloud account is required for the backend at all with the quick-tunnel
  option below.

## 1. Bring up the backend

```bash
docker compose up -d --build
docker compose exec api alembic upgrade head
curl http://localhost:8000/health   # {"status": "ok", ...}
```

To actually serve real forecasts instead of the stub, switch the predictor on — the
trained 5-seed ensemble is already baked into the image (`deploy/model/`, see
`deploy/model/README.md`), so this is one env var, not a file copy:

```bash
NIDRA_PREDICTOR_IMPL=nidra docker compose up -d --build api inference
```

Add `NIDRA_PREDICTOR_IMPL=nidra` to `.env` instead if you want this to survive future
plain `docker compose up` calls without re-exporting it each time.

## 2. Expose it publicly with Cloudflare Tunnel

**Quickest — no Cloudflare account, no domain, gives you a URL right now:**

```bash
cloudflared tunnel --url http://localhost:8000
```

This prints a `https://<random-words>.trycloudflare.com` URL within a few seconds.
Traffic to it — including WebSocket upgrades on `/api/v1/ws/` — is proxied straight to
your local `api` container. Leave this process running (it's the tunnel); closing it
tears the URL down. Good enough for a demo; the URL is different every time you start
it, so don't hardcode it anywhere durable.

**Stable — a URL that survives restarts, needs a free Cloudflare account + a domain on
Cloudflare's nameservers** (a domain you already own, or a free one from any registrar
pointed at Cloudflare's nameservers — Cloudflare itself doesn't sell domains, just DNS):

```bash
cloudflared tunnel login                       # opens a browser, authorizes once
cloudflared tunnel create nidra-api
cloudflared tunnel route dns nidra-api api.yourdomain.com
cloudflared tunnel run --url http://localhost:8000 nidra-api
```

Run the last command under something that keeps it alive across reboots — `pm2`,
a `launchd`/`systemd` unit, or simply a `screen`/`tmux` session on the Kali VM if this
is a short-lived hackathon demo rather than a long-running service.

## 3. Deploy the frontend to Vercel

From the `web/` directory (or point Vercel's dashboard import at this repo with `web/`
as the root directory):

```bash
cd web
npm install -g vercel   # if not already installed
vercel login
vercel link             # links this directory to a Vercel project
vercel env add NEXT_PUBLIC_API_URL production
# paste the Cloudflare Tunnel URL from step 2, e.g. https://random-words.trycloudflare.com
vercel --prod
```

If importing via the Vercel dashboard instead: set **Root Directory** to `web`,
**Framework Preset** to Next.js (auto-detected), and add the same
`NEXT_PUBLIC_API_URL` environment variable under Project Settings → Environment
Variables before the first production deploy.

## 4. Turn on CORS for the Vercel origin

Once Vercel gives you the production URL (`https://<project>.vercel.app`, or a custom
domain under Project Settings → Domains):

```bash
NIDRA_CORS_ORIGINS=https://<project>.vercel.app docker compose up -d api
```

(comma-separate more than one origin, e.g. a preview-deployment URL alongside
production). Skipping this doesn't break the API — it means every browser request from
the frontend is silently blocked by CORS before it reaches a route, while `curl`/Postman
keep working, which is a confusing failure mode to debug blind.

## 5. Smoke-test end to end

- The Cloudflare Tunnel URL's `/health` returns `{"status": "ok", ...}`.
- `https://<your-vercel-domain>` loads the frontend.
- The frontend's forecast/demo views successfully call the tunnel URL (check the
  browser network tab for CORS errors if step 4's origin doesn't match exactly).
- The frontend's live console view receives forecasts over the WebSocket route —
  Cloudflare proxies the upgrade automatically, no nginx config needed.

## Rollback

Nothing here is destructive to data: `docker compose down` stops the backend without
touching `pgdata` (a named volume); `docker compose up -d` brings it back. To roll back
to the stub predictor, drop `NIDRA_PREDICTOR_IMPL` (or set it to `stub`) and
`docker compose up -d api inference` again.

---

## Optional, paid alternative: Render

`render.yaml` in this repo still defines the same backend as a Render Blueprint —
managed Postgres, managed Redis, the API service, and three background workers — for
anyone who'd rather not keep a personal machine running and doesn't mind linking a card
(Render's own free tier doesn't support background workers or managed Redis at all,
which is why applying this Blueprint prompts for billing).

1. In the Render dashboard: **New → Blueprint**, pick this repo, branch `main`. Render
   parses `render.yaml` and shows every resource it's about to create — Apply.
2. Two env vars are marked `sync: false` because Render can't fill them in for you:
   - **`NIDRA_POSTGRES_URL`** (on all four app services): copy `nidra-postgres`'s
     Internal Database URL and change its scheme from `postgresql://` to
     `postgresql+asyncpg://` (the app's async engine needs the driver named explicitly).
   - **`NIDRA_CORS_ORIGINS`** (on `nidra-api` only): set once you have the Vercel URL
     (same value as step 4 above).
   Manual Deploy each service after saving its env vars.
3. Open `nidra-api`'s **Shell** tab and run `alembic upgrade head` against the fresh
   database.
4. Point `NEXT_PUBLIC_API_URL` (Vercel) at `nidra-api`'s `.onrender.com` URL instead of
   a Cloudflare Tunnel URL — everything else in steps 3–5 above is identical.

`ingest` is co-located inside `nidra-api` here (`scripts/render_api_start.sh`) rather
than a separate service: `api/ingest.py` writes an uploaded capture to disk and
`services/ingest/worker.py` reads it back from the same path, which requires a shared
filesystem that Render's separate services don't have (a Render Disk attaches to
exactly one service) — self-hosting via docker compose doesn't have this problem since
every container already shares the host's Docker volumes.

## Local development is unaffected

`docker compose up -d` still runs the whole stack against `config/default.yaml`'s
`predictor.impl: stub` default exactly as before — none of the above changes what a
local `make demo` / `make e2e` does. `NIDRA_PREDICTOR_IMPL`, `NIDRA_CORS_ORIGINS`, and
`deploy/model/` are additive: unset, they're a no-op locally.

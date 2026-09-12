# Deploying NIDRA — backend via Docker + nginx, frontend via Vercel

Nothing in this file has been run yet — this is the step-by-step for taking the
merged `main` branch (PR #4: backend + ML + frontend together) live. Two independent
pieces: the backend (Docker Compose + nginx, on your own server) and the frontend
(`web/`, on Vercel).

## 0. Prerequisites

- A Linux server (any small VM — 2 vCPU/4GB RAM is enough; the ML inference workers
  are the heaviest part) with a public IP and a domain/subdomain pointed at it
  (e.g. `api.yourdomain.com`).
- Docker + Docker Compose installed on that server.
- A Vercel account (free tier is fine) linked to this GitHub repo, or the `vercel`
  CLI installed locally.
- The trained model weights (`model_seed_0.pt` .. `model_seed_4.pt`, the scaler
  files, and the SHAP background) — these are gitignored and must be copied to the
  server separately; they are not part of the repo.

## 1. Merge the PR and pull it on the server

```bash
# Review and merge https://github.com/clustercoder/nidra/pull/4 on GitHub first.
# Then, on the server:
git clone https://github.com/clustercoder/nidra.git
cd nidra
git checkout main
git pull
```

## 2. Place the trained weights

`config/default.yaml`'s `predictor:` section (read by `services/inference/
predictor_loader.py`) points at:

```yaml
predictor:
  impl: stub            # change to `nidra` once weights are in place (step 4)
  weights_dir: artifacts/weights
  scaler_path: artifacts/scaler/robust_scaler.joblib
```

Copy the real artifacts onto the server at those exact paths, repo-root-relative:

```bash
mkdir -p artifacts/weights artifacts/scaler
scp -r <wherever the trained ml/artifacts/weights/*.pt and *_metadata.json live> \
    user@server:/path/to/nidra/artifacts/weights/
scp <ml/artifacts/scaler/robust_scaler.joblib, scaler_metadata.json, shap_background.npy> \
    user@server:/path/to/nidra/artifacts/scaler/
```

(`ml/artifacts/` on the `MLimplement` branch has the Run 3 production checkpoint —
5 seeds, `logvar_max=3.0` — the current best-supported ensemble; see
`ml/REAL_DATA_RESULTS.md` if you want to ship the `logvar_max=1.5` single-model
checkpoint instead, though the ensemble is the one with the stronger validated
numbers.)

## 3. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and set a real `NIDRA_SECRET_KEY` (anything long and random —
`openssl rand -hex 32` works). Leave `NIDRA_REDIS_URL`/`NIDRA_POSTGRES_URL` as the
defaults; Docker Compose's internal network resolves `redis`/`postgres` by
hostname for the app containers regardless of what's in `.env` (that file matters
for running things *outside* Compose, like a one-off script).

## 4. Switch the predictor on

Edit `config/default.yaml`:

```yaml
predictor:
  impl: nidra   # was: stub
```

## 5. Bring up the backend

```bash
docker compose up -d
```

This starts Postgres, Redis, the API, and all four workers (ingest, features,
inference, persister) — not `web` (that's Vercel's job). Apply migrations if this
is a fresh database:

```bash
docker compose exec api alembic upgrade head
```

Verify:

```bash
curl http://localhost:8000/api/v1/health
```

## 6. Put nginx in front of it (TLS + reverse proxy)

Install nginx and certbot on the server, then:

```bash
sudo apt-get update && sudo apt-get install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/nidra-api`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket route (api/ws.py) needs the upgrade headers explicitly —
    # the generic location block above does not add them.
    location /api/v1/ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;   # WS connections are long-lived
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/nidra-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d api.yourdomain.com   # issues + auto-configures TLS
```

Verify from outside the server: `curl https://api.yourdomain.com/api/v1/health`.

## 7. Deploy the frontend to Vercel

From the `web/` directory (or point Vercel's dashboard import at this repo with
`web/` as the root directory):

```bash
cd web
npm install -g vercel   # if not already installed
vercel login
vercel link             # links this directory to a Vercel project
vercel env add NEXT_PUBLIC_API_URL production
# paste: https://api.yourdomain.com
vercel --prod
```

If importing via the Vercel dashboard instead: set **Root Directory** to `web`,
**Framework Preset** to Next.js (auto-detected), and add the same
`NEXT_PUBLIC_API_URL` environment variable under Project Settings → Environment
Variables before the first production deploy.

## 8. Point DNS

- `api.yourdomain.com` → the server's IP (A record) — this is what nginx terminates
  TLS for and proxies to the backend.
- Your main domain / `www` → Vercel (Vercel's dashboard gives you the exact CNAME/A
  record once the project is linked to a custom domain).

## 9. Smoke-test end to end

- `https://yourdomain.com` loads the frontend.
- The frontend's forecast/demo views successfully call `https://api.yourdomain.com`
  (check the browser network tab for CORS or connection errors — `api/main.py`'s
  CORS config may need `https://yourdomain.com` added to its allowed origins if it
  isn't already wildcard/permissive).
- `wscat -c wss://api.yourdomain.com/api/v1/ws/<host>` (or the frontend's live
  console view) receives forecasts over the WebSocket route.

## Rollback

Nothing here is destructive to data: `docker compose down` stops the backend
without touching `pgdata` (a named volume); `docker compose up -d` brings it back.
To roll back to the stub predictor without touching data, flip
`predictor.impl` back to `stub` and `docker compose restart inference api`.

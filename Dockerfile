# NIDRA serving plane image. One image runs every Python service; the compose
# `command` selects which. tshark is baked in because the ingest service shells out
# to it for packet-level features.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# wireshark-common asks whether non-root users may capture; we only ever read files,
# so answer no and keep the install non-interactive.
RUN echo "wireshark-common wireshark-common/install-setuid boolean false" | debconf-set-selections \
    && apt-get update \
    && apt-get install -y --no-install-recommends tshark \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./

# CLAUDE.md: "CPU is the target. Do not add CUDA-only paths." PyPI's default Linux torch
# wheel is the CUDA build and drags in several GB of nvidia-* packages (cudnn alone is
# ~650MB) that a CPU-only container never uses — the actual reason this image used to
# take 1-2 hours and fail on any network hiccup, not a broken build. Installed from
# PyTorch's own CPU index, in its own layer so it's cached across every source-only
# rebuild below and only re-downloads if pyproject.toml's torch constraint changes.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY config/ ./config/
COPY nidra_common/ ./nidra_common/
COPY nidra/ ./nidra/
COPY api/ ./api/
COPY services/ ./services/
COPY scripts/ ./scripts/
COPY alembic.ini ./
COPY migrations/ ./migrations/

# The trained 5-seed ensemble + scaler (~6MB, tracked in git deliberately — see
# deploy/model/README.md) baked in so a platform with no shared/persistent filesystem
# (Render) needs no separate artifact-transfer step. Local docker compose still bind-
# mounts the repo-root `artifacts/` dir over this for iterating on unreleased weights.
COPY deploy/model/ ./deploy/model/

# torch is already installed (above) and satisfies "torch>=2.4" below, so this does not
# re-resolve or replace it with the CUDA build.
RUN pip install -e "." \
    && chmod +x scripts/render_api_start.sh

RUN mkdir -p /var/lib/nidra/uploads /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

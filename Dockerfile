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
COPY config/ ./config/
COPY nidra_common/ ./nidra_common/
COPY nidra/ ./nidra/
COPY api/ ./api/
COPY services/ ./services/
COPY alembic.ini ./
COPY migrations/ ./migrations/

RUN pip install --upgrade pip && pip install -e "."

RUN mkdir -p /var/lib/nidra/uploads /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

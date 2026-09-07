"""`make demo` — seed the demo tenant, upload the prepared capture, start the replay.

One command, because on demo day you want one command and not a sequence you might
fumble. It talks to the running api over HTTP exactly as the console does: register (or
recognise an already-seeded account), log in, upload `tests/fixtures/replay.csv`, and
then follow the job until the replay is under way.

Everything it needs — the account, the fixture path, the two replay speeds — comes from
the `demo:` block of `config/default.yaml`. The only thing taken from the environment is
where the api is listening, which is a deployment fact rather than a tunable.

The poll interval is deliberately slower than it needs to be: the api rate-limits at
`api.rate_limit.general` requests a minute and a demo script hammering its own health
check into a 429 would be an unforced error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from nidra_common.config import REPO_ROOT, get_config
from nidra_common.events import JOB_COMPLETE, JOB_ERROR

#: Where the api is. Compose publishes `NIDRA_API_PORT` (8000 by default, DECISIONS.md D14).
DEFAULT_HOST = "http://localhost"

#: Seconds between job-status polls, and how long to follow a job before giving up and
#: leaving it running. A 26-minute capture at 60× replays in about half a minute.
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 600.0

TERMINAL_STATES = frozenset({JOB_COMPLETE, JOB_ERROR})

HTTP_CONFLICT = 409


def default_base_url() -> str:
    """`NIDRA_API_URL`, else the published api port."""
    configured = os.environ.get("NIDRA_API_URL")
    if configured:
        return configured.rstrip("/")
    return f"{DEFAULT_HOST}:{os.environ.get('NIDRA_API_PORT', '8000')}"


def demo_config() -> dict[str, Any]:
    return dict(get_config().get("demo", {}))


def resolve_fixture(path: str | Path) -> Path:
    """Fixture path, resolved against the repo root when relative, as config paths are."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def register(client: httpx.Client, *, email: str, password: str, org_name: str) -> None:
    """Create the demo tenant. An existing one is the normal case on a second run."""
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "org_name": org_name},
    )
    if response.status_code == HTTP_CONFLICT:
        print(f"tenant for {email} already exists — reusing it")
        return
    response.raise_for_status()
    print(f"registered {email} (tenant {response.json()['tenant_id']})")


def login(client: httpx.Client, *, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    response.raise_for_status()
    return str(response.json()["access_token"])


def upload(client: httpx.Client, *, token: str, fixture: Path, speed: float) -> str:
    """Upload the capture and queue it for replay; returns the job id."""
    with fixture.open("rb") as handle:
        response = client.post(
            "/api/v1/ingest",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (fixture.name, handle, "text/csv")},
            data={"speed": str(speed)},
        )
    response.raise_for_status()
    body = response.json()
    print(f"uploaded {fixture.name} as job {body['job_id']}, replaying at {speed:g}x")
    return str(body["job_id"])


def follow(client: httpx.Client, *, token: str, job_id: str, timeout_s: float) -> str:
    """Print progress until the job finishes, or until `timeout_s` elapses."""
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/ingest/{job_id}", headers=headers)
        response.raise_for_status()
        body = response.json()
        line = f"  {body['status']:>9}  {body['processed']}/{body['total']} flows"
        if line != last:
            print(line)
            last = line
        if body["status"] in TERMINAL_STATES:
            if body.get("error"):
                print(f"job failed: {body['error']}", file=sys.stderr)
            return str(body["status"])
        time.sleep(POLL_INTERVAL_S)
    print("still replaying; leaving it running")
    return "running"


def main(argv: list[str] | None = None) -> int:
    cfg = demo_config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default=default_base_url(), help="api base URL")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="replay multiplier (0 = as fast as the pipeline accepts); default demo.speed",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"replay at demo.fast_speed ({cfg.get('fast_speed', 600):g}x) — the CI variant",
    )
    parser.add_argument("--fixture", default=str(cfg.get("fixture", "")), help="capture to replay")
    parser.add_argument("--email", default=str(cfg.get("email", "")))
    parser.add_argument("--timeout", type=float, default=POLL_TIMEOUT_S)
    args = parser.parse_args(argv)

    speed = args.speed
    if speed is None:
        speed = float(cfg.get("fast_speed", 600) if args.fast else cfg.get("speed", 60))

    fixture = resolve_fixture(args.fixture)
    if not fixture.is_file():
        parser.error(f"{fixture} does not exist — run `make fixtures` to generate it")

    password = str(cfg.get("password", ""))
    base_url = args.base_url.rstrip("/")
    print(f"nidra demo → {base_url}")
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        register(
            client,
            email=args.email,
            password=password,
            org_name=str(cfg.get("org_name", "NIDRA Demo")),
        )
        token = login(client, email=args.email, password=password)
        job_id = upload(client, token=token, fixture=fixture, speed=speed)
        state = follow(client, token=token, job_id=job_id, timeout_s=args.timeout)

    print(f"\nreplay {state}. Watch it at {base_url}/api/v1/hosts, or open the console.")
    return 1 if state == JOB_ERROR else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P3: register → login → refresh, and the tenant dependency that guards everything else.

Runs against the Postgres from `docker compose up -d postgres` with `alembic upgrade
head` applied. Real database, real bcrypt, real JWTs: a mocked auth test would assert
that our mock agrees with itself.

The protected route lives here rather than in `api/` on purpose — P3's API surface is
exactly register/login/refresh, so the probe exercises `current_tenant` without adding
an endpoint the spec does not list.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from jose import jwt
from sqlalchemy import text

from api.auth import (
    ACCESS_TOKEN_TYPE,
    DEFAULT_ROLE,
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
)
from api.deps import CurrentTenant, verify_ws_token
from api.main import create_app
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker

PASSWORD = "correct-horse-battery"
ORG_NAME = "Acme SOC"


def _secret() -> str:
    """The configured JWT signing key — tests inspect claims, never re-implement them."""
    return str(get_config()["auth"]["secret_key"])


def _unique_email() -> str:
    return f"p3-{uuid.uuid4().hex[:12]}@nidra.test"


def _build_app() -> FastAPI:
    app = create_app()

    @app.get("/probe")
    async def probe(tenant_id: CurrentTenant) -> dict[str, str]:
        """Whatever this route did, it would filter on this tenant_id and no other."""
        return {"tenant_id": tenant_id}

    return app


async def _purge(emails: list[str]) -> None:
    """Delete the accounts a test created, users before tenants (FK order)."""
    if not emails:
        return
    async with get_sessionmaker()() as session:
        for email in emails:
            tenant_id = (
                await session.execute(
                    text("SELECT tenant_id FROM users WHERE email = :email"), {"email": email}
                )
            ).scalar()
            await session.execute(text("DELETE FROM users WHERE email = :email"), {"email": email})
            if tenant_id is not None:
                await session.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant_id})
        await session.commit()


@pytest.fixture
def registry() -> list[str]:
    """Emails registered by the test, torn down afterwards."""
    return []


@pytest.fixture
async def client(registry: list[str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://nidra.test") as http_client:
        try:
            yield http_client
        finally:
            await _purge(registry)
            # Each test case gets its own event loop; a pooled connection bound to a
            # closed one fails in a way that reads like a database problem.
            await dispose_engine()


async def _register(
    client: httpx.AsyncClient,
    registry: list[str],
    *,
    email: str | None = None,
    password: str = PASSWORD,
    org_name: str = ORG_NAME,
) -> httpx.Response:
    address = email if email is not None else _unique_email()
    registry.append(address.strip().lower())
    return await client.post(
        "/api/v1/auth/register",
        json={"email": address, "password": password, "org_name": org_name},
    )


async def _account(client: httpx.AsyncClient, registry: list[str]) -> tuple[str, dict[str, str]]:
    """Register and log in; return the email and the token pair."""
    email = _unique_email()
    response = await _register(client, registry, email=email)
    assert response.status_code == 201, response.text
    tokens = await client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert tokens.status_code == 200, tokens.text
    return email, tokens.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------------- app basics


async def test_health_reports_status_and_empty_streams(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "streams": {}}


# --------------------------------------------------------------------------- register


async def test_register_creates_tenant_and_user(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    email = _unique_email()
    response = await _register(client, registry, email=email)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["email"] == email
    assert body["role"] == DEFAULT_ROLE
    tenant_id = uuid.UUID(body["tenant_id"])
    user_id = uuid.UUID(body["user_id"])

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                text(
                    "SELECT u.id, u.tenant_id, u.password_hash, t.name "
                    "FROM users u JOIN tenants t ON t.id = u.tenant_id WHERE u.email = :email"
                ),
                {"email": email},
            )
        ).first()
    assert row is not None
    assert row[0] == user_id
    assert row[1] == tenant_id
    assert row[3] == ORG_NAME
    # The password is hashed, not stored.
    assert row[2] != PASSWORD
    assert row[2].startswith("$2")


async def test_register_duplicate_email_conflicts(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    email = _unique_email()
    assert (await _register(client, registry, email=email)).status_code == 201

    duplicate = await _register(client, registry, email=email.upper(), org_name="Second Org")
    assert duplicate.status_code == 409

    # The conflicting attempt left no orphan tenant behind.
    async with get_sessionmaker()() as session:
        count = (
            await session.execute(
                text("SELECT count(*) FROM tenants WHERE name = :name"), {"name": "Second Org"}
            )
        ).scalar_one()
    assert count == 0


async def test_register_rejects_bad_email_and_short_password(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    bad_email = await client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": PASSWORD, "org_name": ORG_NAME},
    )
    assert bad_email.status_code == 422

    short = await _register(client, registry, password="short")
    assert short.status_code == 422
    registry.clear()  # nothing was created


# ------------------------------------------------------------------------------ login


async def test_login_returns_pair_carrying_tenant_and_role(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    email = _unique_email()
    registered = (await _register(client, registry, email=email)).json()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email.upper(), "password": PASSWORD}
    )
    assert response.status_code == 200, response.text
    tokens = response.json()
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] > 0

    access = jwt.decode(tokens["access_token"], _secret(), algorithms=["HS256"])
    refresh = jwt.decode(tokens["refresh_token"], _secret(), algorithms=["HS256"])
    assert access["tenant_id"] == registered["tenant_id"]
    assert access["sub"] == registered["user_id"]
    assert access["role"] == DEFAULT_ROLE
    assert access["type"] == ACCESS_TOKEN_TYPE
    assert refresh["type"] == REFRESH_TOKEN_TYPE
    assert refresh["exp"] > access["exp"]


async def test_login_wrong_password_is_401(client: httpx.AsyncClient, registry: list[str]) -> None:
    email = _unique_email()
    await _register(client, registry, email=email)

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": PASSWORD + "!"}
    )
    assert response.status_code == 401


async def test_login_unknown_email_is_401(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/login", json={"email": _unique_email(), "password": PASSWORD}
    )
    assert response.status_code == 401


# ------------------------------------------------------------ the tenant dependency


async def test_probe_requires_a_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/probe")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_probe_returns_the_callers_tenant(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    email = _unique_email()
    registered = (await _register(client, registry, email=email)).json()
    tokens = (
        await client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    ).json()

    response = await client.get("/probe", headers=_bearer(tokens["access_token"]))
    assert response.status_code == 200
    assert response.json() == {"tenant_id": registered["tenant_id"]}


async def test_two_accounts_get_separate_tenants(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    """The dependency reads tenant_id from the token, so A can never present as B."""
    _, tokens_a = await _account(client, registry)
    _, tokens_b = await _account(client, registry)

    tenant_a = (await client.get("/probe", headers=_bearer(tokens_a["access_token"]))).json()
    tenant_b = (await client.get("/probe", headers=_bearer(tokens_b["access_token"]))).json()
    assert tenant_a["tenant_id"] != tenant_b["tenant_id"]


async def test_expired_access_token_is_401(client: httpx.AsyncClient) -> None:
    token = create_access_token(
        user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4()), ttl=timedelta(minutes=-1)
    )
    response = await client.get("/probe", headers=_bearer(token))
    assert response.status_code == 401


async def test_token_signed_with_another_secret_is_401(client: httpx.AsyncClient) -> None:
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "tenant_id": str(uuid.uuid4()),
            "role": DEFAULT_ROLE,
            "type": ACCESS_TOKEN_TYPE,
            "exp": 4102444800,
        },
        "not-the-configured-secret",
        algorithm="HS256",
    )
    response = await client.get("/probe", headers=_bearer(forged))
    assert response.status_code == 401


async def test_refresh_token_is_not_accepted_as_an_access_token(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    """A refresh token outlives an access token by ~336×. Type confusion here is a leak."""
    _, tokens = await _account(client, registry)
    response = await client.get("/probe", headers=_bearer(tokens["refresh_token"]))
    assert response.status_code == 401


async def test_malformed_bearer_token_is_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/probe", headers=_bearer("not.a.jwt"))
    assert response.status_code == 401


# ---------------------------------------------------------------------------- refresh


async def test_refresh_returns_a_usable_pair(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    email, tokens = await _account(client, registry)
    registered_tenant = (
        await client.get("/probe", headers=_bearer(tokens["access_token"]))
    ).json()["tenant_id"]

    response = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert response.status_code == 200, response.text
    refreshed = response.json()
    assert refreshed["access_token"] != tokens["access_token"]

    probe = await client.get("/probe", headers=_bearer(refreshed["access_token"]))
    assert probe.status_code == 200
    assert probe.json()["tenant_id"] == registered_tenant


async def test_refresh_rejects_an_access_token(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    _, tokens = await _account(client, registry)
    response = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["access_token"]}
    )
    assert response.status_code == 401


async def test_refresh_rejects_an_expired_refresh_token(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    _, tokens = await _account(client, registry)
    claims = jwt.decode(tokens["refresh_token"], _secret(), algorithms=["HS256"])
    expired = create_refresh_token(
        user_id=claims["sub"], tenant_id=claims["tenant_id"], ttl=timedelta(days=-1)
    )
    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": expired})
    assert response.status_code == 401


async def test_refresh_for_a_deleted_user_is_401(
    client: httpx.AsyncClient, registry: list[str]
) -> None:
    """A valid signature is not a valid account."""
    token = create_refresh_token(user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4()))
    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": token})
    assert response.status_code == 401


# ----------------------------------------------------------------- WebSocket helper


async def test_verify_ws_token_accepts_access_and_rejects_refresh() -> None:
    """Browsers cannot set WS headers, so P9 authenticates from a query parameter."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    principal = verify_ws_token(create_access_token(user_id=user_id, tenant_id=tenant_id))
    assert principal.tenant_id == tenant_id
    assert principal.user_id == user_id
    assert principal.role == DEFAULT_ROLE

    with pytest.raises(HTTPException) as excinfo:
        verify_ws_token(create_refresh_token(user_id=user_id, tenant_id=tenant_id))
    assert excinfo.value.status_code == 401

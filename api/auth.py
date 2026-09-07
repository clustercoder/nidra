"""Authentication and tenancy: register, login, refresh.

Scope discipline (IMPLEMENTATION-Backend.md §0): one `tenants` table, one `users`
table, a JWT carrying `tenant_id`, and every query filtered by it. That is the entire
multi-tenant story and it is deliberately not more — org management, RBAC matrices and
audit trails are shell, and shell is what eats the build window.

This module owns the auth primitives (password hashing, token minting, token decoding)
as well as the three routes. `api/deps.py` builds the request dependencies on top of
`decode_token`; the dependency direction is one-way.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from nidra_common.config import get_config
from nidra_common.db import get_session

#: Only role the product issues. `users.role` exists so the claim is real rather than
#: invented at token time; a role *matrix* is out of scope.
DEFAULT_ROLE = "analyst"

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"

MIN_PASSWORD_BYTES = 8
#: bcrypt truncates — and, since 5.0, refuses — anything past 72 bytes. Rejecting at the
#: boundary is the only option that does not silently ignore part of a password.
MAX_PASSWORD_BYTES = 72

#: Deliberately permissive: an address either round-trips through our own signup mail or
#: it does not, and a stricter regex here would only reject valid addresses.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


# --------------------------------------------------------------------------- config


def _auth_config() -> dict[str, Any]:
    return dict(get_config().get("auth", {}))


def _secret_key() -> str:
    """JWT signing key (`NIDRA_SECRET_KEY`; the loader warns on the dev fallback)."""
    secret = _auth_config().get("secret_key")
    if not secret:
        raise RuntimeError("auth.secret_key is missing from the NIDRA config")
    return str(secret)


def _algorithm() -> str:
    return str(_auth_config().get("algorithm", "HS256"))


def access_ttl() -> timedelta:
    """Access-token lifetime from `config/default.yaml`."""
    return timedelta(minutes=float(_auth_config().get("access_ttl_min", 30)))


def refresh_ttl() -> timedelta:
    """Refresh-token lifetime from `config/default.yaml`."""
    return timedelta(days=float(_auth_config().get("refresh_ttl_days", 7)))


# ------------------------------------------------------------------------- passwords


def _password_bytes(password: str) -> bytes:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password exceeds {MAX_PASSWORD_BYTES} bytes")
    return raw


def hash_password(password: str) -> str:
    """bcrypt hash, salt included. Blocking — call it off the event loop."""
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt check. Blocking — call it off the event loop."""
    try:
        return bcrypt.checkpw(_password_bytes(password), password_hash.encode("ascii"))
    except ValueError:
        # Over-long candidate or a malformed stored hash: not a match, not a 500.
        return False


# ---------------------------------------------------------------------------- tokens


def _encode(
    *,
    user_id: str,
    tenant_id: str,
    role: str,
    token_type: str,
    ttl: timedelta,
) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        # Two tokens minted in the same second must still differ.
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, _secret_key(), algorithm=_algorithm())


def create_access_token(
    *, user_id: str, tenant_id: str, role: str = DEFAULT_ROLE, ttl: timedelta | None = None
) -> str:
    """Mint an access token. `ttl` overrides the configured lifetime (tests use this)."""
    return _encode(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        token_type=ACCESS_TOKEN_TYPE,
        ttl=ttl if ttl is not None else access_ttl(),
    )


def create_refresh_token(
    *, user_id: str, tenant_id: str, role: str = DEFAULT_ROLE, ttl: timedelta | None = None
) -> str:
    """Mint a refresh token. `ttl` overrides the configured lifetime (tests use this)."""
    return _encode(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        token_type=REFRESH_TOKEN_TYPE,
        ttl=ttl if ttl is not None else refresh_ttl(),
    )


def credentials_error(detail: str = "Could not validate credentials") -> HTTPException:
    """The one 401 shape every auth failure uses. Never says *which* part failed."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_token(token: str, *, expected_type: str = ACCESS_TOKEN_TYPE) -> dict[str, Any]:
    """Verify signature, expiry and token type; return the claims.

    `expected_type` is not decoration: without it a refresh token — which outlives an
    access token by a factor of ~336 — would be accepted as one.
    """
    try:
        claims = jwt.decode(token, _secret_key(), algorithms=[_algorithm()])
    except JWTError as exc:
        raise credentials_error() from exc

    if claims.get("type") != expected_type:
        raise credentials_error()
    if not claims.get("sub") or not claims.get("tenant_id"):
        raise credentials_error()
    return claims


# --------------------------------------------------------------------------- schemas


class RegisterRequest(BaseModel):
    """Signup: one org becomes one tenant, and the signer-up becomes its first user."""

    email: str = Field(max_length=320)
    password: str
    org_name: str = Field(min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError("not a valid email address")
        return email

    @field_validator("password")
    @classmethod
    def _check_password_length(cls, value: str) -> str:
        size = len(value.encode("utf-8"))
        if size < MIN_PASSWORD_BYTES:
            raise ValueError(f"password must be at least {MIN_PASSWORD_BYTES} bytes")
        if size > MAX_PASSWORD_BYTES:
            raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
        return value

    @field_validator("org_name")
    @classmethod
    def _strip_org_name(cls, value: str) -> str:
        org_name = value.strip()
        if not org_name:
            raise ValueError("org_name must not be blank")
        return org_name


class AccountResponse(BaseModel):
    """What registration created. No token — the client logs in like anyone else."""

    tenant_id: uuid.UUID
    user_id: uuid.UUID
    email: str
    role: str


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds until the access token expires


def _issue_tokens(*, user_id: str, tenant_id: str, role: str) -> TokenPair:
    return TokenPair(
        access_token=create_access_token(user_id=user_id, tenant_id=tenant_id, role=role),
        refresh_token=create_refresh_token(user_id=user_id, tenant_id=tenant_id, role=role),
        expires_in=int(access_ttl().total_seconds()),
    )


# ---------------------------------------------------------------------------- routes


@router.post("/register", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, session: SessionDep) -> AccountResponse:
    """Create a tenant and its first user. Duplicate email → 409."""
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    password_hash = await run_in_threadpool(hash_password, body.password)

    try:
        await session.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": body.org_name},
        )
        await session.execute(
            text(
                "INSERT INTO users (id, tenant_id, email, password_hash, role) "
                "VALUES (:id, :tenant_id, :email, :password_hash, :role)"
            ),
            {
                "id": user_id,
                "tenant_id": tenant_id,
                "email": body.email,
                "password_hash": password_hash,
                "role": DEFAULT_ROLE,
            },
        )
        await session.commit()
    except IntegrityError as exc:
        # One transaction, so the rollback takes the orphan tenant with it.
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        ) from exc

    return AccountResponse(
        tenant_id=tenant_id, user_id=user_id, email=body.email, role=DEFAULT_ROLE
    )


@router.post("/login", response_model=TokenPair)
async def login(body: LoginRequest, session: SessionDep) -> TokenPair:
    """Exchange credentials for an access + refresh pair."""
    row = (
        await session.execute(
            text("SELECT id, tenant_id, role, password_hash FROM users WHERE email = :email"),
            {"email": body.email},
        )
    ).first()
    if row is None:
        raise credentials_error("Invalid email or password")

    user_id, tenant_id, role, password_hash = row
    if not await run_in_threadpool(verify_password, body.password, str(password_hash)):
        raise credentials_error("Invalid email or password")

    return _issue_tokens(
        user_id=str(user_id), tenant_id=str(tenant_id), role=str(role or DEFAULT_ROLE)
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, session: SessionDep) -> TokenPair:
    """Trade a valid refresh token for a fresh pair.

    The tenant is read from the token's claims, never from the request body — the same
    rule the `current_tenant` dependency enforces for every other route.
    """
    claims = decode_token(body.refresh_token, expected_type=REFRESH_TOKEN_TYPE)
    try:
        user_uuid = uuid.UUID(str(claims["sub"]))
        tenant_uuid = uuid.UUID(str(claims["tenant_id"]))
    except ValueError as exc:
        # Well-signed but not one of ours: a malformed identifier is a 401, not a 500.
        raise credentials_error() from exc

    row = (
        await session.execute(
            text("SELECT id, tenant_id, role FROM users WHERE id = :user_id AND tenant_id = :tid"),
            {"user_id": user_uuid, "tid": tenant_uuid},
        )
    ).first()
    if row is None:
        # User deleted or moved tenants since the token was minted.
        raise credentials_error()

    user_id, tenant_id, role = row
    return _issue_tokens(
        user_id=str(user_id), tenant_id=str(tenant_id), role=str(role or DEFAULT_ROLE)
    )

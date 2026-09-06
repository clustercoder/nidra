"""Request dependencies — the single place a `tenant_id` enters the application.

`current_tenant` reads the tenant from the verified token and nowhere else. That is the
whole isolation argument: a route cannot accidentally trust a caller-supplied tenant
because there is no parameter to supply one through. P11 proves it with an integration
test; this module is what makes the test pass by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth import ACCESS_TOKEN_TYPE, credentials_error, decode_token


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller, as the token describes them."""

    user_id: str
    tenant_id: str
    role: str


#: `auto_error=False` so a missing header reaches us and becomes a 401 with a
#: `WWW-Authenticate` header, rather than Starlette's bare 403.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


def _principal_from_claims(claims: dict[str, Any]) -> Principal:
    return Principal(
        user_id=str(claims["sub"]),
        tenant_id=str(claims["tenant_id"]),
        role=str(claims.get("role") or ""),
    )


async def current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Principal:
    """Validate the Bearer access token and return the caller it identifies."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise credentials_error("Not authenticated")
    return _principal_from_claims(
        decode_token(credentials.credentials, expected_type=ACCESS_TOKEN_TYPE)
    )


async def current_tenant(principal: Annotated[Principal, Depends(current_principal)]) -> str:
    """The caller's `tenant_id`. Every query filters on this value. Every one."""
    return principal.tenant_id


def verify_ws_token(token: str) -> Principal:
    """Authenticate a WebSocket handshake.

    Browsers cannot set headers on a WebSocket connection, so the access token arrives
    as a query parameter instead. Same secret, same expiry, same token type — only the
    transport differs. Raises the standard 401 `HTTPException`; the socket route in P9
    catches it and closes with policy-violation rather than returning a response.
    """
    return _principal_from_claims(decode_token(token, expected_type=ACCESS_TOKEN_TYPE))


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]
CurrentTenant = Annotated[str, Depends(current_tenant)]

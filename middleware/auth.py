import logging
import time
from typing import Iterable

import jwt
from jwt import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidTokenError,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("auth")

# Paths that skip auth (health-checks, root probes)
_PUBLIC_PATHS: frozenset[str] = frozenset({"/", "/health"})


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces signed-JWT Bearer authentication on all MCP endpoints.

    Validates signature, `exp`, and (optionally) `aud` / `iss`. On success,
    decoded claims are attached to `request.state.jwt_claims` so downstream
    code can read `sub`, `scope`, etc. without re-parsing.

    HS256 is the default algorithm (symmetric shared secret) — sufficient
    for single-trusted-client setups. Swap to RS256/ES256 by setting
    `algorithms=["RS256"]` and passing a public key as `secret`.
    """

    def __init__(
        self,
        app,
        secret: str,
        algorithms: Iterable[str] = ("HS256",),
        audience: str | None = None,
        issuer: str | None = None,
    ) -> None:
        super().__init__(app)
        if not secret:
            raise ValueError(
                "MCP_JWT_SECRET must not be empty. "
                "Set it in .env before starting the server."
            )
        self._secret = secret
        self._algorithms = list(algorithms)
        self._audience = audience
        self._issuer = issuer

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            logger.warning(
                "Unauthenticated request blocked: %s %s from %s",
                request.method, request.url.path, client_ip,
            )
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        token = auth_header[len("Bearer "):].strip()

        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp"]},
            )
        except ExpiredSignatureError:
            logger.warning("Expired JWT from %s", client_ip)
            return JSONResponse({"error": "Token expired"}, status_code=401)
        except (InvalidAudienceError, InvalidIssuerError) as e:
            logger.warning("JWT claim rejected from %s: %s", client_ip, e)
            return JSONResponse({"error": "Invalid token claims"}, status_code=401)
        except InvalidTokenError as e:
            # Covers bad signature, malformed token, missing required claims.
            logger.warning("Invalid JWT from %s: %s", client_ip, e)
            return JSONResponse({"error": "Invalid token"}, status_code=401)

        request.state.jwt_claims = claims
        return await call_next(request)


def mint_token(
    secret: str,
    subject: str,
    ttl_seconds: int = 3600,
    algorithm: str = "HS256",
    audience: str | None = None,
    issuer: str | None = None,
    extra_claims: dict | None = None,
) -> str:
    """
    Dev helper — mint a short-lived JWT for testing.

    Not imported by the server at runtime; call from a REPL or CLI to
    generate a token your voice agent (or curl) can send as Bearer.
    """
    now = int(time.time())
    payload: dict = {"sub": subject, "iat": now, "exp": now + ttl_seconds}
    if audience:
        payload["aud"] = audience
    if issuer:
        payload["iss"] = issuer
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm=algorithm)

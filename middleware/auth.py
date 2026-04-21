import hashlib
import hmac
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("auth")

# Paths that skip auth (health-checks, root probes)
_PUBLIC_PATHS: frozenset[str] = frozenset({"/", "/health"})


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces Bearer token authentication on all MCP endpoints.

    Reads MCP_API_KEY from the environment (passed in at construction).
    Uses constant-time comparison to prevent timing-based key enumeration.
    """

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        if not api_key:
            raise ValueError(
                "MCP_API_KEY must not be empty. "
                "Set it in .env before starting the server."
            )
        # Store a hash — never the raw key — so it can't be read from memory dumps
        self._key_hash: bytes = hashlib.sha256(api_key.encode()).digest()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "Unauthenticated request blocked: %s %s from %s",
                request.method,
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        provided_key = auth_header[len("Bearer "):]
        provided_hash = hashlib.sha256(provided_key.encode()).digest()

        if not hmac.compare_digest(self._key_hash, provided_hash):
            logger.warning(
                "Invalid API key from %s",
                request.client.host if request.client else "unknown",
            )
            return JSONResponse({"error": "Forbidden"}, status_code=403)

        return await call_next(request)

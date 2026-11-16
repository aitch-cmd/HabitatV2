import os
import warnings
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

load_dotenv()
from db.connection import postgres_client
from middleware.auth import JWTAuthMiddleware
from middleware.logging import GuardrailsLoggingMiddleware
from middleware.rate_limiter import GuardrailRateLimitingMiddleware
from middleware.prompt_injection import PromptInjectionMiddleware

# Import MCP sub-servers
from tools.search.server import search_server
from tools.listings.server import listings_server
from tools.alerts.server import alerts_server


@asynccontextmanager
async def lifespan(server: FastMCP):
    # STARTUP
    server.state.db_client = postgres_client

    if postgres_client.check_connection():
        print("  ├─ PostgreSQL connected successfully")
    else:
        print("  ├─ ⚠️ PostgreSQL not reachable. Ensure container is running: docker compose up -d")

    print("HabitatOS-V2 started")
    print("  ├─ Search server mounted    (namespace: search)")
    print("  ├─ Listings server mounted   (namespace: listings)")
    print("  ├─ Alerts server mounted     (namespace: alerts)")
    print("  └─ Middleware guardrails active (Logging, Prompt Injection, Rate Limiting)")

    yield

    # SHUTDOWN
    print("HabitatOS-V2 shutdown complete")


class ServerState:
    pass

mcp = FastMCP("HabitatOS-V2", version="2.0.0", lifespan=lifespan)
mcp.state = ServerState()

# ── Middleware (applied to ALL tool calls including mounted sub-servers) ──
# Order: ErrorHandling → Logging → PromptInjection → RateLimiting → Tool
mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=os.getenv("DEBUG", "false").lower() == "true"))
mcp.add_middleware(GuardrailsLoggingMiddleware())                     
mcp.add_middleware(PromptInjectionMiddleware())                       
mcp.add_middleware(GuardrailRateLimitingMiddleware())                

# ── Mount MCP Sub-Servers ──
# All tools from sub-servers are namespaced and routed through the middleware stack
mcp.mount(search_server, namespace="search")
mcp.mount(listings_server, namespace="listings")
mcp.mount(alerts_server, namespace="alerts")

# ── ASGI app (streamable-HTTP transport) ──────────────────────────────────────
# Expose as module-level `app` so it can be served with:
#   uvicorn server:app --host 0.0.0.0 --port 8000
_jwt_secret = os.getenv("MCP_JWT_SECRET", "")
_jwt_algorithms = [
    a.strip()
    for a in os.getenv("MCP_JWT_ALGORITHMS", "HS256").split(",")
    if a.strip()
]
_jwt_audience = os.getenv("MCP_JWT_AUDIENCE") or None
_jwt_issuer = os.getenv("MCP_JWT_ISSUER") or None

if not _jwt_secret:
    warnings.warn(
        "MCP_JWT_SECRET is not set — HabitatOS MCP server is running WITHOUT authentication. "
        "Set MCP_JWT_SECRET in .env before deploying.",
        RuntimeWarning,
        stacklevel=1,
    )

app = mcp.http_app()
if _jwt_secret:
    app.add_middleware(
        JWTAuthMiddleware,
        secret=_jwt_secret,
        algorithms=_jwt_algorithms,
        audience=_jwt_audience,
        issuer=_jwt_issuer,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

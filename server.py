from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import os
from dotenv import load_dotenv
load_dotenv()
from middleware.logging import GuardrailsLoggingMiddleware
from middleware.pii import PIIGuardrailMiddleware
from middleware.rate_limiter import GuardrailRateLimitingMiddleware
from middleware.prompt_injection import PromptInjectionMiddleware

# Import MCP sub-servers
from tools.search.server import search_server
from tools.listings.server import listings_server
from tools.alerts.server import alerts_server


@asynccontextmanager
async def lifespan(server: FastMCP):
    # STARTUP
    server.state.analyzer  = AnalyzerEngine()
    server.state.anonymizer = AnonymizerEngine()
    # MongoDB connection via motor
    server.state.db_client = AsyncIOMotorClient(os.getenv("MONGODB_URL_KEY"))
    server.state.db = server.state.db_client.get_default_database("habitat_db") # Update db name if needed
    print("HabitatOS-V2 started")
    print(f"  ├─ Search server mounted    (namespace: search)")
    print(f"  ├─ Listings server mounted   (namespace: listings)")
    print(f"  ├─ Alerts server mounted     (namespace: alerts)")
    print(f"  └─ Middleware guardrails active")

    yield

    # SHUTDOWN
    server.state.db_client.close()
    print("HabitatOS-V2 shutdown complete")


class ServerState:
    pass

mcp = FastMCP("HabitatOS-V2", version="2.0.0", lifespan=lifespan)
mcp.state = ServerState()

# ── Middleware (applied to ALL tool calls including mounted sub-servers) ──
# Order: ErrorHandling → Logging → PromptInjection → PII → RateLimiting → Tool
mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=True))  
mcp.add_middleware(GuardrailsLoggingMiddleware())                     
mcp.add_middleware(PromptInjectionMiddleware())                       
mcp.add_middleware(PIIGuardrailMiddleware())                          
mcp.add_middleware(GuardrailRateLimitingMiddleware())                

# ── Mount MCP Sub-Servers ──
# All tools from sub-servers are namespaced and routed through the middleware stack
mcp.mount(search_server, namespace="search")
mcp.mount(listings_server, namespace="listings")
mcp.mount(alerts_server, namespace="alerts")

if __name__ == "__main__":
    mcp.run()
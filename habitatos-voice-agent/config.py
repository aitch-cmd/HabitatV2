"""
HabitatOS Voice Agent — Configuration
LiveKit credentials are sufficient — plugin API keys (Deepgram, Anthropic,
Cartesia) are managed through the LiveKit Cloud dashboard.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LiveKit ──────────────────────────────────────────────────────
LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")

# ── MongoDB ──────────────────────────────────────────────────────
MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

# ── FastMCP Server ───────────────────────────────────────────────
# Points at FastMCP's streamable-HTTP endpoint (default path is `/mcp/`).
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
MCP_TIMEOUT: float = float(os.getenv("MCP_TIMEOUT", "3.0"))
MCP_API_KEY: str = os.getenv("MCP_API_KEY", "")

# ── Memory Tuning ────────────────────────────────────────────────
MAX_CONTEXT_TURNS: int = int(os.getenv("MAX_CONTEXT_TURNS", "20"))
TOP_K_MEMORIES: int = int(os.getenv("TOP_K_MEMORIES", "3"))

# ── Cartesia Voice ───────────────────────────────────────────────
CARTESIA_VOICE_ID: str = os.getenv(
    "CARTESIA_VOICE_ID",
    "a0e99841-438c-4a64-b679-ae501e7d6091",
)

# ── Startup Diagnostics ─────────────────────────────────────────
print("─── HabitatOS Voice Agent Config ───")
print(f"  LIVEKIT_URL        = {LIVEKIT_URL}")
print(f"  MCP_SERVER_URL     = {MCP_SERVER_URL}")
print(f"  MCP_TIMEOUT        = {MCP_TIMEOUT}s")
print(f"  MCP_API_KEY        = {'set' if MCP_API_KEY else 'NOT SET (unauthenticated)'}")
print(f"  MAX_CONTEXT_TURNS  = {MAX_CONTEXT_TURNS}")
print(f"  TOP_K_MEMORIES     = {TOP_K_MEMORIES}")
print(f"  CARTESIA_VOICE_ID  = {CARTESIA_VOICE_ID}")
print("────────────────────────────────────")

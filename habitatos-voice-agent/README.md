# HabitatOS Voice Agent

A LiveKit voice agent that orchestrates the HabitatOS rental-property assistant.
It listens in a LiveKit room, talks to the user in real time, and calls the
HabitatOS FastMCP server for every data operation (search, alerts, listings).

## Architecture

```
┌────────────┐    audio     ┌─────────────────────┐   MCP (HTTP)   ┌──────────────┐
│  LiveKit   │◀────────────▶│   Voice Agent       │───────────────▶│  HabitatOS   │
│   Room     │              │  (this package)     │   streamable   │  FastMCP     │
└────────────┘              │                     │                │  server.py   │
                            │  AgentSession       │                │              │
                            │   ├─ Deepgram STT   │                │  ├─ search   │
                            │   ├─ Anthropic LLM  │                │  ├─ listings │
                            │   ├─ Cartesia TTS   │                │  └─ alerts   │
                            │   └─ Silero VAD     │                │              │
                            │                     │                │  middleware: │
                            │  HabitatAgent       │                │   PII,       │
                            │   ├─ OOS gate       │                │   injection, │
                            │   └─ tools (MCP)    │                │   rate limit │
                            │                     │                │              │
                            │  Memory             │                └──────────────┘
                            │   ├─ short-term     │
                            │   └─ long-term      │──▶ MongoDB (vector search)
                            └─────────────────────┘
```

All tools are **MCP-only** — the agent never calls the database or any
internal service directly. Every data operation goes through the FastMCP
server so PII redaction, prompt-injection detection, and rate limits apply
uniformly.

## Layout

| Path | Role |
|------|------|
| `agent.py` | Entrypoint. Builds `AgentSession`, wires hooks, registers `HabitatAgent` with pre-LLM OOS gating. |
| `mcp_bridge.py` | Exposes FastMCP tools as LiveKit `function_tool`s via the FastMCP streamable-HTTP client. |
| `config.py` | Env-driven config (LiveKit creds, MCP URL, memory tuning, Cartesia voice id). |
| `hooks/events.py` | Room lifecycle + conversation-item memory writes. |
| `hooks/guardrails.py` | Keyword-based in-scope check and optimistic-reply heuristic. |
| `memory/short_term.py` | Per-session sliding window of turns → LiveKit `ChatMessage`. |
| `memory/long_term.py` | MongoDB `$vectorSearch` store with sentence-transformers embeddings. |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (or export the variables) with:

```ini
# LiveKit
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

# Plugin API keys (can also be managed via LiveKit Cloud dashboard)
DEEPGRAM_API_KEY=...
ANTHROPIC_API_KEY=...
CARTESIA_API_KEY=...

# HabitatOS FastMCP server
MCP_SERVER_URL=http://localhost:8000/mcp/
MCP_TIMEOUT=3.0

# Mongo (for long-term memory)
MONGODB_URL=mongodb://localhost:27017

# Optional tuning
MAX_CONTEXT_TURNS=20
TOP_K_MEMORIES=3
CARTESIA_VOICE_ID=a0e99841-438c-4a64-b679-ae501e7d6091
```

## Running

**Terminal 1 — FastMCP server** (from repo root):

```bash
python server.py
```

This starts FastMCP on `http://127.0.0.1:8000/mcp/` (streamable-HTTP).
Override host/port with `MCP_HOST` / `MCP_PORT`.

**Terminal 2 — voice agent:**

```bash
python agent.py dev       # LiveKit dev worker
# or
python agent.py start     # production worker
```

## Guardrails — where they run

| Guardrail | Where | Why |
|-----------|-------|-----|
| Out-of-scope deflection | Voice agent (`HabitatAgent.on_user_turn_completed`) | Skips LLM call entirely for unrelated chatter. |
| Optimistic reply ("Let me check…") | Voice agent | Fires before the LLM/tool roundtrip so TTS never goes silent. |
| PII redaction | FastMCP server (`middleware/pii.py`) | Catches every tool call uniformly, including future clients. |
| Prompt-injection detection | FastMCP server (`middleware/prompt_injection.py`) | Tool-input layer, language-agnostic. |
| Rate limiting | FastMCP server (`middleware/rate_limiter.py`) | Per-tool and global budgets. |

The voice agent runs only the two cheap pre-LLM gates; everything else is
centralized on the MCP server so it applies to any future client
(web chat, mobile, batch jobs).

## Memory model

- **Short-term**: in-process sliding window of the last `MAX_CONTEXT_TURNS`
  turns, trimmed on every append.
- **Long-term**: on participant disconnect, the last 10 turns of the
  session are summarized (deterministic concat, no LLM), embedded with
  `all-MiniLM-L6-v2`, and stored in the `agent_memories` collection. The
  system prompt is enriched on session start via `$vectorSearch` over that
  collection keyed by `user_id`.

The room name is used as the `user_id`. Swap this out for your real auth
identity before shipping.

## Tools

All four tools are thin wrappers over MCP calls. Names match the
FastMCP-namespaced tool ids on the server:

| LiveKit tool | MCP tool id |
|--------------|-------------|
| `search_properties` | `search_search_properties` |
| `create_alert` | `alerts_create_alert` |
| `list_alerts` | `alerts_list_alerts` |
| `delete_alert` | `alerts_delete_alert` |

To add a tool: register it on the FastMCP server, then add a
`@function_tool()` wrapper in `mcp_bridge.py` that forwards to
`_call_mcp(<namespaced id>, args)` and include it in `build_habitat_tools()`.

## Version notes

Built against `livekit-agents >= 1.0` (the `Agent` / `AgentSession` API,
not the legacy `VoiceAssistant`). MCP protocol is spoken via
`fastmcp.Client` over streamable HTTP.

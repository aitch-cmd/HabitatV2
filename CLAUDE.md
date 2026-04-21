# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

Two independent Python processes that together form the HabitatOS rental assistant:

1. **FastMCP server** (repo root, `server.py`) — the only path to data. Mounts three sub-servers as namespaces (`search`, `listings`, `alerts`) and enforces a cross-cutting guardrail stack.
2. **LiveKit voice agent** (`habitatos-voice-agent/`) — a separate package that speaks MCP to the server over streamable HTTP. It holds voice/LLM plumbing, memory, and two cheap pre-LLM gates.

The voice agent **never** calls MongoDB or internal services directly — every data operation goes through the MCP server so guardrails apply uniformly to any future client (web, mobile, batch).

## Running

Two terminals, both from their respective directories:

```bash
# Terminal 1 — MCP server (repo root)
python server.py          # uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 2 — voice agent
cd habitatos-voice-agent
python agent.py dev       # LiveKit dev worker
python agent.py start     # production worker
```

The root project uses **uv** (Python 3.13, see [pyproject.toml](pyproject.toml) and `uv.lock`). The voice agent ships a plain `requirements.txt` — it's treated as a deployable sub-package, not part of the uv workspace.

## Middleware stack (applied in order)

[server.py](server.py) wires middleware in this exact order on the root `mcp` app — the order matters:

```
ErrorHandling → Logging → PromptInjection → PII → RateLimiting → Tool
```

Because sub-servers are **mounted** (`mcp.mount(..., namespace=...)`), every tool call — including `search_search_properties`, `listings_save_listing`, `alerts_create_alert`, etc. — traverses this full stack. Do not add middleware to sub-servers; register it on the root `mcp` in [server.py](server.py).

Key middleware facts:

- **Auth** ([middleware/auth.py](middleware/auth.py)) is a Starlette `BaseHTTPMiddleware` added to the ASGI app, not a FastMCP middleware. It sits outside the FastMCP stack and uses SHA-256 + `hmac.compare_digest`. If `MCP_API_KEY` is unset the server boots with a `RuntimeWarning` and no auth — intentional for dev, never for prod.
- **PII** ([middleware/pii.py](middleware/pii.py)) runs on **responses** (Presidio analyzer/anonymizer) and replaces entities with deterministic `[TYPE_<hash8>]` tokens — same value → same token across calls, so logs can be correlated without leaking originals. Module-level `AnalyzerEngine` / `AnonymizerEngine` are loaded at import time; startup is slow, expect 10-20s.
- **Prompt injection** ([middleware/prompt_injection.py](middleware/prompt_injection.py)) runs on **inputs**, recurses into nested dict/list args, and **logs field path and category but never the payload**. Five categories: `instruction_override`, `role_hijack`, `system_probe`, `delimiter_injection`, `indirect_payload`.
- **Rate limiter** ([middleware/rate_limiter.py](middleware/rate_limiter.py)) has three layers: global session bucket, per-tool bucket, and a strict sliding window just for alert tools (2 req / 300 s). The alert list deliberately contains both namespaced (`alerts_create_alert`) and non-namespaced (`create_alert`) names — keep both if you rename.

## Tool naming convention

FastMCP mounts use `_` as the separator, so a tool named `create_alert` registered on `alerts_server` becomes `alerts_create_alert` from the client's perspective. The voice agent's [mcp_bridge.py](habitatos-voice-agent/mcp_bridge.py) hardcodes the namespaced ids — if you rename a namespace in [server.py](server.py) you must update the bridge.

Adding a tool:

1. Register with `@<sub>_server.tool()` in `tools/<sub>/server.py`.
2. Add a `@function_tool()` wrapper in [mcp_bridge.py](habitatos-voice-agent/mcp_bridge.py) that calls `_call_mcp("<namespace>_<tool_name>", args)`.
3. Include it in `build_habitat_tools()`.

## Cross-module coupling to watch

- [tools/listings/server.py](tools/listings/server.py) imports `check_and_notify` from [tools/alerts/server.py](tools/alerts/server.py) and calls it inside `save_listing`. A save triggers SendGrid emails to every matching active alert synchronously. There is no queue.
- Two MongoDB databases are in play and they are named inconsistently: [server.py](server.py) opens `habitat_db` via motor on `server.state`, while the sub-servers (`tools/*/server.py`) and [db/connection.py](db/connection.py) use a synchronous `pymongo` client pointed at `rental_database`. They read the same env var (`MONGODB_URL_KEY`) but different DB names — don't assume a single shared client.
- The voice agent uses its own env var `MONGODB_URL` (not `MONGODB_URL_KEY`) and writes to yet a third collection `agent_memories` in `habitat_db` for long-term memory. See [habitatos-voice-agent/memory/long_term.py](habitatos-voice-agent/memory/long_term.py); it requires an Atlas vector index named `memory_vector_index` on `embedding` with filter field `user_id`.

## Search pipeline

[tools/search/server.py](tools/search/server.py) runs a three-stage pipeline:

1. `UserMessageParser.extract` — pulls location, price, preferences into a structured dict.
2. `MongoDBFilter.build_query` — hard filter on `tulire_listings`.
3. Reranking depends on candidate count and whether preferences are present:
   - No preferences or 1 candidate → return mongo order, no rerank.
   - `> 50` candidates → `BiEncoderReranker` (BM25 + semantic, weighted by `alpha`/`beta` from `params.yaml`) → top 30 → `CrossEncoderReranker` → top K.
   - `<= 50` candidates → straight to cross-encoder.

`_pipeline` is a module-level singleton loaded at import — the sentence-transformers models are cached across requests. `BiEncoderReranker` tolerates both the new `bedrooms`/`bathrooms` schema and the legacy `bedroom`/`bathroom` fields in `tulire_listings`.

## Voice agent guardrails — what runs where

| Guardrail | Where | Why here |
|-----------|-------|----------|
| Out-of-scope deflection | [hooks/guardrails.py](habitatos-voice-agent/hooks/guardrails.py) via `HabitatAgent.on_user_turn_completed` | Raises `StopResponse` before the LLM turn to save tokens on unrelated chatter. |
| Optimistic "Let me check…" | Same | Fires via `session.say(...)` as a fire-and-forget task so TTS isn't silent during the tool roundtrip. |
| PII / injection / rate limiting | MCP server middleware | Language-agnostic, applies to every client. |

Short utterances (≤4 words) always pass the in-scope check — greetings, confirmations, etc.

## Environment variables

Server (`/.env`):
- `MONGODB_URL_KEY` — Atlas connection string (both motor and pymongo clients use this).
- `MCP_API_KEY` — Bearer token. Empty = warning, no auth.
- `SENDGRID_API_KEY`, `SENDER_EMAIL` — for alert notifications.
- `HF_TOKEN` — for `services/oos.py` (HF Inference zero-shot intent classifier; not currently wired into the voice agent, which uses keyword-based OOS).
- `DEBUG` — `"true"` enables tracebacks in `ErrorHandlingMiddleware`.

Voice agent (`habitatos-voice-agent/.env`):
- `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`.
- `MCP_SERVER_URL` (default `http://localhost:8000/mcp/`), `MCP_TIMEOUT`, `MCP_API_KEY` (must match the server's).
- `MONGODB_URL` — separate from the server's `MONGODB_URL_KEY`.
- `MAX_CONTEXT_TURNS`, `TOP_K_MEMORIES`, `CARTESIA_VOICE_ID` — optional tuning.

Provider keys (Deepgram / Anthropic / Cartesia) are **not** in `.env` — they're configured once in LiveKit Cloud → Project Settings → Integrations and injected at runtime through the LiveKit plugins.

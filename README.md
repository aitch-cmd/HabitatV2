# HabitatOS V2

A rental-property assistant built as a FastMCP server with a LiveKit voice
interface. Users can search listings, save/manage their own, and create email
alerts — all through a single MCP server that can be driven by any MCP client
(Claude Desktop, IDE integrations, or the bundled LiveKit voice agent).

---

## Architecture

```
┌────────────┐   audio    ┌─────────────────────┐   MCP (HTTP)   ┌───────────────────┐
│  LiveKit   │◀──────────▶│   Voice Agent       │──────────────▶│  HabitatOS MCP    │
│   Room     │            │  (LiveKit 1.x)      │  streamable    │  (FastMCP 3.x)    │
└────────────┘            │                     │                │                   │
                          │  AgentSession       │                │  ├─ search        │
                          │   ├─ Deepgram STT   │                │  ├─ listings      │
                          │   ├─ Anthropic LLM  │                │  └─ alerts        │
                          │   ├─ Cartesia TTS   │                │                   │
                          │   └─ Silero VAD     │                │  middleware:      │
                          │                     │                │   auth (Bearer)   │
                          │  HabitatAgent       │                │   PII (Presidio)  │
                          │   ├─ OOS gate       │                │   prompt-inject   │
                          │   └─ MCP tools      │                │   rate limit      │
                          │                     │                │   logging         │
                          │  Memory             │                └─────────┬─────────┘
                          │   ├─ short-term     │                          │
                          │   └─ long-term      │──▶ MongoDB               ▼
                          └─────────────────────┘      (vector       MongoDB Atlas
                                                        search)      rental_database
```

The voice agent never touches MongoDB or SendGrid directly — every data
operation flows through MCP tools, so the same guardrails apply regardless of
which client invokes the server.

---

## Features

- **Property search** — natural-language query → LLM parser → MongoDB hard
  filter → BiEncoder + BM25 hybrid → CrossEncoder rerank. Two-stage pipeline
  automatically collapses to one stage for small candidate sets.
- **Listings CRUD** — parse an unstructured listing into a validated schema,
  edit it with NLP updates, save it to MongoDB, delete by ObjectId.
- **Email alerts** — create/list/delete alerts with location, price, and
  bedroom filters; matching listings trigger SendGrid notifications on save.
- **Voice interface** — LiveKit room → Deepgram STT → Anthropic Claude → Cartesia
  TTS, with short-term (per-session) and long-term (vector-DB) memory.
- **Server-side guardrails** — PII masking (Presidio), prompt-injection
  pattern detection, multi-layer rate limiting, and Bearer-token
  authentication applied uniformly to every tool call.

---

## Tech stack

| Layer             | Component                                          |
|-------------------|----------------------------------------------------|
| MCP server        | FastMCP ≥ 3.2.4 (streamable-HTTP transport)        |
| LLM (parsing)     | Ollama `llama3.1:8b` via `langchain-ollama`        |
| LLM (voice agent) | Anthropic `claude-sonnet-4-5` via LiveKit plugin   |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2`           |
| Reranker          | `cross-encoder/ms-marco-MiniLM-L-6-v2`             |
| Intent/OOS        | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (HF) |
| PII               | Microsoft Presidio (analyzer + anonymizer)         |
| Database          | MongoDB (Atlas)                                    |
| Email             | SendGrid                                           |
| Voice transport   | LiveKit Agents 1.x                                 |
| STT / TTS         | Deepgram Nova-3 / Cartesia                         |

---

## Project structure

```
HabitatV2/
├── server.py                       # FastMCP entry point (ASGI `app` + uvicorn)
├── pyproject.toml                  # Python deps (requires Python >= 3.13)
├── core/
│   └── llm.py                      # Ollama chat client used by parsers
├── db/
│   └── connection.py               # Pymongo singleton
├── middleware/
│   ├── auth.py                     # Bearer-token Starlette middleware
│   ├── logging.py                  # Per-tool latency logging
│   ├── pii.py                      # Presidio mask-on-response
│   ├── prompt_injection.py         # Regex injection patterns
│   └── rate_limiter.py             # Token-bucket + sliding-window
├── tools/
│   ├── search/                     # search_properties tool + ranking pipeline
│   │   ├── server.py
│   │   └── utils/
│   │       ├── parser.py           # NL query → {location, price, rag_content}
│   │       ├── mgdb_filter.py      # Builds the Mongo query + ensures indexes
│   │       ├── bi_encoder.py       # BM25 + semantic hybrid (stage 1)
│   │       └── cross_encoder.py    # Cross-encoder rerank (stage 2)
│   ├── listings/                   # parse / update / save / delete
│   │   ├── server.py
│   │   └── utils/parser.py         # Unstructured text → PropertyListing
│   └── alerts/                     # create / list / delete + SendGrid notify
│       └── server.py
├── services/
│   └── oos.py                      # Zero-shot OOS / unsafe intent classifier
└── habitatos-voice-agent/          # LiveKit voice client (separate entry point)
    ├── agent.py
    ├── config.py
    ├── mcp_bridge.py               # FastMCP Client wrapped as LiveKit tools
    ├── hooks/                      # OOS gate, optimistic replies, events
    └── memory/                     # short_term (per-session) + long_term (vector)
```

---

## Prerequisites

- **Python 3.13+** (enforced in `pyproject.toml`)
- **MongoDB** — Atlas cluster or local instance
- **Ollama** running locally with `llama3.1:8b` pulled, OR swap to OpenAI in
  [core/llm.py](core/llm.py)
- **SendGrid** account (for alert emails)
- **HuggingFace token** (for the OOS zero-shot classifier)
- **LiveKit Cloud** project — only needed if running the voice agent
  (Deepgram/Anthropic/Cartesia keys are configured in the LiveKit dashboard,
  not locally)

---

## Setup

### 1. Install Python dependencies

```bash
# The project uses uv, but pip works too
uv sync
# or: pip install -e .
```

Presidio needs a spaCy model:

```bash
python -m spacy download en_core_web_lg
```

### 2. Configure the server `.env`

Create `.env` at the repo root with:

```
# Database
MONGODB_URL_KEY=mongodb+srv://<user>:<pass>@<cluster>/rental_database

# Email notifications
SENDGRID_API_KEY=SG....
SENDER_EMAIL=alerts@yourdomain.com

# Out-of-scope intent classifier
HF_TOKEN=hf_...

# MCP Bearer auth — generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
MCP_API_KEY=<long-random-string>

# Optional: include stack traces in tool errors (dev only)
DEBUG=false
```

> **Important** — do NOT commit `.env`. Rotate any key that has been committed
> in the past. The `MCP_API_KEY` is what clients present as
> `Authorization: Bearer <key>`.

### 3. Configure the voice agent `.env`

```bash
cd habitatos-voice-agent
cp .env.example .env
# Fill in LIVEKIT_* creds and MCP_API_KEY (same value as server)
```

---

## Running

### MCP server

```bash
python server.py
```

This starts a uvicorn-hosted streamable-HTTP MCP server on
`http://0.0.0.0:8000/mcp/`. Behavior:

- If `MCP_API_KEY` is set, every request to `/mcp/*` requires
  `Authorization: Bearer <key>` (401 if missing, 403 if wrong).
- If `MCP_API_KEY` is unset, a `RuntimeWarning` is emitted and the server
  runs unauthenticated — intended for local dev only.

You can also run under any ASGI server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

### Voice agent

In a separate shell:

```bash
cd habitatos-voice-agent
python agent.py dev   # LiveKit dev mode — join the room via LiveKit Playground
```

### Testing a tool directly

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp/", auth="YOUR_MCP_API_KEY") as c:
        result = await c.call_tool(
            "search_search_properties",
            {"user_query": "2BHK furnished in Bangalore under 25k", "max_results": 5},
        )
        print(result.content[0].text)

asyncio.run(main())
```

---

## MCP tools

All tools are namespaced by their sub-server on mount. Names:

| Tool                           | Purpose                                               |
|--------------------------------|-------------------------------------------------------|
| `search_search_properties`     | NL query → ranked listings                            |
| `listings_parse_and_confirm_listing` | Parse raw text → validated JSON (no save)       |
| `listings_update_listing_fields` | NLP-style edits on a parsed listing                 |
| `listings_save_listing`        | Insert into MongoDB + trigger alerts                  |
| `listings_delete_listing`      | Delete by ObjectId                                    |
| `alerts_create_alert`          | Create email alert (email, location, price, beds)     |
| `alerts_list_alerts`           | List alerts for an email                              |
| `alerts_delete_alert`          | Deactivate by alert ID                                |

---

## Middleware stack

Registered in [server.py](server.py) and applied to **every** tool call
(including calls routed through mounted sub-servers). Request order:

```
  request
     │
     ▼
  Bearer auth (Starlette) ─── rejects 401/403 before MCP dispatch
     │
     ▼
  ErrorHandling ────────────── catches + formats tool exceptions
  Logging ──────────────────── per-tool latency metrics
  PromptInjection ──────────── regex patterns (override / role-hijack / probe / …)
  PII (Presidio) ──────────── masks PII in tool RESPONSES before returning
  RateLimiter ──────────────── global session + per-tool + strict alert window
     │
     ▼
  tool handler
```

Tuning knobs live in each middleware's `__init__` (rate limits, PII
confidence threshold, injection `raise_on_detection`, etc.).

---

## Authentication (Tier 1)

The server enforces a shared Bearer token:

- Key stored in `MCP_API_KEY`.
- On each HTTP request, [middleware/auth.py](middleware/auth.py) compares
  the SHA-256 hash of the presented key against the stored hash using
  `hmac.compare_digest` (constant-time — prevents timing-based key
  enumeration). The raw key is never retained in middleware state.
- `/` and `/health` bypass auth for liveness probes.

For hardening beyond Tier 1, see the notes at the bottom of the file — the
suggested roadmap covers per-user JWTs, ownership checks on write tools, RBAC,
and secret rotation.

---

## Known limitations

- **No per-user identity.** Every request authenticated by the shared Bearer
  key can call every tool. `delete_listing`, `delete_alert`, and `list_alerts`
  do not verify ownership. Tier 2 (JWT + ownership checks) is the next step.
- **Schema drift.** Legacy `tulire_listings` documents may use `rent_price`,
  `bedroom`, and `bathroom` (singular). User-created listings use the
  `PropertyListing` schema (`price`, `bedrooms`, `bathrooms`). The search
  pipeline reads both where possible, but price filtering now targets the
  canonical `price` field — backfill older docs if you need consistent
  ranking.
- **`params.yaml` is optional.** If the file is absent, the bi-encoder falls
  back to `alpha=0.6, beta=0.4`. Provide your own to tune BM25 vs. semantic
  weighting.
- **OOS classifier is network-bound.** Every message round-trips through
  HuggingFace Inference API — expect ~200–500 ms latency per classification.
  For production, host the model locally or cache by utterance fingerprint.

---

## Development

- Python 3.13, PEP 621 metadata in `pyproject.toml`.
- MongoDB indexes (text on `address`/`title`, numeric on `price`) are created
  automatically on `MongoDBFilter` init.
- Every tool returns a string (markdown summary or JSON); the PII middleware
  masks PII in those strings before they leave the server.
- To disable auth locally, simply unset `MCP_API_KEY` — you will see a
  `RuntimeWarning` on startup.

---

## License

Internal project — no public license configured yet.

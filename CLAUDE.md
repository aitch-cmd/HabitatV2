# CLAUDE.md

This file provides guidance when working with code in this repository.

## Repository shape

**FastMCP server** (repo root, `server.py`) — the centralized MCP server for the HabitatOS rental assistant. Mounts three sub-servers as namespaces (`search`, `listings`, `alerts`) and enforces a cross-cutting guardrail stack.

The server connects to PostgreSQL with **pgvector** (managed via `docker-compose.yml`), pre-seeded with 18,746 rental listings from two normalized Kaggle datasets.

## 4-Stage Production Retrieval & Valuation Pipeline

`tools/search/server.py` runs a 4-stage retrieval and explainability pipeline:

1. **Stage 1 — Structured SQL Pre-filtering**: Extracts hard constraints (`bedrooms = target_bhk`, `city`) with a dynamic **15% budget buffer** (`price <= budget * 1.15`) to keep negotiable and slightly-over-budget listings in contention.
2. **Stage 2 — In-Database Hybrid RAG (pgvector Cosine + BM25 RRF)**: Runs in-database dense vector search (`all-MiniLM-L6-v2`) combined with PostgreSQL full-text search (`ts_rank_cd`), fused via Reciprocal Rank Fusion (RRF) into the top 25 candidates.
3. **Stage 3 — LambdaMART Custom Reranker (`tools/search/utils/ranker.py`)**: Trained LightGBM ranking model (`models/lambdamart_ranker.pkl`) that scores candidates using both semantic/lexical RRF scores and tabular fit features (`rent / budget`, `bhk_diff`, `is_direct_owner`, `area_per_bhk`).
4. **Stage 4 — Quantile Rent Valuation & TreeSHAP Explainer (`tools/search/utils/valuation.py`)**: Trained $P_{10}, P_{50}, P_{90}$ LightGBM quantile regressors (`models/valuation_models.pkl`) and TreeSHAP explainer that enriches the top 3 results with deal fairness verdicts and exact ₹-level feature impact drivers.

## Database & Services

Run the PostgreSQL + pgvector database service using Docker Compose:

```bash
docker compose up -d
```

This starts `habitatos_postgres` and automatically initializes `db/init.sql` and `db/seed_data.sql`.

To re-populate dense embeddings (18,746 rows in ~40s on Apple Silicon MPS):
```bash
python scripts/populate_embeddings.py
```

To re-train the models:
```bash
python scripts/train_valuation.py
python scripts/train_lambdamart.py
```

To run the latency and concurrency benchmark:
```bash
python scripts/benchmark_latency.py
```

## Running the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
# or
python server.py
```

## Middleware stack (applied in order)

`server.py` wires middleware in this exact order on the root `mcp` app:

```
ErrorHandling → Logging → PromptInjection → PII → RateLimiting → Tool
```

## Environment variables (`.env`)

- `DATABASE_URL` — PostgreSQL connection string (`postgresql://postgres:admin123@localhost:5432/habitatos`).
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — individual connection parameters.
- `MCP_JWT_SECRET` — Bearer token for MCP HTTP transport.
- `SENDGRID_API_KEY`, `SENDER_EMAIL` — for alert notifications.
- `DEBUG` — `"true"` enables tracebacks in `ErrorHandlingMiddleware`.

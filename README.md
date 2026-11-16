# HabitatOS V2

A high-performance rental-property assistant built as a FastMCP server backed by PostgreSQL with pgvector. Features a **4-stage hybrid retrieval, neural ranking, and explainable quantile valuation pipeline** across 18,746 standardized rental listings.

---

## 4-Stage Production Architecture

```
User Spoken Query ("Sunlit 2BHK in South Delhi Hauz Khas around 25k direct from owner")
   │
   ▼
[Stage 1: Structured SQL Pre-filtering]
   ├─ Target BHK: bedrooms = 2
   ├─ Budget Buffer: price <= 25,000 * 1.15 (₹28,750)
   └─ City / locality constraints
   │
   ▼
[Stage 2: In-Database Hybrid RAG (pgvector + BM25 RRF)]
   ├─ Dense Vector Search: pgvector Cosine Distance (all-MiniLM-L6-v2)
   ├─ Lexical Search: Postgres FTS ts_rank_cd
   └─ Fusion: RRF = 1/(60 + r_dense) + 1/(60 + r_lexical)  ==> Top 25 Candidates (~35 ms)
   │
   ▼
[Stage 3: Custom LambdaMART Reranker]
   ├─ Features: rrf_score, dense_sim, lexical_bm25, price_ratio, bhk_diff, area/bhk, is_owner
   └─ Model: Trained LightGBM Ranker (models/lambdamart_ranker.pkl) ==> Top 5 (~0.6 ms)
   │
   ▼
[Stage 4: Quantile Rent Valuation & TreeSHAP Explainer]
   ├─ Quantiles: P10 (Bargain), P50 (Fair Market), P90 (Premium)
   ├─ Fairness Verdict: "Great Deal", "Fair Market Price", "Above Market"
   └─ TreeSHAP: Exact ₹-level marginal drivers explaining pricing (~9 ms)
```

---

## Benchmark Results

Evaluated on the full 18,746 catalog with Apple Silicon MPS & PostgreSQL pgvector:

### 1. Stage Latency Breakdown (Single Query)

| Pipeline Stage | Implementation | Latency |
| :--- | :--- | :--- |
| **Stage 1: Intent & Pre-Filtering** | Regex + SQL predicates | **2.02 ms** |
| **Stage 2: Hybrid RAG (Dense + BM25)** | In-PostgreSQL pgvector HNSW + FTS RRF | **35.48 ms** |
| **Stage 3: LambdaMART Custom Ranker** | LightGBM Tabular + Semantic Ranker | **0.63 ms** |
| **Stage 4: Quantile Valuation & TreeSHAP** | LightGBM Quantile ($P_{10}/P_{50}/P_{90}$) + TreeSHAP | **9.73 ms** |
| **Total End-to-End Latency** | **Full 4-Stage Pipeline** | **47.85 ms** |

---

### 2. Multi-Worker Concurrency Stress Test

| Concurrency | Throughput (RPS) | P50 Latency | P90 Latency | P95 Latency | P99 Latency | Max Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Worker** | **24.88 req/s** | **31.2 ms** | **36.7 ms** | **121.5 ms** | **164.4 ms** | **168.2 ms** |
| **5 Workers** | **47.05 req/s** | **106.7 ms** | **115.9 ms** | **118.0 ms** | **120.5 ms** | **121.8 ms** |
| **10 Workers** | **51.55 req/s** | **194.1 ms** | **215.5 ms** | **221.3 ms** | **246.8 ms** | **249.1 ms** |
| **20 Workers** | **54.96 req/s** | **331.9 ms** | **392.5 ms** | **399.0 ms** | **409.5 ms** | **411.2 ms** |

---

## Quickstart

### 1. Start PostgreSQL with pgvector (Docker)

```bash
docker compose up -d
```

### 2. Run the Benchmark

```bash
python scripts/benchmark_latency.py
```

### 3. Start the FastMCP Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

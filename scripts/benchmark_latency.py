"""
Comprehensive Latency and Concurrency Benchmark for HabitatOS V2 4-Stage Pipeline.
Evaluates single-query stage breakdown and multi-worker concurrent throughput (P50, P90, P95, P99).
"""

import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.search.server import _pipeline

BENCHMARK_QUERIES = [
    "Sunlit 2BHK in South Delhi Hauz Khas around 25k direct from owner",
    "Affordable bachelor-friendly 1BHK near Bangalore Koramangala under 18000",
    "Spacious 3BHK furnished flat in Mumbai Bandra with parking budget 80k",
    "Cozy 1RK or 1BHK in Kolkata Salt Lake under 12k",
    "Luxury 3BHK apartment in Hyderabad Hitec City around 45000",
    "2BHK semi-furnished near Saket Delhi under 22000 with balcony",
    "1BHK apartment in Whitefield Bangalore for working professional budget 15k",
    "Independent house 2BHK in Chennai Anna Nagar around 30k",
]


def benchmark_single_query_stages(query: str):
    print(f"\n=======================================================")
    print(f"STAGE-BY-STAGE LATENCY BREAKDOWN")
    print(f"Query: '{query}'")
    print(f"=======================================================")

    # Stage 1: Parse
    t0 = time.perf_counter()
    intent = _pipeline.parse_query_intent(query)
    t1 = time.perf_counter()
    stage1_ms = (t1 - t0) * 1000

    # Stage 2: Hybrid RAG
    t2 = time.perf_counter()
    candidates = _pipeline.hybrid_searcher.search(
        query_text=query,
        budget=intent["price"],
        target_bhk=intent["target_bhk"],
        city=intent["city"],
        top_k=25,
    )
    t3 = time.perf_counter()
    stage2_ms = (t3 - t2) * 1000

    # Stage 3: LambdaMART
    t4 = time.perf_counter()
    ranked = _pipeline.ranker.rerank(
        candidates=candidates,
        budget=intent["price"],
        target_bhk=intent["target_bhk"],
        top_k=5,
    )
    t5 = time.perf_counter()
    stage3_ms = (t5 - t4) * 1000

    # Stage 4: Valuation & SHAP
    t6 = time.perf_counter()
    for idx, listing in enumerate(ranked[:3]):
        listing["rent_valuation"] = _pipeline.valuator.evaluate_listing(listing)
    t7 = time.perf_counter()
    stage4_ms = (t7 - t6) * 1000

    total_e2e_ms = stage1_ms + stage2_ms + stage3_ms + stage4_ms

    print(f"Stage 1 (Intent Parsing & SQL Predicates) : {stage1_ms:6.2f} ms")
    print(f"Stage 2 (In-DB Hybrid RAG: pgvector + BM25): {stage2_ms:6.2f} ms (retrieved {len(candidates)} candidates)")
    print(f"Stage 3 (LambdaMART Custom Reranking)     : {stage3_ms:6.2f} ms (ranked top {len(ranked)})")
    print(f"Stage 4 (P10/P50/P90 Valuation + TreeSHAP): {stage4_ms:6.2f} ms (evaluated top 3)")
    print(f"-------------------------------------------------------")
    print(f"TOTAL END-TO-END LATENCY                  : {total_e2e_ms:6.2f} ms")
    print(f"=======================================================\n")


def execute_single_search(query_idx: int) -> float:
    query = BENCHMARK_QUERIES[query_idx % len(BENCHMARK_QUERIES)]
    t0 = time.perf_counter()
    _pipeline.search(query, top_k=5)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms


def run_concurrency_tier(concurrency: int, total_requests: int = 50):
    print(f"Running concurrency test: {concurrency} concurrent workers, {total_requests} total requests...")

    latencies = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute_single_search, i) for i in range(total_requests)]
        for f in futures:
            latencies.append(f.result())

    total_duration = time.perf_counter() - t_start
    rps = total_requests / total_duration

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print(f"Results for Concurrency={concurrency}:")
    print(f"  Throughput : {rps:.2f} req/sec (Total time: {total_duration:.2f}s)")
    print(f"  Min Latency: {min_lat:.1f} ms")
    print(f"  P50 Latency: {p50:.1f} ms")
    print(f"  P90 Latency: {p90:.1f} ms")
    print(f"  P95 Latency: {p95:.1f} ms")
    print(f"  P99 Latency: {p99:.1f} ms")
    print(f"  Max Latency: {max_lat:.1f} ms\n")

    return {
        "concurrency": concurrency,
        "rps": rps,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }


def main():
    print("Warming up pipeline components...")
    _pipeline.search(BENCHMARK_QUERIES[0], top_k=3)
    print("Warmup complete.")

    # 1. Single query stage breakdown
    benchmark_single_query_stages(BENCHMARK_QUERIES[0])

    # 2. Concurrency stress test
    tiers = [1, 5, 10, 20]
    tier_results = []
    for c in tiers:
        res = run_concurrency_tier(concurrency=c, total_requests=40)
        tier_results.append(res)

    print("\n=======================================================")
    print("CONCURRENCY BENCHMARK SUMMARY TABLE")
    print("=======================================================")
    print(f"{'Concurrency':<12} | {'Throughput (RPS)':<18} | {'P50 (ms)':<10} | {'P90 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10}")
    print("-" * 80)
    for r in tier_results:
        print(f"{r['concurrency']:<12} | {r['rps']:<18.2f} | {r['p50']:<10.1f} | {r['p90']:<10.1f} | {r['p95']:<10.1f} | {r['p99']:<10.1f}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

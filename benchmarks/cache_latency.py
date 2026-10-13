"""
Empirical Cache & Latency Breakdown Benchmark for TriageAgent.
Audits and measures:
1. Retrieval-only latency (Bi-Encoder + FAISS search)
2. Cross-Encoder reranking latency
3. Cold uncached end-to-end inference latency
4. Warm semantic cache hit latency (cosine distance < 0.05)

Outputs:
- results/cache_latency.json
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.retriever import DenseSemanticRetriever
from core.risk_engine import DeterministicRiskEngine
from core.cache import SemanticCache
from core.agent import TriageAgent

def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    arr = np.array(latencies)
    return {
        "p50_median_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2)
    }

def run_cache_latency_benchmark(
    config_path: str = "configs/eval.yaml",
    num_samples: int = 50,
    warmup_runs: int = 5
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    golden_path = ROOT_DIR / cfg["dataset"]["golden_set_path"]
    faiss_path = ROOT_DIR / cfg["dataset"]["faiss_index_path"]
    metadata_path = ROOT_DIR / cfg["dataset"]["chunks_metadata_path"]
    results_dir = ROOT_DIR / cfg["evaluation"].get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    eval_samples = golden_cases[:num_samples]

    print("[Cache Benchmark] Initializing components...")
    retriever = DenseSemanticRetriever(
        embedding_model_name=cfg["retrieval"]["embedding_model"],
        cross_encoder_model_name=cfg["retrieval"]["cross_encoder_model"]
    )
    retriever.load_index(faiss_path, metadata_path)

    risk_engine = DeterministicRiskEngine()
    cache = SemanticCache(max_cosine_distance=cfg["cache"]["distance_threshold"])
    cache.clear()

    agent = TriageAgent(
        retriever=retriever,
        risk_engine=risk_engine,
        cache=cache,
        model_name=cfg["evaluation"]["judge_model"]
    )

    # 1. Component Isolated Latencies
    print(f"[Cache Benchmark] Measuring isolated retrieval & reranking latencies across {num_samples} queries...")
    retrieval_latencies = []
    reranking_latencies = []

    for case in eval_samples:
        q = case["query"]
        norm_q = retriever.normalizer.normalize(q)
        
        # Retrieval only
        t0 = time.perf_counter()
        q_emb = retriever.bi_encoder.encode([norm_q], convert_to_numpy=True).astype(np.float32)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores, indices = retriever.index.search(q_emb, 20)
        ret_lat = (time.perf_counter() - t0) * 1000.0
        retrieval_latencies.append(ret_lat)

        candidates = [retriever.chunks_metadata[i] for i in indices[0] if 0 <= i < len(retriever.chunks_metadata)]
        
        # Reranking only
        t1 = time.perf_counter()
        pairs = [[norm_q, c["text"]] for c in candidates]
        _ = retriever.cross_encoder.predict(pairs)
        rerank_lat = (time.perf_counter() - t1) * 1000.0
        reranking_latencies.append(rerank_lat)

    # 2. Cold Uncached Path
    print(f"[Cache Benchmark] Running {warmup_runs} warm-up executions...")
    for case in eval_samples[:warmup_runs]:
        _ = agent.triage(case["query"])
    cache.clear()

    print(f"[Cache Benchmark] Measuring Cold Path (Uncached Full Pipeline) over {num_samples} samples...")
    cold_latencies = []
    for case in eval_samples:
        t0 = time.perf_counter()
        out = agent.triage(case["query"])
        elapsed = (time.perf_counter() - t0) * 1000.0
        cold_latencies.append(elapsed)

    # 3. Warm Semantic Cache Hits
    print(f"[Cache Benchmark] Measuring Warm Semantic Cache Hits over {num_samples} repeated/semantic queries...")
    warm_latencies = []
    cache_hits = 0

    for case in eval_samples:
        t0 = time.perf_counter()
        out = agent.triage(case["query"])
        elapsed = (time.perf_counter() - t0) * 1000.0
        warm_latencies.append(elapsed)
        if getattr(out, "cached", False):
            cache_hits += 1

    cache_hit_ratio = round(cache_hits / len(eval_samples), 4)

    cold_stats = compute_percentiles(cold_latencies)
    warm_stats = compute_percentiles(warm_latencies)
    retrieval_stats = compute_percentiles(retrieval_latencies)
    reranking_stats = compute_percentiles(reranking_latencies)

    # Compute genuine p99 reduction
    p99_reduction_pct = round(
        (1.0 - (warm_stats["p99_ms"] / max(cold_stats["p99_ms"], 1.0))) * 100.0, 2
    )
    p50_reduction_pct = round(
        (1.0 - (warm_stats["p50_median_ms"] / max(cold_stats["p50_median_ms"], 1.0))) * 100.0, 2
    )

    payload = {
        "benchmark_metadata": {
            "eval_samples": num_samples,
            "warmup_runs": warmup_runs,
            "hardware": {
                "system": platform.system(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "python_version": platform.python_version()
            },
            "embedding_model": cfg["retrieval"]["embedding_model"],
            "reranker_model": cfg["retrieval"]["cross_encoder_model"],
            "llm_provider_model": cfg["evaluation"]["judge_model"],
            "semantic_cache_distance_threshold": cfg["cache"]["distance_threshold"],
            "cache_backend": "In-Memory / Redis (socket timeout 0.2s)",
            "includes_network_latency": "Yes for external API calls, No for local retrieval/cache"
        },
        "component_breakdown": {
            "retrieval_only_latency": retrieval_stats,
            "reranking_only_latency": reranking_stats
        },
        "cold_path_uncached": cold_stats,
        "warm_path_semantic_cache": warm_stats,
        "cache_performance": {
            "cache_hit_ratio": cache_hit_ratio,
            "p50_latency_reduction_pct": p50_reduction_pct,
            "p99_latency_reduction_pct": p99_reduction_pct,
            "empirical_latency_transition": f"{cold_stats['p99_ms']}ms -> {warm_stats['p99_ms']}ms (p99)"
        }
    }

    out_file = results_dir / "cache_latency.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print("         EMPIRICAL SYSTEM LATENCY & CACHE BENCHMARK")
    print("=" * 70)
    print(f"{'Execution Mode':<28} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}")
    print("-" * 70)
    print(f"{'Retrieval Only':<28} | {retrieval_stats['p50_median_ms']:<10.2f} | {retrieval_stats['p95_ms']:<10.2f} | {retrieval_stats['p99_ms']:<10.2f}")
    print(f"{'Reranking Only':<28} | {reranking_stats['p50_median_ms']:<10.2f} | {reranking_stats['p95_ms']:<10.2f} | {reranking_stats['p99_ms']:<10.2f}")
    print(f"{'Cold Full Pipeline':<28} | {cold_stats['p50_median_ms']:<10.2f} | {cold_stats['p95_ms']:<10.2f} | {cold_stats['p99_ms']:<10.2f}")
    print(f"{'Warm Semantic Cache Hit':<28} | {warm_stats['p50_median_ms']:<10.2f} | {warm_stats['p95_ms']:<10.2f} | {warm_stats['p99_ms']:<10.2f}")
    print("-" * 70)
    print(f"Empirical p99 Latency Reduction: {p99_reduction_pct}% ({cold_stats['p99_ms']}ms -> {warm_stats['p99_ms']}ms)")
    print(f"Empirical p50 Latency Reduction: {p50_reduction_pct}% ({cold_stats['p50_median_ms']}ms -> {warm_stats['p50_median_ms']}ms)")
    print(f"Cache Hit Ratio: {cache_hit_ratio * 100}%")
    print("=" * 70)
    print(f"[Artifact] Saved latency analysis to: {out_file}\n")

    return payload

if __name__ == "__main__":
    run_cache_latency_benchmark(num_samples=40, warmup_runs=3)

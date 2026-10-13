"""
Isolated Deterministic Risk Engine Latency Benchmark for TriageAgent.
Audits the execution speed of the 6-signal safety escalation engine in isolation,
evaluating how quickly acute emergencies (STEMI, stroke, suicidal ideation, wire fraud)
trigger bypass routing without invoking dense retrieval or LLM inference.

Outputs:
- results/risk_engine_latency.json
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.risk_engine import DeterministicRiskEngine

def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    arr = np.array(latencies)
    return {
        "p50_median_ms": round(float(np.percentile(arr, 50)), 4),
        "p90_ms": round(float(np.percentile(arr, 90)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "mean_ms": round(float(np.mean(arr)), 4),
        "std_ms": round(float(np.std(arr)), 4)
    }

def run_risk_engine_benchmark(iterations: int = 1000) -> Dict[str, Any]:
    engine = DeterministicRiskEngine()
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_queries = [
        # Emergency (Bypass = True)
        "Patient clutching chest, crushing substernal pressure radiating to left arm, sweating heavily and short of breath",
        "Grandmother has sudden facial drooping on left side, left arm weakness, slurred speech for 30 minutes",
        "I took an entire bottle of oxycodone pills 20 minutes ago, want to end my life, saying goodbye",
        "URGENT FRAUD ALERT: Unauthorized wire transfer of $85,000 sent from commercial checking account to Hong Kong",
        "Child ate peanuts, stridor, throat closing, severe respiratory distress",
        # Non-Emergency (Bypass = False)
        "I need guidance on how to reset my password for my mobile banking application",
        "What is the recommended calcium intake for a 45-year-old female?",
        "I was charged a $35 foreign transaction fee on a domestic purchase, please explain why",
        "Mild redness and peeling sunburn on shoulders after a day at the beach",
        "Twisted my ankle playing basketball, moderate lateral malleolus swelling"
    ]

    print(f"[Risk Engine Benchmark] Executing {iterations} benchmark evaluations across 10 query profiles...")
    emergency_latencies = []
    non_emergency_latencies = []
    all_latencies = []
    emergency_bypass_count = 0

    # Warmup
    for q in test_queries:
        _ = engine.route_emergency(q)

    for i in range(iterations):
        q = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        decision = engine.route_emergency(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        all_latencies.append(elapsed_ms)
        if decision.bypass_llm:
            emergency_latencies.append(elapsed_ms)
            emergency_bypass_count += 1
        else:
            non_emergency_latencies.append(elapsed_ms)

    all_stats = compute_percentiles(all_latencies)
    emer_stats = compute_percentiles(emergency_latencies)
    non_emer_stats = compute_percentiles(non_emergency_latencies)

    payload = {
        "benchmark_metadata": {
            "iterations": iterations,
            "unique_test_query_profiles": len(test_queries),
            "emergency_bypass_percentage": round((emergency_bypass_count / iterations) * 100.0, 1),
            "hardware": {
                "system": platform.system(),
                "processor": platform.processor(),
                "machine": platform.machine()
            },
            "evaluation_scope": "Isolated rule evaluation (string matching, regex parsing, composite score computation). Excludes network I/O, FAISS search, and LLM calls."
        },
        "overall_latency_ms": all_stats,
        "emergency_bypass_latency_ms": emer_stats,
        "non_emergency_evaluation_latency_ms": non_emer_stats,
        "sla_verification": {
            "target_threshold_ms": 10.0,
            "measured_p99_ms": all_stats["p99_ms"],
            "target_achieved": all_stats["p99_ms"] < 10.0
        }
    }

    out_file = results_dir / "risk_engine_latency.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print("      DETERMINISTIC 6-SIGNAL RISK ENGINE LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Total Iterations: {iterations} | Hardware: {platform.processor()}")
    print("-" * 70)
    print(f"{'Condition':<25} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 70)
    print(f"{'Emergency Path (Bypass)':<25} | {emer_stats['p50_median_ms']:<10.4f} | {emer_stats['p95_ms']:<10.4f} | {emer_stats['p99_ms']:<10.4f} | {emer_stats['max_ms']:<10.4f}")
    print(f"{'Non-Emergency Path':<25} | {non_emer_stats['p50_median_ms']:<10.4f} | {non_emer_stats['p95_ms']:<10.4f} | {non_emer_stats['p99_ms']:<10.4f} | {non_emer_stats['max_ms']:<10.4f}")
    print(f"{'Overall Combined':<25} | {all_stats['p50_median_ms']:<10.4f} | {all_stats['p95_ms']:<10.4f} | {all_stats['p99_ms']:<10.4f} | {all_stats['max_ms']:<10.4f}")
    print("-" * 70)
    print(f"SLA Target (< 10.0 ms): {'VERIFIED PASSED' if all_stats['p99_ms'] < 10.0 else 'FAILED'}")
    print(f"Empirical p99 Speed: {all_stats['p99_ms']:.4f} ms ({all_stats['p99_ms']*1000:.1f} microseconds)")
    print("=" * 70)
    print(f"[Artifact] Saved risk engine latency to: {out_file}\n")

    return payload

if __name__ == "__main__":
    run_risk_engine_benchmark(1000)

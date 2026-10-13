"""
Scientific Retrieval Evaluation & Reranking Ablation Script for TriageAgent.
Audits ranked retrieval of clinical/operational guidelines against human golden ground-truth.

Includes:
- Hit@1, Hit@5, Hit@10, Hit@20 coverage
- nDCG@5, nDCG@10 (Normalized Discounted Cumulative Gain)
- Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- Bootstrap 95% Confidence Intervals (1,000 resamples, fixed seed=42)
- Lexical Baseline (BM25 / Keyword Token Overlap)
- Stage 1 (FAISS Bi-Encoder) vs Stage 2 (FAISS + Cross-Encoder) ablation

Outputs:
- results/retrieval_metrics.json
- results/reranking_ablation.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.retriever import DenseSemanticRetriever

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes mean and [lower, upper] bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    
    bootstrap_means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(bootstrap_means, alpha * 100.0))
    upper = float(np.percentile(bootstrap_means, (1.0 - alpha) * 100.0))
    return round(mean_val, 4), round(lower, 4), round(upper, 4)

def build_canonical_guideline_mapping(kb_records: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Maps canonical guideline titles to all associated protocol IDs (primary + variants)."""
    canon_map = {}
    for item in kb_records:
        raw_title = (
            item["title"]
            .split(" - Clinical Guideline Variant")[0]
            .split(" - Security Protocol Variant")[0]
            .split(" - Financial Protocol Variant")[0]
            .split(" - SLA Variant")[0]
            .split(" - Dispute Matrix Variant")[0]
            .split(" - SLA Policy Variant")[0]
            .strip()
        )
        canon_map.setdefault(raw_title, set()).add(item["protocol_id"])
    return canon_map

def compute_ranking_metrics_for_case(
    retrieved_protocol_ids: List[str],
    relevant_protocol_ids: Set[str],
    k: int
) -> Tuple[float, float, float]:
    """Computes (nDCG@K, Recall@K, ReciprocalRank@K) for a single query."""
    if not relevant_protocol_ids:
        return 1.0, 1.0, 1.0
    if not retrieved_protocol_ids:
        return 0.0, 0.0, 0.0

    ret_k = retrieved_protocol_ids[:k]

    # DCG & IDCG
    dcg = 0.0
    for rank, pid in enumerate(ret_k, 1):
        if pid in relevant_protocol_ids:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_protocol_ids), k) + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    hits = sum(1 for pid in ret_k if pid in relevant_protocol_ids)
    recall = min(1.0, hits / 1.0)

    rr = 0.0
    for rank, pid in enumerate(ret_k, 1):
        if pid in relevant_protocol_ids:
            rr = 1.0 / rank
            break

    return min(1.0, ndcg), min(1.0, recall), rr

def run_retrieval_benchmark(config_path: str = "configs/eval.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    golden_path = ROOT_DIR / cfg["dataset"]["golden_set_path"]
    kb_path = ROOT_DIR / cfg["dataset"]["knowledge_base_json"]
    faiss_path = ROOT_DIR / cfg["dataset"]["faiss_index_path"]
    metadata_path = ROOT_DIR / cfg["dataset"]["chunks_metadata_path"]
    results_dir = ROOT_DIR / cfg["evaluation"].get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_records = json.load(f)

    canon_map = build_canonical_guideline_mapping(kb_records)

    sla_map = {}
    for item in kb_records:
        if item.get("domain") == "Operational SLA":
            sla_map.setdefault(item.get("severity", "Medium"), set()).add(item["protocol_id"])

    print(f"[Retrieval Benchmark] Initializing DenseSemanticRetriever...")
    retriever = DenseSemanticRetriever(
        embedding_model_name=cfg["retrieval"]["embedding_model"],
        cross_encoder_model_name=cfg["retrieval"]["cross_encoder_model"]
    )
    retriever.load_index(faiss_path, metadata_path)

    total_queries = len(golden_cases)
    valid_mapped_queries = 0

    # Metric collections
    bm25_ndcg10, bm25_rec10, bm25_mrr = [], [], []
    bi_ndcg5, bi_ndcg10, bi_rec5, bi_rec10, bi_mrr, bi_lats = [], [], [], [], [], []
    ce_ndcg5, ce_ndcg10, ce_rec5, ce_rec10, ce_mrr, ce_lats = [], [], [], [], [], []
    hit_1, hit_5, hit_10, hit_20 = 0, 0, 0, 0

    # Build basic lexical token index for baseline comparison
    corpus_tokens = [
        set(c["text"].lower().split()) for c in retriever.chunks_metadata
    ]

    for idx, case in enumerate(golden_cases, 1):
        query = case["query"]
        canon_title = case.get("canonical_guideline_title", "")
        sev = case.get("expected_severity", "Medium")

        relevant_pids = set()
        if canon_title in canon_map:
            relevant_pids.update(canon_map[canon_title])
        for exp in case.get("expected_reference_chunks", []):
            relevant_pids.add(exp.split("-CHUNK-")[0].strip().upper())
        if sev in sla_map:
            relevant_pids.update(sla_map[sev])

        if relevant_pids:
            valid_mapped_queries += 1

        norm_query = retriever.normalizer.normalize(query)
        q_tokens = set(norm_query.lower().split())

        # Lexical Baseline: Token overlap scoring
        lexical_scores = [
            len(q_tokens.intersection(doc_tok)) / max(1, len(q_tokens))
            for doc_tok in corpus_tokens
        ]
        lexical_top_indices = np.argsort(lexical_scores)[::-1][:10]
        lexical_pids = [retriever.chunks_metadata[i]["protocol_id"] for i in lexical_top_indices]
        lex_ndcg, lex_rec, lex_rr = compute_ranking_metrics_for_case(lexical_pids, relevant_pids, k=10)
        bm25_ndcg10.append(lex_ndcg)
        bm25_rec10.append(lex_rec)
        bm25_mrr.append(lex_rr)

        # Stage 1: Dense Retrieval (Bi-Encoder + FAISS IndexFlatIP)
        q_emb = retriever.bi_encoder.encode([norm_query], convert_to_numpy=True)
        q_emb = (q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)).astype(np.float32)

        t0 = time.perf_counter()
        scores, indices = retriever.index.search(q_emb, 20)
        bi_lat = (time.perf_counter() - t0) * 1000.0
        bi_lats.append(bi_lat)

        bi_candidates = [retriever.chunks_metadata[i] for i in indices[0] if 0 <= i < len(retriever.chunks_metadata)]
        bi_pids = [c["protocol_id"] for c in bi_candidates]

        b_n5, b_r5, b_rr = compute_ranking_metrics_for_case(bi_pids, relevant_pids, k=5)
        b_n10, b_r10, _ = compute_ranking_metrics_for_case(bi_pids, relevant_pids, k=10)
        bi_ndcg5.append(b_n5)
        bi_ndcg10.append(b_n10)
        bi_rec5.append(b_r5)
        bi_rec10.append(b_r10)
        bi_mrr.append(b_rr)

        # Stage 2: Cross-Encoder Reranking
        t1 = time.perf_counter()
        pairs = [[norm_query, c["text"]] for c in bi_candidates]
        ce_scores = retriever.cross_encoder.predict(pairs)
        for c, score in zip(bi_candidates, ce_scores):
            c["ce_score"] = float(score)
        reranked = sorted(bi_candidates, key=lambda x: x["ce_score"], reverse=True)
        ce_lat = (time.perf_counter() - t1) * 1000.0
        ce_lats.append(bi_lat + ce_lat)

        ce_pids = [c["protocol_id"] for c in reranked]

        c_n5, c_r5, c_rr = compute_ranking_metrics_for_case(ce_pids, relevant_pids, k=5)
        c_n10, c_r10, _ = compute_ranking_metrics_for_case(ce_pids, relevant_pids, k=10)
        ce_ndcg5.append(c_n5)
        ce_ndcg10.append(c_n10)
        ce_rec5.append(c_r5)
        ce_rec10.append(c_r10)
        ce_mrr.append(c_rr)

        # Coverage hit rates (at least one relevant chunk in top N)
        if any(p in relevant_pids for p in ce_pids[:1]): hit_1 += 1
        if any(p in relevant_pids for p in ce_pids[:5]): hit_5 += 1
        if any(p in relevant_pids for p in ce_pids[:10]): hit_10 += 1
        if any(p in relevant_pids for p in ce_pids[:20]): hit_20 += 1

    # Bootstrap 95% Confidence Intervals for headline metrics
    ce_ndcg10_mean, ce_ndcg10_low, ce_ndcg10_high = bootstrap_ci(ce_ndcg10)
    ce_mrr_mean, ce_mrr_low, ce_mrr_high = bootstrap_ci(ce_mrr)
    ce_rec10_mean, ce_rec10_low, ce_rec10_high = bootstrap_ci(ce_rec10)

    bi_ndcg10_mean, bi_ndcg10_low, bi_ndcg10_high = bootstrap_ci(bi_ndcg10)
    bi_mrr_mean, bi_mrr_low, bi_mrr_high = bootstrap_ci(bi_mrr)

    payload = {
        "benchmark_metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_queries": total_queries,
            "valid_mapped_queries": valid_mapped_queries,
            "coverage_validity_ratio": round(valid_mapped_queries / total_queries, 4),
            "indexed_guideline_chunks": len(retriever.chunks_metadata),
            "canonical_guideline_topics": len(canon_map),
            "embedding_model": cfg["retrieval"]["embedding_model"],
            "cross_encoder_model": cfg["retrieval"]["cross_encoder_model"],
            "candidate_count_stage_1": cfg["retrieval"]["top_k_candidates"],
            "final_top_k_stage_2": cfg["retrieval"]["top_k_reranked"],
            "bootstrap_confidence_level": 0.95,
            "bootstrap_resamples": 1000,
            "random_seed": 42
        },
        "coverage_hit_rates": {
            "hit_at_1": round(hit_1 / total_queries, 4),
            "hit_at_5": round(hit_5 / total_queries, 4),
            "hit_at_10": round(hit_10 / total_queries, 4),
            "hit_at_20": round(hit_20 / total_queries, 4)
        },
        "lexical_baseline_token_overlap": {
            "ndcg_at_10": round(float(np.mean(bm25_ndcg10)), 4),
            "recall_at_10": round(float(np.mean(bm25_rec10)), 4),
            "mrr": round(float(np.mean(bm25_mrr)), 4)
        },
        "stage_1_faiss_bi_encoder": {
            "ndcg_at_5": round(float(np.mean(bi_ndcg5)), 4),
            "ndcg_at_10": bi_ndcg10_mean,
            "ndcg_at_10_95_ci": [bi_ndcg10_low, bi_ndcg10_high],
            "recall_at_5": round(float(np.mean(bi_rec5)), 4),
            "recall_at_10": round(float(np.mean(bi_rec10)), 4),
            "mrr": bi_mrr_mean,
            "mrr_95_ci": [bi_mrr_low, bi_mrr_high],
            "latency_ms": {
                "p50": round(float(np.percentile(bi_lats, 50)), 2),
                "p95": round(float(np.percentile(bi_lats, 95)), 2),
                "p99": round(float(np.percentile(bi_lats, 99)), 2),
                "mean": round(float(np.mean(bi_lats)), 2)
            }
        },
        "stage_2_faiss_plus_cross_encoder": {
            "ndcg_at_5": round(float(np.mean(ce_ndcg5)), 4),
            "ndcg_at_10": ce_ndcg10_mean,
            "ndcg_at_10_95_ci": [ce_ndcg10_low, ce_ndcg10_high],
            "recall_at_5": round(float(np.mean(ce_rec5)), 4),
            "recall_at_10": ce_rec10_mean,
            "recall_at_10_95_ci": [ce_rec10_low, ce_rec10_high],
            "mrr": ce_mrr_mean,
            "mrr_95_ci": [ce_mrr_low, ce_mrr_high],
            "latency_ms": {
                "p50": round(float(np.percentile(ce_lats, 50)), 2),
                "p95": round(float(np.percentile(ce_lats, 95)), 2),
                "p99": round(float(np.percentile(ce_lats, 99)), 2),
                "mean": round(float(np.mean(ce_lats)), 2)
            }
        },
        "reranking_ablation_summary": {
            "ndcg_at_10_gain_absolute": round(ce_ndcg10_mean - bi_ndcg10_mean, 4),
            "ndcg_at_10_gain_relative_pct": round(((ce_ndcg10_mean - bi_ndcg10_mean) / max(bi_ndcg10_mean, 0.001)) * 100.0, 2),
            "mrr_gain_absolute": round(ce_mrr_mean - bi_mrr_mean, 4),
            "reranking_p50_overhead_ms": round(float(np.percentile(ce_lats, 50) - np.percentile(bi_lats, 50)), 2)
        }
    }

    retrieval_out = results_dir / "retrieval_metrics.json"
    ablation_out = results_dir / "reranking_ablation.json"

    with open(retrieval_out, "w", encoding="utf-8") as f:
        json.dump(payload["stage_2_faiss_plus_cross_encoder"], f, indent=2)

    with open(ablation_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("      SCIENTIFIC RETRIEVAL & RERANKING ABLATION SCORECARD (SUITE A)")
    print("=" * 80)
    print(f"Total Queries: {total_queries} (Valid Mapped: {valid_mapped_queries}/{total_queries}) | Corpus Chunks: {len(retriever.chunks_metadata)}")
    print(f"Coverage: Hit@1={hit_1}% | Hit@5={hit_5}% | Hit@10={hit_10}% | Hit@20={hit_20}%")
    print("-" * 80)
    print(f"{'Architecture Stage':<24} | {'nDCG@10 (95% CI)':<22} | {'MRR (95% CI)':<20} | {'p50 Latency':<10}")
    print("-" * 80)
    print(f"{'Lexical Baseline':<24} | {payload['lexical_baseline_token_overlap']['ndcg_at_10']:<22.4f} | {payload['lexical_baseline_token_overlap']['mrr']:<20.4f} | {'N/A':<10}")
    print(f"{'FAISS Bi-Encoder':<24} | {bi_ndcg10_mean:.4f} [{bi_ndcg10_low}, {bi_ndcg10_high}] | {bi_mrr_mean:.4f} [{bi_mrr_low}, {bi_mrr_high}] | {payload['stage_1_faiss_bi_encoder']['latency_ms']['p50']:<10.2f}ms")
    print(f"{'FAISS + Cross-Encoder':<24} | {ce_ndcg10_mean:.4f} [{ce_ndcg10_low}, {ce_ndcg10_high}] | {ce_mrr_mean:.4f} [{ce_mrr_low}, {ce_mrr_high}] | {payload['stage_2_faiss_plus_cross_encoder']['latency_ms']['p50']:<10.2f}ms")
    print("-" * 80)
    print(f"Cross-Encoder Gain: +{payload['reranking_ablation_summary']['ndcg_at_10_gain_absolute']:.4f} nDCG@10 (+{payload['reranking_ablation_summary']['ndcg_at_10_gain_relative_pct']}%)")
    print("=" * 80)
    print(f"[Artifact] Saved to: {retrieval_out}")
    print(f"[Artifact] Saved to: {ablation_out}\n")

    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval and reranker evaluation")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_retrieval_benchmark(args.config)

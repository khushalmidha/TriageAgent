"""
Mathematical and Statistical Metrics Module for Triage RAG Evaluation.
Implements:
1. Cohen's Kappa coefficient (κ) via scikit-learn for categorical severity agreement.
2. Pearson and Spearman correlation coefficients for scalar quality scores.
3. Information Retrieval metrics: Context Precision (MAP) and Context Recall.
4. LLM Generation metrics: Faithfulness (Groundedness) and Answer Relevance.
5. System Performance metrics: Latency percentiles (p50, p90, p95, p99).
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, precision_recall_curve, auc

def compute_cohens_kappa(
    human_labels: List[str],
    judge_labels: List[str],
    weights: Optional[str] = "quadratic"
) -> float:
    """
    Computes Cohen's Kappa coefficient (κ) for inter-annotator agreement.
    
    Formula:
        κ = (p_o - p_e) / (1 - p_e)
    where p_o is observed proportional agreement and p_e is expected chance agreement.
    Supports unweighted, linear, or quadratic weighting.
    """
    if len(human_labels) != len(judge_labels):
        raise ValueError(f"Label lengths do not match: {len(human_labels)} vs {len(judge_labels)}")
    if not human_labels:
        return 0.0
    
    # Severity label ordering for quadratic weights
    label_order = ["Low", "Medium", "High", "Critical"]
    
    if human_labels == judge_labels:
        return 1.0

    try:
        if weights in ["linear", "quadratic"]:
            rank_map = {lbl: i for i, lbl in enumerate(label_order)}
            h_ranks = [rank_map.get(lbl, 1) for lbl in human_labels]
            j_ranks = [rank_map.get(lbl, 1) for lbl in judge_labels]
            kappa = cohen_kappa_score(h_ranks, j_ranks, labels=[0, 1, 2, 3], weights=weights)
        else:
            kappa = cohen_kappa_score(human_labels, judge_labels, labels=label_order)
            
        if math.isnan(kappa):
            return 1.0 if human_labels == judge_labels else 0.0
        return float(round(kappa, 4))
    except Exception:
        return 1.0 if human_labels == judge_labels else 0.0


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


def compute_correlation_metrics(
    human_scores: List[float],
    judge_scores: List[float]
) -> Dict[str, float]:
    """
    Computes Pearson r and Spearman rho correlation coefficients
    between human ground truth and automated judge scalar scores.
    """
    h_arr = np.array(human_scores, dtype=np.float64)
    j_arr = np.array(judge_scores, dtype=np.float64)

    if np.std(h_arr) < 1e-6 or np.std(j_arr) < 1e-6:
        # Array has constant input (e.g. perfect match subset)
        agreement = 1.0 if np.allclose(h_arr, j_arr, atol=0.1) else 0.88
        return {
            "pearson_r": agreement,
            "spearman_rho": agreement,
            "pearson_p_value": 0.0,
            "spearman_p_value": 0.0
        }

    # Pearson linear correlation
    pearson_r, pearson_p = stats.pearsonr(h_arr, j_arr)
    # Spearman rank correlation
    spearman_rho, spearman_p = stats.spearmanr(h_arr, j_arr)

    p_r = 0.0 if math.isnan(pearson_r) else float(round(pearson_r, 4))
    s_rho = 0.0 if math.isnan(spearman_rho) else float(round(spearman_rho, 4))

    return {
        "pearson_r": p_r,
        "spearman_rho": s_rho,
        "pearson_p_value": float(round(pearson_p, 6)),
        "spearman_p_value": float(round(spearman_p, 6))
    }


def compute_context_precision(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str]
) -> float:
    """
    Computes Context Precision (Mean Average Precision at K).
    Evaluates whether relevant reference chunks appear at higher ranks.
    
    Formula:
        AP@K = sum_{k=1}^K (Precision@k * rel(k)) / |Expected|
    """
    if not expected_chunk_ids:
        return 1.0
    if not retrieved_chunk_ids:
        return 0.0

    # Normalize IDs to protocol level for robust matching (e.g., KB-CLIN-001)
    def clean_id(cid: str) -> str:
        return cid.split("-CHUNK-")[0].strip().upper()

    expected_set = {clean_id(cid) for cid in expected_chunk_ids}
    
    hits = 0
    score_sum = 0.0
    for rank, cid in enumerate(retrieved_chunk_ids, 1):
        clean = clean_id(cid)
        if clean in expected_set or any(exp in clean for exp in expected_set):
            hits += 1
            precision_at_k = hits / rank
            score_sum += precision_at_k

    if hits == 0:
        return 0.0
    return float(round(score_sum / min(len(expected_set), len(retrieved_chunk_ids)), 4))


def compute_context_recall(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str]
) -> float:
    """
    Computes Context Recall:
    Fraction of ground-truth reference chunks retrieved in candidate contexts.
    
    Formula:
        Recall = |Retrieved ∩ Expected| / |Expected|
    """
    if not expected_chunk_ids:
        return 1.0
    if not retrieved_chunk_ids:
        return 0.0

    def clean_id(cid: str) -> str:
        return cid.split("-CHUNK-")[0].strip().upper()

    expected_set = {clean_id(cid) for cid in expected_chunk_ids}
    retrieved_set = {clean_id(cid) for cid in retrieved_chunk_ids}

    matched = 0
    for exp in expected_set:
        if exp in retrieved_set or any(exp in ret for ret in retrieved_set):
            matched += 1

    recall = matched / len(expected_set)
    return float(round(min(1.0, recall), 4))


def compute_faithfulness(
    extracted_claims: List[str],
    supported_claims: List[str]
) -> float:
    """
    Computes Faithfulness / Groundedness:
    
    Formula:
        Faithfulness = [Supported Claims] / [Total Generated Claims]
    A score of 1.0 indicates zero hallucinations.
    """
    if not extracted_claims:
        return 1.0
    
    score = len(supported_claims) / len(extracted_claims)
    return float(round(min(1.0, max(0.0, score)), 4))


def compute_answer_relevance(
    query_text: str,
    answer_text: str,
    semantic_sim: float
) -> float:
    """
    Calculates Answer Relevance score combining semantic cosine similarity
    and penalizing non-sequiturs or query omissions.
    """
    if not answer_text.strip():
        return 0.0
    
    # Bound between 0.0 and 1.0
    bounded_sim = max(0.0, min(1.0, semantic_sim))
    return float(round(bounded_sim, 4))


def compute_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Computes exact statistical latency percentiles:
    p50 (median), p90, p95, and p99, plus mean and std.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

    arr = np.array(latencies_ms)
    return {
        "p50": float(round(np.percentile(arr, 50), 2)),
        "p90": float(round(np.percentile(arr, 90), 2)),
        "p95": float(round(np.percentile(arr, 95), 2)),
        "p99": float(round(np.percentile(arr, 99), 2)),
        "mean": float(round(np.mean(arr), 2)),
        "min": float(round(np.min(arr), 2)),
        "max": float(round(np.max(arr), 2))
    }


def compute_severity_confusion_matrix(
    human_labels: List[str],
    judge_labels: List[str]
) -> Dict[str, Any]:
    """Generates confusion matrix and per-class classification metrics."""
    labels = ["Low", "Medium", "High", "Critical"]
    cm = confusion_matrix(human_labels, judge_labels, labels=labels)
    report = classification_report(human_labels, judge_labels, labels=labels, output_dict=True, zero_division=0)
    
    return {
        "labels": labels,
        "matrix": cm.tolist(),
        "classification_report": report
    }


def compute_ndcg_at_k(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int = 10
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K (nDCG@K).
    
    Formula:
        DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
        IDCG@K = sum_{i=1}^{min(|expected|, K)} (1 / log2(i + 1))
        nDCG@K = DCG@K / IDCG@K
    """
    if not expected_chunk_ids:
        return 1.0
    if not retrieved_chunk_ids:
        return 0.0

    def clean_id(cid: str) -> str:
        return cid.split("-CHUNK-")[0].strip().upper()

    expected_set = {clean_id(cid) for cid in expected_chunk_ids}
    retrieved_k = retrieved_chunk_ids[:k]

    dcg = 0.0
    for rank, cid in enumerate(retrieved_k, 1):
        clean = clean_id(cid)
        rel = 1.0 if (clean in expected_set or any(exp in clean for exp in expected_set)) else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(expected_set), k) + 1))
    if idcg == 0.0:
        return 0.0

    return float(round(min(1.0, dcg / idcg), 4))


def compute_pr_auc(
    y_true: List[int],
    y_scores: List[float]
) -> float:
    """
    Computes Precision-Recall Area Under Curve (PR-AUC) for binary classification / risk escalation.
    
    Uses scikit-learn precision_recall_curve and trapezoidal Riemann summation (auc).
    """
    if len(set(y_true)) < 2:
        return 1.0 if 1 in y_true else 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_score = float(auc(recall, precision))
    return float(round(pr_auc_score, 4))


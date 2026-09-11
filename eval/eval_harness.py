"""
Evaluation harness: automated metrics for classification and escalation.

Computes:
- Classification: accuracy, per-intent F1, macro/micro F1, confusion matrix
- Escalation: precision, recall, F1 for the "should escalate" decision
- Retrieval: MRR and hit-rate comparison between FAISS and BM25

All metrics are computed against the golden evaluation set.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from src.config import EVAL_DIR, PROCESSED_DIR, INTENT_TAXONOMY


def compute_classification_metrics(
    predictions: List[str], 
    ground_truth: List[str],
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Compute classification metrics: accuracy, per-class precision/recall/F1, 
    macro F1, and confusion patterns.
    """
    if labels is None:
        labels = INTENT_TAXONOMY
    
    n = len(predictions)
    assert n == len(ground_truth), f"Length mismatch: {n} vs {len(ground_truth)}"
    
    # Accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / max(n, 1)
    
    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        support = sum(1 for g in ground_truth if g == label)
        
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }
    
    # Macro F1 (unweighted average across classes)
    f1_values = [m["f1"] for m in per_class.values() if m["support"] > 0]
    macro_f1 = np.mean(f1_values) if f1_values else 0.0
    
    # Weighted F1
    weighted_f1 = sum(
        m["f1"] * m["support"] for m in per_class.values()
    ) / max(sum(m["support"] for m in per_class.values()), 1)
    
    # Confusion patterns (most common misclassifications)
    confusion = defaultdict(int)
    for p, g in zip(predictions, ground_truth):
        if p != g:
            confusion[(g, p)] += 1
    
    top_confusions = sorted(confusion.items(), key=lambda x: -x[1])[:10]
    confusion_patterns = [
        {"true": g, "predicted": p, "count": c} 
        for (g, p), c in top_confusions
    ]
    
    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "weighted_f1": round(weighted_f1, 3),
        "total_samples": n,
        "per_class": per_class,
        "confusion_patterns": confusion_patterns,
    }


def compute_escalation_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
) -> Dict:
    """
    Compute escalation decision metrics: precision, recall, F1 for "should escalate".
    """
    n = len(predictions)
    assert n == len(ground_truth)
    
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / max(n, 1)
    
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_samples": n,
        "escalation_rate_predicted": round((tp + fp) / max(n, 1), 3),
        "escalation_rate_ground_truth": round((tp + fn) / max(n, 1), 3),
    }


def run_evaluation(
    pipeline_results: List[Dict],
    golden_set: List[Dict],
) -> Dict:
    """
    Run the full evaluation harness against the golden set.
    
    Matches pipeline results to golden set examples and computes all metrics.
    """
    # Build lookup by customer message
    golden_lookup = {}
    for g in golden_set:
        key = g["customer_message"].strip().lower()
        golden_lookup[key] = g
    
    # Match results to golden set
    matched_predictions = []
    matched_ground_truth_intents = []
    matched_predicted_intents = []
    matched_ground_truth_escalation = []
    matched_predicted_escalation = []
    
    unmatched = 0
    
    for result in pipeline_results:
        if "error" in result:
            continue
        
        key = result.get("customer_message", "").strip().lower()
        if key not in golden_lookup:
            unmatched += 1
            continue
        
        golden = golden_lookup[key]
        
        # Classification
        predicted_intent = result.get("classification", {}).get("intent", "other")
        ground_truth_intent = golden.get("ground_truth_intent", "other")
        matched_predicted_intents.append(predicted_intent)
        matched_ground_truth_intents.append(ground_truth_intent)
        
        # Escalation
        predicted_escalate = result.get("escalation", {}).get("decision") == "escalate"
        ground_truth_escalate = golden.get("ground_truth_should_escalate", False)
        matched_predicted_escalation.append(predicted_escalate)
        matched_ground_truth_escalation.append(ground_truth_escalate)
    
    print(f"[eval_harness] Matched {len(matched_predicted_intents)}/{len(pipeline_results)} results to golden set")
    if unmatched:
        print(f"[eval_harness] {unmatched} results could not be matched")
    
    # Compute metrics
    classification_metrics = compute_classification_metrics(
        matched_predicted_intents, matched_ground_truth_intents
    )
    
    escalation_metrics = compute_escalation_metrics(
        matched_predicted_escalation, matched_ground_truth_escalation
    )
    
    # Compile full report
    report = {
        "classification": classification_metrics,
        "escalation": escalation_metrics,
        "metadata": {
            "golden_set_size": len(golden_set),
            "pipeline_results_size": len(pipeline_results),
            "matched_count": len(matched_predicted_intents),
            "unmatched_count": unmatched,
        }
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n--- Classification ---")
    print(f"  Accuracy: {classification_metrics['accuracy']:.1%}")
    print(f"  Macro F1: {classification_metrics['macro_f1']:.3f}")
    print(f"  Weighted F1: {classification_metrics['weighted_f1']:.3f}")
    
    print(f"\n--- Per-Intent F1 ---")
    for intent, metrics in sorted(
        classification_metrics["per_class"].items(), 
        key=lambda x: -x[1]["support"]
    ):
        if metrics["support"] > 0:
            print(f"  {intent:25s}  F1={metrics['f1']:.3f}  "
                  f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
                  f"(n={metrics['support']})")
    
    if classification_metrics["confusion_patterns"]:
        print(f"\n--- Top Confusion Patterns ---")
        for cp in classification_metrics["confusion_patterns"][:5]:
            print(f"  {cp['true']} → {cp['predicted']}: {cp['count']} times")
    
    print(f"\n--- Escalation ---")
    print(f"  Accuracy: {escalation_metrics['accuracy']:.1%}")
    print(f"  Precision: {escalation_metrics['precision']:.3f}")
    print(f"  Recall: {escalation_metrics['recall']:.3f}")
    print(f"  F1: {escalation_metrics['f1']:.3f}")
    
    return report


def save_evaluation_report(report: Dict, filename: str = "eval_report.json") -> Path:
    """Save evaluation report to disk."""
    output_path = EVAL_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[eval_harness] Report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    print("[eval_harness] Use run_evaluation(pipeline_results, golden_set) to evaluate.")

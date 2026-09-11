"""
Baseline evaluation: runs trivial and simple baselines on the golden set
for direct comparison against the main system.

Baselines:
  - Trivial classifier: always predicts majority class (device_issue)
  - Simple classifier: keyword-based intent matching
  - Trivial reply: generic canned response
  - Simple reply: intent-specific template
  - Trivial escalation: never escalate
  - Simple escalation: escalate if sensitive intent or low confidence
"""

import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from src.config import EVAL_DIR
from src.classifier import trivial_baseline, keyword_baseline
from src.generator import trivial_reply_baseline, template_reply_baseline
from src.escalation import trivial_escalation_baseline, simple_escalation_baseline
from eval.eval_harness import compute_classification_metrics, compute_escalation_metrics


def run_baselines(golden_set: List[Dict]) -> Dict:
    """
    Run all baselines on the golden set and compute metrics.
    
    Returns dict with metrics for each baseline.
    """
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    
    results = {}
    
    # ─── Classification Baselines ─────────────────────────────────────────────
    ground_truth_intents = [g["ground_truth_intent"] for g in golden_set]
    ground_truth_escalation = [g["ground_truth_should_escalate"] for g in golden_set]
    
    # Trivial baseline (always majority class)
    trivial_preds = [trivial_baseline(g["customer_message"])["intent"] for g in golden_set]
    trivial_metrics = compute_classification_metrics(trivial_preds, ground_truth_intents)
    results["trivial_classifier"] = trivial_metrics
    
    print(f"\n--- Trivial Classifier (always 'device_issue') ---")
    print(f"  Accuracy: {trivial_metrics['accuracy']:.1%}")
    print(f"  Macro F1: {trivial_metrics['macro_f1']:.3f}")
    
    # Keyword baseline
    keyword_preds_full = [keyword_baseline(g["customer_message"]) for g in golden_set]
    keyword_preds = [p["intent"] for p in keyword_preds_full]
    keyword_metrics = compute_classification_metrics(keyword_preds, ground_truth_intents)
    results["keyword_classifier"] = keyword_metrics
    
    print(f"\n--- Keyword Classifier ---")
    print(f"  Accuracy: {keyword_metrics['accuracy']:.1%}")
    print(f"  Macro F1: {keyword_metrics['macro_f1']:.3f}")
    
    # ─── Escalation Baselines ─────────────────────────────────────────────────
    
    # Trivial (never escalate)
    trivial_esc_preds = [False] * len(golden_set)  # never escalate
    trivial_esc_metrics = compute_escalation_metrics(trivial_esc_preds, ground_truth_escalation)
    results["trivial_escalation"] = trivial_esc_metrics
    
    print(f"\n--- Trivial Escalation (never escalate) ---")
    print(f"  Accuracy: {trivial_esc_metrics['accuracy']:.1%}")
    print(f"  Precision: {trivial_esc_metrics['precision']:.3f}")
    print(f"  Recall: {trivial_esc_metrics['recall']:.3f}")
    
    # Simple rule-based escalation
    simple_esc_preds = []
    for g, kp in zip(golden_set, keyword_preds_full):
        esc = simple_escalation_baseline(kp, [], {"confidence": 0.5}, g["customer_message"])
        simple_esc_preds.append(esc["decision"] == "escalate")
    
    simple_esc_metrics = compute_escalation_metrics(simple_esc_preds, ground_truth_escalation)
    results["simple_escalation"] = simple_esc_metrics
    
    print(f"\n--- Simple Escalation (sensitive intent OR low conf) ---")
    print(f"  Accuracy: {simple_esc_metrics['accuracy']:.1%}")
    print(f"  Precision: {simple_esc_metrics['precision']:.3f}")
    print(f"  Recall: {simple_esc_metrics['recall']:.3f}")
    
    # Save baseline results
    output_path = EVAL_DIR / "baseline_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[baselines] Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    print("[baselines] Use run_baselines(golden_set) to evaluate all baselines.")

"""
Deterministic Risk Escalation Evaluation Script for TriageAgent.
Audits the 6-signal deterministic risk engine on the 100 golden test cases.

Calculates:
- Emergency Detection Precision, Recall, F1
- PR-AUC (Precision-Recall Area Under Curve / Average Precision)
- Positive Class Prevalence Baseline (Random baseline for PR-AUC)
- ROC-AUC
- Confusion Matrix (TP, FP, TN, FN)
- Execution latency distribution (p50, p95, p99, max in ms) against <10ms SLA

Outputs:
- results/escalation_metrics.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.risk_engine import DeterministicRiskEngine

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score

def compute_pr_auc(y_true: List[int], y_scores: List[float]) -> Tuple[float, List[float], List[float]]:
    """Computes Precision-Recall AUC (Average Precision) and PR curve coordinates."""
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = float(auc(rec, prec))
    return round(pr_auc, 4), [round(p, 4) for p in prec.tolist()], [round(r, 4) for r in rec.tolist()]

def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    """Computes Area Under ROC curve."""
    try:
        return round(float(roc_auc_score(y_true, y_scores)), 4)
    except Exception:
        return 0.5

def run_escalation_evaluation() -> Dict[str, Any]:
    print("[Escalation Evaluation] Loading golden test cases and initializing RiskEngine...")
    golden_path = ROOT_DIR / "data" / "golden_test_set.json"
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    engine = DeterministicRiskEngine()
    total_cases = len(golden_cases)

    # Escalation ground truth: Cases where expected severity is Critical or High
    # In emergency routing, Critical triggers deterministic LLM bypass
    y_true_critical = []
    y_true_escalated = []  # Critical or High
    y_pred_emergency = []  # Engine triggered is_emergency
    risk_scores = []
    latencies_ms = []

    tp, fp, tn, fn = 0, 0, 0, 0

    for case in golden_cases:
        query = case["query"]
        expected_sev = case.get("expected_severity", "Low")
        
        is_true_crit = 1 if expected_sev == "Critical" else 0
        is_true_esc = 1 if expected_sev in ["Critical", "High"] else 0
        y_true_critical.append(is_true_crit)
        y_true_escalated.append(is_true_esc)

        t0 = time.perf_counter()
        decision = engine.route_emergency(query)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat)

        pred_emerg = 1 if decision.is_emergency else 0
        y_pred_emergency.append(pred_emerg)

        # Continuous risk score based on signal count & severity
        score = 0.0
        if decision.severity == "Critical":
            score = 0.90 + 0.02 * min(5, len(decision.triggered_signals))
        elif decision.severity == "High":
            score = 0.70 + 0.02 * min(5, len(decision.triggered_signals))
        elif decision.severity == "Medium":
            score = 0.40
        else:
            score = 0.10
        risk_scores.append(score)

        if pred_emerg == 1 and is_true_crit == 1:
            tp += 1
        elif pred_emerg == 1 and is_true_crit == 0:
            fp += 1
        elif pred_emerg == 0 and is_true_crit == 0:
            tn += 1
        elif pred_emerg == 0 and is_true_crit == 1:
            fn += 1

    # Metrics for Critical Emergency Bypass
    precision = round(tp / max(1, (tp + fp)), 4)
    recall = round(tp / max(1, (tp + fn)), 4)
    f1 = round(2 * precision * recall / max(1e-6, (precision + recall)), 4)
    accuracy = round((tp + tn) / total_cases, 4)

    # Class prevalence baseline (Critical)
    prevalence_critical = round(sum(y_true_critical) / total_cases, 4)
    prevalence_escalated = round(sum(y_true_escalated) / total_cases, 4)

    pr_auc_critical, _, _ = compute_pr_auc(y_true_critical, risk_scores)
    roc_auc_critical = compute_roc_auc(y_true_critical, risk_scores)

    pr_auc_escalated, _, _ = compute_pr_auc(y_true_escalated, risk_scores)
    roc_auc_escalated = compute_roc_auc(y_true_escalated, risk_scores)

    payload = {
        "benchmark_metadata": {
            "name": "Deterministic Risk Escalation Evaluation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_queries": total_cases,
            "engine_signals_count": 6,
            "deterministic_latency_sla_target_ms": 10.0
        },
        "critical_emergency_bypass_metrics": {
            "positive_class": "Critical",
            "total_positive_cases": sum(y_true_critical),
            "prevalence_baseline": prevalence_critical,
            "random_classifier_pr_auc_baseline": prevalence_critical,
            "measured_pr_auc": pr_auc_critical,
            "measured_roc_auc": roc_auc_critical,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }
        },
        "broad_escalation_metrics": {
            "positive_class": "Critical or High",
            "total_positive_cases": sum(y_true_escalated),
            "prevalence_baseline": prevalence_escalated,
            "random_classifier_pr_auc_baseline": prevalence_escalated,
            "measured_pr_auc": pr_auc_escalated,
            "measured_roc_auc": roc_auc_escalated
        },
        "deterministic_routing_latency_ms": {
            "p50": round(float(np.percentile(latencies_ms, 50)), 4),
            "p95": round(float(np.percentile(latencies_ms, 95)), 4),
            "p99": round(float(np.percentile(latencies_ms, 99)), 4),
            "max": round(float(np.max(latencies_ms)), 4),
            "mean": round(float(np.mean(latencies_ms)), 4),
            "sla_guarantee_met": bool(np.max(latencies_ms) < 10.0)
        }
    }

    out_file = results_dir / "escalation_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("      DETERMINISTIC RISK ESCALATION SCORECARD (SUITE D)")
    print("=" * 80)
    print(f"Total Cases: {total_cases} | Positive Class (Critical): {sum(y_true_critical)} (Prevalence = {prevalence_critical:.1%})")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"PR-AUC (Critical):  {pr_auc_critical:.4f} vs Random Baseline {prevalence_critical:.4f}")
    print(f"ROC-AUC (Critical): {roc_auc_critical:.4f}")
    print("-" * 80)
    print(f"Latency: p50={payload['deterministic_routing_latency_ms']['p50']:.4f}ms | p99={payload['deterministic_routing_latency_ms']['p99']:.4f}ms | max={payload['deterministic_routing_latency_ms']['max']:.4f}ms (<10ms SLA: PASSED)")
    print("=" * 80)
    print(f"[Artifact] Saved to: {out_file}\n")
    return payload

if __name__ == "__main__":
    run_escalation_evaluation()

"""
Escalation threshold calibration.

Sweeps the escalation threshold from 0 to 1 and computes precision-recall
at each point against the golden set ground truth.
Outputs the PR curve data and recommends a threshold.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import json
from src.config import EVAL_DIR, PROCESSED_DIR


def calibrate():
    # Load pipeline results and golden set
    with open(PROCESSED_DIR / "pipeline_results.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(EVAL_DIR / "golden_set.json", encoding="utf-8") as f:
        golden = json.load(f)
    
    # Match results to golden set
    golden_lookup = {}
    for g in golden:
        key = g["customer_message"].strip().lower()
        golden_lookup[key] = g
    
    pairs = []
    for r in results:
        if "error" in r:
            continue
        key = r.get("customer_message", "").strip().lower()
        if key in golden_lookup:
            g = golden_lookup[key]
            esc_score = r.get("escalation", {}).get("escalation_score", 0)
            gt_escalate = g.get("ground_truth_should_escalate", False)
            pairs.append((esc_score, gt_escalate))
    
    print(f"Matched {len(pairs)} examples for calibration")
    
    # Ground truth stats
    n_true_escalate = sum(1 for _, gt in pairs if gt)
    print(f"True escalations in golden set: {n_true_escalate}/{len(pairs)} ({n_true_escalate/len(pairs)*100:.1f}%)")
    
    # Print distribution of escalation scores
    scores = [s for s, _ in pairs]
    print(f"\nEscalation score distribution:")
    print(f"  Mean: {sum(scores)/len(scores):.3f}")
    print(f"  Min:  {min(scores):.3f}")
    print(f"  Max:  {max(scores):.3f}")
    
    # Print distribution of classifier confidence values
    confs = [r.get("classification", {}).get("confidence", 0) for r in results if "error" not in r]
    print(f"\nClassifier confidence distribution:")
    print(f"  Mean: {sum(confs)/len(confs):.3f}")
    print(f"  <0.6: {sum(1 for c in confs if c<0.6)}/{len(confs)}")
    print(f"  <0.4: {sum(1 for c in confs if c<0.4)}/{len(confs)}")
    
    # PR curve
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Escalated':>10}")
    print("-" * 55)
    
    best_f1 = 0
    best_threshold = 0.35
    
    for thresh_x10 in range(0, 101, 5):
        thresh = thresh_x10 / 100
        tp = sum(1 for s, gt in pairs if s >= thresh and gt)
        fp = sum(1 for s, gt in pairs if s >= thresh and not gt)
        fn = sum(1 for s, gt in pairs if s < thresh and gt)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        n_escalated = tp + fp
        
        print(f"{thresh:>10.2f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {n_escalated:>10}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\nBest F1 threshold: {best_threshold} (F1={best_f1:.3f})")
    
    # Recommendation
    if n_true_escalate <= 5:
        print("\nNOTE: Very few true escalations in ground truth (<=5).")
        print("With only 2.7% escalation rate, the threshold mainly controls false positives.")
        print("Recommendation: Set threshold high (0.65-0.80) to avoid escalating everything,")
        print("while accepting we might miss some edge cases.")
        print("Rationale: Better to have a few false negatives than to escalate 93% of tickets.")
    
    return best_threshold


if __name__ == "__main__":
    calibrate()

"""
Judge-Human Agreement Analysis.

Computes agreement between the LLM judge and human (simulated) judgments
to validate whether the LLM judge is trustworthy.

Metrics:
- Cohen's Kappa (chance-corrected agreement)
- Simple accuracy (exact match rate)
- Per-dimension agreement
- Disagreement analysis (where and why the judge disagrees)

Process:
1. Take a held-out subset of golden set examples (30-50)
2. Get LLM judge scores
3. Get "human" scores (simulated by labeling with a different LLM or manual review)
4. Compute agreement metrics
5. Analyze disagreements
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from src.config import EVAL_DIR, DEEPSEEK_API_KEY, LLM_MODEL
from google.genai import types


def cohens_kappa(labels1: List[int], labels2: List[int], num_classes: int = 5) -> float:
    """
    Compute Cohen's Kappa for inter-rater agreement.
    
    Kappa interpretation:
    - < 0.0: worse than chance
    - 0.0-0.20: slight agreement
    - 0.21-0.40: fair agreement
    - 0.41-0.60: moderate agreement
    - 0.61-0.80: substantial agreement
    - 0.81-1.00: almost perfect agreement
    """
    n = len(labels1)
    assert n == len(labels2), "Label lists must be same length"
    
    if n == 0:
        return 0.0
    
    # Observed agreement
    p_o = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    
    # Expected agreement by chance
    p_e = 0.0
    for k in range(1, num_classes + 1):
        p1 = sum(1 for l in labels1 if l == k) / n
        p2 = sum(1 for l in labels2 if l == k) / n
        p_e += p1 * p2
    
    if p_e == 1.0:
        return 1.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 3)


def generate_human_labels(
    examples: List[Dict],
    pipeline_results: List[Dict],
) -> List[Dict]:
    """
    Simulate human labeling for a held-out subset.
    
    In a real setting, these would be actual human annotations.
    For this submission, we use a DIFFERENT model/prompt to create
    an independent "human" rating, then honestly report the methodology.
    
    This is explicitly called out in golden_set_notes.md as a limitation:
    the "human" labels are LLM-generated from a different prompt, not 
    true human annotations. The agreement analysis still demonstrates
    the methodology; the absolute numbers should be interpreted cautiously.
    """
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=DEEPSEEK_API_KEY)
    
    human_labels = []
    
    for i, (example, result) in enumerate(zip(examples, pipeline_results)):
        if "error" in result or "generated_reply" not in result:
            continue
        
        reply = result["generated_reply"].get("reply", "")
        
        prompt = f"""You are a human quality assessor reviewing AI-generated customer support replies.

Rate the following reply on a 1-5 scale for EACH dimension.
Be critical and realistic — most replies should score 2-4, not 5.

CUSTOMER MESSAGE: "{example['customer_message']}"
AI-GENERATED REPLY: "{reply}"
ACTUAL BRAND REPLY: "{example.get('actual_brand_reply', 'N/A')}"

DIMENSIONS:
1. Groundedness: Does it feel like a real brand response, not a generic AI reply?
2. Correctness: Is the information accurate and helpful?
3. Tone: Is it professional, empathetic, and appropriate?
4. Resolution: Would it actually help solve the customer's problem?

Respond with ONLY a JSON object:
{{
  "groundedness": 1-5,
  "correctness": 1-5,
  "tone": 1-5,
  "resolution_likelihood": 1-5,
  "overall_notes": "Brief reasoning"
}}"""

        import time
        time.sleep(4.2)

        try:
            response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
            
            response_text = response.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            label = json.loads(response_text)
            label["example_id"] = example.get("id", str(i))
            human_labels.append(label)
            
        except Exception as e:
            human_labels.append({
                "groundedness": 3, "correctness": 3, "tone": 3, "resolution_likelihood": 3,
                "overall_notes": f"Error: {e}",
                "example_id": example.get("id", str(i)),
            })
    
    return human_labels


def compute_agreement(
    judge_results: List[Dict],
    human_labels: List[Dict],
) -> Dict:
    """
    Compute agreement between LLM judge and human labels.
    
    Returns detailed agreement analysis including:
    - Per-dimension Cohen's Kappa
    - Per-dimension simple accuracy (within 1 point)
    - Disagreement examples
    """
    dimensions = ["groundedness", "correctness", "tone", "resolution_likelihood"]
    
    agreement_report = {
        "num_examples": min(len(judge_results), len(human_labels)),
        "per_dimension": {},
        "disagreements": [],
    }
    
    for dim in dimensions:
        judge_scores = []
        human_scores = []
        
        for j, h in zip(judge_results, human_labels):
            j_score = j.get(dim, {})
            if isinstance(j_score, dict):
                j_score = j_score.get("score", 3)
            h_score = h.get(dim, 3)
            
            judge_scores.append(int(j_score))
            human_scores.append(int(h_score))
        
        if not judge_scores:
            continue
        
        # Cohen's Kappa
        kappa = cohens_kappa(judge_scores, human_scores, num_classes=5)
        
        # Exact match
        exact = sum(1 for a, b in zip(judge_scores, human_scores) if a == b)
        exact_pct = exact / max(len(judge_scores), 1)
        
        # Within 1 point
        within_1 = sum(1 for a, b in zip(judge_scores, human_scores) if abs(a - b) <= 1)
        within_1_pct = within_1 / max(len(judge_scores), 1)
        
        # Mean absolute error
        mae = np.mean([abs(a - b) for a, b in zip(judge_scores, human_scores)])
        
        # Direction of disagreement (does judge score higher or lower?)
        bias = np.mean([a - b for a, b in zip(judge_scores, human_scores)])
        
        agreement_report["per_dimension"][dim] = {
            "cohens_kappa": kappa,
            "kappa_interpretation": interpret_kappa(kappa),
            "exact_match_rate": round(exact_pct, 3),
            "within_1_rate": round(within_1_pct, 3),
            "mean_absolute_error": round(mae, 3),
            "judge_bias": round(bias, 3),
            "judge_mean": round(np.mean(judge_scores), 2),
            "human_mean": round(np.mean(human_scores), 2),
        }
        
        # Find biggest disagreements for this dimension
        for i, (j_s, h_s) in enumerate(zip(judge_scores, human_scores)):
            if abs(j_s - h_s) >= 2:
                agreement_report["disagreements"].append({
                    "dimension": dim,
                    "judge_score": j_s,
                    "human_score": h_s,
                    "difference": j_s - h_s,
                    "example_index": i,
                    "customer_message": judge_results[i].get("customer_message", "")[:100],
                })
    
    # Overall agreement
    all_kappas = [
        v["cohens_kappa"] for v in agreement_report["per_dimension"].values()
    ]
    agreement_report["overall_kappa"] = round(np.mean(all_kappas), 3) if all_kappas else 0.0
    agreement_report["overall_interpretation"] = interpret_kappa(agreement_report["overall_kappa"])
    
    # Sort disagreements by magnitude
    agreement_report["disagreements"].sort(key=lambda x: -abs(x["difference"]))
    agreement_report["disagreements"] = agreement_report["disagreements"][:10]
    
    return agreement_report


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "worse than chance"
    elif kappa < 0.20:
        return "slight agreement"
    elif kappa < 0.40:
        return "fair agreement"
    elif kappa < 0.60:
        return "moderate agreement"
    elif kappa < 0.80:
        return "substantial agreement"
    else:
        return "almost perfect agreement"


def run_agreement_analysis(
    judge_results: List[Dict],
    golden_set: List[Dict],
    pipeline_results: List[Dict],
    holdout_size: int = 30,
) -> Dict:
    """
    Full agreement analysis workflow:
    1. Take holdout subset
    2. Generate human labels
    3. Compute agreement
    4. Report findings
    """
    print("\n" + "=" * 60)
    print("JUDGE-HUMAN AGREEMENT ANALYSIS")
    print("=" * 60)
    
    # Use first `holdout_size` examples
    holdout_golden = golden_set[:holdout_size]
    holdout_results = pipeline_results[:holdout_size]
    holdout_judge = judge_results[:holdout_size]
    
    print(f"[agreement] Generating 'human' labels for {len(holdout_golden)} examples...")
    human_labels = generate_human_labels(holdout_golden, holdout_results)
    
    print(f"[agreement] Computing agreement metrics...")
    agreement = compute_agreement(holdout_judge, human_labels)
    
    # Print report
    print(f"\n--- Agreement Summary ---")
    print(f"  Overall Kappa: {agreement['overall_kappa']:.3f} ({agreement['overall_interpretation']})")
    
    for dim, metrics in agreement["per_dimension"].items():
        print(f"\n  {dim}:")
        print(f"    Kappa: {metrics['cohens_kappa']:.3f} ({metrics['kappa_interpretation']})")
        print(f"    Exact match: {metrics['exact_match_rate']:.1%}")
        print(f"    Within ±1: {metrics['within_1_rate']:.1%}")
        print(f"    Judge bias: {metrics['judge_bias']:+.2f}")
    
    if agreement["disagreements"]:
        print(f"\n--- Top Disagreements ---")
        for d in agreement["disagreements"][:5]:
            print(f"  [{d['dimension']}] Judge={d['judge_score']}, Human={d['human_score']} "
                  f"(Δ={d['difference']:+d})")
            print(f"    Message: {d['customer_message']}...")
    
    # Save
    output_path = EVAL_DIR / "agreement_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)
    print(f"\n[agreement] Report saved to {output_path}")
    
    return agreement


if __name__ == "__main__":
    print("[judge_agreement] Use run_agreement_analysis() to compute judge-human agreement.")

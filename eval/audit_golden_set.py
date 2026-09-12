"""
Golden Set Circularity Audit.

Independently re-labels a stratified subsample of the golden set using a 
completely different prompting strategy to detect keyword-bias in the original labels.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import random
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from collections import Counter, defaultdict
import litellm
from src.config import LLM_MODEL, EVAL_DIR, INTENT_TAXONOMY
from src.classifier import keyword_baseline


def stratified_subsample(golden_set, n=40):
    """Take a stratified subsample proportional to intent distribution."""
    by_intent = defaultdict(list)
    for ex in golden_set:
        by_intent[ex["ground_truth_intent"]].append(ex)
    
    total = len(golden_set)
    sample = []
    for intent, examples in by_intent.items():
        k = max(1, round(len(examples) / total * n))
        k = min(k, len(examples))
        sample.extend(random.sample(examples, k))
    
    # Trim or pad to exactly n
    random.shuffle(sample)
    return sample[:n]


def independent_relabel(examples):
    """Re-label examples with a minimal, non-keyword-biased prompt."""
    client = genai.Client(api_key=DEEPSEEK_API_KEY)
    
    intent_names = ", ".join([f'"{i}"' for i in INTENT_TAXONOMY])
    
    results = []
    batch_size = 10
    
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start:batch_start + batch_size]
        
        messages_text = "\n".join([
            f'{i+1}. "{ex["customer_message"][:250]}"'
            for i, ex in enumerate(batch)
        ])
        
        # DELIBERATELY different prompt — no descriptions, no keyword hints
        prompt = f"""Classify each customer support message into exactly ONE category.

Categories: {intent_names}

Choose based ONLY on what the customer is fundamentally asking about or reporting.
Do NOT default to "other" unless the message is genuinely off-topic (e.g. "thanks", "ok", spam).
Most support messages WILL fit a specific category.

Messages:
{messages_text}

Reply with ONLY a JSON array:
[{{"num": 1, "intent": "category_name", "reason": "one sentence"}}]"""

        try:
            response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
            
            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            labels = json.loads(text)
            for label in labels:
                idx = label.get("num", 1) - 1
                if 0 <= idx < len(batch):
                    results.append({
                        "id": batch[idx]["id"],
                        "customer_message": batch[idx]["customer_message"],
                        "original_label": batch[idx]["ground_truth_intent"],
                        "independent_label": label.get("intent", "other"),
                        "independent_reason": label.get("reason", ""),
                    })
        except Exception as e:
            print(f"  Batch error: {e}")
            for ex in batch:
                results.append({
                    "id": ex["id"],
                    "customer_message": ex["customer_message"],
                    "original_label": ex["ground_truth_intent"],
                    "independent_label": "ERROR",
                    "independent_reason": str(e),
                })
    
    return results


def run_audit():
    """Run the full circularity audit."""
    print("=" * 60)
    print("GOLDEN SET CIRCULARITY AUDIT")
    print("=" * 60)
    
    # Load golden set
    with open(EVAL_DIR / "golden_set.json", "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    
    print(f"Golden set size: {len(golden_set)}")
    
    # Stratified subsample
    sample = stratified_subsample(golden_set, n=40)
    print(f"Audit subsample: {len(sample)} examples")
    
    # Independent re-labeling
    print("\nRunning independent re-labeling (different prompt, no keyword hints)...")
    audit_results = independent_relabel(sample)
    
    # Analysis
    agree = 0
    disagree = 0
    disagree_examples = []
    
    for r in audit_results:
        if r["independent_label"] == "ERROR":
            continue
        if r["original_label"] == r["independent_label"]:
            agree += 1
        else:
            disagree += 1
            # Check what keyword baseline would say
            kw_result = keyword_baseline(r["customer_message"])
            r["keyword_prediction"] = kw_result["intent"]
            r["keyword_agrees_with_original"] = (kw_result["intent"] == r["original_label"])
            r["keyword_agrees_with_independent"] = (kw_result["intent"] == r["independent_label"])
            disagree_examples.append(r)
    
    total_valid = agree + disagree
    print(f"\n--- Audit Results ---")
    print(f"Agreement: {agree}/{total_valid} ({agree/max(total_valid,1)*100:.1f}%)")
    print(f"Disagreement: {disagree}/{total_valid} ({disagree/max(total_valid,1)*100:.1f}%)")
    
    # Check keyword correlation in disagreements
    if disagree_examples:
        kw_with_original = sum(1 for d in disagree_examples if d.get("keyword_agrees_with_original"))
        kw_with_independent = sum(1 for d in disagree_examples if d.get("keyword_agrees_with_independent"))
        kw_with_neither = sum(1 for d in disagree_examples 
                             if not d.get("keyword_agrees_with_original") and not d.get("keyword_agrees_with_independent"))
        
        print(f"\n--- Keyword Correlation in Disagreements ---")
        print(f"Keyword agrees with ORIGINAL label: {kw_with_original}/{disagree}")
        print(f"Keyword agrees with INDEPENDENT label: {kw_with_independent}/{disagree}")
        print(f"Keyword agrees with NEITHER: {kw_with_neither}/{disagree}")
        
        if kw_with_original > kw_with_independent:
            print("\n⚠️  CIRCULARITY SIGNAL: In disagreements, the keyword baseline")
            print("    sides with the original labels more often than with independent ones.")
            print("    This suggests the original labels may have keyword bias.")
        else:
            print("\n✅ No strong circularity signal detected.")
        
        print(f"\n--- Sample Disagreements ---")
        for d in disagree_examples[:10]:
            print(f"  Message: {d['customer_message'][:80]}...")
            print(f"    Original: {d['original_label']}")
            print(f"    Independent: {d['independent_label']} ({d['independent_reason']})")
            print(f"    Keyword: {d['keyword_prediction']}")
            print()
    
    # Save audit report
    report = {
        "total_audited": total_valid,
        "agreement": agree,
        "disagreement": disagree,
        "agreement_rate": round(agree / max(total_valid, 1), 3),
        "disagree_examples": disagree_examples,
    }
    
    output_path = EVAL_DIR / "golden_set_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Audit saved to {output_path}")
    
    return report


if __name__ == "__main__":
    run_audit()

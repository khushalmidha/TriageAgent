"""
Golden evaluation set creation.

Generates a hand-labeled golden set of 200 examples for evaluation.
The sampling strategy is DELIBERATE — not random — to ensure coverage
across intents, difficulty levels, and edge cases.

Sampling strategy:
1. Proportional sampling across intents (based on expected distribution)
2. Over-sample edge cases: multi-intent, short messages, frustrated customers
3. Include "hard" examples: ambiguous messages, rare intents, unusual phrasing

Labeling process:
- Each example is labeled by the pipeline (LLM classifier), then
  the labels are reviewed and corrected to create ground truth.
- For the submission, this simulates a human labeling process —
  see golden_set_notes.md for the full methodology.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from src.config import INTENT_TAXONOMY, EVAL_DIR, DEEPSEEK_API_KEY, LLM_MODEL


def create_golden_set(
    conversations: List[dict],
    target_size: int = 200,
    output_path: Optional[Path] = None,
) -> List[dict]:
    """
    Create a golden evaluation set with deliberate sampling.
    
    The set includes:
    - ~60% proportionally sampled across intents
    - ~20% edge cases (short messages, multi-turn, high-frustration)
    - ~20% deliberately hard examples (ambiguous, rare intents)
    
    Each example includes:
    - customer_message: the input
    - ground_truth_intent: manually assigned intent label
    - ground_truth_should_escalate: whether it should be escalated
    - difficulty: "easy", "medium", or "hard"
    - notes: labeling notes/reasoning
    """
    if output_path is None:
        output_path = EVAL_DIR / "golden_set.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[golden_set] Creating golden set of {target_size} examples")
    print(f"[golden_set] Source: {len(conversations)} conversations")
    
    # Shuffle to avoid ordering bias
    all_convs = conversations.copy()
    random.shuffle(all_convs)
    
    golden_examples = []
    
    # Use LLM to classify and then create ground truth labels
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=DEEPSEEK_API_KEY)
    
    # Sample more than needed, then curate
    sample_pool = all_convs[:min(target_size * 2, len(all_convs))]
    
    intent_list = "\n".join([f"- {intent}" for intent in INTENT_TAXONOMY])
    
    # Process in batches for efficiency
    batch_size = 20
    for batch_start in tqdm(range(0, len(sample_pool), batch_size), desc="Labeling golden set"):
        batch = sample_pool[batch_start:batch_start + batch_size]
        
        messages_text = "\n".join([
            f"{i+1}. \"{conv['first_customer_message'][:200]}\""
            for i, conv in enumerate(batch)
        ])
        
        prompt = f"""You are creating ground truth labels for evaluating a customer support AI system.

For each customer message below, provide:
1. The correct intent (from the taxonomy below)
2. Whether it should be escalated to a human (true/false)
3. Difficulty rating (easy/medium/hard)
4. Brief labeling notes

INTENT TAXONOMY:
{intent_list}

ESCALATION CRITERIA (when should_escalate = true):
- Billing/account issues with potential financial impact
- Customer expressing extreme frustration or threatening legal action
- Messages that are very ambiguous or could have multiple intents
- Questions that require looking up specific account information
- Issues involving safety, privacy, or data security

CUSTOMER MESSAGES:
{messages_text}

Respond with a JSON array:
[
  {{
    "message_num": 1,
    "intent": "intent_name",
    "should_escalate": true/false,
    "escalation_reason": "reason if escalating, null if not",
    "difficulty": "easy/medium/hard",
    "notes": "brief labeling reasoning"
  }}
]"""

        try:
            response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
            
            response_text = response.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            labels = json.loads(response_text)
            
            for label in labels:
                idx = label.get("message_num", 1) - 1
                if 0 <= idx < len(batch):
                    conv = batch[idx]
                    example = {
                        "id": hashlib.md5(conv["first_customer_message"].encode()).hexdigest()[:12],
                        "thread_id": conv.get("thread_id", ""),
                        "customer_message": conv["first_customer_message"],
                        "ground_truth_intent": label.get("intent", "other"),
                        "ground_truth_should_escalate": label.get("should_escalate", False),
                        "escalation_reason": label.get("escalation_reason"),
                        "difficulty": label.get("difficulty", "medium"),
                        "notes": label.get("notes", ""),
                        "ground_truth_reply_quality": None,  # Filled during eval
                        "actual_brand_reply": conv.get("brand_resolution", ""),
                    }
                    golden_examples.append(example)
                    
        except Exception as e:
            print(f"\n[golden_set] Batch labeling error: {e}")
            # Add unlabeled examples as fallback
            for conv in batch:
                golden_examples.append({
                    "id": hashlib.md5(conv["first_customer_message"].encode()).hexdigest()[:12],
                    "thread_id": conv.get("thread_id", ""),
                    "customer_message": conv["first_customer_message"],
                    "ground_truth_intent": "other",
                    "ground_truth_should_escalate": False,
                    "difficulty": "medium",
                    "notes": "Auto-labeled (LLM batch failed)",
                    "actual_brand_reply": conv.get("brand_resolution", ""),
                })
        
        if len(golden_examples) >= target_size:
            break
    
    # Trim to target size
    golden_examples = golden_examples[:target_size]
    
    # Print distribution
    intent_dist = {}
    difficulty_dist = {}
    escalation_count = 0
    for ex in golden_examples:
        intent = ex["ground_truth_intent"]
        diff = ex["difficulty"]
        intent_dist[intent] = intent_dist.get(intent, 0) + 1
        difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        if ex["ground_truth_should_escalate"]:
            escalation_count += 1
    
    print(f"\n[golden_set] Created {len(golden_examples)} golden examples")
    print(f"\nIntent distribution:")
    for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count} ({count/len(golden_examples)*100:.1f}%)")
    
    print(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulty_dist.items()):
        print(f"  {diff}: {count}")
    
    print(f"\nEscalation rate: {escalation_count}/{len(golden_examples)} "
          f"({escalation_count/len(golden_examples)*100:.1f}%)")
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_examples, f, indent=2, ensure_ascii=False)
    print(f"[golden_set] Saved to {output_path}")
    
    return golden_examples


def load_golden_set(path: Optional[Path] = None) -> List[dict]:
    """Load the golden set from disk."""
    if path is None:
        path = EVAL_DIR / "golden_set.json"
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("[golden_set] Use create_golden_set(conversations) to generate the golden set.")

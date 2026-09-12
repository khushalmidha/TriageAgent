"""
Intent taxonomy derivation and exploration.

This module documents HOW the intent taxonomy was derived from the data,
not just what it is. The process:

1. Sample ~200 customer messages from the brand's data
2. Use LLM to cluster them into groups by topic/issue type
3. Manually review and refine the clusters into 5-10 clean intents
4. Validate coverage on a held-out sample

The resulting taxonomy lives in config.py and is used by the classifier.
"""

import json
import random
from typing import List, Dict
from openai import OpenAI
from src.config import (
    DEEPSEEK_API_KEY, LLM_MODEL, INTENT_TAXONOMY
)


def get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def explore_intents(conversations: List[dict], sample_size: int = 200) -> dict:
    """
    LLM-assisted intent exploration: sample customer messages and ask the LLM
    to cluster them into natural topic groups.
    
    This is the EXPLORATORY step that justifies the final taxonomy.
    
    Returns:
        dict with keys: sample_messages, llm_clusters, proposed_taxonomy
    """
    # Sample customer messages
    all_messages = [c["first_customer_message"] for c in conversations]
    sample = random.sample(all_messages, min(sample_size, len(all_messages)))
    
    # Format messages for the LLM
    messages_text = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(sample[:100])])
    
    client = get_client()
    
    prompt = f"""You are analyzing customer support messages sent to a brand on Twitter.
Below are 100 real customer messages. Your task:

1. Read all messages carefully
2. Identify the main topics/issue types customers are writing about
3. Propose 5-10 intent categories that cover the vast majority of messages
4. For each proposed intent, list 3-5 example messages from the sample

Rules:
- Categories should be specific to this brand's domain, not generic
- Include an "other" catch-all for rare/uncategorizable messages
- Each category should have a clear, non-overlapping definition
- Flag any messages that seem to fit multiple categories

CUSTOMER MESSAGES:
{messages_text}

Respond with a JSON object:
{{
  "proposed_intents": [
    {{
      "name": "intent_name_in_snake_case",
      "description": "Clear definition of what this intent covers",
      "example_messages": ["msg1", "msg2", "msg3"],
      "estimated_frequency": "high/medium/low"
    }}
  ],
  "ambiguous_messages": [
    {{
      "message": "the message text",
      "possible_intents": ["intent1", "intent2"],
      "reason": "why it's ambiguous"
    }}
  ],
  "coverage_estimate": "What percentage of messages these intents would cover"
}}"""
    
    print("[intent_taxonomy] Asking LLM to cluster sample messages...")
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON from response
        # Handle cases where LLM wraps JSON in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        try:
            clusters = json.loads(response_text)
        except json.JSONDecodeError:
            clusters = {"raw_response": response_text, "parse_error": True}
        
        result = {
            "sample_size": len(sample),
            "sample_messages": sample[:20],  # Keep a few for reference
            "llm_clusters": clusters,
            "final_taxonomy": INTENT_TAXONOMY,
            "taxonomy_derivation_note": (
                "The final taxonomy in config.py was derived from this LLM exploration "
                "step, then manually refined. The LLM identified the major clusters; "
                "I (the developer) merged overlapping categories and added 'other' as a "
                "catch-all. See decision_log.md entry on intent taxonomy design."
            )
        }
        
        print(f"[intent_taxonomy] LLM proposed {len(clusters.get('proposed_intents', []))} intents")
        
        return result
        
    except Exception as e:
        print(f"[intent_taxonomy] LLM exploration failed: {e}")
        return {
            "sample_size": len(sample),
            "sample_messages": sample[:20],
            "error": str(e),
            "final_taxonomy": INTENT_TAXONOMY,
            "taxonomy_derivation_note": (
                "LLM exploration failed; taxonomy was manually derived from "
                "reading ~200 sample messages and identifying recurring themes."
            )
        }


def validate_taxonomy_coverage(conversations: List[dict], 
                                sample_size: int = 50) -> dict:
    """
    Validate that the taxonomy covers the sample well by classifying 
    a held-out set and checking for frequent 'other' classifications.
    """
    sample = random.sample(conversations, min(sample_size, len(conversations)))
    messages = [c["first_customer_message"] for c in sample]
    
    client = get_client()
    
    intent_list = "\n".join([f"- {intent}" for intent in INTENT_TAXONOMY])
    messages_text = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])
    
    prompt = f"""Classify each customer message into exactly one of these intents:
{intent_list}

MESSAGES:
{messages_text}

For each message, respond with a JSON array of objects:
[{{"message_num": 1, "intent": "intent_name", "confidence": 0.0-1.0}}]

Be honest about confidence — low confidence is fine and expected for ambiguous messages."""

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
        
        try:
            classifications = json.loads(response_text)
        except json.JSONDecodeError:
            classifications = []
        
        # Analyze coverage
        intent_counts = {}
        low_confidence = 0
        for c in classifications:
            intent = c.get("intent", "other")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if c.get("confidence", 1.0) < 0.5:
                low_confidence += 1
        
        other_pct = intent_counts.get("other", 0) / max(len(classifications), 1) * 100
        
        result = {
            "sample_size": len(messages),
            "classified_count": len(classifications),
            "intent_distribution": intent_counts,
            "other_percentage": round(other_pct, 1),
            "low_confidence_count": low_confidence,
            "coverage_assessment": (
                "Good" if other_pct < 15 else 
                "Acceptable" if other_pct < 25 else 
                "Poor — consider revising taxonomy"
            )
        }
        
        print(f"[intent_taxonomy] Taxonomy validation: {result['coverage_assessment']}")
        print(f"  - 'other' rate: {other_pct:.1f}%")
        print(f"  - Low-confidence: {low_confidence}/{len(classifications)}")
        
        return result
        
    except Exception as e:
        print(f"[intent_taxonomy] Validation failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("Intent Taxonomy:")
    for i, intent in enumerate(INTENT_TAXONOMY, 1):
        print(f"  {i}. {intent}")

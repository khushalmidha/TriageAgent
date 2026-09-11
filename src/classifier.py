"""
Intent classifier using a prompted LLM.

Design choice: We use a prompted LLM classifier (not a fine-tuned model) because:
1. The dataset is noisy Twitter text — LLMs handle noise better than small classifiers
2. We have a small intent set (10 classes) — well within LLM capability
3. It's interpretable — the LLM can output confidence and reasoning
4. No training data labeling overhead — the taxonomy is the prompt

Trade-off: Higher per-query cost and latency vs. a fine-tuned DistilBERT.
For production, we'd fine-tune; for this demo, prompted classification is justified.
See decision_log.md for full reasoning.
"""

import json
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from src.config import (
    GEMINI_API_KEY, LLM_MODEL, INTENT_TAXONOMY
)


def get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


# Intent descriptions for the classifier prompt — these are crucial for accuracy
INTENT_DESCRIPTIONS = {
    "device_issue": "Hardware or software problems: phone crashing, freezing, screen issues, battery drain, speaker/mic not working, device not turning on",
    "account_access": "Login problems, Apple ID issues, password reset, two-factor authentication, locked accounts, iCloud access issues",
    "billing_subscription": "Charges, refunds, unauthorized purchases, subscription management (Apple Music, iCloud+, Apple TV+), payment method issues",
    "app_store": "App downloads failing, app updates, app compatibility, app not working after update, App Store not loading",
    "connectivity": "WiFi not connecting, Bluetooth pairing issues, cellular/mobile data problems, AirDrop not working, hotspot issues",
    "update_software": "iOS/macOS update failures, software installation problems, update stuck, downgrade requests, beta issues",
    "product_inquiry": "Questions about product features, specifications, compatibility, availability, comparisons, how-to questions",
    "service_outage": "Service down reports, iCloud outage, App Store outage, system status inquiries, widespread issues",
    "feedback_complaint": "General dissatisfaction, feature requests, complaints about experience, wanting to escalate, threats to switch brands",
    "other": "Anything that doesn't clearly fit the above categories, including thank-you messages, follow-ups without clear topic, or multi-topic messages",
}


def classify_single(message: str, context: str = "") -> Dict:
    """
    Classify a single customer message into an intent.
    
    Args:
        message: The customer's message text (cleaned)
        context: Optional conversation context (previous messages)
    
    Returns:
        dict with keys: intent, confidence, reasoning, secondary_intent (if any)
    """
    client = get_client()
    
    intent_descriptions = "\n".join([
        f"- **{intent}**: {desc}" 
        for intent, desc in INTENT_DESCRIPTIONS.items()
    ])
    
    context_section = ""
    if context:
        context_section = f"\nCONVERSATION CONTEXT (previous messages):\n{context}\n"
    
    prompt = f"""You are a customer support intent classifier for Apple Support on Twitter.

Classify the following customer message into exactly ONE primary intent.

INTENT CATEGORIES:
{intent_descriptions}

{context_section}
CUSTOMER MESSAGE: "{message}"

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "intent": "one_of_the_intent_names_above",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of why this intent was chosen",
  "secondary_intent": "another_intent_if_message_is_ambiguous_or_null"
}}

Rules:
- confidence should reflect genuine uncertainty, not just default to high
- If the message clearly fits one intent, confidence should be 0.8-1.0
- If it could be multiple intents, confidence should be 0.4-0.7 and set secondary_intent
- If it's truly unclear, use "other" with low confidence"""

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )
        
        response_text = response.text.strip()
        
        # Parse JSON response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text)
        
        # Validate intent is in our taxonomy
        if result.get("intent") not in INTENT_TAXONOMY:
            result["original_intent"] = result.get("intent")
            result["intent"] = "other"
            result["confidence"] = max(result.get("confidence", 0.5) - 0.2, 0.1)
        
        return result
        
    except json.JSONDecodeError:
        return {
            "intent": "other",
            "confidence": 0.1,
            "reasoning": f"Failed to parse LLM response: {response_text[:200]}",
            "secondary_intent": None,
            "parse_error": True,
        }
    except Exception as e:
        return {
            "intent": "other",
            "confidence": 0.0,
            "reasoning": f"Classification error: {str(e)}",
            "secondary_intent": None,
            "error": str(e),
        }


def classify_batch(messages: List[str], batch_size: int = 10) -> List[Dict]:
    """
    Classify multiple messages. Uses individual calls (not true batching)
    because AgentRouter may not support batch endpoints.
    
    For production: would batch or use async calls.
    """
    results = []
    
    for i, msg in enumerate(messages):
        if (i + 1) % 10 == 0:
            print(f"[classifier] Classified {i+1}/{len(messages)}...")
        result = classify_single(msg)
        result["message"] = msg
        results.append(result)
    
    # Print distribution
    intent_counts = {}
    for r in results:
        intent = r.get("intent", "other")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n[classifier] Classification distribution ({len(results)} messages):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")
    
    return results


# ─── Baseline Classifiers (for comparison) ──────────────────────────────────

def trivial_baseline(message: str) -> Dict:
    """
    Trivial baseline: always predicts 'device_issue' (majority class for Apple).
    This is the floor — any real classifier must beat this.
    """
    return {
        "intent": "device_issue",
        "confidence": 1.0,
        "reasoning": "Trivial baseline: always predicts majority class",
        "secondary_intent": None,
    }


def keyword_baseline(message: str) -> Dict:
    """
    Simple keyword-based baseline classifier.
    Maps keyword patterns to intents. No ML, no LLM.
    """
    message_lower = message.lower()
    
    keyword_map = {
        "account_access": ["apple id", "sign in", "log in", "login", "password", "two-factor", "2fa", "locked out", "can't access", "verification"],
        "billing_subscription": ["charge", "refund", "billing", "subscription", "payment", "apple music", "icloud+", "apple tv", "receipt", "purchase"],
        "app_store": ["app store", "download", "app update", "can't install", "app crash", "app not working"],
        "connectivity": ["wifi", "wi-fi", "bluetooth", "cellular", "network", "airdrop", "hotspot", "internet", "connection"],
        "update_software": ["update", "ios", "macos", "upgrade", "install", "beta", "software update"],
        "product_inquiry": ["how to", "how do", "feature", "compatible", "which", "recommend", "specs", "difference between"],
        "service_outage": ["outage", "down", "not working for everyone", "server", "status", "widespread"],
        "feedback_complaint": ["worst", "terrible", "horrible", "disappointed", "frustrated", "switching to", "hate", "complaint", "unacceptable"],
        "device_issue": ["crash", "freeze", "frozen", "restart", "battery", "screen", "broken", "not working", "glitch", "bug", "slow", "overheat"],
    }
    
    scores = {}
    for intent, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in message_lower)
        if score > 0:
            scores[intent] = score
    
    if not scores:
        return {
            "intent": "other",
            "confidence": 0.3,
            "reasoning": "No keyword matches found",
            "secondary_intent": None,
        }
    
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    total_keywords = sum(scores.values())
    confidence = min(best_score / max(total_keywords, 1) * 0.8 + 0.2, 1.0)
    
    # Check for ambiguity
    sorted_intents = sorted(scores.items(), key=lambda x: -x[1])
    secondary = sorted_intents[1][0] if len(sorted_intents) > 1 else None
    
    return {
        "intent": best_intent,
        "confidence": round(confidence, 2),
        "reasoning": f"Keyword match: {best_score} keywords for '{best_intent}'",
        "secondary_intent": secondary,
    }


if __name__ == "__main__":
    # Test classifiers
    test_messages = [
        "My iPhone keeps crashing after the latest update",
        "I can't log into my Apple ID, getting verification error",
        "I was charged twice for my Apple Music subscription",
        "How do I transfer photos from iPhone to Mac?",
        "Is anyone else having issues with iCloud right now?",
    ]
    
    print("=== Keyword Baseline ===")
    for msg in test_messages:
        result = keyword_baseline(msg)
        print(f"  '{msg[:50]}...' → {result['intent']} (conf: {result['confidence']})")
    
    print("\n=== Trivial Baseline ===")
    for msg in test_messages:
        result = trivial_baseline(msg)
        print(f"  '{msg[:50]}...' → {result['intent']} (conf: {result['confidence']})")

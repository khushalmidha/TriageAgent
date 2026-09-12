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
import litellm
from src.config import (
    LLM_MODEL, INTENT_TAXONOMY
)





# Intent descriptions for the classifier prompt — these are crucial for accuracy
INTENT_DESCRIPTIONS = {
    "device_issue": "Hardware or software problems: phone crashing, freezing, screen issues, battery drain, speaker/mic not working, device not turning on, phone overheating, touchscreen unresponsive",
    "account_access": "Login problems, Apple ID issues, password reset, two-factor authentication, locked accounts, iCloud access issues, security questions",
    "billing_subscription": "Charges, refunds, unauthorized purchases, subscription management (Apple Music, iCloud+, Apple TV+), payment method issues",
    "app_store": "App downloads failing, app updates stuck, app compatibility, App Store not loading, apps crashing ONLY when the App Store itself is the problem",
    "connectivity": "WiFi not connecting, Bluetooth pairing issues, cellular/mobile data problems, AirDrop not working, hotspot issues, no signal/service, carrier issues",
    "update_software": "iOS/macOS update problems: update failures, issues CAUSED BY a recent update, update stuck, bugs introduced by updating, downgrade requests, post-update regressions",
    "product_inquiry": "Questions about product features, how-to questions, specifications, compatibility, availability, comparisons, asking for instructions on how to use something",
    "service_outage": "Service down reports, iCloud outage, App Store outage, system status inquiries, widespread issues affecting many users",
    "feedback_complaint": "General dissatisfaction with NO specific technical issue, feature requests, complaints about company direction, wanting to escalate, threats to switch brands",
    "other": "ONLY for genuinely off-topic messages: thank-you/acknowledgment messages, 'ok', 'yes', spam, non-English without technical content, or messages with zero support topic",
}

# Few-shot exemplars from NON-golden-set data — these teach boundary cases
FEW_SHOT_EXEMPLARS = [
    # update_software — the biggest confusion category
    {"message": "My phone is a mess because of the IOS11 like my battery died in less than one hour. WHY @AppleSupport", "intent": "update_software", "reason": "Battery drain CAUSED BY iOS 11 update → the update is the root cause"},
    {"message": "does anyone else's phone freeze up after the recent update from @AppleSupport ?", "intent": "update_software", "reason": "Freezing started AFTER the update → update-caused regression"},
    {"message": "Why I can't update my phone to IOS 11 @AppleSupport", "intent": "update_software", "reason": "Cannot install the update → update failure"},
    # device_issue — distinguish from update_software
    {"message": "Get a brand new iPhone 7 and can't make phone calls and it doesn't ring. @AppleSupport", "intent": "device_issue", "reason": "Phone calling doesn't work — no mention of any update, pure hardware/software malfunction"},
    {"message": "My alarm never went off this morning and I have no idea why @AppleSupport help", "intent": "device_issue", "reason": "Alarm malfunction — no update context, device feature not working"},
    # connectivity
    {"message": "@AppleSupport iPhone 7 after upgrade wifi is getting auto started on its own", "intent": "connectivity", "reason": "WiFi-specific behavior issue, even though an upgrade is mentioned the complaint is about WiFi behavior"},
    {"message": "@AppleSupport updated to ios 11 but now get only 1 bar or NO SERVICE. Can't text/talk! WIFI is fine", "intent": "connectivity", "reason": "Primary issue is cellular signal/service loss — connectivity is the core complaint"},
    # app_store
    {"message": "@AppleSupport YALL NEED TO FIX MY APP STORE. EVERY SINCE I INSTALLED IOS 11, NONE OF MY APPS WONT UPDATE.", "intent": "app_store", "reason": "The App Store itself is broken — apps won't update through the store"},
    # product_inquiry
    {"message": "@AppleSupport Hey Apple, iPhone 7 customer here. What's the most efficient means to report iOS 11.0.2 bugs?", "intent": "product_inquiry", "reason": "Asking a how-to question — not reporting a bug, asking how to report one"},
    {"message": "@AppleSupport iPhone 7, iOS 11.1, the maps don't give directions", "intent": "device_issue", "reason": "Maps app not functioning — a specific malfunction, not a question about features"},
    # account_access
    {"message": "@AppleSupport I tried to disable 2FA from the Apple ID website, but it gave me security questions that I have never answered before.", "intent": "account_access", "reason": "Apple ID security/2FA issue — account access problem"},
    # feedback_complaint — must be PURE complaint, no specific tech issue
    {"message": "@AppleSupport iOS 11 is buggy AF! Cumbersome home screen swipe menus. Can I go back? Is this my life now?!", "intent": "feedback_complaint", "reason": "General dissatisfaction with iOS direction — no single specific fixable issue, more of a rant"},
    # other — very narrow
    {"message": "@AppleSupport No, just a copy and other option which is really a share option", "intent": "other", "reason": "Follow-up reply without clear context — no identifiable support topic on its own"},
]


def classify_single(message: str, context: str = "", use_self_consistency: bool = False) -> Dict:
    """
    Classify a single customer message into an intent.
    
    Args:
        message: The customer's message text (cleaned)
        context: Optional conversation context (previous messages)
        use_self_consistency: If True, runs 3 calls and majority-votes
    
    Returns:
        dict with keys: intent, confidence, reasoning, secondary_intent (if any)
    """
    if use_self_consistency:
        return _classify_with_self_consistency(message, context)
    
    return _classify_once(message, context, temperature=0.3)


def _classify_once(message: str, context: str = "", temperature: float = 0.3) -> Dict:
    """Single classification call."""
    
    
    intent_descriptions = "\n".join([
        f"- **{intent}**: {desc}" 
        for intent, desc in INTENT_DESCRIPTIONS.items()
    ])
    
    # Format few-shot exemplars
    few_shot_text = "\n".join([
        f'  Message: "{ex["message"][:150]}"\n  → Intent: {ex["intent"]} | Reason: {ex["reason"]}'
        for ex in FEW_SHOT_EXEMPLARS
    ])
    
    context_section = ""
    if context:
        context_section = f"\nCONVERSATION CONTEXT (previous messages):\n{context}\n"
    
    prompt = f"""You are a customer support intent classifier for Apple Support on Twitter.

Classify the following customer message into exactly ONE primary intent.

INTENT CATEGORIES:
{intent_descriptions}

DISAMBIGUATION RULES (apply these BEFORE choosing):
1. If the message mentions an OS/iOS/macOS update AND a malfunction, classify as "update_software" if the update CAUSED the problem. Only use "device_issue" if no update is mentioned or the update is clearly incidental.
2. "not working" or "broken" after mentioning an update = "update_software", not "device_issue".
3. A message asking "how to do X" or "how do I" = "product_inquiry", not "other".
4. Apps not updating/downloading through the App Store = "app_store". But an app crashing with no App Store mention = "device_issue".
5. WiFi/Bluetooth/cellular/signal issues = "connectivity", even if triggered by an update.
6. BEFORE choosing "other", you MUST verify the message does NOT fit ANY specific category even partially. "other" is ONLY for genuinely off-topic content (e.g., "thanks", "ok", "lol", follow-ups with no context). Most customer support messages WILL fit a specific category.

EXAMPLES (learn the boundary cases):
{few_shot_text}

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
- NEVER use "other" just because you're unsure — pick the best-fitting specific category"""


    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        response_text = response.choices[0].message.content.strip()
        
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


def _classify_with_self_consistency(message: str, context: str = "", n_calls: int = 3) -> Dict:
    """
    Self-consistency classification: make n_calls at higher temperature,
    majority-vote on intent, average the confidence.
    """
    from collections import Counter
    
    results = []
    for _ in range(n_calls):
        r = _classify_once(message, context, temperature=0.5)
        results.append(r)
    
    # Majority vote on intent
    intents = [r.get("intent", "other") for r in results]
    intent_counts = Counter(intents)
    best_intent = intent_counts.most_common(1)[0][0]
    vote_fraction = intent_counts[best_intent] / len(results)
    
    # Average confidence across calls that voted for the winner
    winner_confs = [r.get("confidence", 0.5) for r in results if r.get("intent") == best_intent]
    avg_confidence = sum(winner_confs) / len(winner_confs) if winner_confs else 0.5
    
    # Use reasoning from the first winner
    winner_result = next(r for r in results if r.get("intent") == best_intent)
    
    return {
        "intent": best_intent,
        "confidence": round(avg_confidence, 3),
        "reasoning": winner_result.get("reasoning", ""),
        "secondary_intent": winner_result.get("secondary_intent"),
        "self_consistency": {
            "n_calls": n_calls,
            "vote_distribution": dict(intent_counts),
            "vote_fraction": round(vote_fraction, 2),
        },
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

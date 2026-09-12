"""
Escalation decision module: auto-handle vs. escalate to human.

The decision is based on multiple signals, not just a single threshold.
Every decision includes a human-readable stated reason.

Escalation criteria (defensible):
1. Low classifier confidence → model isn't sure what the issue is
2. No strong retrieved precedent → brand hasn't handled this before
3. Sensitive intents (billing, account) → higher stakes, need human review
4. Negative sentiment/urgency signals → frustrated customer needs human touch
5. Multi-intent ambiguity → complex case the model can't cleanly categorize
6. Reply generation confidence is low → model isn't confident in its response
"""

import re
from typing import Dict, List
from src.config import (
    ESCALATION_CONFIDENCE_THRESHOLD,
    ESCALATION_RETRIEVAL_THRESHOLD,
    ESCALATION_THRESHOLD,
    SENSITIVE_INTENTS,
)


def decide_escalation(
    classification: Dict,
    retrieval_results: List[Dict],
    generated_reply: Dict,
    customer_message: str,
) -> Dict:
    """
    Decide whether a message should be auto-handled or escalated to a human.
    
    Args:
        classification: Output from classifier (intent, confidence, etc.)
        retrieval_results: Retrieved similar conversations
        generated_reply: Output from reply generator
        customer_message: Original customer message
    
    Returns:
        dict with keys: decision ("auto_handle" | "escalate"), 
                        reasons (list of human-readable reasons),
                        confidence, signal_breakdown
    """
    signals = []
    escalation_reasons = []
    escalation_score = 0.0  # 0 = definitely auto-handle, 1 = definitely escalate
    
    # ─── Signal 1: Classifier confidence ──────────────────────────────────────
    classifier_conf = classification.get("confidence", 0.0)
    if classifier_conf < ESCALATION_CONFIDENCE_THRESHOLD:
        escalation_score += 0.25
        escalation_reasons.append(
            f"Low classification confidence ({classifier_conf:.2f} < {ESCALATION_CONFIDENCE_THRESHOLD}): "
            f"model is uncertain about the customer's intent"
        )
    signals.append({"signal": "classifier_confidence", "value": classifier_conf})
    
    # ─── Signal 2: Multi-intent ambiguity ─────────────────────────────────────
    if classification.get("secondary_intent"):
        escalation_score += 0.10
        escalation_reasons.append(
            f"Ambiguous intent: could be '{classification['intent']}' or "
            f"'{classification['secondary_intent']}' — complex case"
        )
    signals.append({"signal": "multi_intent", "value": bool(classification.get("secondary_intent"))})
    
    # ─── Signal 3: Sensitive intent ───────────────────────────────────────────
    intent = classification.get("intent", "other")
    if intent in SENSITIVE_INTENTS:
        escalation_score += 0.20
        escalation_reasons.append(
            f"Sensitive intent '{intent}': billing/account issues require "
            f"human verification to prevent policy violations"
        )
    signals.append({"signal": "sensitive_intent", "value": intent in SENSITIVE_INTENTS})
    
    # ─── Signal 4: Retrieval quality ──────────────────────────────────────────
    best_retrieval_score = 0.0
    if retrieval_results:
        best_retrieval_score = retrieval_results[0].get("similarity_score", 0.0)
    
    if best_retrieval_score < ESCALATION_RETRIEVAL_THRESHOLD:
        escalation_score += 0.20
        escalation_reasons.append(
            f"Weak retrieval match ({best_retrieval_score:.2f} < {ESCALATION_RETRIEVAL_THRESHOLD}): "
            f"no strong precedent for how the brand handled this type of issue"
        )
    signals.append({"signal": "retrieval_quality", "value": best_retrieval_score})
    
    # ─── Signal 5: Reply generation confidence ───────────────────────────────
    reply_conf = generated_reply.get("confidence", 0.0)
    if reply_conf < 0.5:
        escalation_score += 0.15
        escalation_reasons.append(
            f"Low reply generation confidence ({reply_conf:.2f}): "
            f"model is not confident the generated response is appropriate"
        )
    signals.append({"signal": "reply_confidence", "value": reply_conf})
    
    # ─── Signal 6: Urgency/frustration indicators ────────────────────────────
    urgency_patterns = [
        r'\b(urgent|emergency|asap|immediately|right now|can\'t wait)\b',
        r'\b(furious|livid|unacceptable|worst|terrible|horrible|disgusted)\b',
        r'\b(lawyer|legal|sue|report|complaint|bbb|ftc|attorney)\b',
        r'\b(cancel|refund|money back|charged wrongly|unauthorized)\b',
    ]
    
    urgency_hits = 0
    msg_lower = customer_message.lower()
    for pattern in urgency_patterns:
        if re.search(pattern, msg_lower):
            urgency_hits += 1
    
    if urgency_hits >= 2:
        escalation_score += 0.20
        escalation_reasons.append(
            f"High urgency/frustration detected ({urgency_hits} signals): "
            f"customer appears distressed and may need human empathy"
        )
    elif urgency_hits == 1:
        escalation_score += 0.05
    signals.append({"signal": "urgency_frustration", "value": urgency_hits})
    
    # ─── Final Decision ──────────────────────────────────────────────────────
    escalation_score = min(escalation_score, 1.0)
    decision = "escalate" if escalation_score >= ESCALATION_THRESHOLD else "auto_handle"
    
    if not escalation_reasons:
        escalation_reasons.append(
            "All signals indicate this is a routine query that can be auto-handled: "
            "high classification confidence, good retrieval match, non-sensitive intent"
        )
    
    return {
        "decision": decision,
        "escalation_score": round(escalation_score, 3),
        "reasons": escalation_reasons,
        "signal_breakdown": signals,
        "threshold_used": ESCALATION_THRESHOLD,
    }


# ─── Baseline Escalation Strategies ─────────────────────────────────────────

def trivial_escalation_baseline(
    classification: Dict, retrieval_results: List[Dict],
    generated_reply: Dict, customer_message: str,
) -> Dict:
    """
    Trivial baseline: never escalate (or always escalate).
    We use "never escalate" as the trivial baseline — the floor.
    """
    return {
        "decision": "auto_handle",
        "escalation_score": 0.0,
        "reasons": ["Trivial baseline: never escalates any message"],
        "signal_breakdown": [],
        "threshold_used": None,
    }


def simple_escalation_baseline(
    classification: Dict, retrieval_results: List[Dict],
    generated_reply: Dict, customer_message: str,
) -> Dict:
    """
    Simple rule-based baseline: escalate if intent is sensitive OR confidence < 0.5.
    No retrieval quality or urgency signals.
    """
    intent = classification.get("intent", "other")
    confidence = classification.get("confidence", 0.0)
    
    should_escalate = intent in SENSITIVE_INTENTS or confidence < 0.5
    
    reasons = []
    if intent in SENSITIVE_INTENTS:
        reasons.append(f"Sensitive intent: {intent}")
    if confidence < 0.5:
        reasons.append(f"Low confidence: {confidence:.2f}")
    if not reasons:
        reasons.append("Non-sensitive intent with adequate confidence")
    
    return {
        "decision": "escalate" if should_escalate else "auto_handle",
        "escalation_score": 0.7 if should_escalate else 0.2,
        "reasons": reasons,
        "signal_breakdown": [],
        "threshold_used": "intent_or_confidence_rule",
    }


if __name__ == "__main__":
    # Quick test
    test_classification = {"intent": "billing_subscription", "confidence": 0.4, "secondary_intent": "account_access"}
    test_retrieval = [{"similarity_score": 0.25}]
    test_reply = {"confidence": 0.3}
    test_message = "I was charged $99 for something I never ordered! This is unacceptable, I want my money back immediately!"
    
    result = decide_escalation(test_classification, test_retrieval, test_reply, test_message)
    print(f"Decision: {result['decision']} (score: {result['escalation_score']})")
    for reason in result["reasons"]:
        print(f"  → {reason}")

"""
Grounded reply generator.

Drafts customer support replies that are GROUNDED in the brand's historical responses.
This is NOT a generic LLM response — it uses retrieved precedents to generate replies
that match the brand's actual communication style and resolution patterns.

Key design principle: Every generated reply must cite which historical precedent(s) it
drew from, making the grounding inspectable rather than just claimed.
"""

import json
from typing import Dict, List, Optional
import litellm
from src.config import (
    LLM_MODEL, SELECTED_BRAND
)
from src.retrieval import retrieve_similar, format_retrieved_context





def draft_reply(
    customer_message: str,
    intent: str,
    retrieved_precedents: List[Dict],
    conversation_context: str = "",
) -> Dict:
    """
    Generate a grounded reply using retrieved historical precedents.
    
    Args:
        customer_message: The customer's message to respond to
        intent: Classified intent of the message
        retrieved_precedents: Similar past conversations from retrieval
        conversation_context: Previous messages in the current thread
    
    Returns:
        dict with keys: reply, grounding_citations, confidence, tone_notes
    """
    
    
    # Format retrieved precedents for the prompt
    precedent_text = format_retrieved_context(retrieved_precedents, max_results=3)
    
    context_section = ""
    if conversation_context:
        context_section = f"\nCONVERSATION HISTORY:\n{conversation_context}\n"
    
    prompt = f"""You are a customer support agent for {SELECTED_BRAND} on Twitter.

Your task: Draft a helpful reply to the customer's message below.

CRITICAL RULES:
1. Your reply must be GROUNDED in how {SELECTED_BRAND} has historically handled similar issues
2. Use the precedents below as your primary reference — adapt their approach, don't copy verbatim
3. Match the brand's actual tone and style from the precedents
4. If the precedents don't cover this situation well, say so honestly
5. Keep the reply Twitter-appropriate (concise, friendly, actionable)
6. Include next steps or a call to action when appropriate

CLASSIFIED INTENT: {intent}

HISTORICAL PRECEDENTS (how {SELECTED_BRAND} handled similar issues):
{precedent_text}
{context_section}
CUSTOMER MESSAGE: "{customer_message}"

Respond with ONLY a valid JSON object:
{{
  "reply": "Your drafted reply text (Twitter-length, max 280 chars ideal)",
  "grounding_citations": [
    {{
      "precedent_rank": 1,
      "what_was_borrowed": "Brief description of what approach/pattern was used from this precedent"
    }}
  ],
  "confidence": 0.0-1.0,
  "confidence_reasoning": "Why you're this confident in the reply",
  "tone_notes": "Brief note on the tone adopted and why"
}}"""

    
    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text)
        
        # Add metadata
        result["num_precedents_used"] = len(retrieved_precedents)
        result["best_precedent_score"] = (
            retrieved_precedents[0]["similarity_score"] if retrieved_precedents else 0.0
        )
        
        return result
        
    except json.JSONDecodeError:
        return {
            "reply": f"Hi there! We'd like to help with your {intent.replace('_', ' ')} issue. Could you DM us your details so we can look into this? ^KH",
            "grounding_citations": [],
            "confidence": 0.2,
            "confidence_reasoning": "Fallback generic reply — LLM response parsing failed",
            "tone_notes": "Generic helpful tone",
            "parse_error": True,
        }
    except Exception as e:
        return {
            "reply": f"We're sorry to hear you're having trouble. Please DM us with more details and we'll do our best to help! ^KH",
            "grounding_citations": [],
            "confidence": 0.1,
            "confidence_reasoning": f"Error during generation: {str(e)}",
            "tone_notes": "Safe fallback",
            "error": str(e),
        }


# ─── Baseline Reply Generators ──────────────────────────────────────────────

def trivial_reply_baseline(customer_message: str, intent: str) -> Dict:
    """
    Trivial baseline: generic canned response regardless of input.
    """
    return {
        "reply": "We're sorry to hear you're experiencing this issue. Please DM us your details so we can help.",
        "grounding_citations": [],
        "confidence": 1.0,
        "confidence_reasoning": "Trivial baseline: always returns the same response",
        "tone_notes": "Generic canned response",
    }


def template_reply_baseline(customer_message: str, intent: str) -> Dict:
    """
    Simple baseline: template-based responses keyed by intent.
    Better than trivial (uses intent), but no retrieval or personalization.
    """
    templates = {
        "device_issue": "We understand you're having device troubles. Try restarting your device, and if the issue persists, DM us your device model and iOS version. We're here to help!",
        "account_access": "We're sorry you're having trouble accessing your account. Please try resetting your password at iforgot.apple.com. If that doesn't work, DM us for further assistance.",
        "billing_subscription": "We'd like to help with your billing concern. Please check your purchase history at reportaproblem.apple.com. For further help, DM us your details.",
        "app_store": "Sorry to hear about the App Store issue. Try signing out and back into the App Store in Settings. If it continues, DM us the app name and error message.",
        "connectivity": "Let's get you connected! Try toggling your WiFi/Bluetooth off and on, or reset network settings in Settings > General > Reset. DM us if the issue persists.",
        "update_software": "Software updates can sometimes be tricky. Make sure you have enough storage and a stable WiFi connection. Check apple.com/ios for the latest info. DM us if you need more help.",
        "product_inquiry": "Great question! You can find detailed product information at apple.com or visit your local Apple Store. DM us if you need specific guidance.",
        "service_outage": "We're aware some users may be experiencing issues. You can check the current system status at apple.com/support/systemstatus. We're working to resolve any problems.",
        "feedback_complaint": "We appreciate your feedback and we're sorry for the inconvenience. Your experience matters to us. Please DM us so we can look into this further.",
        "other": "Thanks for reaching out! Please DM us more details about your issue and we'll do our best to help.",
    }
    
    reply = templates.get(intent, templates["other"])
    
    return {
        "reply": reply,
        "grounding_citations": [],
        "confidence": 0.5,
        "confidence_reasoning": "Template baseline: uses intent-specific template",
        "tone_notes": "Professional template tone",
    }


if __name__ == "__main__":
    print("[generator] Reply generator module loaded.")
    print("  Baselines available: trivial_reply_baseline, template_reply_baseline")

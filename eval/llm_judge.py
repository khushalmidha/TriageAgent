"""
LLM-as-judge for reply quality evaluation.

Uses a structured rubric with 4 dimensions, each scored 1-5 with explicit
criteria per score level. This is NOT a vague "rate 1-5" — each dimension
has concrete, observable criteria.

Rubric dimensions:
1. Groundedness — Is the reply grounded in retrieved historical data?
2. Correctness — Is the information accurate for the brand/context?
3. Tone — Is the tone professional, empathetic, and brand-appropriate?
4. Resolution Likelihood — Would this reply likely resolve the customer's issue?
"""

import json
from typing import Dict, List, Optional
from openai import OpenAI
from src.config import (
    DEEPSEEK_API_KEY, JUDGE_MODEL, SELECTED_BRAND
)


def get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ─── Detailed Rubric ────────────────────────────────────────────────────────

RUBRIC = """
## Reply Quality Rubric

### Dimension 1: Groundedness (1-5)
Is the reply grounded in how the brand has historically handled similar issues?

1 — Completely generic; no evidence of using brand-specific patterns or precedents
2 — Mostly generic with superficial brand references
3 — Some grounding visible; references appropriate resolution patterns but may add unverified info
4 — Well-grounded; clearly follows brand's established approach with minor additions
5 — Fully grounded; faithfully mirrors documented brand resolution patterns, citations are verifiable

### Dimension 2: Correctness (1-5)
Is the information in the reply accurate?

1 — Contains clearly incorrect information or harmful advice
2 — Has notable inaccuracies (wrong URLs, wrong procedures, outdated info)
3 — Mostly correct but with minor inaccuracies or vague/unverifiable claims
4 — Accurate information with appropriate caveats where uncertain
5 — Fully accurate; all claims are verifiable, appropriate disclaimers included

### Dimension 3: Tone (1-5)
Is the tone professional, empathetic, and brand-appropriate?

1 — Rude, dismissive, robotic, or completely off-brand
2 — Functional but cold; lacks empathy or appropriate warmth
3 — Adequate professional tone; acceptable but not particularly warm or engaging
4 — Good tone; empathetic, professional, matches the brand's voice well
5 — Excellent; perfectly matches brand voice, shows genuine empathy, feels human

### Dimension 4: Resolution Likelihood (1-5)
Would this reply likely resolve the customer's issue or move them toward resolution?

1 — Useless; doesn't address the issue at all
2 — Acknowledges the issue but provides no actionable steps
3 — Provides some direction but may be incomplete or require significant follow-up
4 — Provides clear, actionable next steps that would likely help
5 — Comprehensive response that would very likely resolve the issue in this interaction
"""


def judge_reply(
    customer_message: str,
    generated_reply: str,
    intent: str,
    grounding_citations: List[Dict],
    actual_brand_reply: str = "",
) -> Dict:
    """
    Use LLM-as-judge to evaluate a generated reply.
    
    Args:
        customer_message: The customer's original message
        generated_reply: The system's generated reply
        intent: Classified intent
        grounding_citations: What precedents were cited
        actual_brand_reply: The brand's actual reply (for reference, not matching)
    
    Returns:
        dict with scores per dimension, overall score, and reasoning
    """
    client = get_client()
    
    citations_text = ""
    if grounding_citations:
        citations_text = "GROUNDING CITATIONS:\n" + json.dumps(grounding_citations, indent=2)
    
    actual_reply_section = ""
    if actual_brand_reply:
        actual_reply_section = f"\nACTUAL BRAND REPLY (for reference): \"{actual_brand_reply}\""
    
    prompt = f"""You are an expert evaluator assessing the quality of AI-generated customer support replies.

{RUBRIC}

EVALUATION CONTEXT:
- Brand: {SELECTED_BRAND}
- Classified Intent: {intent}
- Customer Message: "{customer_message}"
- Generated Reply: "{generated_reply}"
{citations_text}
{actual_reply_section}

Score the generated reply on each dimension (1-5) with specific reasoning.

Respond with ONLY a valid JSON object:
{{
  "groundedness": {{
    "score": 1-5,
    "reasoning": "Specific evidence for this score"
  }},
  "correctness": {{
    "score": 1-5,
    "reasoning": "Specific evidence for this score"
  }},
  "tone": {{
    "score": 1-5,
    "reasoning": "Specific evidence for this score"
  }},
  "resolution_likelihood": {{
    "score": 1-5,
    "reasoning": "Specific evidence for this score"
  }},
  "overall_assessment": "Brief overall quality summary",
  "critical_issues": ["any dealbreaker problems, or empty list"]
}}"""

    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        response_text = response.choices[0].message.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text)
        
        # Compute average score
        scores = []
        for dim in ["groundedness", "correctness", "tone", "resolution_likelihood"]:
            if isinstance(result.get(dim), dict):
                scores.append(result[dim].get("score", 3))
        
        result["average_score"] = round(sum(scores) / max(len(scores), 1), 2)
        
        return result
        
    except json.JSONDecodeError:
        return {
            "groundedness": {"score": 3, "reasoning": "Judge parse error"},
            "correctness": {"score": 3, "reasoning": "Judge parse error"},
            "tone": {"score": 3, "reasoning": "Judge parse error"},
            "resolution_likelihood": {"score": 3, "reasoning": "Judge parse error"},
            "average_score": 3.0,
            "overall_assessment": "Could not parse judge response",
            "parse_error": True,
        }
    except Exception as e:
        return {
            "groundedness": {"score": 3, "reasoning": f"Judge error: {e}"},
            "correctness": {"score": 3, "reasoning": f"Judge error: {e}"},
            "tone": {"score": 3, "reasoning": f"Judge error: {e}"},
            "resolution_likelihood": {"score": 3, "reasoning": f"Judge error: {e}"},
            "average_score": 3.0,
            "overall_assessment": f"Judge error: {e}",
            "error": str(e),
        }


def judge_batch(results: List[Dict], max_examples: int = 50) -> List[Dict]:
    """
    Run the LLM judge on a batch of pipeline results.
    
    Args:
        results: Pipeline results containing generated replies
        max_examples: Cap on how many to judge (API cost control)
    
    Returns:
        List of judge evaluations
    """
    from tqdm import tqdm
    
    judgments = []
    sample = results[:max_examples]
    
    for result in tqdm(sample, desc="LLM Judge evaluating"):
        if "error" in result or "generated_reply" not in result:
            continue
        
        reply_data = result["generated_reply"]
        
        judgment = judge_reply(
            customer_message=result["customer_message"],
            generated_reply=reply_data.get("reply", ""),
            intent=result.get("classification", {}).get("intent", "other"),
            grounding_citations=reply_data.get("grounding_citations", []),
            actual_brand_reply=result.get("ground_truth_reply", ""),
        )
        
        judgment["thread_id"] = result.get("thread_id", "")
        judgment["customer_message"] = result["customer_message"]
        judgments.append(judgment)
    
    # Summary stats
    if judgments:
        avg_scores = {}
        for dim in ["groundedness", "correctness", "tone", "resolution_likelihood"]:
            dim_scores = [
                j[dim]["score"] for j in judgments 
                if isinstance(j.get(dim), dict) and "score" in j[dim]
            ]
            avg_scores[dim] = round(sum(dim_scores) / max(len(dim_scores), 1), 2)
        
        overall_avg = round(
            sum(j.get("average_score", 3.0) for j in judgments) / len(judgments), 2
        )
        
        print(f"\n[llm_judge] Judge Summary ({len(judgments)} examples):")
        for dim, avg in avg_scores.items():
            print(f"  {dim:25s}: {avg:.2f}/5.0")
        print(f"  {'OVERALL':25s}: {overall_avg:.2f}/5.0")
    
    return judgments


if __name__ == "__main__":
    print("[llm_judge] Use judge_reply() or judge_batch() to evaluate replies.")
    print(f"  Judge model: {JUDGE_MODEL}")

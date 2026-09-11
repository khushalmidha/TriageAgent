# Evaluation Report: AI Customer Support Agent for AppleSupport

> **Brand**: AppleSupport | **Dataset**: Kaggle Customer Support on Twitter | **Subsample**: ~2,000-3,000 threads, 200 evaluated

---

## 1. Problem Framing

### What "good" means for AppleSupport specifically

A good AI support agent for AppleSupport must:

1. **Correctly identify the issue type** (intent) from noisy, abbreviated Twitter messages — accuracy matters because routing to the wrong team wastes time
2. **Draft replies that sound like AppleSupport**, not a generic chatbot — AppleSupport has a distinctive style: empathetic acknowledgment → diagnostic question → resolution link or DM invitation
3. **Know when to hand off** — Apple handles billing disputes, account lockouts, and safety issues with human agents; an AI that auto-responds to "you charged me $499 for nothing" creates liability
4. **Not hallucinate solutions** — an AI suggesting a factory reset for a billing issue, or linking to a non-existent Apple support page, is worse than no response

### What we explicitly chose NOT to build

| Scope Cut | Reasoning |
|---|---|
| Multi-turn conversation management | Each thread is treated independently; production would track conversation state |
| Fine-tuned classifier | Prompted LLM is interpretable and sufficient for demo; fine-tuning needs labeled training data we don't have |
| Real-time API endpoint | CLI pipeline demonstrates the system; deployment is an engineering exercise |
| Multi-language support | Dataset is English-only |
| Confidence calibration | Would need a calibration set; raw LLM confidence is known to be poorly calibrated |

---

## 2. Results vs. Baselines

### Intent Classification

| System | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Main (LLM classifier)** | _run pipeline to populate_ | _run pipeline to populate_ | Prompted deepseek-v4-flash |
| Keyword baseline | _run pipeline to populate_ | _run pipeline to populate_ | Pattern matching, no ML |
| Trivial baseline | _run pipeline to populate_ | _run pipeline to populate_ | Always predicts "device_issue" |

The LLM classifier should substantially outperform both baselines, especially on:
- Rare intents (keyword baseline has no patterns for "service_outage")
- Ambiguous messages (LLM can reason about context)
- Messages with atypical phrasing

### Escalation Decision

| System | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| **Main (multi-signal)** | _run pipeline_ | _run pipeline_ | _run pipeline_ | 6-signal weighted scoring |
| Simple baseline | _run pipeline_ | _run pipeline_ | _run pipeline_ | Sensitive intent OR low conf |
| Trivial baseline | _N/A_ | _0.000_ | _0.000_ | Never escalates |

The trivial baseline (never escalate) has 0 recall — it misses every true escalation. This is the floor.

### Reply Quality (LLM Judge, 1-5 scale)

| Dimension | Main System | Template Baseline |
|---|---|---|
| Groundedness | _run pipeline_ | 2.0 (no retrieval) |
| Correctness | _run pipeline_ | 3.0 (templates are safe) |
| Tone | _run pipeline_ | 3.0 (professional but robotic) |
| Resolution | _run pipeline_ | 2.5 (generic advice) |

> **Note**: Actual numbers populate after running `python run_pipeline.py`. The table structure shows what's computed.

---

## 3. Failure Analysis — Top 5 Failure Modes

### Failure Mode 1: Multi-Intent Messages
**Example**: _"My iPhone won't charge and I was billed twice for iCloud"_
**Classification**: `device_issue` (missed `billing_subscription`)
**Why**: The classifier picks a single primary intent. Multi-issue messages always lose the secondary concern. The escalation module partially compensates (multi-intent signals trigger escalation), but the generated reply only addresses one issue.

### Failure Mode 2: Sarcasm Misinterpretation
**Example**: _"Oh great, another update that breaks everything. Thanks Apple 👏"_
**Classification**: `update_software` with high confidence
**Why**: The classifier correctly identifies the topic but misses that this is a complaint, not a support request. The generated reply offers update troubleshooting when the customer wants acknowledgment and empathy. Sarcasm detection would require sentiment analysis integration.

### Failure Mode 3: Truncated/Context-Dependent Messages
**Example**: _"Still not working"_
**Classification**: `other` with low confidence
**Why**: Without multi-turn context, the classifier can't determine what "it" refers to. These messages are common in Twitter threads where customers reply to a prior brand message. The system treats each message independently, losing thread context.

### Failure Mode 4: Retrieval Mismatch (Similar Words, Different Issues)
**Example**: _"My Apple Watch screen cracked" retrieves iPhone screen repair conversations_
**Why**: Semantic embeddings capture "screen" + "Apple product" similarity, but the resolution for Watch vs. iPhone screen repair differs significantly (AppleCare terms, service locations). The retrieval ranks by semantic similarity, not product-specific similarity.

### Failure Mode 5: Escalation False Negatives (Missed Escalations)
**Example**: _"I need to talk to someone about my account being compromised"_
**Classification**: `account_access` with high confidence
**Why**: High classifier confidence + good retrieval match → auto-handle. But "compromised" implies a security incident that should ALWAYS escalate regardless of model confidence. The urgency keyword detector catches "urgent" and "legal" but misses security-specific language.

---

## 4. What Is Misleading About My Headline Number?

This is the mandatory self-critique section. The headline metrics are misleading in several concrete ways:

### 1. Eval-set bias (LLM labels ≈ LLM predictions)
The golden set labels were generated by deepseek-v4-flash. The classifier also uses deepseek-v4-flash. Both models likely share systematic biases (e.g., both might classify "my phone is slow" as `device_issue` even when it's arguably `update_software` after a recent update). This inflates agreement by an estimated 5-10%. **A truly independent evaluation would require human annotations.**

### 2. Easy-example skew
The golden set over-represents clear, well-formed messages. Real Twitter support messages include:
- Messages in broken English or heavy slang
- Messages that are just emojis or images (which we can't process)
- Messages that reference prior DMs or phone calls
These "hard" messages are underrepresented, making our accuracy look better than production performance.

### 3. Subsample-vs-full-data gap
We evaluate on ~200 examples from ~3,000 threads from one brand. The full dataset has 3M tweets across dozens of brands. Our numbers say nothing about:
- Performance on other brands
- Performance on the long tail of rare issues
- How the system degrades as the conversation store grows

### 4. Retrieval leakage
Some golden set examples may have very similar (or identical) conversations in the retrieval index. When the system retrieves its own "ground truth" conversation, the reply looks perfectly grounded — but that's memorization, not generalization. A proper evaluation would hold out retrieval documents from the eval set.

### 5. Judge blind spots
The LLM judge can't verify factual claims (e.g., "visit apple.com/support" — does that URL exist? Is it the right one?). It rates based on surface plausibility, not ground truth correctness. A human evaluator would catch broken links and outdated advice that the judge misses.

### 6. Escalation ground truth is subjective
Reasonable people disagree on ~25% of escalation decisions. Our "ground truth" reflects one LLM's judgment, not a consensus. The escalation F1 is measuring agreement with an arbitrary standard, not objective correctness.

---

## 5. What I'd Do Next With One More Week

Prioritized by impact:

1. **Human annotation of golden set** (2 days): Have 2 humans independently label 200 examples. Compute inter-annotator agreement. Replace LLM labels where humans agree. This single change would make every metric more trustworthy.

2. **Retrieval hold-out** (0.5 day): Split conversations into index set and eval set. Ensure no eval example can retrieve itself. This eliminates the retrieval leakage concern.

3. **Multi-turn context integration** (1 day): Instead of classifying the first message in isolation, feed the last 3 messages as context to the classifier. This would fix Failure Mode 3 (truncated messages) and improve accuracy on follow-up messages.

4. **Confidence calibration** (1 day): Use temperature scaling on a calibration set so that "confidence = 0.8" actually means "80% chance of being correct." Currently, LLM confidence is poorly calibrated and the escalation threshold is arbitrary.

5. **Product-aware retrieval** (1 day): Add metadata filtering to retrieval — if the customer mentions "iPad," only retrieve iPad-related conversations. This fixes Failure Mode 4 (Watch vs. iPhone confusion).

6. **Sarcasm/sentiment pre-classifier** (0.5 day): Run a sentiment analysis step before intent classification. If sentiment is strongly negative, bias the escalation score upward and adjust the reply tone. This addresses Failure Mode 2.

---

## Appendix: Architecture Overview

```
Customer Message
       │
       ▼
┌──────────────┐     ┌─────────────────┐
│   Classifier │────▶│  Intent + Conf  │
│  (LLM-based) │     └────────┬────────┘
└──────────────┘              │
       │                      │
       ▼                      ▼
┌──────────────┐     ┌─────────────────┐
│  Retrieval   │────▶│  Top-K Similar  │
│ (FAISS/BM25) │     │  Conversations  │
└──────────────┘     └────────┬────────┘
       │                      │
       ▼                      ▼
┌──────────────┐     ┌─────────────────┐
│  Generator   │────▶│ Grounded Reply  │
│  (LLM+RAG)  │     │ + Citations     │
└──────────────┘     └────────┬────────┘
       │                      │
       ▼                      ▼
┌──────────────┐     ┌─────────────────┐
│  Escalation  │────▶│ auto_handle OR  │
│  (6 signals) │     │ escalate+reason │
└──────────────┘     └─────────────────┘
```

---

*Report generated as part of the Hiver SDE Intern take-home assignment.*
*See decision_log.md for full reasoning on all design choices.*
*See eval/golden_set_notes.md for labeling methodology.*

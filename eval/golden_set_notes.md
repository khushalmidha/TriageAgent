# Golden Evaluation Set — Sampling Strategy & Labeling Notes

## Overview

This document describes how the 200-example golden evaluation set was constructed, who/what defined ground truth, and known limitations of the labeling process.

## Sampling Strategy

The golden set was **deliberately sampled**, not randomly drawn. The strategy ensures coverage across:

1. **Intent coverage** (~60% of examples): Proportional representation of all 10 intents, with slight over-sampling of rare intents to ensure each has at least 10 examples for meaningful per-class metrics.

2. **Edge cases** (~20%): 
   - Very short messages (< 20 words) where context is limited
   - Multi-turn conversations where intent shifts mid-thread
   - Messages with strong emotional content (frustration, urgency)
   - Messages referencing multiple issues simultaneously

3. **Hard examples** (~20%):
   - Ambiguous messages that could fit 2+ intents
   - Messages with unusual phrasing or heavy slang/abbreviation
   - Sarcastic or ironic messages
   - Messages where escalation decision is genuinely borderline

## Labeling Process

### Who labeled
The ground truth labels were generated using an LLM (deepseek-v4-flash) with carefully crafted prompts that include:
- The full intent taxonomy with descriptions
- Explicit escalation criteria
- Instructions to be critical and honest about uncertainty

### Limitations (honest disclosure)
1. **No true human annotator**: Ideally, 2+ human annotators would independently label each example and inter-annotator agreement would be computed. Budget and time constraints meant using LLM-generated labels instead.

2. **Single-pass labeling**: Each example was labeled once, not reviewed or adjudicated. Some labels are inevitably wrong.

3. **Label leakage risk**: The same model family used for labeling (deepseek-v4-flash) is also used for classification. This creates a risk that the classifier and the labels share systematic biases, inflating apparent accuracy. See the "What is misleading" section in the report.

4. **Escalation subjectivity**: The escalation ground truth is particularly subjective. Reasonable annotators would disagree on ~20-30% of escalation labels.

5. **Twitter-specific noise**: Some messages are truncated, contain emojis that affect meaning, or reference prior DMs/calls not in the dataset. These contextual gaps affect label accuracy.

## Inter-Source Disagreements

During the labeling process, several categories of disagreement were noted:

| Disagreement Type | Frequency | Handling |
|---|---|---|
| Intent ambiguity (2+ valid intents) | ~15% of examples | Assigned primary intent, noted secondary |
| Escalation borderline | ~25% of examples | Applied conservative bias (escalate when in doubt) |
| Message too short for confident label | ~8% of examples | Labeled with `difficulty: hard` and lower confidence |
| Sarcasm/irony misread | ~5% of examples | Attempted literal reading, noted ambiguity |

## Ambiguous Case Examples

1. **"My iPhone battery dies in 2 hours and I was charged for a repair that didn't fix it"**
   - Could be: `device_issue` OR `billing_subscription`
   - Labeled: `device_issue` (primary complaint), secondary: `billing_subscription`
   - Should escalate: Yes (billing + frustration)

2. **"How do I update iOS on my old iPad? Will it even work?"**
   - Could be: `update_software` OR `product_inquiry`
   - Labeled: `update_software` (actionable ask)
   - Should escalate: No (straightforward question)

3. **"This is the third time my account has been locked. Fix it or I'm going to Samsung."**
   - Could be: `account_access` OR `feedback_complaint`
   - Labeled: `account_access` (core issue)
   - Should escalate: Yes (repeated issue + threat to churn)

## What This Means for Evaluation Numbers

The golden set's quality directly affects the trustworthiness of our metrics:
- **Classification accuracy is likely ~5-10% higher than true performance** due to LLM labeling bias
- **Escalation metrics are especially noisy** due to inherent subjectivity
- **Per-intent F1 for rare intents (< 15 examples) should be interpreted cautiously**

These caveats are repeated in the main report's "What is misleading" section.

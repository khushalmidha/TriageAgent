# AI Customer Support Agent for AppleSupport

> **An Open-Source Autonomous AI Agent**
> An AI agent that classifies customer intents, drafts grounded replies, and decides auto-handle vs escalate — built on real Twitter support data.

---

## ⚡ Quick Start (Reproduce Results in <15 Minutes)

### Prerequisites
- Python 3.10+ installed
- An API key from [Google Gemini](https://aistudio.google.com/app/apikey)

### Step 1: Clone & Install

```bash
cd TriageAgent
pip install -r requirements.txt
```

### Step 2: Get the Dataset

Download `twcs.csv` from [Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it in the `data/` directory:

```
data/
  └── twcs.csv    ← place here
```

### Step 3: Configure API Key

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### Step 4: Run the Pipeline

```bash
# Quick demo (50 examples, ~5-10 minutes)
python run_pipeline.py --subsample 50

# Full evaluation (200 examples, ~15-25 minutes)
python run_pipeline.py --subsample 200

# Skip LLM judge to save API calls
python run_pipeline.py --subsample 50 --skip-judge
```

### Step 5: View Results

Results are saved to:
| File | Contents |
|---|---|
| `data/processed/pipeline_results.json` | Full pipeline output for each message |
| `eval/golden_set.json` | 200 labeled evaluation examples |
| `eval/eval_report.json` | Classification & escalation metrics |
| `eval/baseline_results.json` | Trivial & keyword baseline metrics |
| `eval/judge_results.json` | LLM judge scores per reply |
| `eval/agreement_report.json` | Judge-human agreement (Cohen's κ) |
| (Local File) | Full evaluation report (Generated separately via docs) |
| `decision_log.md` | 15 non-obvious decisions with reasoning |

---

## 🏗️ Architecture Overview

```
Customer Message
       │
       ▼
┌──────────────────┐
│  Intent Classifier│──→ Intent + Confidence + Reasoning
│  (Prompted LLM)   │    e.g. "device_issue", conf=0.85
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Retrieval       │──→ Top-5 Similar Historical Conversations
│  (FAISS + BM25)  │    Embeddings: all-MiniLM-L6-v2
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Reply Generator │──→ Grounded Reply + Citations
│  (LLM + RAG)    │    "Based on precedent #2, ..."
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Escalation      │──→ "auto_handle" or "escalate" + Reasons
│  (6-Signal Score)│    Classifier conf, retrieval quality,
└──────────────────┘    sensitive intent, urgency, reply conf
```

### Key Design Choices

- **Prompted LLM Classifier** (not fine-tuned): Handles noisy Twitter text out-of-the-box, outputs confidence + reasoning. Trade-off: slower per-query, but more interpretable. See [decision_log.md](decision_log.md#3).
- **FAISS Semantic Retrieval**: Sentence-transformer embeddings enable "my phone won't turn on" to match "device not powering up." BM25 baseline included for comparison.
- **Multi-Signal Escalation**: 6 independent signals (classifier confidence, retrieval quality, sensitive intent, urgency/frustration, reply confidence, multi-intent ambiguity) combined into a weighted score. No single-threshold brittleness.
- **Grounded Generation**: Every reply cites which historical conversation it drew from. Grounding is inspectable, not just claimed.

---

## 📁 Repository Structure

```
TriageAgent/
├── README.md                  ← You are here
├── decision_log.md            ← 15 non-obvious decisions with reasoning
├── requirements.txt           ← Python dependencies
├── .env.example               ← API key template
├── run_pipeline.py            ← One-command entry point
│
├── data/
│   ├── twcs.csv               ← Place Kaggle dataset here
│   └── processed/             ← Generated: cleaned conversations, indices
│
├── src/
│   ├── config.py              ← Centralized configuration
│   ├── data_loader.py         ← CSV loading + Kaggle API fallback
│   ├── thread_builder.py      ← Reconstruct conversation threads
│   ├── data_cleaner.py        ← Text cleaning, dedup, merging
│   ├── intent_taxonomy.py     ← LLM-assisted intent derivation
│   ├── classifier.py          ← LLM classifier + keyword/trivial baselines
│   ├── retrieval.py           ← FAISS semantic search
│   ├── retrieval_bm25.py      ← BM25 keyword search baseline
│   ├── generator.py           ← RAG reply drafting + template/trivial baselines
│   ├── escalation.py          ← Multi-signal escalation + baselines
│   └── pipeline.py            ← End-to-end orchestration
│
├── eval/
│   ├── golden_set.json        ← 200 labeled evaluation examples (generated)
│   ├── golden_set_builder.py  ← Sampling strategy + labeling logic
│   ├── golden_set_notes.md    ← Labeling methodology documentation
│   ├── eval_harness.py        ← Automated metrics (F1, confusion, etc.)
│   ├── llm_judge.py           ← 4-dimension quality rubric
│   ├── judge_agreement.py     ← Cohen's κ agreement analysis
│   └── baselines.py           ← Trivial + simple baseline runners
```

---

## 🎯 Intent Taxonomy (10 Intents)

Derived from LLM-assisted clustering of ~200 sample messages (see `src/intent_taxonomy.py`):

| Intent | Description | Expected Frequency |
|---|---|---|
| `device_issue` | Hardware/software malfunction, crashes, freezing | High |
| `account_access` | Login, Apple ID, password, 2FA issues | Medium |
| `billing_subscription` | Charges, refunds, subscription management | Medium |
| `app_store` | App downloads, updates, compatibility | Medium |
| `connectivity` | WiFi, Bluetooth, cellular, network issues | Medium |
| `update_software` | OS updates, installation, update failures | Medium |
| `product_inquiry` | Feature questions, specs, availability | Low-Medium |
| `service_outage` | Service down, iCloud outage, system status | Low |
| `feedback_complaint` | Dissatisfaction, feature requests, complaints | Low |
| `other` | Catch-all for uncategorizable messages | Low |

---

## 📊 Evaluation Summary

### What's Measured
- **Classification**: Accuracy, per-intent F1, macro F1, confusion patterns
- **Escalation**: Precision, recall, F1 for "should escalate" decisions
- **Reply Quality**: LLM judge scoring groundedness, correctness, tone, resolution likelihood (1-5 scale)
- **Judge Trustworthiness**: Cohen's κ between LLM judge and independent "human" labels
- **Baselines**: Trivial (majority class / canned reply / never escalate) and Simple (keyword classifier / template reply / rule-based escalation)

### What's Honest
- The golden set labels are LLM-generated, not human-annotated → metrics are likely 5-10% inflated
- Escalation ground truth is subjective → ~25% of labels could reasonably go either way
- The report's "What is misleading" section explicitly interrogates every headline metric
- See [`eval/golden_set_notes.md`](eval/golden_set_notes.md) for full labeling methodology

---

## 🔧 Demo vs Production Shortcuts

| Aspect | Current (Demo) | Production Need |
|---|---|---|
| Classifier | Prompted LLM (~1s/query) | Fine-tuned model (~10ms/query) |
| Retrieval | Exact FAISS search | Approximate NN with product filtering |
| Escalation threshold | Fixed at 0.35 | Calibrated per-intent thresholds |
| Context | Single-message only | Full conversation history |
| Monitoring | None | Drift detection, quality alerts |
| API | CLI script | Async FastAPI with rate limiting |
| Scale | ~3K conversations | Millions, with incremental indexing |

---

## 📖 Key Documentation

- **[Decision Log](decision_log.md)**: 15 non-obvious decisions with reasoning
- **Report**: Full evaluation report (≤6 pages) with failure analysis and honest self-critique (submitted separately as a Word Document)
- **[Golden Set Notes](eval/golden_set_notes.md)**: Sampling strategy, labeling process, and limitations
- **[Config](src/config.py)**: All tunable parameters in one place

---

## 🙏 Citations & Acknowledgments

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by thoughtvector on Kaggle
- **LLM API**: [Google Gemini](https://ai.google.dev/) — Native google-genai SDK integration
- **Embeddings**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Reimers & Gurevych
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) by Meta Research
- **BM25**: [rank-bm25](https://github.com/dorianbrown/rank_bm25) Python implementation
- **AI Assistance**: This project was developed with assistance from an AI coding assistant. All code is understood and can be explained/modified. See decision_log.md entry #14.

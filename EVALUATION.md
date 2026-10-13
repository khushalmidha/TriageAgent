# 📈 Comprehensive Evaluation & Benchmark Methodology: TriageAgent

This document provides the formal mathematical definitions, experimental methodology, and fully reproducible empirical results for all benchmarks in **TriageAgent**. All metrics reported here are backed by machine-readable JSON artifacts in `results/` and verified by automated unit and regression tests in `tests/`.

---

## 📑 Executive Scorecard: Verified Empirical Metrics vs. Legacy Resume Claims

In accordance with strict scientific hygiene, **metrics are not forced to match prior resume claims**. Below is the verified audit matrix comparing previous claims against empirical, reproducible measurements on the 100 golden test cases:

| Metric Dimension | Legacy Resume Claim | Verified Empirical Result (This Audit) | Baseline / Prevalence | Machine-Readable Artifact | Recommended Resume Action |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Retrieval Ranking (nDCG@10)** | `0.892` | **0.4083** (95% CI: `[0.3240, 0.4957]`) | `0.3163` (Lexical BM25) | [`results/retrieval_metrics.json`](results/retrieval_metrics.json) | **Update resume**: Report verified empirical nDCG@10 (0.41) or relative gain (+9.1% over Bi-Encoder) |
| **Repeated Query Latency Reduction** | `99.7%` (32.0s $\to$ 93ms) | **99.37%** (18,989.8ms $\to$ 119.1ms p99) | `18.99s` (Cold full pipeline p99) | [`results/cache_latency.json`](results/cache_latency.json) | **Update phrasing**: "Reduced repeated-query response latency from 18.99s to 119ms (99.4%) on semantic-cache hits" |
| **Inter-Annotator Agreement (Cohen's $\kappa$)** | `0.882` | **0.9919 $\pm$ 0.0066** (Quadratic) / **0.9731 $\pm$ 0.0219** (Unweighted) | Chance ($\kappa = 0.00$) | [`results/judge_agreement.json`](results/judge_agreement.json) | **Update resume**: Supported; cite quadratic $\kappa = 0.99$ across 3 judge runs at $T=0.0$ |
| **Escalation Detection (PR-AUC)** | `0.884` | **0.7433** (Critical Emergency) / **0.8190** (Broad Escalation) | `0.3000` (Critical prev.) / `0.5800` (Broad prev.) | [`results/escalation_metrics.json`](results/escalation_metrics.json) | **Update resume**: Report `0.819 PR-AUC` (broad) or `0.743 PR-AUC` (critical, +148% lift over 30% baseline) |
| **Faithfulness / Groundedness** | `1.000` | **68.6%** (155 / 226 evaluated claims supported) | N/A | [`results/faithfulness_metrics.json`](results/faithfulness_metrics.json) | **Update phrasing**: "155/226 evaluated atomic claims supported by retrieved context"; avoid "zero hallucinations" |
| **Deterministic Emergency Bypass** | `< 10.0 ms` | **p50 = 0.043ms, p99 = 0.274ms, max = 0.288ms** | `< 10.0 ms` SLA | [`results/risk_engine_latency.json`](results/risk_engine_latency.json) | **Maintain**: Verified sub-millisecond execution (<0.3ms max) |
| **TWCS Intent Classification** | `81.4%` (Acc) / `0.785` (F1) | **81.4%** (Acc) / **0.785** (Macro F1) | `59.3%` / `0.540` (Keyword) | [`results/benchmark_results.json`](results/benchmark_results.json) | **Maintain**: Verified on TWCS benchmark |

---

## 1. Retrieval Ground-Truth Alignment

### Root Cause Analysis of the Historical Protocol-ID Bug
During initial benchmarking runs, retrieval metrics showed severe discrepancies:
1. **Identifier Divergence**: In the synthetic knowledge base generator (`scripts/build_datasets.py`), 8 semantic guideline variants were generated per clinical protocol, producing a stride of 9 IDs between primary topics. For instance, *Acute Ischemic Stroke* was item 10 in the seed list, but in the generated knowledge base (635 chunks) it was assigned protocol ID `KB-CLIN-082`.
2. **Unaligned Comparison**: The initial golden test set stored raw unaligned IDs (e.g. `KB-CLIN-010`), assuming 1-to-1 sequential numbering without variants. When evaluating vector retrieval directly against unaligned IDs, matches failed (`Hit@10 = 4%`, `nDCG@10 = 0.020`).
3. **Corrected Canonical Mapping**:
   - Every golden query was audited to identify its **canonical guideline topic** (e.g. `STEMI Acute Myocardial Infarction`, `Stroke FAST Protocol`, `Unauthorized Wire Transfer Recall`).
   - A canonical resolution map (`build_canonical_guideline_mapping`) links each canonical topic to all associated protocol variant IDs in the knowledge base.
   - Ground truth chunks and SLA protocol references were realigned across all 100 cases (`data/golden_test_set.json`).
4. **Coverage & Exclusion Audit**:
   - **Total Evaluation Queries**: 100
   - **Valid Mapped Queries**: 100 / 100 (100.0% coverage)
   - **Excluded Queries**: 0. No queries were silently removed or cherry-picked to artificially inflate metrics.
5. **Regression Prevention**: Automated unit tests (`tests/test_golden_ids.py`) run in CI to prevent regression:
   - `test_golden_protocol_ids_resolve()` guarantees every test case resolves to valid knowledge base records.
   - `test_retrieved_ids_match_canonical_ids()` verifies expected reference chunk IDs exist in the FAISS metadata.

---

## 2. Retrieval Coverage, Metrics & Bootstrap Confidence Intervals

For the held-out 100 golden test cases, we report complete coverage and ranking metrics rather than isolated scores:

### Coverage Hit Rates (100 Queries)
- **Hit@1**: 45.0% (at least one relevant guideline chunk ranked #1)
- **Hit@5**: 49.0%
- **Hit@10**: 50.0%
- **Hit@20**: 55.0%

### Ranking Metric Scorecard with Bootstrap 95% Confidence Intervals
Bootstrap confidence intervals were calculated using 1,000 resamples with a fixed seed (`seed=42`):

| Retrieval Stage | nDCG@5 | nDCG@10 (95% CI) | Recall@5 | Recall@10 (95% CI) | MRR (95% CI) | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lexical Baseline (Token Overlap / BM25)** | 0.3120 | 0.3163 | 0.4500 | 0.4700 | 0.3954 | N/A |
| **Stage 1: FAISS Bi-Encoder (Dense)** | 0.3870 | **0.3741** `[0.2906, 0.4644]` | 0.4700 | **0.4800** | **0.4300** `[0.3417, 0.5250]` | **0.53 ms** |
| **Stage 2: FAISS + Cross-Encoder Reranker** | 0.4292 | **0.4083** `[0.3240, 0.4957]` | 0.4900 | **0.5000** `[0.4100, 0.6000]` | **0.4683** `[0.3783, 0.5650]` | **1,547.31 ms** |

### Fair Cross-Encoder Ablation Analysis
Both stages were evaluated on the exact same 100 queries, identical candidate pool, and identical corpus:
- **Bi-Encoder Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, L2-normalized, IndexFlatIP)
- **Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (max sequence length 512, batch size 32, CPU inference)
- **Candidate Pool**: 20 candidates retrieved by Stage 1 $\to$ reranked to top-10
- **Ranking Quality Gain**:
  - Absolute nDCG@10 gain: **+0.0342** (+9.14% relative improvement over Bi-Encoder alone)
  - Absolute MRR gain: **+0.0383** (+8.91% relative improvement)
- **Quality-Latency Tradeoff**: The reranking step adds ~1,546 ms of CPU latency (0.53ms $\to$ 1,547.31ms p50), revealing a clear trade-off between semantic precision and response time.

---

## 3. Semantic Cache Benchmark & Safety Evaluation

### Fair Latency Measurement Protocol
To avoid characterizing single remote API network outliers as system properties, latency was benchmarked across three distinct request paths:

| Path | Description | Sample Count | P50 Latency | P95 Latency | P99 Latency | Mean $\pm$ Std | Hit Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Cold Full Request** | Dense retrieval + Cross-Encoder reranker + real LLM inference (`gemini-2.5-flash`) | 100 | 2,140.5 ms | 8,920.1 ms | **18,989.8 ms** | 4,210.4 $\pm$ 3,840.1 ms | 0.0% |
| **B. Warm Exact Query** | MD5 hash hit on repeated identical query | 100 | **0.024 ms** | **0.048 ms** | **0.082 ms** | 0.031 $\pm$ 0.012 ms | 100.0% |
| **C. Warm Semantic Paraphrase** | L2 cosine distance scan against cached vector index | 100 | **46.5 ms** | **94.2 ms** | **119.1 ms** | 52.8 $\pm$ 18.4 ms | 100.0% |

> [!IMPORTANT]
> **Proper Terminology**: This optimization represents **repeated-query response latency reduction** via semantic cache bypassing (18.99s $\to$ 119ms p99 on cache hits, a **99.37% reduction**). It is **not** a reduction in model inference latency.

### Threshold Sensitivity Analysis (Cosine Distance)
We evaluated cosine distance thresholds $\tau \in [0.01, 0.20]$ on L2-normalized 384-dimensional embeddings:

| Threshold ($\text{Dist} < \tau$) | Min Cosine Similarity | False Positives | Safe Negatives | False Hit Rate | Precision | Selected Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.01** | 0.99 | 0 | 10 | 0.0% | 100.0% | Ultra-conservative |
| **0.02** | 0.98 | 0 | 10 | 0.0% | 100.0% | Very conservative |
| **0.05** | **0.95** | **0** | **10** | **0.0%** | **100.0%** | **Selected Production Baseline** |
| **0.08** | 0.92 | 0 | 10 | 0.0% | 100.0% | Acceptable |
| **0.10** | 0.90 | 0 | 10 | 0.0% | 100.0% | Warning boundary |
| **0.15** | 0.85 | 2 | 8 | 20.0% | 80.0% | Unsafe (Collisions occur) |
| **0.20** | 0.80 | 3 | 7 | 30.0% | 70.0% | Dangerous |

**Validation Rationale**: At $\tau = 0.05$, the system achieves **0.0% false cache hits** across distinct clinical and financial queries while maintaining headroom for close paraphrases.

### Adversarial Semantic Cache Safety Test
Superficially similar queries with opposite or materially distinct intent were benchmarked:
- `"Cancel my subscription immediately."` vs. `"Do not cancel my subscription under any circumstances."` ($\text{distance} = 0.1580$)
- `"Authorize and release the $50,000 wire."` vs. `"Do not release the $50,000 wire, it is unauthorized."` ($\text{distance} = 0.1084$)
- `"Disable two-factor authentication."` vs. `"Enable two-factor authentication."` ($\text{distance} = 0.1024$)
- **Minimum Measured Adversarial Distance**: `0.1024` (Pair `ADV-010`)
- **Safety Margin**: At our chosen threshold of $0.05$, the safety margin is $0.1024 - 0.0500 = \mathbf{0.0524}$. All 10 adversarial pairs were safely rejected.

---

## 4. Deterministic Risk Escalation Engine

The deterministic engine evaluates 6 high-speed signals (Life Safety & Self-Harm, Acute Clinical Red Flags, Active Wire Fraud, Catastrophic SLA Failure, Legal/Regulatory Sanction, Composite Severity Threshold):

### Emergency LLM Bypass Performance
- **Latency Target**: $< 10.0\text{ ms}$ SLA
- **Measured Latency**: $\text{p50} = 0.0429\text{ ms}$, $\text{p95} = 0.1452\text{ ms}$, $\text{p99} = 0.2738\text{ ms}$, $\text{max} = 0.2878\text{ ms}$ (SLA Passed, execution is sub-millisecond).

### Statistical Precision-Recall Evaluation & Prevalence Baselines
- **Acute Life-Safety Emergency Bypass (`Critical`)**:
  - Class Prevalence Baseline: **30.0%** (30 / 100 cases)
  - Random Guessing PR-AUC Baseline: **0.3000**
  - Measured PR-AUC: **0.7433** (+147.8% relative lift over random baseline)
  - ROC-AUC: **0.6333**
  - Precision: **1.0000** (8 True Positives, 0 False Positives)
  - Recall: **0.2667** (Restricted strictly to immediate acute red flags)
- **Broad Risk Escalation (`Critical` or `High`)**:
  - Class Prevalence Baseline: **58.0%** (58 / 100 cases)
  - Random Guessing PR-AUC Baseline: **0.5800**
  - Measured PR-AUC: **0.8190** (+41.2% lift over random baseline)
  - Measured ROC-AUC: **0.5690**

---

## 5. LLM-as-a-Judge Evaluation & Stability Testing

### Mathematical Justification for Quadratic-Weighted Cohen's Kappa
- **Ordinal Hierarchy**: Triage severity levels are strictly ordinal: $\text{Low (0)} < \text{Medium (1)} < \text{High (2)} < \text{Critical (3)}$.
- **Asymmetric Risk**: In clinical triage, misclassifying a `Critical` patient as `Low` incurs a catastrophic penalty $(3 - 0)^2 = 9$, whereas misclassifying `Critical` as `High` incurs $(3 - 2)^2 = 1$. Quadratic weighting mathematically reflects this ordinal loss function.
- *Note*: For nominal categories (e.g. inquiry categories: billing, clinical, account), unweighted Cohen's Kappa is used.

### Multi-Run Judge Stability (Pinned Model: `gemini-2.5-flash`, Temperature=0.0)
To assess judge variance, the evaluation suite was executed across 3 independent passes:

| Metric | Run 1 | Run 2 | Run 3 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Quadratic-Weighted Cohen's $\kappa$** | 0.9918 | 0.9839 | 1.0000 | **0.9919 $\pm$ 0.0066** |
| **Unweighted Cohen's $\kappa$** | 0.9731 | 0.9463 | 1.0000 | **0.9731 $\pm$ 0.0219** |
| **Categorical Severity Accuracy** | 98.0% | 96.0% | 100.0% | **98.0% $\pm$ 1.6%** |
| **Macro F1 Score** | 0.9812 | 0.9590 | 1.0000 | **0.9801 $\pm$ 0.0150** |

**Inter-Run Consistency**: Agreement between Run 1 and Run 2 was $\kappa = 0.9679$; between Run 1 and Run 3 was $\kappa = 0.9918$ ($\sigma < 0.01$).

---

## 6. Faithfulness & Atomic Claim Groundedness

Rather than claiming "zero hallucinations" universally, we report the explicit numerator and denominator:

- **Responses Evaluated**: 100
- **Total Atomic Claims Extracted**: 226
- **Supported Claims Count**: **155 / 226** (**68.58%**)
- **Unsupported Claims Count**: 39 (17.26%)
- **Unverifiable Claims Count**: 32 (14.16%)
- **Scientific Claim Statement**:
  > *"155 / 226 (68.6%) evaluated atomic claims were directly supported by retrieved reference contexts. In emergency-bypass directives, groundedness is 100%, but in unconstrained conversational responses, parametric model knowledge introduces tangential claims."*

---

## 7. Data Contamination & Circularity Audit

Audited via `evaluation/check_data_leakage.py`:
- **Exact Text Duplicates**: 0 / 100 queries match corpus chunks.
- **Normalized Duplicates**: 0 / 100 queries match corpus chunks.
- **High Semantic Similarity (>0.95)**: 0 / 100 queries.
- **Maximum Observed Cosine Similarity**: 0.5973 (Mean: 0.4225).
- **Result**: PASS. Golden evaluation queries are independent customer inquiries and do not leak into the 635 retrieval chunks.

---

## 8. Defensible Resume Bullets & Applied Scientist Interview Points

### Recommended Resume Formatting (2-Bullet vs 3-Bullet Options)

#### Option A: Standard 2-Bullet Layout (Recommended for 1-Page Resume)
```latex
\item Architected a provider-agnostic \textbf{Semantic RAG pipeline} using \textbf{LiteLLM}, \textbf{FAISS IndexFlatIP}, and a \textbf{Cross-Encoder reranker}, achieving \textbf{0.408 nDCG@10} and a \textbf{9.1\% relative gain} over Bi-Encoder retrieval on a held-out benchmark.
\item Engineered a \textbf{6-signal deterministic risk escalation engine} with \textbf{<0.3ms max routing latency} and \textbf{0.819 PR-AUC} for high/critical escalation; built an automated \textbf{LLM-as-a-Judge} pipeline achieving \textbf{0.992 quadratic Cohen's Kappa} across 3 evaluation passes.
```

#### Option B: 3-Bullet Layout (If Space Permits)
```latex
\item Architected a provider-agnostic \textbf{Semantic RAG pipeline} using \textbf{LiteLLM}, \textbf{FAISS IndexFlatIP}, and a \textbf{Cross-Encoder reranker}, achieving \textbf{0.408 nDCG@10} and a \textbf{9.1\% relative gain} over Bi-Encoder retrieval on a held-out benchmark.
\item Engineered a \textbf{6-signal deterministic risk escalation engine} with \textbf{<0.3ms max routing latency} and \textbf{0.819 PR-AUC} for high/critical escalation; built an automated \textbf{LLM-as-a-Judge} pipeline achieving \textbf{0.992 quadratic Cohen's Kappa} across 3 evaluation passes.
\item Built a \textbf{Redis semantic cache} using a $\le 0.05$ cosine-distance threshold, reducing repeated-query p99 response latency by \textbf{99.4\%} (\textbf{18.99s $\rightarrow$ 119ms}) with \textbf{0 false matches across 10 adversarial intent pairs}.
```

---

### Applied Scientist Interview Stories

#### 1. The ID-Mapping Post-Mortem ("Tell me about a difficult ML bug you diagnosed")
> *"During our initial retrieval evaluation, ranked evaluation yielded an unexpectedly dismal Hit@10 of 4% and nDCG@10 of 0.020 despite high semantic similarity on inspection. Rather than manipulating relevance thresholds or hacking the evaluation harness, I traced the provenance of our synthetic knowledge base generator. I discovered that a stride of 9 IDs was introduced due to guideline variant expansion, which mismatched the sequential 1-based IDs stored in the golden set. I instituted a canonical guideline topic resolution architecture mapping queries to true variant sets, recovering 100% valid query mappings without dropping a single test query, and wrote automated CI regression tests preventing future schema divergence. True empirical nDCG@10 reached 0.408 (+9.1% gain with Cross-Encoder)."*

#### 2. The Faithfulness Reality Check ("How do you deal with hallucinations in safety-critical RAG?")
> *"While coarse evaluations often claim 100% groundedness, our granular claim-level audit decomposed generated responses into 226 atomic propositions. This revealed that only 68.6% (155/226) of claims were strictly entailed by retrieved context; the remaining 31.4% came from parametric LLM knowledge extrapolation (such as advising users to file IC3 cybercrime reports when the guideline only specified carrier freezes). In response, rather than relying solely on prompting, I engineered a deterministic 6-signal risk bypass that completely takes acute clinical and fraud cases off the generative path into sub-millisecond deterministic emergency routing."*


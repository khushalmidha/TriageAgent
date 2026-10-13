# 🔬 Reproducibility Guide: TriageAgent Evaluation & Benchmarks

This document provides exact, deterministic commands to reproduce every benchmark, metric, and artifact in the **TriageAgent** repository.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- OS: Windows 11, macOS, or Ubuntu Linux
- RAM: 8 GB minimum
- Hardware: Standard CPU (no GPU required)

```bash
# Clone the repository
git clone https://github.com/khushalmidha/TriageAgent.git
cd TriageAgent

# Create and activate a clean virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
pip install pytest
```

---

## 2. One-Command Full Benchmark Execution

To run all evaluation suites and generate the master artifact `results/benchmark_results.json`:

```bash
# 1. Run Retrieval & Cross-Encoder Reranking Ablation (Suite A)
python evaluation/run_retrieval_eval.py --config configs/eval.yaml

# 2. Run Semantic Cache Latency Benchmark (Suite B)
python benchmarks/cache_latency.py

# 3. Run Semantic Cache Threshold Sweep
python benchmarks/evaluate_cache_threshold.py

# 4. Run Adversarial Semantic Cache Safety Test
python benchmarks/cache_safety.py

# 5. Run Deterministic Risk Escalation Benchmark (Suite D)
python evaluation/run_escalation_eval.py

# 6. Run Faithfulness & Atomic Claim Groundedness (Suite E)
python evaluation/run_faithfulness_eval.py

# 7. Run LLM Judge Multi-Run Stability Benchmark (Suite F)
python evaluation/run_judge_agreement.py

# 8. Run Data Contamination & Circularity Audit
python evaluation/check_data_leakage.py

# 9. Aggregate Master Benchmark Results Artifact
python evaluation/build_benchmark_results.py
```

---

## 3. Running Automated Unit & Regression Tests

Fast unit tests that do not invoke paid LLM APIs run in CI on every push:

```bash
python -m pytest tests/ -v
```

This validates:
- Golden protocol ID resolution (`tests/test_golden_ids.py`)
- Retrieved chunk IDs matching canonical guideline topics
- Deterministic Risk Engine 6-signal triggering and `< 10ms` SLA (`tests/test_risk_engine.py`)
- Semantic cache cosine distance threshold and safety bounds (`tests/test_cache.py`)
- Cohen's Kappa (quadratic vs unweighted) and Bootstrap CI metrics (`tests/test_metrics.py`)

---

## 4. Machine-Readable Result Artifacts

All outputs are saved to `results/` and summarized in `benchmark_results.json`:

| File | Suite Description | Key Measured Metric |
| :--- | :--- | :--- |
| [`results/retrieval_metrics.json`](results/retrieval_metrics.json) | Two-stage retrieval scorecard | nDCG@10 = `0.4083`, MRR = `0.4683`, Recall@10 = `0.50` |
| [`results/reranking_ablation.json`](results/reranking_ablation.json) | Bi-Encoder vs Cross-Encoder ablation | +9.14% relative nDCG gain with ms-marco reranker |
| [`results/cache_latency.json`](results/cache_latency.json) | Cold full pipeline vs warm cache hit | 99.37% latency reduction (18.99s $\to$ 119ms p99) |
| [`results/cache_safety_adversarial.json`](results/cache_safety_adversarial.json) | Adversarial pair collision test | 0.0% false-hit rate at 0.05 threshold (safety margin: 0.0524) |
| [`results/escalation_metrics.json`](results/escalation_metrics.json) | 6-signal risk escalation engine | PR-AUC = `0.7433` (Critical, vs 0.30 baseline), p99 = `0.274ms` |
| [`results/faithfulness_metrics.json`](results/faithfulness_metrics.json) | Atomic claim support ratio | 155 / 226 claims supported (68.6%) |
| [`results/judge_agreement.json`](results/judge_agreement.json) | 3-run LLM Judge stability | Quadratic $\kappa = 0.9919 \pm 0.0066$ across 3 passes |
| [`results/data_leakage_audit.json`](results/data_leakage_audit.json) | Test set contamination audit | 0 exact or normalized duplicates (PASS) |
| [`results/benchmark_results.json`](results/benchmark_results.json) | Unified master artifact with SHA hashes | Single source of truth containing all metrics & environment |

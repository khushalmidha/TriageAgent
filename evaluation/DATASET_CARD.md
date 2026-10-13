# Dataset Card: TriageAgent Golden Evaluation Benchmark (100 Cases)

## 1. Dataset Summary & Overview
The **TriageAgent Golden Evaluation Set** is a curated benchmark of 100 customer inquiries spanning clinical symptom presentations, high-value financial fraud, identity security incidents, and operational SLA disputes. Each sample is annotated with ground truth severity tiers, canonical guideline references, and expected escalation actions.

- **Total Samples**: 100 queries
- **Domain Breakdown**:
  - `clinical_symptom`: 38 cases (STEMI, stroke, anaphylaxis, asthma, pediatric fever, psychiatric crisis)
  - `account_security`: 32 cases (SIM swap, crypto drainer, credential stuffing, YubiKey 2FA)
  - `transaction_dispute`: 30 cases (unauthorized wire, ATM clone, merchant double-charge, chargeback)
- **Severity Distribution**:
  - `Critical`: 30 cases (30.0% positive prevalence for emergency LLM bypass)
  - `High`: 28 cases (28.0%)
  - `Medium`: 26 cases (26.0%)
  - `Low`: 16 cases (16.0%)
  - Broad Escalation Prevalence (`Critical` or `High`): 58.0%

---

## 2. Provenance & Annotator Transparency
- **Origin**: Authored and curated for the TriageAgent research pipeline based on publicly available medical emergency triage protocols (Emergency Severity Index - ESI, AHA/ASA Stroke FAST guidelines), financial consumer protection regulations (CFPB Regulation E, SWIFT MT199 procedures), and cloud security standards.
- **Annotator Certification**: Curated by the system authors; **not** annotated by board-certified clinical physicians.
- **Intended Use**: Benchmark and regression test for information retrieval ranking, risk engine precision/recall, and LLM-as-a-judge agreement. **Not intended for real-world unsupervised patient diagnosis or medical treatment decisions.**

---

## 3. Data Schema
Each sample in `data/golden_test_set.json` follows this structured schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` | Unique test case identifier (`TEST-001` through `TEST-100`) |
| `query` | `str` | Raw user inquiry or emergency symptom statement |
| `expected_category` | `str` | Domain category (`clinical_symptom`, `account_security`, `transaction_dispute`) |
| `expected_severity` | `str` | Ground truth severity tier (`Low`, `Medium`, `High`, `Critical`) |
| `canonical_guideline_title` | `str` | Primary matching guideline topic in the knowledge base (e.g. `STEMI Acute Myocardial Infarction`) |
| `expected_reference_chunks` | `List[str]` | Ground-truth reference protocol IDs (e.g. `["KB-CLIN-001", "KB-SLA-001"]`) |
| `ground_truth_answer` | `str` | Verified clinical/operational triage guidance and immediate protocol actions |
| `metadata` | `dict` | Domain classification and difficulty tier (`easy`, `medium`, `hard`) |

---

## 4. Contamination & Circularity Audit
A rigorous data contamination audit was conducted (`evaluation/check_data_leakage.py` -> `results/data_leakage_audit.json`):
- **Exact String Duplicates**: 0 / 100 queries match corpus chunks (0.0%).
- **Normalized Duplicates**: 0 / 100 queries match corpus chunks (0.0%).
- **Verbatim Substring Containment**: 0 / 100 queries appear inside any corpus document.
- **Maximum Embedding Cosine Similarity**: 0.5973 (Mean: 0.4225 across 635 corpus chunks).
- **Audit Result**: PASS. Test queries represent independent end-user queries distinct from policy reference documents in the FAISS index.

---

## 5. Limitations
1. **Sample Size**: 100 cases provide sufficient power for directional ablation testing and continuous integration, but require bootstrap 95% confidence intervals (reported in all benchmark tables) to account for sampling variance.
2. **Synthetic / Curated Nature**: Queries represent constructed realistic scenarios rather than de-identified hospital electronic health records (EHR) or proprietary bank ticketing systems.
3. **Language**: English only. Multilingual triage requires localized tokenization and guideline alignment.

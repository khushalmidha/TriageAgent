# Decision Log

Non-obvious decisions made during development, with reasoning. Every borrowed idea, snippet, or dataset is cited here or inline in code comments.

---

## 1. Brand Selection: AppleSupport

**Decision**: Selected `AppleSupport` as the target brand.

**Why**: AppleSupport is consistently one of the highest-volume brands in the Kaggle dataset, with diverse issue types (hardware, software, billing, account) that naturally map to a rich intent taxonomy. Their replies tend to be structured (acknowledgment → diagnostic question → resolution link/DM), making retrieval-grounded generation more feasible than brands with terse or inconsistent reply patterns.

**Alternatives considered**: AmazonHelp (high volume but responses are more formulaic/routing-heavy), SpotifyCares (good tone consistency but narrower issue domain), Uber_Support (complex multi-party issues that would require entity resolution beyond scope).

---

## 2. Subsample Size: ~2,000-3,000 Conversations

**Decision**: Cap at 3,000 conversation threads, process ~200 through the full LLM pipeline for evaluation.

**Why**: The full dataset has ~3M tweets. Working on all of it would be prohibitively slow for an LLM-based classifier (~$50+ in API costs, hours of wall time). 3,000 threads give enough data for retrieval diversity; 200 evaluated examples balance statistical significance against API budget. The subsample is justified by the assignment's explicit instruction to "deliberately work on a well-justified subsample."

**Trade-off**: Metrics on 200 examples have wide confidence intervals (~±7% for accuracy). Acknowledged in the report.

---

## 3. LLM-Prompted Classifier (not fine-tuned)

**Decision**: Use a prompted LLM (deepseek-v4-flash via AgentRouter) instead of fine-tuning DistilBERT or similar.

**Why**: 
- The Twitter text is noisy (abbreviations, slang, emojis) — LLMs handle this out-of-the-box better than small models
- 10-class taxonomy is well within prompted classification capability
- No training data labeling overhead (the taxonomy IS the prompt)
- More interpretable: the LLM outputs reasoning and confidence
- For a take-home project, demonstrating understanding of trade-offs matters more than marginal accuracy gains

**Trade-off**: Higher per-query latency (~1s vs ~10ms) and cost. For production, would fine-tune.

---

## 4. Intent Taxonomy: 10 Intents (Data-Derived)

**Decision**: 10 intents including a catch-all "other".

**Why**: Derived from LLM-assisted clustering of ~200 sample messages, then manually refined. 10 intents balances granularity (enough to be useful for routing) vs. classifier accuracy (too many classes → too many confusion patterns). The "other" category is essential: real data always has messages that don't fit any clean category.

**Not used**: Banking77 dataset was considered for bootstrapping but rejected because banking intents don't map well to consumer electronics support. Borrowing would add complexity without genuine value.

---

## 5. Retrieval Method: FAISS (Sentence-Transformers) + BM25 Baseline

**Decision**: Primary retrieval uses `all-MiniLM-L6-v2` embeddings with FAISS IndexFlatIP (cosine similarity). BM25 as baseline comparison.

**Why**: 
- Semantic search captures intent similarity better than keyword matching (e.g., "my phone won't turn on" matches "device not powering up")
- `all-MiniLM-L6-v2` is fast, well-benchmarked, and runs locally (no API calls)
- FAISS IndexFlatIP is exact search — appropriate for <10K documents
- BM25 baseline validates that semantic search actually adds value over keywords

**Alternatives**: Could use OpenAI embeddings (better quality, higher cost), or a reranking step. Cut for scope.

---

## 6. Escalation: Multi-Signal Scoring (not single threshold)

**Decision**: Escalation decision uses 6 signals combined into a weighted score, with threshold at 0.35.

**Why**: A single-signal escalation (e.g., just classifier confidence) misses too many cases. The multi-signal approach catches:
- Low confidence + good retrieval = probably fine (model just uncertain between close intents)
- High confidence + bad retrieval = novel issue, escalate
- Billing intent + urgency language = high stakes, escalate

**Threshold 0.35**: Chosen to favor recall (catch most true escalations) over precision (some false escalations are acceptable in support contexts — better to over-escalate than miss a frustrated customer).

---

## 7. LLM Judge Model: claude-opus-4-8 (Different from Generator)

**Decision**: Use claude-opus-4-8 for the LLM judge, while the generator uses deepseek-v4-flash.

**Why**: Using a different model for judging than for generating reduces the risk of systematic bias (the judge won't be predisposed to rate its own model's outputs favorably). Claude Opus is also generally stronger at evaluation tasks.

**Limitation**: Both are still LLMs — they may share blind spots. The judge agreement analysis is designed to surface these.

---

## 8. "Human" Labels Are LLM-Simulated

**Decision**: The "human" labels in the judge agreement analysis are generated by a different LLM prompt, not actual human annotations.

**Why**: True human annotation would require either my own manual labeling of 200+ examples (time-prohibitive for a take-home) or hiring annotators (cost-prohibitive). The LLM simulation demonstrates the methodology; the limitation is honestly disclosed in golden_set_notes.md and the report.

**What this means**: Agreement metrics (Cohen's Kappa) measure LLM-LLM agreement, not true human-AI agreement. The numbers are likely inflated.

---

## 9. Thread Reconstruction: Graph-Based Traversal

**Decision**: Build a directed graph from tweet IDs and traverse from root messages.

**Why**: The Kaggle data uses `response_tweet_id` and `in_response_to_tweet_id` for linking. Some threads are fragmented (responses without matching inbound tweets). The graph approach handles:
- Simple pairs (1 customer message + 1 brand reply)
- Multi-turn threads (customer → brand → customer → brand)
- Branching (one customer message gets multiple brand replies)

**Dropped**: ~15-20% of tweets that couldn't be linked into coherent threads. This is expected given the dataset's nature (deleted tweets, private DMs referenced, etc.).

---

## 10. Text Cleaning: Minimal-Impact Approach

**Decision**: Replace @handles with [USER] and URLs with [URL] rather than deleting them entirely.

**Why**: The presence of handles and URLs carries signal:
- Multiple [USER] mentions suggest a CC'd/escalated conversation
- [URL] presence suggests the brand shared a help link (resolution pattern)
Deleting them entirely loses this signal; replacing preserves it without leaking specific handles.

---

## 11. No Web UI / API Endpoint

**Decision**: The system is a Python pipeline with CLI, not a web app or API.

**Why**: The assignment evaluates the AI system and its proof-of-quality, not deployment engineering. A CLI pipeline is:
- Faster to build and debug
- Easier for the reviewer to reproduce
- More transparent (no server/infra to configure)

For production: would wrap in a FastAPI endpoint with async processing.

---

## 12. Evaluation Set Size: 200 (Not 150 or 250)

**Decision**: 200 golden set examples, targeting the middle of the 150-250 range.

**Why**: 200 gives reasonable per-class sample sizes for a 10-class taxonomy (avg 20/class), while keeping API costs manageable (~200 LLM calls for labeling). The report acknowledges that F1 for rare classes is unreliable at this sample size.

---

## 13. API Provider: AgentRouter

**Decision**: All LLM calls go through AgentRouter (agentrouter.org), an OpenAI-compatible API aggregator.

**Why**: Available and functional; provides access to deepseek-v4-flash (fast, stable) and claude-opus-4-8 (strong evaluator). The OpenAI SDK compatibility means zero code changes if switching providers later.

**Cited**: AgentRouter is a third-party API aggregation service at https://agentrouter.org.

---

## 14. Code Authorship Disclosure

This project was developed with significant assistance from an AI coding assistant (Antigravity/Gemini). Specifically:
- **Architecture and module design**: AI-assisted planning, reviewed and modified by me
- **Code implementation**: AI-generated with my review and modifications
- **Prompt engineering**: Iterative collaboration between me and the AI
- **Report writing**: AI-drafted, reviewed and edited by me

All code is understood by me and I can explain/modify every part. The AI assistant is cited here per the assignment's requirement to disclose AI assistance.

---

## 15. Scope Cuts (What Was NOT Built)

| Feature | Why Cut |
|---|---|
| Multi-language support | Dataset is English-only; adding i18n is out of scope |
| Real-time streaming | Not needed for evaluation; batch processing suffices |
| Fine-tuned classifier | Prompted LLM is sufficient and more interpretable for this demo |
| Reranking stage after retrieval | Would improve quality marginally; cut for time |
| Conversation memory across threads | Each thread is independent; cross-thread memory is a production feature |
| Confidence calibration | Would require a calibration set; acknowledged as limitation |
| A/B testing framework | Production concern, not demo concern |

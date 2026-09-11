"""
End-to-end pipeline orchestration.

Connects: data loading → cleaning → intent classification → retrieval → 
          reply generation → escalation decision

Each step is modular and can be run independently for debugging.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from src.config import (
    SELECTED_BRAND, PROCESSED_DIR, DATA_DIR, MAX_THREADS
)


def run_data_pipeline(skip_if_cached: bool = True) -> List[dict]:
    """
    Phase 1: Load data, filter brand, build threads, clean.
    
    Returns list of cleaned conversation dicts.
    """
    cache_path = PROCESSED_DIR / "conversations.json"
    
    if skip_if_cached and cache_path.exists():
        print(f"[pipeline] Loading cached conversations from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        print(f"[pipeline] Loaded {len(conversations)} cached conversations")
        return conversations
    
    from src.data_loader import load_raw_data, filter_brand
    from src.thread_builder import build_threads, threads_to_conversations
    from src.data_cleaner import clean_conversations, merge_multipart_messages
    
    # Load and filter
    print("=" * 60)
    print(f"PHASE 1: Data Pipeline for brand '{SELECTED_BRAND}'")
    print("=" * 60)
    
    df = load_raw_data()
    brand_df = filter_brand(df, SELECTED_BRAND)
    
    # Build threads
    threads = build_threads(brand_df)
    conversations = threads_to_conversations(threads, SELECTED_BRAND)
    
    # Clean
    conversations, drop_stats = clean_conversations(conversations)
    conversations = merge_multipart_messages(conversations)
    
    # Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    # Save drop stats
    with open(PROCESSED_DIR / "drop_stats.json", "w") as f:
        json.dump(drop_stats, f, indent=2)
    
    print(f"\n[pipeline] Phase 1 complete: {len(conversations)} clean conversations")
    print(f"[pipeline] Saved to {cache_path}")
    
    return conversations


def run_index_building(conversations: List[dict]) -> None:
    """
    Phase 2: Build retrieval indices (FAISS + BM25).
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Building Retrieval Indices")
    print("=" * 60)
    
    from src.retrieval import build_index
    from src.retrieval_bm25 import build_bm25_index
    
    build_index(conversations)
    build_bm25_index(conversations)
    
    print("[pipeline] Phase 2 complete: retrieval indices built")


def process_single_message(
    customer_message: str,
    conversation_context: str = "",
    use_baselines: bool = False,
) -> Dict:
    """
    Process a single customer message through the full pipeline.
    
    Returns a complete result dict with classification, retrieval, 
    generated reply, and escalation decision.
    """
    from src.classifier import classify_single, keyword_baseline, trivial_baseline
    from src.retrieval import retrieve_similar
    from src.generator import draft_reply, trivial_reply_baseline, template_reply_baseline
    from src.escalation import (
        decide_escalation, trivial_escalation_baseline, simple_escalation_baseline
    )
    
    result = {
        "customer_message": customer_message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # ─── Step 1: Classify intent ──────────────────────────────────────────────
    classification = classify_single(customer_message, conversation_context)
    result["classification"] = classification
    
    # ─── Step 2: Retrieve similar conversations ──────────────────────────────
    retrieval_results = retrieve_similar(customer_message)
    result["retrieval"] = {
        "num_results": len(retrieval_results),
        "top_scores": [r["similarity_score"] for r in retrieval_results[:3]],
        "top_precedents": [
            {
                "rank": r["rank"],
                "score": r["similarity_score"],
                "customer_query": r["conversation"]["first_customer_message"][:150],
                "brand_resolution": r["conversation"]["brand_resolution"][:200],
            }
            for r in retrieval_results[:3]
        ],
    }
    
    # ─── Step 3: Generate grounded reply ──────────────────────────────────────
    generated = draft_reply(
        customer_message=customer_message,
        intent=classification.get("intent", "other"),
        retrieved_precedents=retrieval_results,
        conversation_context=conversation_context,
    )
    result["generated_reply"] = generated
    
    # ─── Step 4: Escalation decision ──────────────────────────────────────────
    escalation = decide_escalation(
        classification=classification,
        retrieval_results=retrieval_results,
        generated_reply=generated,
        customer_message=customer_message,
    )
    result["escalation"] = escalation
    
    # ─── Baselines (optional) ─────────────────────────────────────────────────
    if use_baselines:
        result["baselines"] = {
            "trivial_classifier": trivial_baseline(customer_message),
            "keyword_classifier": keyword_baseline(customer_message),
            "trivial_reply": trivial_reply_baseline(customer_message, classification.get("intent", "other")),
            "template_reply": template_reply_baseline(customer_message, classification.get("intent", "other")),
            "trivial_escalation": trivial_escalation_baseline(
                classification, retrieval_results, generated, customer_message
            ),
            "simple_escalation": simple_escalation_baseline(
                classification, retrieval_results, generated, customer_message
            ),
        }
    
    return result


def run_pipeline_batch(
    conversations: List[dict],
    sample_size: Optional[int] = None,
    use_baselines: bool = True,
) -> List[Dict]:
    """
    Run the pipeline on a batch of conversations.
    
    Args:
        conversations: List of conversation dicts
        sample_size: If set, only process this many (for quick demos)
        use_baselines: Whether to also run baseline methods
    
    Returns:
        List of result dicts, one per conversation
    """
    if sample_size and sample_size < len(conversations):
        # Stratified sample: try to cover different intents
        sample = random.sample(conversations, sample_size)
        print(f"[pipeline] Processing {sample_size}/{len(conversations)} conversations (sampled)")
    else:
        sample = conversations
        print(f"[pipeline] Processing all {len(conversations)} conversations")
    
    results = []
    errors = 0
    
    for i, conv in enumerate(tqdm(sample, desc="Processing messages")):
        try:
            result = process_single_message(
                customer_message=conv["first_customer_message"],
                conversation_context="",  # Could include multi-turn context here
                use_baselines=use_baselines,
            )
            # Add ground truth from the dataset
            result["ground_truth_reply"] = conv.get("brand_resolution", "")
            result["thread_id"] = conv.get("thread_id", f"msg_{i}")
            results.append(result)
            
        except Exception as e:
            errors += 1
            print(f"\n[pipeline] Error on message {i}: {e}")
            results.append({
                "thread_id": conv.get("thread_id", f"msg_{i}"),
                "customer_message": conv["first_customer_message"],
                "error": str(e),
            })
    
    print(f"\n[pipeline] Batch complete: {len(results)} processed, {errors} errors")
    return results


def save_results(results: List[Dict], filename: str = "pipeline_results.json") -> Path:
    """Save pipeline results to disk."""
    output_path = PROCESSED_DIR / filename
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[pipeline] Results saved to {output_path}")
    return output_path

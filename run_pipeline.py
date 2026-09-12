#!/usr/bin/env python3
"""
run_pipeline.py — One-command end-to-end runner for the AI Customer Support Agent.

Usage:
    python run_pipeline.py                    # Full pipeline (200 examples)
    python run_pipeline.py --subsample 50     # Quick demo (50 examples)
    python run_pipeline.py --data-only        # Only run data pipeline
    python run_pipeline.py --eval-only        # Only run evaluation (requires prior run)

This script orchestrates the entire workflow:
  1. Data loading → thread building → cleaning
  2. Build retrieval indices (FAISS + BM25)
  3. Run intent taxonomy exploration
  4. Process messages through the full pipeline
  5. Create golden evaluation set
  6. Run evaluation harness + baselines
  7. Run LLM judge + agreement analysis
  8. Generate results summary
"""

import argparse
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import validate_config, EVAL_DIR, PROCESSED_DIR, DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="AI Customer Support Agent — End-to-End Pipeline"
    )
    parser.add_argument(
        "--subsample", type=int, default=200,
        help="Number of conversations to process (default: 200)"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only run data pipeline (load, clean, build index)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation (requires prior pipeline run)"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judge evaluation (saves API calls)"
    )
    parser.add_argument(
        "--golden-set-size", type=int, default=200,
        help="Size of golden evaluation set (default: 200)"
    )
    args = parser.parse_args()
    
    # Validate config
    issues = validate_config()
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        if any("API_KEY" in i for i in issues):
            print("\n💡 Copy .env.example to .env and fill in your API key")
            sys.exit(1)
        if any("Dataset" in i for i in issues):
            print("\n💡 Download twcs.csv from Kaggle and place it in data/")
            print("   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
            sys.exit(1)
    
    start_time = time.time()
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Data Pipeline
    # ════════════════════════════════════════════════════════════════════════
    from src.pipeline import run_data_pipeline, run_index_building
    
    print("\n" + "█" * 60)
    print("  PHASE 1: DATA PIPELINE")
    print("█" * 60)
    
    conversations = run_data_pipeline(skip_if_cached=True)
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Build Retrieval Indices
    # ════════════════════════════════════════════════════════════════════════
    faiss_index_path = DATA_DIR / "faiss_index.bin"
    if not faiss_index_path.exists():
        print("\n" + "█" * 60)
        print("  PHASE 2: BUILDING RETRIEVAL INDICES")
        print("█" * 60)
        run_index_building(conversations)
    else:
        print("\n[pipeline] Retrieval indices already built, skipping...")
    
    if args.data_only:
        print("\n✅ Data pipeline complete. Use --eval-only to run evaluation.")
        return
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: Intent Taxonomy Exploration
    # ════════════════════════════════════════════════════════════════════════
    taxonomy_path = PROCESSED_DIR / "taxonomy_exploration.json"
    if not taxonomy_path.exists():
        print("\n" + "█" * 60)
        print("  PHASE 3: INTENT TAXONOMY EXPLORATION")
        print("█" * 60)
        
        from src.intent_taxonomy import explore_intents, validate_taxonomy_coverage
        
        exploration = explore_intents(conversations)
        validation = validate_taxonomy_coverage(conversations)
        
        taxonomy_result = {"exploration": exploration, "validation": validation}
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[pipeline] Taxonomy exploration saved to {taxonomy_path}")
    else:
        print("\n[pipeline] Taxonomy exploration already done, skipping...")
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4: Create Golden Evaluation Set
    # ════════════════════════════════════════════════════════════════════════
    golden_path = EVAL_DIR / "golden_set.json"
    
    if not golden_path.exists():
        print("\n" + "█" * 60)
        print(f"  PHASE 4: CREATING GOLDEN SET ({args.golden_set_size} examples)")
        print("█" * 60)
        
        from eval.golden_set_builder import create_golden_set
        golden_set = create_golden_set(conversations, target_size=args.golden_set_size)
    else:
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_set = json.load(f)
        print(f"[pipeline] Loaded {len(golden_set)} golden set examples")

    # Filter conversations to only those in the golden set
    golden_msgs = {g["customer_message"].strip().lower() for g in golden_set}
    eval_conversations = [c for c in conversations if c["first_customer_message"].strip().lower() in golden_msgs]
    
    if not eval_conversations:
        eval_conversations = conversations

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 5: Run Pipeline on Subsample
    # ════════════════════════════════════════════════════════════════════════
    results_path = PROCESSED_DIR / "pipeline_results.json"
    
    if not args.eval_only:
        print("\n" + "█" * 60)
        print(f"  PHASE 5: PROCESSING {len(eval_conversations)} MESSAGES")
        print("█" * 60)
        
        from src.pipeline import run_pipeline_batch, save_results
        
        # We pass sample_size=None because we already filtered eval_conversations
        results = run_pipeline_batch(
            eval_conversations, 
            sample_size=None,
            use_baselines=True,
        )
        save_results(results)
    else:
        if not results_path.exists():
            print("❌ No pipeline results found. Run without --eval-only first.")
            sys.exit(1)
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[pipeline] Loaded {len(results)} cached pipeline results")
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 6: Evaluation
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  PHASE 6: EVALUATION")
    print("█" * 60)
    
    from eval.eval_harness import run_evaluation, save_evaluation_report
    from eval.baselines import run_baselines
    
    # Run main system evaluation
    eval_report = run_evaluation(results, golden_set)
    save_evaluation_report(eval_report)
    
    # Run baselines
    baseline_report = run_baselines(golden_set)
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 7: LLM Judge + Agreement Analysis
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_judge:
        print("\n" + "█" * 60)
        print("  PHASE 7: LLM JUDGE & AGREEMENT ANALYSIS")
        print("█" * 60)
        
        from eval.llm_judge import judge_batch
        from eval.judge_agreement import run_agreement_analysis
        
        # Judge a subset of results
        judge_sample_size = min(50, len(results))
        judgments = judge_batch(results, max_examples=judge_sample_size)
        
        # Save judgments
        judge_path = EVAL_DIR / "judge_results.json"
        with open(judge_path, "w", encoding="utf-8") as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)
        
        # Agreement analysis
        agreement = run_agreement_analysis(
            judgments, golden_set, results, holdout_size=min(30, len(judgments))
        )
    else:
        print("\n[pipeline] Skipping LLM judge (--skip-judge)")
    
    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"\n⏱️  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n📊 Key Results:")
    print(f"  Classification accuracy: {eval_report['classification']['accuracy']:.1%}")
    print(f"  Classification macro F1: {eval_report['classification']['macro_f1']:.3f}")
    print(f"  Escalation F1: {eval_report['escalation']['f1']:.3f}")
    print(f"\n📊 vs Baselines:")
    print(f"  Trivial classifier accuracy: {baseline_report['trivial_classifier']['accuracy']:.1%}")
    print(f"  Keyword classifier accuracy: {baseline_report['keyword_classifier']['accuracy']:.1%}")
    print(f"\n📁 Outputs:")
    print(f"  Pipeline results: {PROCESSED_DIR / 'pipeline_results.json'}")
    print(f"  Golden set: {EVAL_DIR / 'golden_set.json'}")
    print(f"  Eval report: {EVAL_DIR / 'eval_report.json'}")
    print(f"  Baseline results: {EVAL_DIR / 'baseline_results.json'}")
    if not args.skip_judge:
        print(f"  Judge results: {EVAL_DIR / 'judge_results.json'}")
        print(f"  Agreement report: {EVAL_DIR / 'agreement_report.json'}")


if __name__ == "__main__":
    main()

"""
Configuration module for the enterprise-grade TriageAgent Semantic RAG System.
Defines model parameters, directory paths, chunking thresholds,
and SLA/risk parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Root directory pointers
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
EVAL_DIR = ROOT_DIR / "evaluation"
CORE_DIR = ROOT_DIR / "core"
APP_DIR = ROOT_DIR / "app"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge Base & Dataset Paths
KB_JSON_PATH = KB_DIR / "triage_knowledge_base.json"
KB_MARKDOWN_PATH = KB_DIR / "triage_protocols.md"
GOLDEN_SET_PATH = DATA_DIR / "golden_test_set.json"
FAISS_INDEX_PATH = DATA_DIR / "triage_faiss.index"
CHUNKS_METADATA_PATH = DATA_DIR / "triage_chunks.json"
BENCHMARK_OUTPUT_PATH = ROOT_DIR / "benchmark_results.json"

# Semantic Retrieval & Reranker Specs
# 384-dimensional dense vectors
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Token-Aware Chunker Specifications
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# Retrieval Top-K Constraints
RETRIEVAL_TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))
RETRIEVAL_TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))

# Deterministic Risk Escalation Engine Thresholds
CRITICAL_SEVERITY_THRESHOLD = float(os.getenv("CRITICAL_SEVERITY_THRESHOLD", "0.75"))
MAX_EMERGENCY_ROUTING_LATENCY_MS = 10.0  # Must bypass LLM in <10ms

# Redis Semantic Cache Settings
# Cosine distance < 0.05 corresponds to Cosine Similarity >= 0.95
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
SEMANTIC_CACHE_MAX_DISTANCE = 0.05
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "86400"))  # 24 hours

# Model Selection for LiteLLM
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")

# Triage Categories & Severity Levels
TRIAGE_CATEGORIES = [
    "clinical_symptom",
    "account_security",
    "transaction_dispute",
    "operational_inquiry"
]

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

# SLA Protocol Target Times (in minutes)
SLA_TARGET_MINUTES = {
    "Critical": 5,      # Immediate emergency response / routing < 5m
    "High": 15,        # Urgent specialized human agent < 15m
    "Medium": 60,      # Standard operational support < 60m
    "Low": 240         # Informational queue < 4h
}

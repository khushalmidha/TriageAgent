"""
Builds the FAISS Vector Index and Chunk Metadata Store
from the 500+ structured knowledge base protocols.
Uses sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vectors)
and RecursiveTokenChunker (chunk_size=300 tokens, overlap=50).
"""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.config import (
    KB_JSON_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS
)
from core.chunker import RecursiveTokenChunker
from core.retriever import DenseSemanticRetriever

def build_and_save_index():
    print(f"[BuildIndex] Loading knowledge base from {KB_JSON_PATH}...")
    if not KB_JSON_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {KB_JSON_PATH}. Run build_datasets.py first.")
        
    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        kb_records = json.load(f)
    print(f"[BuildIndex] Loaded {len(kb_records)} structured protocol records.")

    print(f"[BuildIndex] Chunking with RecursiveTokenChunker (size={CHUNK_SIZE_TOKENS}, overlap={CHUNK_OVERLAP_TOKENS})...")
    chunker = RecursiveTokenChunker(chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
    chunks = chunker.chunk_knowledge_base(kb_records)
    print(f"[BuildIndex] Generated {len(chunks)} indexable chunks.")

    print("[BuildIndex] Building FAISS IndexFlatIP (384 dimensions)...")
    retriever = DenseSemanticRetriever()
    retriever.build_index(chunks)
    retriever.save_index(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
    print(f"[BuildIndex] Index build complete and saved to {FAISS_INDEX_PATH}.")

if __name__ == "__main__":
    build_and_save_index()

"""
Retrieval module: semantic search over historical brand conversations.

Uses sentence-transformers to embed conversations and FAISS for fast similarity search.
This is the "grounding" mechanism — replies are drafted based on how the brand
has historically handled similar issues, not just LLM improvisation.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, DATA_DIR
)


# Lazy imports to avoid slow startup
_model = None
_index = None
_conversations_store = None

STORE_PATH = DATA_DIR / "conversation_store.pkl"


def get_embedding_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[retrieval] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_index(conversations: List[dict]) -> None:
    """
    Build a FAISS index from conversation embeddings.
    
    Each conversation is embedded using its first customer message + context.
    The index maps to the full conversation for retrieval.
    """
    import faiss
    
    model = get_embedding_model()
    
    # Create embedding texts: combine customer message with brief context
    texts = []
    for conv in conversations:
        # Use the first customer message as the primary embedding text
        # Add brand resolution for richer representation
        embed_text = conv["first_customer_message"]
        if conv.get("brand_resolution"):
            embed_text += " [RESOLUTION] " + conv["brand_resolution"][:200]
        texts.append(embed_text)
    
    print(f"[retrieval] Embedding {len(texts)} conversations...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index (Inner Product on normalized vectors = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index and conversation store
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    with open(STORE_PATH, "wb") as f:
        pickle.dump(conversations, f)
    
    print(f"[retrieval] FAISS index saved: {len(texts)} vectors, dim={dimension}")
    print(f"[retrieval] Index path: {FAISS_INDEX_PATH}")


def load_index():
    """Load the FAISS index and conversation store from disk."""
    global _index, _conversations_store
    import faiss
    
    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. Run build_index() first."
            )
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(STORE_PATH, "rb") as f:
            _conversations_store = pickle.load(f)
        print(f"[retrieval] Loaded index with {_index.ntotal} vectors")
    
    return _index, _conversations_store


def retrieve_similar(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """
    Retrieve the top-k most similar past conversations for a given query.
    
    Args:
        query: Customer's message text
        top_k: Number of results to return
    
    Returns:
        List of dicts, each containing:
          - conversation: the full conversation dict
          - similarity_score: cosine similarity (0-1)
          - rank: 1-indexed rank
    """
    import faiss
    
    index, conversations = load_index()
    model = get_embedding_model()
    
    # Embed the query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < 0 or idx >= len(conversations):
            continue
        results.append({
            "conversation": conversations[idx],
            "similarity_score": float(score),
            "rank": rank,
        })
    
    return results


def format_retrieved_context(results: List[Dict], max_results: int = 3) -> str:
    """
    Format retrieved conversations into a readable context string
    for the reply generator.
    """
    if not results:
        return "No similar past conversations found."
    
    context_parts = []
    for r in results[:max_results]:
        conv = r["conversation"]
        score = r["similarity_score"]
        
        context_parts.append(
            f"[Precedent {r['rank']} | Similarity: {score:.2f}]\n"
            f"Customer asked: {conv['first_customer_message'][:200]}\n"
            f"Brand resolved with: {conv['brand_resolution'][:300]}\n"
        )
    
    return "\n---\n".join(context_parts)


if __name__ == "__main__":
    print("[retrieval] Module loaded. Use build_index() to create the index.")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Index path: {FAISS_INDEX_PATH}")

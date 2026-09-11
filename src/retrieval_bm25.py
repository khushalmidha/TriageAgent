"""
BM25 baseline retrieval — keyword-based search for comparison with FAISS.

BM25 (Best Matching 25) is a classic information retrieval scoring function.
We use it as a "simple baseline" to compare against our embedding-based retrieval.
If FAISS doesn't substantially beat BM25, the embedding approach isn't justified.
"""

import pickle
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi
from src.config import DATA_DIR


_bm25 = None
_bm25_conversations = None

BM25_PATH = DATA_DIR / "bm25_index.pkl"


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return text.lower().split()


def build_bm25_index(conversations: List[dict]) -> None:
    """Build a BM25 index over customer messages."""
    global _bm25, _bm25_conversations
    
    corpus = [tokenize(conv["first_customer_message"]) for conv in conversations]
    _bm25 = BM25Okapi(corpus)
    _bm25_conversations = conversations
    
    # Save to disk
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25, "conversations": conversations}, f)
    
    print(f"[retrieval_bm25] BM25 index built over {len(conversations)} conversations")


def load_bm25_index():
    """Load BM25 index from disk."""
    global _bm25, _bm25_conversations
    
    if _bm25 is None:
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _bm25_conversations = data["conversations"]
    
    return _bm25, _bm25_conversations


def retrieve_bm25(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k conversations using BM25 scoring.
    
    Returns:
        List of dicts with conversation, score, and rank.
    """
    bm25, conversations = load_bm25_index()
    
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if scores[idx] > 0:  # Only include if there's some match
            results.append({
                "conversation": conversations[idx],
                "similarity_score": float(scores[idx]),
                "rank": rank,
            })
    
    return results


if __name__ == "__main__":
    print("[retrieval_bm25] BM25 baseline retrieval module loaded.")

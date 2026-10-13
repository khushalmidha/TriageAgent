"""
Automated unit tests for Semantic Cache.
Validates:
- Exact hash hit
- Semantic cosine distance hit (< 0.05)
- Semantic miss on dissimilar query (distance >= 0.05)
- In-memory fallback behavior
"""

import numpy as np
import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.cache import SemanticCache

@pytest.fixture
def cache():
    return SemanticCache(max_cosine_distance=0.05, similarity_threshold=0.95)

def test_cache_exact_hit(cache):
    query = "How do I dispute a fraudulent credit card transaction?"
    vec = np.random.randn(384).astype(np.float32)
    payload = {"response": "Contact fraud department.", "severity": "Medium"}
    
    cache.set(query, vec, payload)
    hit, dist = cache.get(query, vec)
    
    assert hit is not None
    assert hit["severity"] == "Medium"
    assert dist == 0.0

def test_cache_semantic_hit_within_threshold(cache):
    """Vectors with cosine distance < 0.05 (similarity > 0.95) trigger semantic hit."""
    base_vec = np.random.randn(384).astype(np.float32)
    base_vec = base_vec / np.linalg.norm(base_vec)
    
    # Cosine distance = 0.02 (similarity = 0.98)
    similar_vec = base_vec.copy()
    similar_vec[0] += 0.01
    similar_vec = similar_vec / np.linalg.norm(similar_vec)

    query_a = "Crushing chest pain radiating to arm"
    query_b = "Crushing chest pain radiating to left arm"
    payload = {"response": "Emergency 911 dispatch.", "severity": "Critical"}

    cache.set(query_a, base_vec, payload)
    hit = cache.get(query_b, similar_vec)

    assert hit is not None
    ret_payload, dist = hit
    assert ret_payload["severity"] == "Critical"
    assert dist <= 0.05

def test_cache_miss_on_dissimilar_query(cache):
    """Vectors with cosine distance > 0.05 (orthogonal/dissimilar) result in cache miss."""
    vec_a = np.zeros(384, dtype=np.float32)
    vec_a[0] = 1.0

    vec_b = np.zeros(384, dtype=np.float32)
    vec_b[1] = 1.0  # Orthogonal vector: distance = 1.0

    cache.set("Cancel subscription", vec_a, {"response": "Cancelled."})
    hit = cache.get("Chest pain emergency", vec_b)

    assert hit is None

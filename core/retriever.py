"""
Dense Semantic Retrieval Pipeline with FAISS and Cross-Encoder Re-ranking.
Specifications:
- Bi-Encoder: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vectors)
- Index: FAISS IndexFlatIP over L2-normalized vectors (Exact Cosine Similarity)
- Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 scoring top-20 retrieved candidates down to top-3
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from core.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    CROSS_ENCODER_MODEL_NAME,
    RETRIEVAL_TOP_K_CANDIDATES,
    RETRIEVAL_TOP_K_RERANK,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
)

class QueryNormalizer:
    """Preprocesses and normalizes customer queries for optimal semantic vector matching."""
    
    @staticmethod
    def normalize(query: str) -> str:
        """
        Normalizes query text:
        - Decodes Unicode artifacts
        - Standardizes medical / technical acronyms (e.g., '2fa', 'bp', 'hr')
        - Removes excessive whitespace and non-informative symbols
        """
        if not query:
            return ""
        
        cleaned = query.strip()
        # Collapse multiple spaces and line breaks
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        # Expand common high-signal acronyms for retrieval alignment
        replacements = [
            (r"\b2fa\b", "two-factor authentication"),
            (r"\bmfa\b", "multi-factor authentication"),
            (r"\bato\b", "account takeover"),
            (r"\bhr\b(?=\s*\d+)", "heart rate"),
            (r"\bbp\b(?=\s*\d+)", "blood pressure"),
            (r"\bcp\b", "chest pain"),
            (r"\bsob\b", "shortness of breath"),
            (r"\bach\b", "automated clearing house direct debit"),
            (r"\bswift\b", "international wire transfer"),
        ]
        for pattern, repl in replacements:
            cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
            
        return cleaned


class DenseSemanticRetriever:
    """
    Two-stage Dense Retrieval System:
    Stage 1: Dense Vector Retrieval via FAISS IndexFlatIP (top-20)
    Stage 2: Cross-Encoder Reranking via ms-marco-MiniLM-L-6-v2 (top-3)
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        cross_encoder_model_name: str = CROSS_ENCODER_MODEL_NAME,
        dimension: int = EMBEDDING_DIM
    ):
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.dimension = dimension
        
        print(f"[DenseRetriever] Initializing Bi-Encoder: {embedding_model_name}")
        self.bi_encoder = SentenceTransformer(embedding_model_name)
        
        print(f"[DenseRetriever] Initializing Cross-Encoder: {cross_encoder_model_name}")
        try:
            self.cross_encoder = CrossEncoder(cross_encoder_model_name)
        except Exception as e:
            print(f"[DenseRetriever] Warning: Cross-Encoder initialization fallback: {e}")
            self.cross_encoder = None
            
        self.normalizer = QueryNormalizer()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

        if FAISS_INDEX_PATH.exists() and CHUNKS_METADATA_PATH.exists():
            try:
                self.load_index()
            except Exception as e:
                print(f"[DenseRetriever] Deferred loading: {e}")

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds FAISS IndexFlatIP from document chunks with L2 normalization for exact cosine similarity.
        """
        if not chunks:
            raise ValueError("No chunks provided to build index.")
        
        print(f"[DenseRetriever] Embedding {len(chunks)} chunks with {self.embedding_model_name}...")
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate dense embeddings (batch size 64)
        embeddings = self.bi_encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Exact Cosine Similarity via L2 normalization + Inner Product
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks_metadata = chunks
        
        print(f"[DenseRetriever] Successfully indexed {self.index.ntotal} vectors in FAISS IndexFlatIP.")

    def save_index(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = CHUNKS_METADATA_PATH
    ) -> None:
        """Saves FAISS index and metadata store to disk."""
        if self.index is None:
            raise ValueError("No FAISS index to save. Build index first.")
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2)
            
        print(f"[DenseRetriever] FAISS index saved to: {index_path}")
        print(f"[DenseRetriever] Chunks metadata saved to: {metadata_path}")

    def load_index(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = CHUNKS_METADATA_PATH
    ) -> None:
        """Loads FAISS index and chunk metadata from disk."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"FAISS index or metadata missing at {index_path} or {metadata_path}")
        
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.chunks_metadata = json.load(f)
            
        print(f"[DenseRetriever] Loaded FAISS index with {self.index.ntotal} vectors from {index_path}")

    def retrieve(
        self,
        query: str,
        top_k_candidates: int = RETRIEVAL_TOP_K_CANDIDATES,
        top_k_rerank: int = RETRIEVAL_TOP_K_RERANK
    ) -> List[Dict[str, Any]]:
        """
        Executes two-stage retrieval:
        1. Dense retrieval of top-k candidates (default 20) via FAISS IndexFlatIP
        2. Cross-Encoder re-ranking down to top-k (default 3) highest-relevance contexts
        """
        if self.index is None:
            if FAISS_INDEX_PATH.exists() and CHUNKS_METADATA_PATH.exists():
                self.load_index()
            else:
                raise RuntimeError("FAISS index not loaded. Call load_index() or build_index() first.")
        
        normalized_query = self.normalizer.normalize(query)
        
        # 1. Encode normalized query
        query_vector = self.bi_encoder.encode([normalized_query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        # 2. Stage 1: Dense Retrieval (Top 20)
        candidate_k = min(top_k_candidates, self.index.ntotal)
        scores, indices = self.index.search(query_vector, candidate_k)
        
        candidates = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            chunk_data = dict(self.chunks_metadata[idx])
            chunk_data["bi_encoder_score"] = float(score)
            chunk_data["dense_rank"] = rank
            candidates.append(chunk_data)
            
        if not candidates:
            return []

        # 3. Stage 2: Cross-Encoder Re-ranking (Top 20 down to Top 3)
        if self.cross_encoder is not None:
            pairs = [[normalized_query, c["text"]] for c in candidates]
            cross_scores = self.cross_encoder.predict(pairs)
            
            for c, c_score in zip(candidates, cross_scores):
                c["cross_encoder_score"] = float(c_score)
                # Calibrated hybrid score
                c["final_relevance_score"] = float(c_score)
            
            # Sort descending by Cross-Encoder score
            candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        else:
            # Fallback to bi-encoder score
            for c in candidates:
                c["cross_encoder_score"] = c["bi_encoder_score"]
                c["final_relevance_score"] = c["bi_encoder_score"]
            candidates.sort(key=lambda x: x["bi_encoder_score"], reverse=True)

        # Slice down to top_k_rerank (top-3)
        final_top = candidates[:top_k_rerank]
        for final_rank, item in enumerate(final_top, 1):
            item["final_rank"] = final_rank
            
        return final_top

    def format_context_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats top retrieved contexts into an explicit grounded prompt block."""
        if not retrieved_chunks:
            return "No verified protocol guidelines found."
            
        blocks = []
        for chunk in retrieved_chunks:
            blocks.append(
                f"--- [REFERENCE CHUNK: {chunk.get('chunk_id', 'UNKNOWN')} | Protocol: {chunk.get('protocol_id', '')}] ---\n"
                f"Title: {chunk.get('title', '')}\n"
                f"Severity: {chunk.get('severity', '')} | Target SLA: {chunk.get('target_sla_minutes', '')}m\n"
                f"Protocol Details: {chunk.get('text', '')}"
            )
        return "\n\n".join(blocks)

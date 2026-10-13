"""
Token-Aware Recursive Document Chunker.
Splits semi-structured clinical and operational triage guidelines
into semantically coherent chunks of target size (300 tokens) with overlap (50 tokens).
Preserves structural Markdown hierarchy and domain metadata.
"""

import re
from typing import List, Dict, Any, Optional
import tiktoken

from core.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

class RecursiveTokenChunker:
    """
    Token-aware recursive chunker that respects document structure
    (headers, paragraphs, sentences) while guaranteeing chunk token bounds.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Calculates token length using tiktoken with whitespace fallback."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback approximation: 1 token ~ 0.75 words
        return max(1, int(len(text.split()) * 1.33))

    def split_text_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """Recursively breaks text using ordered hierarchy of split points."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        result = []
        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue
            if self.count_tokens(piece) <= self.chunk_size:
                result.append(piece)
            else:
                if remaining_separators:
                    result.extend(self.split_text_by_separators(piece, remaining_separators))
                else:
                    # Final fallback: split tokens directly
                    tokens = self.tokenizer.encode(piece) if self.tokenizer else piece.split()
                    for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                        slice_toks = tokens[i : i + self.chunk_size]
                        if self.tokenizer:
                            result.append(self.tokenizer.decode(slice_toks))
                        else:
                            result.append(" ".join(slice_toks))
        return result

    def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks text with token overlap while propagating document metadata.
        """
        separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        pieces = self.split_text_by_separators(text, separators)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for piece in pieces:
            piece_tokens = self.count_tokens(piece)
            if current_tokens + piece_tokens <= self.chunk_size:
                current_chunk.append(piece)
                current_tokens += piece_tokens
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    chunks.append(chunk_text)
                
                # Overlap retention
                overlap_chunk = []
                overlap_tokens = 0
                for prev_piece in reversed(current_chunk):
                    p_toks = self.count_tokens(prev_piece)
                    if overlap_tokens + p_toks <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_piece)
                        overlap_tokens += p_toks
                    else:
                        break
                
                current_chunk = overlap_chunk + [piece]
                current_tokens = overlap_tokens + piece_tokens
                
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if not chunks or chunks[-1] != chunk_text:
                chunks.append(chunk_text)
        
        # Package chunks with metadata
        output = []
        doc_meta = metadata or {}
        protocol_id = doc_meta.get("protocol_id", "PROTO-UNK")
        
        for idx, c_text in enumerate(chunks, 1):
            chunk_meta = dict(doc_meta)
            chunk_id = f"{protocol_id}-CHUNK-{idx:02d}"
            chunk_meta.update({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "token_count": self.count_tokens(c_text),
                "text": c_text
            })
            output.append(chunk_meta)
            
        return output

    def chunk_knowledge_base(self, kb_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes entire list of structured knowledge base records into indexable chunks."""
        all_chunks = []
        for record in kb_records:
            full_text = (
                f"Protocol ID: {record['protocol_id']}\n"
                f"Title: {record['title']}\n"
                f"Domain: {record.get('domain', '')} - {record.get('subdomain', '')}\n"
                f"Severity: {record.get('severity', 'Medium')}\n"
                f"Target SLA: {record.get('target_sla_minutes', 60)} minutes\n"
                f"Red Flags: {', '.join(record.get('red_flags', []))}\n"
                f"Triage Criteria: {record.get('triage_criteria', '')}\n"
                f"Escalation Action: {record.get('escalation_action', '')}\n"
                f"Guideline Text: {record.get('guideline_text', '')}"
            )
            chunks = self.chunk_document(full_text, metadata=record)
            all_chunks.extend(chunks)
        return all_chunks

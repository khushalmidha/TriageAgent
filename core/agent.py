"""
LiteLLM Orchestration Agent with Structured Pydantic Output Parsing.
Features:
- Sub-millisecond Semantic Cache integration (Cosine Distance < 0.05)
- Deterministic Emergency Escalation Bypass (< 10ms)
- Two-stage FAISS Dense Retrieval + Cross-Encoder Reranking
- LiteLLM multi-provider LLM calling with schema enforcement
- Robust structured fallback handling for offline or network degradation
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field

import litellm
import numpy as np

from core.config import (
    LLM_MODEL,
    SLA_TARGET_MINUTES,
    CRITICAL_SEVERITY_THRESHOLD,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
)
from core.retriever import DenseSemanticRetriever
from core.risk_engine import DeterministicRiskEngine, EmergencyRoutingDecision
from core.cache import SemanticCache

class TriageOutput(BaseModel):
    """Structured Pydantic response schema for triage decisions."""
    query: str = Field(description="The original user query text")
    category: Literal["clinical_symptom", "account_security", "transaction_dispute", "operational_inquiry"] = Field(
        description="Identified triage domain category"
    )
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Assigned severity risk tier"
    )
    target_sla_minutes: int = Field(description="Maximum response time SLA in minutes")
    escalated_to_human: bool = Field(description="Whether issue requires human specialist intervention")
    escalation_path: str = Field(description="Target routing queue or department")
    grounded_response: str = Field(description="Factually grounded response drafting protocol steps")
    citations: List[str] = Field(default_factory=list, description="IDs of reference chunks used for grounding")
    extracted_claims: List[str] = Field(default_factory=list, description="Atomic factual claims for faithfulness audit")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Model confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Clinical or operational chain-of-thought justification")
    cached: bool = Field(default=False, description="True if retrieved from semantic cache")
    latency_ms: float = Field(description="End-to-end processing latency in milliseconds")


class TriageAgent:
    """
    Enterprise Triage RAG Agent.
    Coordinates Semantic Cache, Deterministic Safety Engine, FAISS Retriever,
    and LiteLLM LLM orchestration.
    """

    def __init__(
        self,
        retriever: Optional[DenseSemanticRetriever] = None,
        risk_engine: Optional[DeterministicRiskEngine] = None,
        cache: Optional[SemanticCache] = None,
        model_name: str = LLM_MODEL
    ):
        self.model_name = model_name
        self.risk_engine = risk_engine or DeterministicRiskEngine()
        self.cache = cache or SemanticCache()
        
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = DenseSemanticRetriever()
            if FAISS_INDEX_PATH.exists() and CHUNKS_METADATA_PATH.exists():
                self.retriever.load_index(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
        self.last_retrieved_chunks: List[Dict[str, Any]] = []

    def _determine_category(self, query: str, top_chunks: List[Dict[str, Any]]) -> str:
        """Determines category based on retrieved contexts and query signals."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["chest pain", "headache", "stroke", "bleeding", "asthma", "fever", "cough", "allergic", "seizure", "vomit", "rash"]):
            return "clinical_symptom"
        if any(w in q_lower for w in ["password", "hacked", "2fa", "mfa", "login", "authenticator", "cloud", "aws", "token", "ransomware", "unauthorized device"]):
            return "account_security"
        if any(w in q_lower for w in ["wire", "dispute", "charged", "billing", "refund", "atm", "debit", "crypto", "drainer", "fee", "chargeback"]):
            return "transaction_dispute"
        
        # Check domain of top retrieved chunk
        if top_chunks:
            domain = top_chunks[0].get("domain", "").lower()
            if "clinical" in domain:
                return "clinical_symptom"
            if "security" in domain:
                return "account_security"
            if "dispute" in domain or "payment" in domain:
                return "transaction_dispute"
                
        return "operational_inquiry"

    def _extract_claims(self, text: str) -> List[str]:
        """Extracts atomic factual claims from text using sentence boundary analysis."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 10 and not s.strip().startswith("#")]
        return claims if claims else [text.strip()]

    def triage(self, query: str) -> TriageOutput:
        """
        Executes end-to-end triage pipeline:
        1. Query embedding + Semantic Cache lookup
        2. Deterministic Risk Signals check (<10ms emergency bypass)
        3. Two-stage Dense Retrieval + Cross-Encoder Reranking
        4. Structured LLM Generation via LiteLLM
        5. Semantic Cache storage
        """
        start_time = time.perf_counter()
        self.last_retrieved_chunks = None
        normalized_query = self.retriever.normalizer.normalize(query)
        
        # Compute query embedding for cache and retrieval
        query_embedding = self.retriever.bi_encoder.encode([normalized_query], convert_to_numpy=True)[0]
        
        # --- Stage 1: Semantic Cache Check ---
        cached_result = self.cache.get(normalized_query, query_embedding)
        if cached_result is not None:
            payload, dist = cached_result
            cached_latency = (time.perf_counter() - start_time) * 1000.0
            payload_copy = dict(payload)
            payload_copy["cached"] = True
            payload_copy["latency_ms"] = round(cached_latency, 2)
            return TriageOutput(**payload_copy)

        # --- Stage 2: Deterministic 6-Signal Risk Engine (<10ms) ---
        emergency_decision = self.risk_engine.route_emergency(query)
        if emergency_decision.bypass_llm:
            total_latency = (time.perf_counter() - start_time) * 1000.0
            
            # Grounded emergency response
            grounded_resp = (
                f"EMERGENCY TRIAGE DISPATCH: {emergency_decision.summary_reason}\n"
                f"Mandatory Actions: {' '.join(emergency_decision.immediate_actions)}"
            )
            claims = self._extract_claims(grounded_resp)
            category = self._determine_category(query, [])
            
            output = TriageOutput(
                query=query,
                category=category,
                severity="Critical",
                target_sla_minutes=emergency_decision.target_sla_minutes,
                escalated_to_human=True,
                escalation_path=emergency_decision.escalation_path,
                grounded_response=grounded_resp,
                citations=["CRITICAL-SAFETY-BYPASS-PROTOCOL"],
                extracted_claims=claims,
                confidence_score=0.99,
                reasoning=f"Deterministic emergency trigger: {', '.join(emergency_decision.triggered_signals)}. {emergency_decision.summary_reason}",
                cached=False,
                latency_ms=round(total_latency, 2)
            )
            self.cache.set(normalized_query, query_embedding, output.model_dump())
            return output

        # --- Stage 3: Dense Retrieval & Cross-Encoder Reranking ---
        retrieved_chunks = self.retriever.retrieve(normalized_query)
        self.last_retrieved_chunks = retrieved_chunks
        context_str = self.retriever.format_context_for_prompt(retrieved_chunks)
        category = self._determine_category(query, retrieved_chunks)
        
        top_chunk_severity = retrieved_chunks[0].get("severity", "Medium") if retrieved_chunks else "Medium"
        top_chunk_sla = retrieved_chunks[0].get("target_sla_minutes", 60) if retrieved_chunks else 60
        top_chunk_citations = [c.get("chunk_id", c.get("protocol_id", "UNK")) for c in retrieved_chunks]
        
        # --- Stage 4: LiteLLM Structured Invocation with Fallback ---
        system_prompt = (
            "You are an enterprise triage intelligence system specializing in clinical and operational customer emergencies.\n"
            "Given the user's inquiry and the retrieved protocol context below, produce a JSON object strictly matching this schema:\n"
            "{\n"
            '  "category": "clinical_symptom" | "account_security" | "transaction_dispute" | "operational_inquiry",\n'
            '  "severity": "Low" | "Medium" | "High" | "Critical",\n'
            '  "target_sla_minutes": int,\n'
            '  "escalated_to_human": bool,\n'
            '  "escalation_path": str,\n'
            '  "grounded_response": str,\n'
            '  "citations": [str],\n'
            '  "confidence_score": float,\n'
            '  "reasoning": str\n'
            "}\n"
            "CRITICAL RULES:\n"
            "1. Ground all recommendations strictly on the provided context.\n"
            "2. Citations must list the exact chunk or protocol IDs provided.\n"
            "3. If clinical emergency or severe financial fraud, severity must be Critical or High."
        )
        
        user_prompt = (
            f"USER QUERY: {query}\n\n"
            f"RETRIEVED GUIDELINES:\n{context_str}\n\n"
            "Respond strictly in valid JSON."
        )

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3500,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content or ""
            raw_content = raw_content.strip()
            # Clean markdown codeblocks if present
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()

            json_match = re.search(r"(\{.*\})", raw_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(raw_content)
            
            sev = parsed.get("severity", top_chunk_severity)
            if sev not in ["Low", "Medium", "High", "Critical"]:
                sev = top_chunk_severity
                
            cat = parsed.get("category", category)
            if cat not in ["clinical_symptom", "account_security", "transaction_dispute", "operational_inquiry"]:
                cat = category
                
            sla = parsed.get("target_sla_minutes", SLA_TARGET_MINUTES.get(sev, top_chunk_sla))
            grounded_resp = parsed.get("grounded_response", retrieved_chunks[0].get("escalation_action", ""))
            claims = self._extract_claims(grounded_resp)
            citations = parsed.get("citations", top_chunk_citations)
            
            total_latency = (time.perf_counter() - start_time) * 1000.0
            output = TriageOutput(
                query=query,
                category=cat,
                severity=sev,
                target_sla_minutes=sla,
                escalated_to_human=parsed.get("escalated_to_human", sev in ["Critical", "High"]),
                escalation_path=parsed.get("escalation_path", "SPECIALIZED_TIER_QUEUE"),
                grounded_response=grounded_resp,
                citations=citations if citations else top_chunk_citations,
                extracted_claims=claims,
                confidence_score=float(parsed.get("confidence_score", 0.92)),
                reasoning=parsed.get("reasoning", "Assigned based on grounded clinical/operational guideline evaluation."),
                cached=False,
                latency_ms=round(total_latency, 2)
            )

        except Exception as e:
            # High-fidelity deterministic fallback
            print(f"[TriageAgent] LiteLLM fallback activated ({e}). Executing deterministic RAG synthesis.")
            top_rec = retrieved_chunks[0] if retrieved_chunks else {}
            sev = top_rec.get("severity", top_chunk_severity)
            action = top_rec.get("escalation_action", "Proceed according to standard verification procedures.")
            grounded_resp = f"Guideline Recommendation: {action}"
            claims = self._extract_claims(grounded_resp)
            
            total_latency = (time.perf_counter() - start_time) * 1000.0
            output = TriageOutput(
                query=query,
                category=category,
                severity=sev,
                target_sla_minutes=top_rec.get("target_sla_minutes", SLA_TARGET_MINUTES.get(sev, 60)),
                escalated_to_human=sev in ["Critical", "High"],
                escalation_path=f"TIER_{sev.upper()}_TRIAGE_UNIT",
                grounded_response=grounded_resp,
                citations=top_chunk_citations,
                extracted_claims=claims,
                confidence_score=0.88,
                reasoning=f"Synthesized from top retrieved protocol {top_rec.get('protocol_id', '')} with high semantic relevance.",
                cached=False,
                latency_ms=round(total_latency, 2)
            )

        # Store in semantic cache
        self.cache.set(normalized_query, query_embedding, output.model_dump())
        return output

    async def triage_stream(self, query: str):
        """
        Asynchronously streams triage inference response tokens.
        Yields Server-Sent Event (SSE) formatted data chunks.
        """
        import asyncio
        start_time = time.perf_counter()
        normalized_query = self.retriever.normalizer.normalize(query)
        query_embedding = self.retriever.bi_encoder.encode([normalized_query], convert_to_numpy=True)[0]

        # 1. Check Semantic Cache
        cached_result = self.cache.get(normalized_query, query_embedding)
        if cached_result is not None:
            payload, dist = cached_result
            payload_copy = dict(payload)
            payload_copy["cached"] = True
            payload_copy["latency_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            yield f"data: {json.dumps({'type': 'cache_hit', 'distance': dist})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': payload_copy.get('grounded_response', '')})}\n\n"
            yield f"data: {json.dumps({'type': 'completed', 'output': payload_copy})}\n\n"
            return

        # 2. Check Deterministic 6-Signal Risk Engine (<10ms)
        emergency_decision = self.risk_engine.route_emergency(query)
        if emergency_decision.bypass_llm:
            output = self.triage(query)
            yield f"data: {json.dumps({'type': 'emergency_bypass', 'signals': emergency_decision.triggered_signals})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': output.grounded_response})}\n\n"
            yield f"data: {json.dumps({'type': 'completed', 'output': output.model_dump()})}\n\n"
            return

        # 3. Dense Retrieval
        retrieved_chunks = self.retriever.retrieve(normalized_query)
        top_citations = [c.get("chunk_id", c.get("protocol_id", "UNK")) for c in retrieved_chunks]
        yield f"data: {json.dumps({'type': 'retrieval', 'top_chunks': top_citations})}\n\n"

        # 4. Stream response via LiteLLM acompletion
        context_str = self.retriever.format_context_for_prompt(retrieved_chunks)
        category = self._determine_category(query, retrieved_chunks)
        top_rec = retrieved_chunks[0] if retrieved_chunks else {}
        top_sev = top_rec.get("severity", "Medium")

        try:
            system_prompt = (
                "You are an enterprise triage intelligence system specializing in clinical and operational customer emergencies.\n"
                "Provide a factually grounded, concise, empathetic response based on the retrieved guidelines below.\n"
                "Clearly outline immediate steps and escalation."
            )
            user_prompt = f"USER QUERY: {query}\n\nRETRIEVED GUIDELINES:\n{context_str}"
            
            stream_resp = await litellm.acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                stream=True
            )
            
            accumulated = []
            async for chunk in stream_resp:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    accumulated.append(delta)
                    yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"

            full_text = "".join(accumulated)
            claims = self._extract_claims(full_text)
            total_latency = (time.perf_counter() - start_time) * 1000.0
            
            final_output = TriageOutput(
                query=query,
                category=category,
                severity=top_sev,
                target_sla_minutes=top_rec.get("target_sla_minutes", SLA_TARGET_MINUTES.get(top_sev, 60)),
                escalated_to_human=top_sev in ["Critical", "High"],
                escalation_path=f"TIER_{top_sev.upper()}_TRIAGE_UNIT",
                grounded_response=full_text,
                citations=top_citations,
                extracted_claims=claims,
                confidence_score=0.90,
                reasoning="Streamed RAG response grounded in verified protocols.",
                cached=False,
                latency_ms=round(total_latency, 2)
            )
            self.cache.set(normalized_query, query_embedding, final_output.model_dump())
            yield f"data: {json.dumps({'type': 'completed', 'output': final_output.model_dump()})}\n\n"

        except Exception as e:
            # Fallback output
            fallback_output = self.triage(query)
            yield f"data: {json.dumps({'type': 'token', 'token': fallback_output.grounded_response})}\n\n"
            yield f"data: {json.dumps({'type': 'completed', 'output': fallback_output.model_dump()})}\n\n"


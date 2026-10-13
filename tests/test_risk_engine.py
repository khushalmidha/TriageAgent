"""
Automated unit tests for Deterministic Risk Engine.
Validates the 6 deterministic signals, LLM bypass logic, and <10ms execution SLA.
"""

import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.risk_engine import DeterministicRiskEngine

@pytest.fixture
def risk_engine():
    return DeterministicRiskEngine()

def test_self_harm_signal_trigger(risk_engine):
    query = "I took an overdose of pills 20 minutes ago and want to end my life."
    decision = risk_engine.route_emergency(query)
    
    assert decision.is_emergency is True
    assert decision.severity == "Critical"
    assert decision.bypass_llm is True
    assert "SIGNAL_LIFE_SAFETY_SELF_HARM" in decision.triggered_signals
    assert decision.routing_latency_ms < 10.0

def test_clinical_red_flag_signal_trigger(risk_engine):
    query = "My husband collapsed with crushing chest pain radiating to left arm and jaw."
    decision = risk_engine.route_emergency(query)
    
    assert decision.is_emergency is True
    assert decision.severity == "Critical"
    assert decision.bypass_llm is True
    assert "SIGNAL_ACUTE_CLINICAL_RED_FLAG" in decision.triggered_signals
    assert decision.routing_latency_ms < 10.0

def test_financial_fraud_signal_trigger(risk_engine):
    query = "An unauthorized wire transfer of $85,000 was sent from my account."
    decision = risk_engine.route_emergency(query)
    
    assert decision.is_emergency is True
    assert decision.severity == "Critical"
    assert decision.bypass_llm is True
    assert "SIGNAL_ACTIVE_FINANCIAL_FRAUD" in decision.triggered_signals
    assert decision.routing_latency_ms < 10.0

def test_non_emergency_routine_query(risk_engine):
    query = "How do I update my billing credit card expiration date in settings?"
    decision = risk_engine.route_emergency(query)
    
    assert decision.is_emergency is False
    assert decision.bypass_llm is False
    assert decision.routing_latency_ms < 10.0

def test_latency_sla_under_10ms_stress(risk_engine):
    """Stress tests 100 consecutive queries to guarantee <10ms SLA."""
    query = "I have a sudden facial droop and cannot move my left arm."
    for _ in range(100):
        decision = risk_engine.route_emergency(query)
        assert decision.routing_latency_ms < 10.0, f"Exceeded 10ms SLA: {decision.routing_latency_ms}ms"

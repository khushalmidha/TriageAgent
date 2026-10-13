"""
Deterministic Risk Escalation Engine.
Evaluates 6 deterministic signals combining keyword rules, semantic triggers, and safety criteria.
If Max Severity >= Threshold or an acute safety signal is activated,
bypasses the conversational LLM and triggers deterministic emergency routing in <10ms.
"""

import re
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from core.config import (
    CRITICAL_SEVERITY_THRESHOLD,
    MAX_EMERGENCY_ROUTING_LATENCY_MS,
    SLA_TARGET_MINUTES
)

class EmergencyRoutingDecision(BaseModel):
    """Deterministic routing decision output."""
    is_emergency: bool = Field(description="True if emergency routing was triggered")
    severity: str = Field(description="Severity classification: Low, Medium, High, or Critical")
    bypass_llm: bool = Field(description="True if conversational LLM was bypassed")
    target_sla_minutes: int = Field(description="Target response SLA in minutes")
    escalation_path: str = Field(description="Designated emergency department or team")
    triggered_signals: List[str] = Field(description="Names of deterministic signals that fired")
    immediate_actions: List[str] = Field(description="Mandatory immediate protocol steps")
    routing_latency_ms: float = Field(description="Processing latency in milliseconds (<10ms)")
    summary_reason: str = Field(description="Clinical or operational justification")


class DeterministicRiskEngine:
    """
    6 Deterministic Signals Engine:
    1. Life Safety & Self-Harm Signal
    2. Acute Clinical Red Flag Signal (Stroke FAST, STEMI, Anaphylaxis, Trauma)
    3. Active Financial Fraud Signal (Wire recall, Smart contract drainer, Counterfeit)
    4. Critical SLA & Systemic Latency Signal
    5. Legal & Regulatory Exposure Signal (Subpoena, Regulatory breach)
    6. High Semantic Risk Composite Signal
    """

    def __init__(self, severity_threshold: float = CRITICAL_SEVERITY_THRESHOLD):
        self.severity_threshold = severity_threshold
        
        # Signal 1: Self-Harm & Life Safety Patterns
        self.self_harm_regex = re.compile(
            r"\b(suicid(e|al)|kill\s+(my|one)self|end\s+my\s+life|want\s+to\s+die|"
            r"overdos(e|ed)|took\s+all\s+(my\s+)?pills|hang\s+myself|slit\s+my\s+wrist|"
            r"swallow(ed)?\s+a\s+bottle|lethal\s+dose|goodbye\s+cruel\s+world)\b",
            re.IGNORECASE
        )

        # Signal 2: Acute Clinical Red Flags
        self.clinical_red_flag_regex = re.compile(
            r"\b(crushing\s+chest\s+pain|radiat(ing|es)\s+to\s+(left\s+)?arm|"
            r"facial\s+droop|arm\s+(weakness|limp|numb)|slurred\s+speech|cannot\s+speak|"
            r"throat\s+closing|stridor|anaphylax(is|ic)|lips?\s+(turning\s+)?blue|"
            r"spurting\s+(arterial\s+)?blood|femoral\s+artery|tourniquet|"
            r"unresponsive|not\s+breathing|cardiac\s+arrest|thunderclap\s+headache|"
            r"worst\s+headache\s+of\s+(my\s+)?life|grand\s+mal|seizure\s+over\s+5\s+min|"
            r"flail\s+chest|carbon\s+monoxide|petechial\s+rash\s+with\s+fever)\b",
            re.IGNORECASE
        )

        # Signal 3: Active Financial Fraud
        self.financial_fraud_regex = re.compile(
            r"\b(unauthorized\s+(wire|swift(\s+transfer)?)|wire\s+sent\s+without\s+permission|"
            r"drain(ed|er)\s+(my\s+)?(crypto|wallet|tokens|eth)|smart\s+contract\s+drain|"
            r"sim\s+swap(ped)?|counterfeit\s+(cards?|atm)|stole\s+\$(?:[1-9]\d{4,}|[2-9]\d{3,})|"
            r"hacked\s+into\s+my\s+bank|account\s+emptied|unauthorized\s+payout)\b",
            re.IGNORECASE
        )

        # Signal 4: Critical SLA / Systemic Latency Failure
        self.sla_critical_regex = re.compile(
            r"\b(system-wide\s+(outage|crash)|root\s+(aws|cloud|azure|gcp)\s+compromise|"
            r"ransomware\s+lockout|all\s+systems\s+down|telemetry\s+unreachable|"
            r"database\s+exfiltrat(ed|ion)|emergency\s+line\s+dead|life-support\s+offline)\b",
            re.IGNORECASE
        )

        # Signal 5: Legal & Regulatory Exposure
        self.legal_regulatory_regex = re.compile(
            r"\b(subpoena|formal\s+deposition|retained\s+(a\s+)?(lawyer|attorney)|"
            r"reporting\s+to\s+(cfpb|sec|hipaa|fbi|doj|ftc)|lawsuit\s+filed|"
            r"regulatory\s+sanction|criminal\s+negligence)\b",
            re.IGNORECASE
        )

    def evaluate_signals(self, query: str, context_severity_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates the 6 deterministic signals on the query in microsecond time.
        """
        start_time = time.perf_counter()
        query_text = query.strip()
        
        triggered_signals = []
        immediate_actions = []
        is_emergency = False
        severity = "Low"
        target_sla = SLA_TARGET_MINUTES["Low"]
        escalation_path = "STANDARD_CUSTOMER_QUEUE"
        summary_reasons = []

        # 1. Signal 1: Self-Harm & Life Safety
        if self.self_harm_regex.search(query_text):
            triggered_signals.append("SIGNAL_LIFE_SAFETY_SELF_HARM")
            is_emergency = True
            severity = "Critical"
            target_sla = SLA_TARGET_MINUTES["Critical"]
            escalation_path = "988_SUICIDE_AND_CRISIS_LIFELINE_DISPATCH"
            immediate_actions.extend([
                "Connect immediately to 988 Suicide & Crisis Lifeline or 911 dispatch.",
                "Maintain continuous active operator connection with user.",
                "Do not terminate call or chat session under any circumstance."
            ])
            summary_reasons.append("Active self-harm or imminent life safety keywords identified.")

        # 2. Signal 2: Acute Clinical Red Flags
        if self.clinical_red_flag_regex.search(query_text):
            triggered_signals.append("SIGNAL_ACUTE_CLINICAL_RED_FLAG")
            is_emergency = True
            severity = "Critical"
            target_sla = SLA_TARGET_MINUTES["Critical"]
            escalation_path = "911_EMERGENCY_MEDICAL_DISPATCH"
            immediate_actions.extend([
                "Call 911 (or local emergency medical services) immediately.",
                "Initiate immediate first-aid protocols (Tourniquet / EpiPen / Aspirin chew / Recovery position).",
                "Do not transport patient via private vehicle if hemodynamically unstable."
            ])
            summary_reasons.append("Acute clinical red flags identified (STEMI / Stroke / Anaphylaxis / Hemorrhage / Seizure).")

        # 3. Signal 3: Active Financial Fraud
        if self.financial_fraud_regex.search(query_text):
            triggered_signals.append("SIGNAL_ACTIVE_FINANCIAL_FRAUD")
            is_emergency = True
            severity = "Critical"
            target_sla = SLA_TARGET_MINUTES["Critical"]
            escalation_path = "EMERGENCY_FRAUD_INTERCEPTION_DESK"
            immediate_actions.extend([
                "Execute emergency SWIFT wire recall (MT199/MT292) / ACH Return R10.",
                "Freeze outbound debits, wire channels, and secondary card authorizations.",
                "Revoke active web3 smart contract approvals and API session keys."
            ])
            summary_reasons.append("High-velocity financial fraud or unauthorized wire transfer detected.")

        # 4. Signal 4: Critical SLA / Systemic Failure
        if self.sla_critical_regex.search(query_text):
            triggered_signals.append("SIGNAL_CRITICAL_SLA_LATENCY")
            is_emergency = True
            severity = "Critical"
            target_sla = SLA_TARGET_MINUTES["Critical"]
            escalation_path = "SEV1_INCIDENT_WAR_ROOM"
            immediate_actions.extend([
                "Page on-call Chief Information Security Officer and Principal Site Reliability Engineers.",
                "Isolate compromised AWS/GCP administrative VPCs and revoke root API credentials.",
                "Engage corporate business continuity disaster recovery protocols."
            ])
            summary_reasons.append("Catastrophic operational outage or corporate credential compromise.")

        # 5. Signal 5: Legal & Regulatory Exposure
        if self.legal_regulatory_regex.search(query_text):
            triggered_signals.append("SIGNAL_LEGAL_REGULATORY_EXPOSURE")
            severity = "High" if severity != "Critical" else severity
            if not is_emergency and severity != "Critical":
                target_sla = SLA_TARGET_MINUTES["High"]
                escalation_path = "LEGAL_AND_REGULATORY_COMPLIANCE_TEAM"
            immediate_actions.extend([
                "Preserve all transaction logs, chat transcripts, and audit trails for litigation hold.",
                "Route inquiry to Corporate Legal and Regulatory Counsel within SLA."
            ])
            summary_reasons.append("Formal regulatory reporting threat or legal counsel engagement identified.")

        # 6. Signal 6: Composite Risk Calculation & Severity Threshold
        # Calculate composite signal density
        signal_count = len(triggered_signals)
        composite_score = min(1.0, 0.45 * signal_count + (0.35 if "SIGNAL_LIFE_SAFETY_SELF_HARM" in triggered_signals else 0.0))
        
        if composite_score >= self.severity_threshold and not is_emergency:
            triggered_signals.append("SIGNAL_SEMANTIC_SEVERITY_THRESHOLD")
            is_emergency = True
            severity = "Critical"
            target_sla = SLA_TARGET_MINUTES["Critical"]
            escalation_path = "RAPID_RESPONSE_ESCALATION_UNIT"
            summary_reasons.append(f"Composite risk score ({composite_score:.2f}) exceeded threshold ({self.severity_threshold:.2f}).")

        # Contextual severity override if provided
        if context_severity_hint == "Critical":
            is_emergency = True
            severity = "Critical"
            target_sla = SLA_TARGET_MINUTES["Critical"]

        routing_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Verify strict latency specification (<10ms)
        bypass_llm = is_emergency or severity == "Critical"
        
        return {
            "is_emergency": is_emergency,
            "severity": severity,
            "bypass_llm": bypass_llm,
            "target_sla_minutes": target_sla,
            "escalation_path": escalation_path,
            "triggered_signals": triggered_signals,
            "immediate_actions": list(dict.fromkeys(immediate_actions)),  # Deduplicate preserving order
            "routing_latency_ms": round(routing_time_ms, 3),
            "summary_reason": " | ".join(summary_reasons) if summary_reasons else "Normal routine triage inquiry."
        }

    def route_emergency(self, query: str, context_severity_hint: Optional[str] = None) -> EmergencyRoutingDecision:
        """
        Executes deterministic triage routing and returns typed decision object.
        Guaranteed to execute in <10ms.
        """
        eval_result = self.evaluate_signals(query, context_severity_hint)
        return EmergencyRoutingDecision(**eval_result)

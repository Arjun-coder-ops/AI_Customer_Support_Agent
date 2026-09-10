import pytest
from src.escalation.policy import EscalationPolicyEngine

def test_escalation_sensitive_request():
    engine = EscalationPolicyEngine()
    res = engine.evaluate(
        customer_message="I am hiring a lawyer to file a lawsuit!",
        predicted_intent="general_inquiry_feedback",
        intent_confidence=0.95,
        evidence_cases=[{"case_id": "c1", "similarity": 0.80}],
    )
    assert res["decision"] == "ESCALATE"
    assert res["reason_code"] == "SENSITIVE_REQUEST"

def test_escalation_low_confidence():
    engine = EscalationPolicyEngine(min_intent_confidence=0.70)
    res = engine.evaluate(
        customer_message="Help me with my order",
        predicted_intent="shipping_delay",
        intent_confidence=0.45,
        evidence_cases=[{"case_id": "c1", "similarity": 0.80}],
    )
    assert res["decision"] == "ESCALATE"
    assert res["reason_code"] == "LOW_INTENT_CONFIDENCE"

def test_escalation_safe_auto_handle():
    engine = EscalationPolicyEngine(min_intent_confidence=0.70, min_retrieval_similarity=0.65)
    res = engine.evaluate(
        customer_message="Where is my delayed package #12345?",
        predicted_intent="shipping_delay",
        intent_confidence=0.90,
        evidence_cases=[{"case_id": "c1", "similarity": 0.85}],
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert res["reason_code"] is None

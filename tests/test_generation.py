import pytest
from src.generation.generator import GroundedReplyGenerator

def test_mock_generation():
    generator = GroundedReplyGenerator(use_mock=True)
    res = generator.generate(
        customer_message="Where is my order?",
        predicted_intent="shipping_delay",
        intent_confidence=0.90,
        evidence_cases=[{"case_id": "c1", "historical_response": "Tracking updated."}],
        should_escalate=False,
    )
    assert res.reply is not None
    assert res.should_escalate is False
    assert "c1" in res.evidence_ids
    assert res.reply.startswith("[MOCK]")

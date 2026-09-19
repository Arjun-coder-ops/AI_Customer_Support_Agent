import os

import pytest

from evaluation.failure_analysis import classify_case, collect_real_case_categories
from src.escalation.policy import EscalationPolicyEngine
from src.generation.generator import GroundedReplyGenerator
from src.generation.schemas import SupportResponseSchema
from src.pipeline import SupportAgentPipeline
from src.retrieval.index import HistoricalCaseRetrievalIndex


def test_escalation_reason_priority_insufficient_before_confidence():
    """Short messages escalate as INSUFFICIENT_CONTEXT even if confidence is low."""
    engine = EscalationPolicyEngine(min_intent_confidence=0.70)
    res = engine.evaluate(
        customer_message="help me",
        predicted_intent="general_inquiry_feedback",
        intent_confidence=0.10,
        evidence_cases=[],
    )
    assert res["decision"] == "ESCALATE"
    assert res["reason_code"] == "INSUFFICIENT_CONTEXT"


def test_escalation_account_action_before_retrieval():
    engine = EscalationPolicyEngine()
    res = engine.evaluate(
        customer_message="Please help me change address on my order tonight",
        predicted_intent="general_inquiry_feedback",
        intent_confidence=0.99,
        evidence_cases=[{"case_id": "c1", "similarity": 0.99}],
    )
    assert res["reason_code"] == "ACCOUNT_SPECIFIC_ACTION_REQUIRED"


def test_failure_analysis_prefers_escalation_reason_over_similarity():
    """
    A case with low similarity AND LOW_INTENT_CONFIDENCE must not be
    labelled LOW_RETRIEVAL_SIMILARITY.
    """
    item = {
        "customer_message": "Please cancel my wrong order request still pending",
        "intent": {"name": "order_cancellation", "confidence": 0.55},
        "retrieval": {"top_similarity": 0.20},
        "escalation": {
            "decision": "ESCALATE",
            "reason_code": "LOW_INTENT_CONFIDENCE",
        },
        "generated_reply": {"reply": "escalated"},
        "evaluation": {
            "conversation_id": "conv_test",
            "reference_intent": "order_cancellation",
            "reference_label_type": "HEURISTIC_NOT_HUMAN",
        },
    }
    cat, kind = classify_case(item)
    assert cat == "LOW_INTENT_CONFIDENCE_ESCALATION"
    assert kind == "expected_safety_escalation"


def test_failure_analysis_insufficient_context_not_low_retrieval():
    item = {
        "customer_message": "Amazon",
        "intent": {"name": "general_inquiry_feedback", "confidence": 0.99},
        "retrieval": {"top_similarity": 0.0},
        "escalation": {
            "decision": "ESCALATE",
            "reason_code": "INSUFFICIENT_CONTEXT",
        },
        "generated_reply": {"reply": "escalated"},
        "evaluation": {
            "conversation_id": "conv_short",
            "reference_intent": "general_inquiry_feedback",
            "reference_label_type": "HEURISTIC_NOT_HUMAN",
        },
    }
    cat, _ = classify_case(item)
    assert cat == "INSUFFICIENT_CONTEXT_ESCALATION"


def test_failure_analysis_intent_mismatch_highest_priority():
    item = {
        "customer_message": "My package is late and tracking stuck",
        "intent": {"name": "general_inquiry_feedback", "confidence": 0.40},
        "retrieval": {"top_similarity": 0.10},
        "escalation": {
            "decision": "ESCALATE",
            "reason_code": "LOW_INTENT_CONFIDENCE",
        },
        "generated_reply": {"reply": "escalated"},
        "evaluation": {
            "conversation_id": "conv_mm",
            "reference_intent": "shipping_delay",
            "reference_label_type": "HEURISTIC_NOT_HUMAN",
        },
    }
    cat, kind = classify_case(item)
    assert cat == "INTENT_MISMATCH_VS_HEURISTIC"
    assert kind == "diagnostic_disagreement"


def test_retrieval_index_excludes_held_out_ids():
    """Train-only indexing: test conversation IDs must not appear in the index."""
    train_cases = [
        {
            "case_id": "train_1",
            "conversation_id": "train_1",
            "customer_message": "Where is my delayed package?",
            "historical_response": "Tracking updated.",
            "turn_count": 2,
        },
        {
            "case_id": "train_2",
            "conversation_id": "train_2",
            "customer_message": "Missing item from box",
            "historical_response": "Replacement issued.",
            "turn_count": 2,
        },
    ]
    test_ids = {"test_9", "test_8"}

    index = HistoricalCaseRetrievalIndex(top_k=2)
    index.build_index(train_cases)
    indexed = {c.get("conversation_id") or c.get("case_id") for c in index.cases}
    assert indexed.isdisjoint(test_ids)
    assert "train_1" in indexed


def test_pipeline_generated_reply_uses_model_dump_schema():
    pipeline = SupportAgentPipeline(brand_name="AmazonHelp", use_mock=True)
    train_cases = [
        {"customer_initial_message": "Where is my delayed package?", "brand": "AmazonHelp"},
        {"customer_initial_message": "Missing item from box", "brand": "AmazonHelp"},
    ]
    resolved_cases = [
        {
            "case_id": "c1",
            "conversation_id": "c1",
            "customer_message": "Where is my delayed package?",
            "historical_response": "Tracking updated.",
            "turn_count": 2,
        },
    ]
    pipeline.train_and_index(train_cases, resolved_cases)
    res = pipeline.process_message("Where is my delayed package order please?")
    assert isinstance(res["generated_reply"], dict)
    assert "reply" in res["generated_reply"]
    assert "should_escalate" in res["generated_reply"]
    # Validate against schema
    SupportResponseSchema(**res["generated_reply"])


def test_generator_mock_mode_deterministic_and_marked():
    gen = GroundedReplyGenerator(use_mock=True)
    assert gen.use_mock is True
    a = gen.generate(
        customer_message="Where is my order?",
        predicted_intent="shipping_delay",
        intent_confidence=0.9,
        evidence_cases=[{"case_id": "c1", "historical_response": "Tracking updated."}],
        should_escalate=False,
    )
    b = gen.generate(
        customer_message="Where is my order?",
        predicted_intent="shipping_delay",
        intent_confidence=0.9,
        evidence_cases=[{"case_id": "c1", "historical_response": "Tracking updated."}],
        should_escalate=False,
    )
    assert a.reply == b.reply
    assert a.reply.startswith("[MOCK]")
    assert a.should_escalate is False


def test_generator_escalation_respects_policy_reason():
    gen = GroundedReplyGenerator(use_mock=True)
    res = gen.generate(
        customer_message="help",
        predicted_intent="general_inquiry_feedback",
        intent_confidence=0.2,
        evidence_cases=[],
        should_escalate=True,
        escalation_reason_code="INSUFFICIENT_CONTEXT",
    )
    assert res.should_escalate is True
    assert res.escalation_reason == "INSUFFICIENT_CONTEXT"
    assert "[MOCK]" in res.reply


def test_generator_gemini_mode_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MOCK_LLM", "false")
    gen = GroundedReplyGenerator(use_mock=None)
    # Without key, constructor should fall back to mock for safety
    assert gen.use_mock is True


def test_generator_explicit_non_mock_without_key_raises_on_llm_path(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gen = GroundedReplyGenerator(use_mock=False)
    assert gen.use_mock is False
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        gen._generate_llm(
            customer_message="hi there friend",
            predicted_intent="general_inquiry_feedback",
            intent_confidence=0.9,
            evidence_cases=[],
            should_escalate=False,
            escalation_reason_code=None,
        )


@pytest.mark.skipif(
    not os.path.exists("results/pipeline_test_outputs.jsonl"),
    reason="pipeline outputs not present",
)
def test_failure_analysis_on_real_outputs_has_priority_categories():
    collected = collect_real_case_categories("results/pipeline_test_outputs.jsonl")
    assert collected["pipeline_outputs_exist"] is True
    assert collected["total_cases_scanned"] > 0
    # If low-confidence escalations exist, they must be counted under that category
    # rather than being swallowed entirely by LOW_RETRIEVAL_SIMILARITY.
    reasons = collected.get("reason_code_counts", {})
    if reasons.get("LOW_INTENT_CONFIDENCE", 0) > 0:
        assert collected["category_counts"].get("LOW_INTENT_CONFIDENCE_ESCALATION", 0) > 0

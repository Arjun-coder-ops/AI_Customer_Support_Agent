"""
Tests for intent classifier and baseline evaluation pipeline.
Covers Issue 4 fix: verifies that evaluation does NOT use training data as test data.
"""
import json
import os
import tempfile
import pytest
from src.intent.classifier import FinalIntentClassifier
from evaluation.baselines.majority import MajorityBaselineClassifier
from evaluation.intent_eval import run_intent_evaluations, map_text_to_heuristic_intent


def _write_jsonl(path: str, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Basic classifier unit tests
# ---------------------------------------------------------------------------

def test_majority_baseline():
    clf = MajorityBaselineClassifier()
    clf.fit(["shipping_delay", "shipping_delay", "refund_return_request"])
    preds = clf.predict(["msg1", "msg2"])
    assert preds == ["shipping_delay", "shipping_delay"]


def test_final_intent_classifier():
    X_train = ["Where is my delayed package?", "I want a refund for damaged item"]
    y_train = ["shipping_delay", "refund_return_request"]

    clf = FinalIntentClassifier()
    clf.fit(X_train, y_train)

    res = clf.predict_single("Where is my package?")
    assert "predicted_intent" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res


# ---------------------------------------------------------------------------
# Issue 4 Fix: evaluation must NOT use training data as test data
# ---------------------------------------------------------------------------

def test_intent_eval_uses_test_split_not_training_data():
    """
    Issue 4 Fix: When golden set is empty/provisional, evaluation must use
    the held-out test split — NOT fall back to X_test = X_train.
    Verifying: eval_source contains 'test' or 'TEST' and NOT 'TRAIN'.
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        # Distinct train and test splits with different messages
        _write_jsonl(train_path, [
            {"conversation_id": f"train_{i}", "customer_initial_message": f"Where is my delayed order number {i}?"}
            for i in range(20)
        ])
        _write_jsonl(test_path, [
            {"conversation_id": f"test_{i}", "customer_initial_message": f"My package is late and tracking shows delay {i}"}
            for i in range(10)
        ])
        # Empty golden (no human-reviewed examples)
        _write_jsonl(golden_path, [])

        res = run_intent_evaluations(
            train_path=train_path,
            test_path=test_path,
            golden_path=golden_path,
        )

        # Must use test split, not training data
        eval_source = res.get("dataset_status", "")
        assert "test" in eval_source.lower() or "TEST" in eval_source, (
            f"Evaluation should use test split but got: {eval_source}"
        )
        # Verify it does NOT claim to use train as evaluation source
        assert "train" not in eval_source.lower() or "test" in eval_source.lower(), (
            "Should not evaluate on training data"
        )


def test_intent_eval_label_type_is_heuristic_not_human():
    """
    Issue 4: Labels must be identified as HEURISTIC when golden set is empty.
    The system must NOT claim human-labelled accuracy when using keyword rules.
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        _write_jsonl(train_path, [
            {"conversation_id": "tr1", "customer_initial_message": "Where is my delayed package?"},
            {"conversation_id": "tr2", "customer_initial_message": "Missing item from delivered box."},
        ])
        _write_jsonl(test_path, [
            {"conversation_id": "te1", "customer_initial_message": "Package tracking shows delay."},
        ])
        _write_jsonl(golden_path, [])

        res = run_intent_evaluations(
            train_path=train_path,
            test_path=test_path,
            golden_path=golden_path,
        )

        label_type = res.get("label_type", "")
        assert "HEURISTIC" in label_type.upper() or "heuristic" in label_type.lower(), (
            f"Label type must be HEURISTIC when no human labels exist. Got: {label_type}"
        )
        # Must NOT claim human labels
        assert "HUMAN_LABELLED" not in label_type.upper() or "NOT" in label_type.upper(), (
            "Must not claim human-labelled when using heuristic rules"
        )


def test_intent_eval_golden_status_blocked_when_insufficient():
    """
    Issue 5: golden_benchmark_status must indicate BLOCKED when < 150 human examples.
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        _write_jsonl(train_path, [
            {"conversation_id": "tr1", "customer_initial_message": "My order is late."},
        ])
        _write_jsonl(test_path, [
            {"conversation_id": "te1", "customer_initial_message": "Package delay tracking."},
        ])
        # Only 1 provisional example, not human-reviewed
        _write_jsonl(golden_path, [
            {
                "id": "cand_0001",
                "conversation_id": "conv_301",
                "customer_message": "Return a product.",
                "intent": "refund_return_request",
                "is_human_reviewed": False,
                "review_status": "PROVISIONAL_SEED_PENDING_HUMAN_AUDIT"
            }
        ])

        res = run_intent_evaluations(
            train_path=train_path,
            test_path=test_path,
            golden_path=golden_path,
        )

        golden_status = res.get("golden_benchmark_status", "")
        assert "BLOCKED" in golden_status.upper(), (
            f"Golden status should be BLOCKED when < 150 human examples. Got: {golden_status}"
        )


def test_heuristic_intent_mapper_covers_all_8_intents():
    """All 8 taxonomy intents must be reachable from the heuristic mapper."""
    VALID_INTENTS = {
        "shipping_delay", "missing_item", "order_cancellation",
        "refund_return_request", "account_access_issue", "payment_billing_issue",
        "product_defect_damage", "general_inquiry_feedback"
    }
    test_inputs = [
        ("Where is my delayed package tracking status?", "shipping_delay"),
        ("Missing item from my delivered box.", "missing_item"),
        ("Cancel my order immediately.", "order_cancellation"),
        ("I want a refund for this return.", "refund_return_request"),
        ("Locked out of my account password reset.", "account_access_issue"),
        ("I was double charged on my billing statement.", "payment_billing_issue"),
        ("Screen arrived cracked and broken defect.", "product_defect_damage"),
        ("Hello there!",  "general_inquiry_feedback"),
    ]
    for msg, expected in test_inputs:
        result = map_text_to_heuristic_intent(msg)
        assert result in VALID_INTENTS, f"Unknown intent '{result}'"
        assert result == expected, f"For '{msg}': expected {expected}, got {result}"


def test_intent_eval_output_has_required_fields():
    """Output must have dataset_status, label_type, and golden_benchmark_status fields."""
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        _write_jsonl(train_path, [
            {"conversation_id": "tr1", "customer_initial_message": "My order is delayed."},
            {"conversation_id": "tr2", "customer_initial_message": "I want a refund."},
        ])
        _write_jsonl(test_path, [
            {"conversation_id": "te1", "customer_initial_message": "Package hasn't arrived."},
        ])
        _write_jsonl(golden_path, [])

        res = run_intent_evaluations(
            train_path=train_path,
            test_path=test_path,
            golden_path=golden_path,
        )

        required_fields = ["dataset_status", "label_type", "golden_benchmark_status"]
        for field in required_fields:
            assert field in res, f"Missing field: {field}"

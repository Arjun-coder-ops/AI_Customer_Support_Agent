"""
Tests for evaluation/retrieval_eval.py

Covers Issue 1 (train-only corpus) and Issue 2 (metric naming) fixes.
"""
import json
import os
import tempfile
import pytest
from evaluation.retrieval_eval import run_retrieval_evaluation, _load_train_resolved_cases, map_intent_heuristic


def _write_jsonl(path: str, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Issue 2: metric naming
# ---------------------------------------------------------------------------

def test_retrieval_output_uses_intent_match_not_recall():
    """
    Issue 2 Fix: The output must use 'intent_match_at_k' keys, NOT 'recall_at_k'.
    This verifies the metric is not mislabelled as Recall@K.
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")

        _write_jsonl(train_path, [
            {
                "conversation_id": f"train_conv_{i}",
                "customer_initial_message": f"Where is my delayed package #{i}?",
                "support_final_response": "We updated tracking.",
                "is_resolved": True,
                "turn_count": 2,
                "brand": "AmazonHelp",
            }
            for i in range(10)
        ])
        _write_jsonl(test_path, [
            {
                "conversation_id": f"test_conv_{i}",
                "customer_initial_message": "Package hasn't arrived and tracking is delayed",
                "support_final_response": "",
                "is_resolved": False,
                "turn_count": 1,
                "brand": "AmazonHelp",
            }
            for i in range(5)
        ])

        res = run_retrieval_evaluation(train_path=train_path, test_path=test_path)

        # Must have intent_match keys
        assert "intent_match_at_1" in res, "Must use 'intent_match_at_1' not 'recall_at_1'"
        assert "intent_match_at_3" in res, "Must use 'intent_match_at_3' not 'recall_at_3'"
        assert "intent_match_at_5" in res, "Must use 'intent_match_at_5' not 'recall_at_5'"

        # Must NOT have recall keys (banned — these would be mislabelled)
        assert "recall_at_1" not in res, "Do NOT use 'recall_at_1' — metric is IntentMatch not Recall"
        assert "recall_at_3" not in res
        assert "recall_at_5" not in res


# ---------------------------------------------------------------------------
# Issue 1: train-only corpus, no test leakage
# ---------------------------------------------------------------------------

def test_retrieval_corpus_contains_only_train_conversations():
    """
    Issue 1 Fix: The retrieval corpus must contain ONLY training conversations.
    No test conversation IDs should appear in the corpus.
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")

        train_conv_ids = {f"train_conv_{i}" for i in range(20)}
        test_conv_ids = {f"test_conv_{i}" for i in range(5)}

        _write_jsonl(train_path, [
            {
                "conversation_id": cid,
                "customer_initial_message": "My delayed package hasn't arrived.",
                "support_final_response": "Tracking updated.",
                "is_resolved": True,
                "turn_count": 2,
                "brand": "AmazonHelp",
            }
            for cid in train_conv_ids
        ])
        _write_jsonl(test_path, [
            {
                "conversation_id": cid,
                "customer_initial_message": "Where is my package?",
                "support_final_response": "",
                "is_resolved": False,
                "turn_count": 1,
                "brand": "AmazonHelp",
            }
            for cid in test_conv_ids
        ])

        res = run_retrieval_evaluation(train_path=train_path, test_path=test_path)

        # Verify zero overlap between retrieval corpus and test queries
        assert res["train_test_conversation_overlap"] == 0, (
            f"Retrieval corpus must not contain test conversations. "
            f"Overlap: {res['train_test_conversation_overlap']}"
        )

        # Verify corpus is built from train, not all conversations
        assert res["index_corpus_resolved_case_count"] > 0
        assert res["index_corpus_resolved_case_count"] <= 20  # only train cases


def test_retrieval_output_has_leakage_status():
    """Retrieval output must include a leakage_status field."""
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")

        _write_jsonl(train_path, [
            {
                "conversation_id": "train_1",
                "customer_initial_message": "Where is my order?",
                "support_final_response": "It's on the way.",
                "is_resolved": True,
                "turn_count": 2,
                "brand": "AmazonHelp",
            }
        ])
        _write_jsonl(test_path, [
            {
                "conversation_id": "test_1",
                "customer_initial_message": "My parcel is delayed.",
                "is_resolved": False,
                "turn_count": 1,
                "brand": "AmazonHelp",
            }
        ])

        res = run_retrieval_evaluation(train_path=train_path, test_path=test_path)
        assert "leakage_status" in res
        assert "PASS" in res["leakage_status"]


def test_retrieval_output_includes_metric_naming_note():
    """Retrieval output must explain why IntentMatch is used instead of Recall."""
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        _write_jsonl(train_path, [])
        _write_jsonl(test_path, [])

        res = run_retrieval_evaluation(train_path=train_path, test_path=test_path)
        assert "metric_naming_note" in res


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_load_train_resolved_cases_only_resolved():
    """_load_train_resolved_cases must exclude unresolved conversations."""
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        _write_jsonl(train_path, [
            {"conversation_id": "c1", "customer_initial_message": "My order is late.", "support_final_response": "We'll investigate.", "is_resolved": True, "turn_count": 2},
            {"conversation_id": "c2", "customer_initial_message": "Hi there.", "support_final_response": "", "is_resolved": False, "turn_count": 1},
        ])
        cases = _load_train_resolved_cases(train_path)
        # Only resolved case should be included
        assert len(cases) == 1
        assert cases[0]["conversation_id"] == "c1"


def test_heuristic_intent_mapper():
    """Intent mapper should return known labels."""
    VALID_INTENTS = {
        "shipping_delay", "missing_item", "order_cancellation",
        "refund_return_request", "account_access_issue", "payment_billing_issue",
        "product_defect_damage", "general_inquiry_feedback"
    }
    test_messages = [
        "Where is my delayed package tracking status?",
        "Missing item from my delivered box.",
        "Cancel my order please.",
        "I want a refund and return.",
        "Locked out of my account password.",
        "I was double charged on billing.",
        "Screen arrived cracked and broken.",
        "Hello there!",
    ]
    for msg in test_messages:
        intent = map_intent_heuristic(msg)
        assert intent in VALID_INTENTS, f"Unknown intent '{intent}' for '{msg}'"

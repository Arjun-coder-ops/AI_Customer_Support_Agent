"""
Tests for evaluation/leakage_check.py

Covers Issue 3 fix: verifies that the leakage checker correctly:
1. Reports PASS on conversation-ID overlap (primary leakage check)
2. Correctly distinguishes conversation-level leakage from
   duplicate-text contamination (informational)
3. Passes even when duplicate messages exist across independent conversations
"""
import json
import os
import tempfile
import pytest
from evaluation.leakage_check import check_split_leakage


def _write_jsonl(path: str, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_split_leakage_check_real_data():
    """Integration test against real data files."""
    res = check_split_leakage()
    assert isinstance(res, dict)
    assert "conversation_id_leakage" in res
    assert "has_leakage" in res
    assert res["train_val_id_overlap"] == 0
    assert res["train_test_id_overlap"] == 0
    assert res["train_golden_id_overlap"] == 0
    # Conversation IDs must not overlap — this is the real leakage check
    assert res["conversation_id_leakage"] == False, (
        f"Conversation-level leakage detected: {res}"
    )


def test_leakage_check_no_leakage_no_duplicates():
    """PASS: no conversation overlap, no message overlap."""
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        val_path = os.path.join(d, "val.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        _write_jsonl(train_path, [
            {"conversation_id": "conv_1", "customer_initial_message": "Where is my package?"},
        ])
        _write_jsonl(test_path, [
            {"conversation_id": "conv_2", "customer_initial_message": "My item is missing."},
        ])
        _write_jsonl(val_path, [
            {"conversation_id": "conv_3", "customer_initial_message": "I want a refund."},
        ])
        _write_jsonl(golden_path, [])

        res = check_split_leakage(train_path, val_path, test_path, golden_path)
        assert res["conversation_id_leakage"] == False
        assert res["has_leakage"] == False
        assert res["train_test_id_overlap"] == 0
        assert res["train_test_exact_message_overlap_count"] == 0


def test_leakage_check_conversation_id_leakage():
    """FAIL: same conversation_id in train and test — genuine leakage."""
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        val_path = os.path.join(d, "val.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        _write_jsonl(train_path, [
            {"conversation_id": "conv_100", "customer_initial_message": "Where is my order?"},
        ])
        _write_jsonl(test_path, [
            # Same conversation_id — genuine leakage!
            {"conversation_id": "conv_100", "customer_initial_message": "Where is my order?"},
        ])
        _write_jsonl(val_path, [])
        _write_jsonl(golden_path, [])

        res = check_split_leakage(train_path, val_path, test_path, golden_path)
        assert res["conversation_id_leakage"] == True
        assert res["has_leakage"] == True
        assert res["train_test_id_overlap"] == 1


def test_leakage_check_duplicate_text_independent_conversations():
    """
    PASS (conversation level): duplicate message text across DIFFERENT conversation IDs.
    This is the real-data case: "@amazonhelp ok" appears in multiple independent conversations.
    The checker should:
    - Report conversation_id_leakage = False (PASS)
    - Report duplicate_text_contamination_risk = PRESENT (informational)
    - NOT report has_leakage = True
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, "train.jsonl")
        test_path = os.path.join(d, "test.jsonl")
        val_path = os.path.join(d, "val.jsonl")
        golden_path = os.path.join(d, "golden.jsonl")

        _write_jsonl(train_path, [
            # Independent conversation with generic message
            {"conversation_id": "conv_10", "customer_initial_message": "@amazonhelp ok"},
        ])
        _write_jsonl(test_path, [
            # Different conversation_id, same trivially generic message
            {"conversation_id": "conv_20", "customer_initial_message": "@amazonhelp ok"},
        ])
        _write_jsonl(val_path, [])
        _write_jsonl(golden_path, [])

        res = check_split_leakage(train_path, val_path, test_path, golden_path)

        # Primary check: no conversation-level leakage
        assert res["conversation_id_leakage"] == False, (
            "Independent conversations with duplicate text should NOT trigger conversation leakage"
        )
        assert res["has_leakage"] == False
        assert res["train_test_id_overlap"] == 0

        # Informational: contamination risk should be flagged
        assert res["train_test_exact_message_overlap_count"] == 1
        assert res["duplicate_text_contamination_risk"] == "PRESENT"
        assert res["duplicate_message_assessment"] == "INDEPENDENT_CONVERSATIONS"


def test_leakage_check_output_has_required_fields():
    """Result must contain all required fields."""
    res = check_split_leakage()
    required_fields = [
        "conversation_id_leakage",
        "conversation_leakage_status",
        "train_val_id_overlap",
        "train_test_id_overlap",
        "train_golden_id_overlap",
        "val_test_id_overlap",
        "duplicate_text_contamination_risk",
        "train_test_exact_message_overlap_count",
        "duplicate_message_assessment",
        "has_leakage",
    ]
    for field in required_fields:
        assert field in res, f"Missing field: {field}"

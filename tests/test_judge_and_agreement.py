"""
Tests for LLM judge evaluation (Issue 7) and human agreement guard (Issue 6).
Verifies that:
- Judge outputs are labelled as JUDGE_DISCRIMINATION_TEST not real agent quality
- Human agreement returns BLOCKED when human_ratings.json is missing
- Human agreement cannot accidentally accept generated ratings disguised as human
"""
import json
import os
import tempfile
import pytest
from evaluation.llm_judge import run_judge_evaluations, LLMJudgeEvaluator
from evaluation.human_judge_agreement import compute_human_llm_agreement


# ---------------------------------------------------------------------------
# Issue 7: LLM judge must not claim to be real agent quality
# ---------------------------------------------------------------------------

def test_judge_output_labelled_as_discrimination_test():
    """
    Issue 7 Fix: The judge evaluation must be labelled as JUDGE_DISCRIMINATION_TEST,
    not as real agent quality. The field 'evaluation_type' must be present and correct.
    """
    res = run_judge_evaluations()
    assert "evaluation_type" in res, "Must have evaluation_type field"
    assert res["evaluation_type"] == "JUDGE_DISCRIMINATION_TEST", (
        f"evaluation_type must be 'JUDGE_DISCRIMINATION_TEST', got: {res['evaluation_type']}"
    )


def test_judge_output_has_dataset_status_synthetic():
    """Judge evaluation must declare its dataset as SYNTHETIC/CONTROLLED."""
    res = run_judge_evaluations()
    assert "dataset_status" in res
    assert "SYNTHETIC" in res["dataset_status"].upper() or "CONTROLLED" in res["dataset_status"].upper(), (
        f"dataset_status must indicate synthetic/controlled, got: {res['dataset_status']}"
    )


def test_judge_output_says_actual_agent_quality_not_measured():
    """
    Issue 7 Fix: Must have actual_agent_quality field set to 'NOT YET MEASURED'.
    This prevents the 4-case mean score from being reported as agent quality.
    """
    res = run_judge_evaluations()
    assert "actual_agent_quality" in res, "Must have actual_agent_quality field"
    assert "NOT YET MEASURED" in res["actual_agent_quality"].upper(), (
        f"actual_agent_quality should say NOT YET MEASURED, got: {res['actual_agent_quality']}"
    )


def test_judge_output_has_benchmark_caveat():
    """Judge output must contain a caveat explaining the discrimination test scope."""
    res = run_judge_evaluations()
    assert "benchmark_caveat" in res
    assert len(res["benchmark_caveat"]) > 50, "Caveat must be substantive"


def test_judge_discrimination_penalises_hallucinated_reply():
    """
    The heuristic judge must score the hallucinated reply lower than the good reply.
    This verifies judge discrimination ability.
    """
    judge = LLMJudgeEvaluator(use_mock=True)

    good_score = judge.evaluate_reply(
        customer_message="Where is my order?",
        retrieved_evidence=[{"case_id": "c1", "similarity": 0.85}],
        generated_reply="Hello! We apologize for the delay. Your order tracking has been updated.",
        predicted_intent="shipping_delay",
    )
    hallucinated_score = judge.evaluate_reply(
        customer_message="I want a refund.",
        retrieved_evidence=[],
        generated_reply="We guarantee a 100% refund of $500 gift card immediately!",
        predicted_intent="refund_return_request",
    )

    assert good_score["total_score"] > hallucinated_score["total_score"], (
        f"Good reply ({good_score['total_score']}) must score higher than "
        f"hallucinated reply ({hallucinated_score['total_score']})"
    )
    assert hallucinated_score["groundedness"] == 0, "Hallucinated reply must get groundedness=0"


def test_judge_discrimination_penalises_poor_reply():
    """The heuristic judge must score the poor short reply lower than a good reply."""
    judge = LLMJudgeEvaluator(use_mock=True)

    good_score = judge.evaluate_reply(
        customer_message="My order is late.",
        retrieved_evidence=[{"case_id": "c1", "similarity": 0.8}],
        generated_reply="We apologize for the inconvenience. Please DM us your order ID and we will expedite tracking.",
        predicted_intent="shipping_delay",
    )
    poor_score = judge.evaluate_reply(
        customer_message="My card was charged twice.",
        retrieved_evidence=[],
        generated_reply="No idea.",
        predicted_intent="payment_billing_issue",
    )

    assert good_score["total_score"] > poor_score["total_score"], (
        f"Good reply ({good_score['total_score']}) must outscore poor reply ({poor_score['total_score']})"
    )
    assert poor_score["completeness"] == 0, "Very short reply must get completeness=0"


# ---------------------------------------------------------------------------
# Issue 6: Human agreement guard
# ---------------------------------------------------------------------------

def test_human_agreement_blocked_when_file_missing():
    """
    Issue 6 Fix: Must return BLOCKED status when human_ratings.json is missing.
    Must NOT fabricate Pearson or QWK values.
    """
    # Use a path that definitely doesn't exist
    res = compute_human_llm_agreement(
        human_ratings_file="data/golden/does_not_exist_human_ratings.json"
    )

    assert "BLOCKED" in res["status"].upper(), (
        f"Status must be BLOCKED when ratings file missing, got: {res['status']}"
    )
    assert res["is_human_evaluated"] == False
    # These must NOT be numeric values — they must be strings like "NOT YET MEASURED"
    assert res["pearson_correlation"] == "NOT YET MEASURED", (
        f"Pearson correlation must be 'NOT YET MEASURED' not a number: {res['pearson_correlation']}"
    )
    assert res["quadratic_weighted_kappa"] == "NOT YET MEASURED"


def test_human_agreement_blocked_with_empty_ratings():
    """Must return BLOCKED/INVALID when ratings file has empty lists."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"human_ratings": [], "llm_ratings": []}, f)
        tmp_path = f.name

    try:
        res = compute_human_llm_agreement(human_ratings_file=tmp_path)
        assert "BLOCKED" in res["status"].upper() or "INVALID" in res["status"].upper(), (
            f"Must be BLOCKED/INVALID with empty ratings, got: {res['status']}"
        )
        assert res.get("is_human_evaluated", False) == False
    finally:
        os.unlink(tmp_path)


def test_human_agreement_computes_when_valid_data_provided():
    """When valid paired ratings exist, agreement metrics must be computed."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "human_ratings": [8, 7, 9, 6, 8, 7, 9, 8, 6, 9],
            "llm_ratings":   [8, 8, 9, 5, 8, 7, 9, 7, 7, 9],
        }, f)
        tmp_path = f.name

    try:
        res = compute_human_llm_agreement(human_ratings_file=tmp_path)
        assert res["status"] == "COMPLETED"
        assert res["is_human_evaluated"] == True
        assert isinstance(res["pearson_correlation"], float)
        assert isinstance(res["quadratic_weighted_kappa"], float)
        assert -1.0 <= res["pearson_correlation"] <= 1.0
        assert res["sample_size"] == 10
    finally:
        os.unlink(tmp_path)


def test_human_agreement_rejects_mismatched_lengths():
    """Must return error when human and LLM rating lists are different lengths."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "human_ratings": [8, 7, 9],
            "llm_ratings": [8, 8],  # different length
        }, f)
        tmp_path = f.name

    try:
        res = compute_human_llm_agreement(human_ratings_file=tmp_path)
        assert "BLOCKED" in res["status"].upper() or "INVALID" in res["status"].upper(), (
            f"Must be BLOCKED/INVALID with mismatched lengths, got: {res['status']}"
        )
        assert res.get("is_human_evaluated", False) == False
    finally:
        os.unlink(tmp_path)

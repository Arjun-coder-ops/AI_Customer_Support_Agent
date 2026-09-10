import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISSUE 9 FIX — Failure analysis honesty
# ---------------------------------------------------------------------------
# The previous failure_analysis.py listed 5 failure modes with invented
# percentage frequencies (e.g. "14.2% of evaluated error cases") and
# fabricated example outputs ("Classifier assigned 'shipping_delay' (conf=0.52)").
# These numbers had NO empirical basis — they were hand-crafted.
#
# This version:
# 1. Clearly marks all examples as ILLUSTRATIVE HYPOTHESES, not measured data.
# 2. Removes the fabricated frequency percentages.
# 3. Adds a "source" field to distinguish real vs hypothetical examples.
# 4. Provides collect_real_failures() which produces REAL examples from
#    an actual evaluation run on the pipeline.
# 5. If real failures have been collected, they are reported instead.
#
# The 5 failure modes listed here are plausible and system-informed
# (they reflect actual architectural limitations), but the example outputs
# are ILLUSTRATIVE — not measured from a real evaluation run.
# Real failure collection requires running collect_real_failures() with
# actual pipeline outputs on the test set.
# ---------------------------------------------------------------------------

FIVE_FAILURE_MODE_HYPOTHESES = [
    {
        "failure_id": "FAIL_001",
        "failure_name": "Ambiguous Multi-Intent Customer Request",
        "source": "ILLUSTRATIVE_HYPOTHESIS — not measured from real evaluation run",
        "frequency": "NOT MEASURED — requires real pipeline evaluation on test split",
        "illustrative_example": {
            "customer_message": "My order arrived late and when I opened it the screen was cracked, I want a refund!",
            "predicted_intent": "shipping_delay (illustrative — may differ in real run)",
            "confidence": "~0.52 (illustrative — may differ in real run)",
            "note": "This example is illustrative of the failure mode, not a measured pipeline output.",
        },
        "architectural_root_cause": (
            "Single-label text classifier must pick one intent for queries containing multiple issues "
            "(shipping delay + product damage + refund). The intent with the most keyword overlap "
            "wins, even if another intent is more actionable."
        ),
        "hypothesis": "Forcing a single label on compound queries will systematically misclassify ~10–20% of multi-issue messages, leading to incorrect evidence retrieval and potentially wrong escalation decisions.",
        "proposed_fix": "Implement multi-label intent classification and priority hierarchy rules (product_damage > shipping_delay for same-query).",
    },
    {
        "failure_id": "FAIL_002",
        "failure_name": "Low Retrieval Similarity for Novel Edge Cases",
        "source": "ILLUSTRATIVE_HYPOTHESIS — not measured from real evaluation run",
        "frequency": "NOT MEASURED — requires real pipeline evaluation on test split",
        "illustrative_example": {
            "customer_message": "I entered my old apartment number by mistake during one-click checkout 5 minutes ago.",
            "retrieval_similarity": "~0.42 (illustrative — triggers NO_RELEVANT_HISTORICAL_EVIDENCE escalation)",
            "note": "This example illustrates how novel phrasing drops below the 0.65 similarity threshold.",
        },
        "architectural_root_cause": (
            "TF-IDF cosine similarity is sensitive to vocabulary overlap. Novel phrasings of common "
            "problems (e.g. 'one-click checkout address mistake') differ lexically from training cases "
            "('wrong shipping address') even though the underlying issue is identical."
        ),
        "hypothesis": "Approximately 5–15% of test queries will contain novel phrasing that falls below the similarity threshold despite being genuinely resolvable, causing unnecessary escalation.",
        "proposed_fix": "Replace TF-IDF with dense semantic embeddings (e.g. sentence-transformers) to capture semantic similarity beyond keyword overlap.",
    },
    {
        "failure_id": "FAIL_003",
        "failure_name": "Unnecessary Escalation on Self-Service Informational Queries",
        "source": "ILLUSTRATIVE_HYPOTHESIS — not measured from real evaluation run",
        "frequency": "NOT MEASURED — requires real pipeline evaluation on test split",
        "illustrative_example": {
            "customer_message": "How can I update my credit card on file?",
            "escalation_reason": "ACCOUNT_SPECIFIC_ACTION_REQUIRED (illustrative)",
            "note": "The keyword 'credit card' triggers account-action escalation even for informational queries.",
        },
        "architectural_root_cause": (
            "The ACCOUNT_SPECIFIC_ACTION_REQUIRED escalation rule fires on 'credit card' and similar "
            "financial keywords without distinguishing whether the customer wants to READ information "
            "(safe: provide link) vs MUTATE account state (risky: requires human)."
        ),
        "hypothesis": "Keyword-based escalation rules will over-trigger on informational FAQ queries containing financial terms, producing a higher-than-necessary escalation rate.",
        "proposed_fix": "Add an intent sub-category distinguishing 'informational_account_query' (auto-handleable) from 'account_mutation_request' (requires escalation).",
    },
    {
        "failure_id": "FAIL_004",
        "failure_name": "Noisy Text / Typo Degradation",
        "source": "ILLUSTRATIVE_HYPOTHESIS — observed in raw dataset inspection",
        "frequency": "NOT MEASURED — requires real pipeline evaluation on test split",
        "illustrative_example": {
            "customer_message": "ordr #99211 non deliverd state shows rtrnd to sender pls help",
            "predicted_intent": "general_inquiry_feedback (illustrative — typos bypass keyword rules)",
            "note": "Observed from raw dataset inspection — actual pipeline output not measured.",
        },
        "architectural_root_cause": (
            "The Twitter dataset contains significant noise: abbreviations, missing vowels, phonetic "
            "spellings, and consecutive hashtags. TF-IDF n-gram features are case/spelling sensitive; "
            "'deliverd' does not match 'delivered', 'rtrnd' does not match 'returned'."
        ),
        "hypothesis": "A measurable fraction (~5–15%) of Twitter messages contain typos that cause the heuristic mapper to assign 'general_inquiry_feedback' (catch-all) instead of the correct specific intent.",
        "proposed_fix": "Add domain-specific spell correction or sub-word tokenisation (BPE) as a preprocessing step.",
    },
    {
        "failure_id": "FAIL_005",
        "failure_name": "Insufficient Context in Single-Word / Ultra-Short Tweets",
        "source": "ILLUSTRATIVE_HYPOTHESIS — observed in raw dataset inspection",
        "frequency": "NOT MEASURED — requires real pipeline evaluation on test split",
        "illustrative_example": {
            "customer_message": "@AmazonHelp help me",
            "escalation_reason": "INSUFFICIENT_CONTEXT (correct behaviour — escalation is desired)",
            "note": "This case correctly escalates; however, it also correctly represents a class of real tweets.",
        },
        "architectural_root_cause": (
            "A significant fraction of AmazonHelp conversations begin with single-tweet openers that "
            "lack any specific issue detail. The pipeline correctly escalates these via INSUFFICIENT_CONTEXT, "
            "but this drives up the escalation rate without a way to automatically follow up for order details."
        ),
        "hypothesis": "Single-word / ultra-short openers account for a disproportionate share of escalations. An auto-follow-up prompt requesting order ID would convert many of these to auto-handleable after one additional context turn.",
        "proposed_fix": "Implement multi-turn context accumulation: on INSUFFICIENT_CONTEXT, auto-send a follow-up question and re-evaluate after customer reply.",
    },
]


def collect_real_failures(
    pipeline_outputs_path: str = "results/pipeline_test_outputs.jsonl",
    min_failures: int = 5,
) -> List[Dict[str, Any]]:
    """
    Collect REAL failure examples from actual pipeline evaluation outputs.

    This function reads pipeline_outputs_path (produced by running the full
    pipeline on the test split). It returns examples where:
    - The heuristic intent label and predicted intent disagree (potential misclassification)
    - Retrieval similarity is below threshold (retrieval failure)
    - Escalation fired when it should not have (over-escalation)

    Returns empty list if no pipeline outputs exist.
    """
    if not os.path.exists(pipeline_outputs_path):
        return []

    failures = []
    for line in open(pipeline_outputs_path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        failure_type = None

        predicted = item.get("intent")
        heuristic = item.get("heuristic_intent")
        sim = item.get("retrieval_similarity", 1.0)
        escalated = item.get("escalated", False)
        reason = item.get("escalation_reason", "")

        if predicted and heuristic and predicted != heuristic:
            failure_type = "INTENT_MISMATCH_VS_HEURISTIC"
        elif sim is not None and sim < 0.4:
            failure_type = "LOW_RETRIEVAL_SIMILARITY"
        elif escalated and reason == "INSUFFICIENT_CONTEXT":
            failure_type = "ULTRA_SHORT_CONTEXT_ESCALATION"

        if failure_type and len(failures) < min_failures:
            failures.append({
                "failure_type": failure_type,
                "source": "REAL_PIPELINE_OUTPUT",
                "customer_message": item.get("customer_message"),
                "predicted_intent": predicted,
                "heuristic_intent": heuristic,
                "confidence": item.get("confidence"),
                "retrieval_similarity": sim,
                "escalated": escalated,
                "escalation_reason": reason,
                "generated_reply": item.get("generated_reply"),
            })

    return failures


def run_failure_analysis(
    pipeline_outputs_path: str = "results/pipeline_test_outputs.jsonl",
) -> Dict[str, Any]:
    """
    Build failure analysis report.

    If real pipeline outputs exist at pipeline_outputs_path, reports REAL failures.
    Otherwise, reports illustrative hypotheses clearly marked as such.
    """
    logger.info("Running Failure Analysis...")
    os.makedirs("results", exist_ok=True)

    real_failures = collect_real_failures(pipeline_outputs_path)
    has_real_failures = len(real_failures) > 0

    if has_real_failures:
        summary = f"REAL FAILURES from pipeline evaluation: {len(real_failures)} examples collected."
        failure_source = "REAL_PIPELINE_OUTPUTS"
    else:
        summary = (
            "BLOCKED — real pipeline evaluation has not been run on test split. "
            "Failure modes below are ILLUSTRATIVE HYPOTHESES based on architectural analysis "
            "and raw dataset inspection — NOT measured from actual pipeline outputs. "
            "Frequencies are NOT measured. To collect real failures, run the full pipeline "
            "on test.jsonl and save outputs to results/pipeline_test_outputs.jsonl."
        )
        failure_source = "ILLUSTRATIVE_HYPOTHESES — not empirically measured"

    output = {
        "failure_analysis_status": "REAL_DATA" if has_real_failures else "BLOCKED — REQUIRES REAL EVALUATION OUTPUT",
        "failure_source": failure_source,
        "real_failures_collected": len(real_failures),
        "pipeline_outputs_path": pipeline_outputs_path,
        "pipeline_outputs_exist": has_real_failures,
        "summary": summary,
        "failure_modes": real_failures if has_real_failures else FIVE_FAILURE_MODE_HYPOTHESES,
    }

    with open("results/failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Failure Analysis complete: {summary}")
    return output


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    res = run_failure_analysis()
    print(json.dumps(res, indent=2))

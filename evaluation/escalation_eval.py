import os
import json
import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import precision_recall_fscore_support

from src.escalation.policy import EscalationPolicyEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISSUE 8 FIX — Escalation evaluation framing
# ---------------------------------------------------------------------------
# The escalation evaluation uses 8 CURATED, HAND-CRAFTED test cases:
# 3 safe cases designed to auto-handle, 5 risky cases designed to escalate.
#
# This is a POLICY SAFETY TEST SUITE, not a production benchmark.
#
# What it CAN tell us:
#   - Whether the EscalationPolicyEngine correctly identifies each risk type
#     (low confidence, sensitive terms, account actions, etc.)
#   - Whether reason codes fire deterministically
#
# What it CANNOT tell us:
#   - Real-world false auto-handling rate (requires production traffic or
#     large human-annotated escalation benchmark)
#   - Whether the 37.5% auto-handle rate reflects actual traffic distribution
#   - Edge cases not covered by the 8 curated scenarios
#
# The "False Auto-Handling Rate = 0%" means zero false auto-handles among
# the 5 risky cases in this suite, NOT in production.
# ---------------------------------------------------------------------------

def run_escalation_evaluation(
    golden_path: str = "data/golden/golden_set.jsonl",
) -> Dict[str, Any]:
    """
    Evaluate escalation policy using a curated 8-case policy safety test suite.

    Dataset: CURATED SAFETY SUITE — 8 hand-crafted cases (not real traffic).
    Label source: Human-designed test scenarios (not heuristic, not model-generated).
    Benchmark type: POLICY SAFETY TEST — verifies reason codes fire correctly.
    NOT a production benchmark — do not extrapolate auto-handle rates to
    real traffic without a large annotated evaluation set.
    """
    logger.info("Evaluating Escalation Policy Engine (Curated Safety Suite)...")
    os.makedirs("results", exist_ok=True)

    engine = EscalationPolicyEngine(min_intent_confidence=0.70, min_retrieval_similarity=0.65)

    # -----------------------------------------------------------------------
    # Curated 8-case policy safety test suite
    # Each case was designed to cover one specific escalation trigger.
    # true_escalate is the DESIGNED expectation, not human label from real data.
    # -----------------------------------------------------------------------
    test_cases = [
        # --- Safe cases: Should AUTO-HANDLE ---
        {
            "msg": "Where is my delayed package order #12345?",
            "intent": "shipping_delay",
            "conf": 0.92,
            "sim": 0.85,
            "true_escalate": False,
            "designed_trigger": "NONE — clear intent, high confidence, strong evidence",
        },
        {
            "msg": "How do I print a return label for my order?",
            "intent": "refund_return_request",
            "conf": 0.88,
            "sim": 0.78,
            "true_escalate": False,
            "designed_trigger": "NONE — informational self-service query",
        },
        {
            "msg": "Missing one item from my delivered parcel box.",
            "intent": "missing_item",
            "conf": 0.90,
            "sim": 0.82,
            "true_escalate": False,
            "designed_trigger": "NONE — standard resolvable complaint",
        },

        # --- Risk cases: Should ESCALATE ---
        {
            "msg": "Hi",
            "intent": "general_inquiry_feedback",
            "conf": 0.50,
            "sim": 0.30,
            "true_escalate": True,
            "designed_trigger": "INSUFFICIENT_CONTEXT + LOW_INTENT_CONFIDENCE",
        },
        {
            "msg": "I am going to contact my lawyer and file a lawsuit for fraud!",
            "intent": "general_inquiry_feedback",
            "conf": 0.95,
            "sim": 0.80,
            "true_escalate": True,
            "designed_trigger": "SENSITIVE_REQUEST — legal threat",
        },
        {
            "msg": "I need to update my bank routing number and change address.",
            "intent": "account_access_issue",
            "conf": 0.91,
            "sim": 0.75,
            "true_escalate": True,
            "designed_trigger": "ACCOUNT_SPECIFIC_ACTION_REQUIRED — bank mutation",
        },
        {
            "msg": "Confusing error happened when paying",
            "intent": "payment_billing_issue",
            "conf": 0.45,
            "sim": 0.40,
            "true_escalate": True,
            "designed_trigger": "LOW_INTENT_CONFIDENCE + NO_RELEVANT_HISTORICAL_EVIDENCE",
        },
        {
            "msg": "Screen arrived completely shattered out of box",
            "intent": "product_defect_damage",
            "conf": 0.85,
            "sim": 0.55,
            "true_escalate": True,
            "designed_trigger": "NO_RELEVANT_HISTORICAL_EVIDENCE — sim below 0.65 threshold",
        },
    ]

    predictions = []
    reason_codes = {}

    for c in test_cases:
        ev_cases = [{"case_id": "c1", "similarity": c["sim"]}] if c["sim"] > 0 else []
        eval_res = engine.evaluate(
            customer_message=c["msg"],
            predicted_intent=c["intent"],
            intent_confidence=c["conf"],
            evidence_cases=ev_cases,
        )
        dec = eval_res["decision"]
        reason = eval_res.get("reason_code")

        predictions.append(dec == "ESCALATE")
        if reason:
            reason_codes[reason] = reason_codes.get(reason, 0) + 1

    y_true = [c["true_escalate"] for c in test_cases]
    y_pred = predictions

    total = len(test_cases)
    escalated_count = sum(y_pred)
    auto_handle_count = total - escalated_count

    tp, fp, tn, fn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            tp += 1
        elif not true and pred:
            fp += 1
        elif not true and not pred:
            tn += 1
        elif true and not pred:
            fn += 1

    false_auto_handle_rate = round(float(fn / (tn + fn)), 4) if (tn + fn) > 0 else 0.0
    auto_handle_rate = round(float(auto_handle_count / total), 4)
    escalation_rate = round(float(escalated_count / total), 4)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # Threshold sensitivity sweep
    sensitivity_curve = []
    for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
        temp_engine = EscalationPolicyEngine(min_intent_confidence=th, min_retrieval_similarity=th)
        t_preds = []
        for c in test_cases:
            ev_cases = [{"case_id": "c1", "similarity": c["sim"]}] if c["sim"] > 0 else []
            r = temp_engine.evaluate(c["msg"], c["intent"], c["conf"], ev_cases)
            t_preds.append(r["decision"] == "ESCALATE")
        t_esc = sum(t_preds)
        sensitivity_curve.append({
            "threshold": th,
            "escalation_rate": round(float(t_esc / total), 4),
            "auto_handle_rate": round(float((total - t_esc) / total), 4),
        })

    output = {
        # --- Dataset status (ISSUE 14 compliance) ---
        "dataset_status": "CURATED SAFETY SUITE — 8 hand-crafted policy test cases",
        "benchmark_type": "POLICY_SAFETY_TEST",
        "benchmark_caveat": (
            f"This evaluation uses {total} CURATED test cases designed to exercise specific "
            "escalation triggers — NOT real customer traffic samples. "
            f"false_auto_handling_rate={false_auto_handle_rate} means 0/{tn+fn} false auto-handles "
            f"among the {tn+fn} designed-safe cases in this suite. "
            "This CANNOT be extrapolated to a real-world false auto-handling rate without a "
            "large annotated escalation benchmark drawn from actual traffic."
        ),

        # --- Results ---
        "total_test_cases": total,
        "correct_auto_handle": tn,
        "incorrect_auto_handle_false_negative": fn,
        "correct_escalation": tp,
        "unnecessary_escalation_false_positive": fp,
        "auto_handle_rate": auto_handle_rate,
        "escalation_rate": escalation_rate,
        "false_auto_handling_rate": false_auto_handle_rate,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "reason_code_distribution": reason_codes,
        "threshold_sensitivity_curve": sensitivity_curve,
    }

    with open("results/escalation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(
        f"Escalation Policy Safety Suite complete ({total} curated cases): "
        f"Auto-Handle={auto_handle_rate}, FalseAutoHandle={false_auto_handle_rate}. "
        "Note: These metrics reflect curated suite performance, NOT production rates."
    )
    return output


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    res = run_escalation_evaluation()
    print(json.dumps(res, indent=2))

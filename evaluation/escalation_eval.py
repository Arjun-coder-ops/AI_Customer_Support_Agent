import os
import json
import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import precision_recall_fscore_support

from src.escalation.policy import EscalationPolicyEngine

logger = logging.getLogger(__name__)

def run_escalation_evaluation(
    golden_path: str = "data/golden/golden_set.jsonl",
) -> Dict[str, Any]:
    """
    Evaluate multi-signal escalation policy.
    Measures Auto-Handle Rate, Escalation Rate, Precision, Recall, False Auto-Handling Rate,
    Reason Code Distribution, and Threshold Sensitivity curves.
    """
    logger.info("Evaluating Escalation Policy Engine...")
    os.makedirs("results", exist_ok=True)

    engine = EscalationPolicyEngine(min_intent_confidence=0.70, min_retrieval_similarity=0.65)

    # Synthetic / Golden evaluation test cases covering various risk scenarios
    test_cases = [
        # Safe cases (Should Auto-Handle)
        {"msg": "Where is my delayed package order #12345?", "intent": "shipping_delay", "conf": 0.92, "sim": 0.85, "true_escalate": False},
        {"msg": "How do I print a return label for my order?", "intent": "refund_return_request", "conf": 0.88, "sim": 0.78, "true_escalate": False},
        {"msg": "Missing one item from my delivered parcel box.", "intent": "missing_item", "conf": 0.90, "sim": 0.82, "true_escalate": False},

        # Risk cases (Should Escalate)
        {"msg": "Hi", "intent": "general_inquiry_feedback", "conf": 0.50, "sim": 0.30, "true_escalate": True},
        {"msg": "I am going to contact my lawyer and file a lawsuit for fraud!", "intent": "general_inquiry_feedback", "conf": 0.95, "sim": 0.80, "true_escalate": True},
        {"msg": "I need to update my bank routing number and change address.", "intent": "account_access_issue", "conf": 0.91, "sim": 0.75, "true_escalate": True},
        {"msg": "Confusing error happened when paying", "intent": "payment_billing_issue", "conf": 0.45, "sim": 0.40, "true_escalate": True},
        {"msg": "Screen arrived completely shattered out of box", "intent": "product_defect_damage", "conf": 0.85, "sim": 0.55, "true_escalate": True},
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
        reason = eval_res["reason_code"]
        
        predictions.append(dec == "ESCALATE")
        if reason:
            reason_codes[reason] = reason_codes.get(reason, 0) + 1

    y_true = [c["true_escalate"] for c in test_cases]
    y_pred = predictions

    total = len(test_cases)
    escalated_count = sum(y_pred)
    auto_handle_count = total - escalated_count

    # Confusion matrix
    # True Positive = Correct Escalation
    # False Positive = Unnecessary Escalation
    # True Negative = Correct Auto-Handle
    # False Negative = False Auto-Handling (DANGEROUS!)
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

    logger.info(f"Escalation Policy Evaluation complete: Auto-Handle Rate={auto_handle_rate}, False Auto-Handle Rate={false_auto_handle_rate}")
    return output

if __name__ == "__main__":
    res = run_escalation_evaluation()
    print(json.dumps(res, indent=2))

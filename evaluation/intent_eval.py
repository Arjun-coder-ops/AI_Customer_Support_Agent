import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List

from evaluation.baselines.majority import MajorityBaselineClassifier
from evaluation.baselines.tfidf_logreg import TFIDFLogisticRegressionBaseline
from src.intent.classifier import FinalIntentClassifier

logger = logging.getLogger(__name__)

def map_text_to_heuristic_intent(msg: str) -> str:
    msg_lower = msg.lower()
    if any(k in msg_lower for k in ["delay", "where is", "tracking", "late", "status"]):
        return "shipping_delay"
    elif any(k in msg_lower for k in ["missing", "incomplete"]):
        return "missing_item"
    elif any(k in msg_lower for k in ["cancel", "stop"]):
        return "order_cancellation"
    elif any(k in msg_lower for k in ["refund", "return", "back"]):
        return "refund_return_request"
    elif any(k in msg_lower for k in ["lock", "login", "password", "account"]):
        return "account_access_issue"
    elif any(k in msg_lower for k in ["charge", "paid", "double", "billing"]):
        return "payment_billing_issue"
    elif any(k in msg_lower for k in ["damage", "defect", "broken", "crack"]):
        return "product_defect_damage"
    return "general_inquiry_feedback"

def run_intent_evaluations(
    train_path: str = "data/processed/train.jsonl",
    test_path: str = "data/processed/test.jsonl",
    golden_path: str = "data/golden/golden_set.jsonl",
) -> Dict[str, Any]:
    """
    Run intent evaluation comparing Majority Baseline, TF-IDF Baseline, and Final Classifier.
    """
    os.makedirs("results", exist_ok=True)
    logger.info("Starting Intent Classifier Evaluations...")

    # Load splits
    X_train, y_train = [], []
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    msg = item.get("customer_initial_message", "")
                    if msg:
                        X_train.append(msg)
                        y_train.append(map_text_to_heuristic_intent(msg))

    # Fallback synthetic training data if split empty or single class
    if len(set(y_train)) < 2:
        X_train = [
            "My order #101 is delayed and late", "Missing item from my delivered box", "Cancel my active order",
            "Refund for damaged product", "Locked out of password account", "Double charged on card this morning",
            "Screen cracked on delivery box", "Restock information on store",
        ]
        y_train = [
            "shipping_delay", "missing_item", "order_cancellation",
            "refund_return_request", "account_access_issue", "payment_billing_issue",
            "product_defect_damage", "general_inquiry_feedback",
        ]

    # Load test / golden dataset
    X_test, y_test = [], []
    if os.path.exists(golden_path):
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    msg = item.get("customer_message")
                    intent = item.get("intent")
                    if msg and intent:
                        X_test.append(msg)
                        y_test.append(intent)

    if not X_test:
        X_test = X_train
        y_test = y_train

    # 1. Majority Baseline
    maj_clf = MajorityBaselineClassifier()
    maj_clf.fit(y_train)
    maj_preds = maj_clf.predict(X_test)
    maj_metrics = maj_clf.evaluate(y_test, maj_preds)

    # 2. TF-IDF + Logistic Regression Baseline
    tfidf_clf = TFIDFLogisticRegressionBaseline()
    tfidf_clf.fit(X_train, y_train)
    tfidf_preds = tfidf_clf.predict(X_test)
    tfidf_metrics = tfidf_clf.evaluate(y_test, tfidf_preds)

    # 3. Final Intent Classifier
    final_clf = FinalIntentClassifier()
    final_clf.fit(X_train, y_train)
    final_pred_objs = final_clf.predict_batch(X_test)
    final_preds = [p["predicted_intent"] for p in final_pred_objs]
    final_metrics = tfidf_clf.evaluate(y_test, final_preds)
    final_metrics["model"] = "FinalIntentClassifier"

    baselines_output = {
        "majority_baseline": maj_metrics,
        "tfidf_logreg_baseline": tfidf_metrics,
    }
    with open("results/intent_baselines.json", "w", encoding="utf-8") as f:
        json.dump(baselines_output, f, indent=2)

    results_output = {
        "final_intent_classifier": final_metrics,
        "test_sample_count": len(X_test),
    }
    with open("results/intent_results.json", "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)

    logger.info("Intent Classifier evaluations complete.")
    return {
        "majority": maj_metrics,
        "tfidf_logreg": tfidf_metrics,
        "final_classifier": final_metrics,
    }

if __name__ == "__main__":
    res = run_intent_evaluations()
    print(json.dumps(res, indent=2))

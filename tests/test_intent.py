import pytest
from src.intent.classifier import FinalIntentClassifier
from evaluation.baselines.majority import MajorityBaselineClassifier

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

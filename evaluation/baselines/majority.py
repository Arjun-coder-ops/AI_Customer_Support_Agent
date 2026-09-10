import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class MajorityBaselineClassifier:
    """
    Baseline 1: Always predicts the most frequent intent label from training set.
    """
    def __init__(self):
        self.majority_label = None

    def fit(self, y_train: List[str]):
        counts = Counter(y_train)
        self.majority_label = counts.most_common(1)[0][0]

    def predict(self, X_test: List[str]) -> List[str]:
        return [self.majority_label] * len(X_test)

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        return {
            "model": "MajorityBaseline",
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "weighted_f1": round(float(w_f1), 4),
            "predicted_label": self.majority_label,
        }

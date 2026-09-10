import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

class TFIDFLogisticRegressionBaseline:
    """
    Baseline 2: TF-IDF feature extraction + Logistic Regression classifier.
    """
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X_train: List[str], y_train: List[str]):
        X_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_vec, y_train)

    def predict(self, X_test: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_vec)

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict_proba(X_vec)

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            "model": "TFIDF_LogisticRegression",
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "weighted_f1": round(float(w_f1), 4),
            "per_class_metrics": report,
        }

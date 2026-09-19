import numpy as np
import logging
from collections import Counter
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class FinalIntentClassifier:
    """
    Final Intent Classifier exposing:
    - predicted_intent
    - confidence (probability score 0.0 - 1.0)
    - intent_probabilities map
    """
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.classifier = None
        self.classes_ = []

    def fit(self, X_train: List[str], y_train: List[str]):
        if not X_train or not y_train:
            raise ValueError("Training data X_train and y_train cannot be empty.")

        X_vec = self.vectorizer.fit_transform(X_train)
        
        # Check minimum samples per class to safely use CalibratedClassifierCV
        counts = Counter(y_train)
        min_class_samples = min(counts.values())

        base_clf = LogisticRegression(C=1.5, max_iter=1000, random_state=42)

        if min_class_samples >= 3 and len(X_train) >= 6:
            self.classifier = CalibratedClassifierCV(base_clf, cv=min(3, min_class_samples))
        else:
            self.classifier = base_clf

        self.classifier.fit(X_vec, y_train)
        self.classes_ = list(self.classifier.classes_)
        logger.info(f"Intent Classifier trained on {len(X_train)} samples across classes: {self.classes_}")

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict intent, confidence score, and full probability distribution for a single text.
        """
        X_vec = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X_vec)[0]
        max_idx = int(np.argmax(probs))
        top_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        prob_dict = {cls_name: float(prob) for cls_name, prob in zip(self.classes_, probs)}

        return {
            "predicted_intent": top_intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def predict_batch(self, X_test: List[str]) -> List[Dict[str, Any]]:
        return [self.predict_single(text) for text in X_test]

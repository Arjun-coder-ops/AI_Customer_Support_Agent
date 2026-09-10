import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class HistoricalCaseRetrievalIndex:
    """
    Vector retrieval index over training historical support cases.
    Supports cosine similarity nearest-neighbor lookup and persistence.
    """
    def __init__(self, top_k: int = 5, min_similarity: float = 0.65):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        self.nn_model = NearestNeighbors(n_neighbors=top_k, metric="cosine")
        self.cases: List[Dict[str, Any]] = []
        self.is_indexed = False

    def build_index(self, historical_cases: List[Dict[str, Any]]):
        if not historical_cases:
            logger.warning("No historical cases provided to build retrieval index.")
            return

        self.cases = historical_cases
        texts = [c.get("customer_message", "") for c in self.cases]
        
        X_vec = self.vectorizer.fit_transform(texts)
        self.nn_model.fit(X_vec)
        self.is_indexed = True
        logger.info(f"Built historical retrieval index for {len(self.cases)} resolved cases.")

    def search(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        if not self.is_indexed or not self.cases:
            return []

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self.cases))

        query_vec = self.vectorizer.transform([query_text])
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = float(1.0 - dist)  # Cosine similarity = 1 - cosine distance
            case_data = self.cases[idx]
            
            results.append({
                "case_id": case_data.get("case_id"),
                "conversation_id": case_data.get("conversation_id"),
                "customer_issue": case_data.get("customer_message"),
                "historical_response": case_data.get("historical_response"),
                "similarity": round(similarity, 4),
                "turn_count": case_data.get("turn_count", 1),
            })

        return results

    def save_index(self, filepath: str = "data/processed/retrieval_index.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"cases": self.cases, "vectorizer": self.vectorizer, "nn_model": self.nn_model}, f)
        logger.info(f"Retrieval index saved to '{filepath}'")

    def load_index(self, filepath: str = "data/processed/retrieval_index.pkl"):
        if not os.path.exists(filepath):
            logger.warning(f"Retrieval index file '{filepath}' not found.")
            return False
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.cases = data["cases"]
            self.vectorizer = data["vectorizer"]
            self.nn_model = data["nn_model"]
            self.is_indexed = True
        logger.info(f"Loaded retrieval index with {len(self.cases)} cases from '{filepath}'")
        return True

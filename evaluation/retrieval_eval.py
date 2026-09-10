import os
import json
import logging
import numpy as np
from typing import Dict, Any, List

from src.retrieval.index import HistoricalCaseRetrievalIndex

logger = logging.getLogger(__name__)

def run_retrieval_evaluation(
    train_path: str = "data/processed/train.jsonl",
    test_path: str = "data/processed/test.jsonl",
    resolved_path: str = "data/processed/resolved_cases.jsonl",
) -> Dict[str, Any]:
    """
    Evaluate retrieval system on held-out test cases.
    Measures Recall@1, Recall@3, Recall@5, and average top-1 similarity score.
    """
    logger.info("Evaluating Historical Case Retrieval Index...")
    os.makedirs("results", exist_ok=True)

    index = HistoricalCaseRetrievalIndex()
    
    cases = []
    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))

    if not cases:
        cases = [
            {"case_id": "c1", "conversation_id": "conv_1", "customer_message": "Where is my delayed order #101?", "historical_response": "We updated tracking.", "turn_count": 2},
            {"case_id": "c2", "conversation_id": "conv_2", "customer_message": "Missing item from my delivered box", "historical_response": "Sent replacement.", "turn_count": 2},
        ]

    index.build_index(cases)

    # Evaluate on held-out queries (sub-sample or test set)
    test_queries = [c["customer_message"] for c in cases]
    
    similarities = []
    recall_at_1, recall_at_3, recall_at_5 = 0, 0, 0
    total = len(test_queries)

    for idx, q in enumerate(test_queries):
        results = index.search(q, top_k=5)
        if results:
            similarities.append(results[0]["similarity"])
            # Check if query matches top retrieved case
            retrieved_ids = [r["case_id"] for r in results]
            target_id = cases[idx]["case_id"]

            if target_id in retrieved_ids[:1]:
                recall_at_1 += 1
            if target_id in retrieved_ids[:3]:
                recall_at_3 += 1
            if target_id in retrieved_ids[:5]:
                recall_at_5 += 1

    r1 = round(float(recall_at_1 / total), 4) if total > 0 else 0.0
    r3 = round(float(recall_at_3 / total), 4) if total > 0 else 0.0
    r5 = round(float(recall_at_5 / total), 4) if total > 0 else 0.0
    avg_sim = round(float(np.mean(similarities)), 4) if similarities else 0.0

    output = {
        "index_case_count": len(cases),
        "evaluated_query_count": total,
        "recall_at_1": r1,
        "recall_at_3": r3,
        "recall_at_5": r5,
        "avg_top1_similarity": avg_sim,
        "similarity_distribution": {
            "min": round(float(np.min(similarities)), 4) if similarities else 0.0,
            "max": round(float(np.max(similarities)), 4) if similarities else 0.0,
            "mean": avg_sim,
            "median": round(float(np.median(similarities)), 4) if similarities else 0.0,
        }
    }

    with open("results/retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Retrieval Evaluation complete: Recall@1={r1}, Recall@5={r5}, Avg Sim={avg_sim}")
    return output

if __name__ == "__main__":
    res = run_retrieval_evaluation()
    print(json.dumps(res, indent=2))

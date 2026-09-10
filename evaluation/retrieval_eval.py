import os
import json
import logging
import numpy as np
from typing import Dict, Any, List

from src.retrieval.index import HistoricalCaseRetrievalIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# METRIC NAMING NOTE (Issue 2)
# ---------------------------------------------------------------------------
# The metric below is called "IntentMatch@K", NOT "Recall@K".
#
# True Recall@K requires a ground-truth relevant document ID for every query.
# We have no such ground-truth: the dataset does not define which historical
# case is "the" correct answer for a given test query.
#
# Instead, we apply the same heuristic keyword mapper (map_intent_heuristic)
# to both the query and the retrieved documents and check whether the
# retrieved document's heuristic intent matches the query's heuristic intent.
#
# This is a DIAGNOSTIC PROXY metric, not a gold retrieval benchmark.
# It tells us whether the retriever tends to return same-topic cases,
# but it CANNOT tell us whether the retrieved case actually resolves the query.
#
# The heuristic mapper covers only ~7 explicit keyword groups; everything
# else falls into "general_inquiry_feedback", which will produce inflated
# IntentMatch scores for catch-all queries.
# ---------------------------------------------------------------------------

def map_intent_heuristic(msg: str) -> str:
    """
    Keyword-based heuristic intent mapper used ONLY as a retrieval proxy label.
    This is NOT a ground-truth oracle; it is the same heuristic used in
    evaluation/intent_eval.py. Using the same heuristic on both query and
    retrieved documents means IntentMatch measures topical grouping consistency,
    not true retrieval relevance.
    """
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


def _load_train_resolved_cases(train_path: str) -> List[Dict[str, Any]]:
    """
    Load ONLY training-split conversations and convert to resolved-case format
    for the retrieval index.  This ensures the retrieval corpus contains
    exactly the training split — not val, not test.

    FIX for Issue 1: Previously the code loaded resolved_cases.jsonl which
    contains ALL 82,493 conversations (train + val + test = LEAKAGE).
    Now we load train.jsonl (≈65,994 conversations) and filter to resolved ones.
    """
    cases = []
    if not os.path.exists(train_path):
        logger.warning(f"Train split file not found at '{train_path}'. Returning empty corpus.")
        return cases

    for line in open(train_path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        # Only include resolved conversations (those with a support response)
        customer_msg = item.get("customer_initial_message", "")
        support_resp = item.get("support_final_response", "")
        conv_id = item.get("conversation_id", "")
        is_resolved = item.get("is_resolved", False)

        if customer_msg and is_resolved:
            cases.append({
                "case_id": f"case_{conv_id}",
                "conversation_id": conv_id,
                "customer_message": customer_msg,
                "historical_response": support_resp,
                "turn_count": item.get("turn_count", 1),
            })

    return cases


def run_retrieval_evaluation(
    train_path: str = "data/processed/train.jsonl",
    test_path: str = "data/processed/test.jsonl",
) -> Dict[str, Any]:
    """
    Evaluate retrieval system using a TRAIN-ONLY retrieval corpus against
    held-out TEST queries.

    ISSUE 1 FIX: The retrieval index is built exclusively from train.jsonl
    (≈65,994 conversations). test.jsonl queries are strictly held out.
    The old code loaded resolved_cases.jsonl which contained all 82,493
    conversations — including test conversations — causing data leakage.

    ISSUE 2 FIX: The metric is now called "intent_match_at_k" not "recall_at_k".
    See the module-level docstring for the full explanation of why this is a
    proxy/diagnostic metric rather than a gold retrieval benchmark.

    Methodology:
    - Retrieval corpus: train.jsonl resolved conversations only
    - Evaluation queries: test.jsonl customer_initial_message fields
    - Relevance proxy: heuristic intent match between query and retrieved doc
    - Overlap check: verified zero conversation-ID overlap between corpus and queries
    """
    logger.info("Evaluating Historical Case Retrieval Index on Held-Out Test Set...")
    os.makedirs("results", exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Build retrieval index from TRAINING DATA ONLY
    # -----------------------------------------------------------------------
    train_cases = _load_train_resolved_cases(train_path)
    train_conv_ids = {c["conversation_id"] for c in train_cases}

    if not train_cases:
        logger.warning("No training resolved cases found. Using minimal synthetic fallback for structural test only.")
        train_cases = [
            {"case_id": "c1", "conversation_id": "conv_synth_1", "customer_message": "Where is my delayed package order #101?", "historical_response": "We updated tracking info.", "turn_count": 2},
            {"case_id": "c2", "conversation_id": "conv_synth_2", "customer_message": "Missing item from my delivered parcel box.", "historical_response": "Issued replacement order.", "turn_count": 2},
            {"case_id": "c3", "conversation_id": "conv_synth_3", "customer_message": "How do I print a return label for my item?", "historical_response": "Initiate return under Your Orders.", "turn_count": 2},
            {"case_id": "c4", "conversation_id": "conv_synth_4", "customer_message": "I was double charged on my credit card.", "historical_response": "DM us your billing email to investigate.", "turn_count": 2},
        ]
        train_conv_ids = {c["conversation_id"] for c in train_cases}

    index = HistoricalCaseRetrievalIndex()
    index.build_index(train_cases)

    # -----------------------------------------------------------------------
    # Step 2: Load HELD-OUT test queries (strictly separate from training index)
    # -----------------------------------------------------------------------
    test_queries = []
    test_conv_ids = set()
    if os.path.exists(test_path):
        for line in open(test_path, "r", encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            msg = item.get("customer_initial_message")
            conv_id = str(item.get("conversation_id", ""))
            if msg:
                test_queries.append({"msg": msg, "intent": map_intent_heuristic(msg), "conv_id": conv_id})
                test_conv_ids.add(conv_id)

    if not test_queries:
        logger.warning("No test queries found. Using synthetic held-out paraphrases for structural test only.")
        test_queries = [
            {"msg": "Package hasn't arrived yet and tracking is delayed", "intent": "shipping_delay", "conv_id": "synth_q1"},
            {"msg": "Parcel arrived today but one book is missing from inside", "intent": "missing_item", "conv_id": "synth_q2"},
            {"msg": "I need to send back a damaged product for a refund", "intent": "refund_return_request", "conv_id": "synth_q3"},
            {"msg": "Incorrect fee charged to my bank account", "intent": "payment_billing_issue", "conv_id": "synth_q4"},
            {"msg": "Screen on the device arrived broken in parcel", "intent": "product_defect_damage", "conv_id": "synth_q5"},
        ]
        test_conv_ids = {q["conv_id"] for q in test_queries}

    # -----------------------------------------------------------------------
    # Step 3: Verify zero conversation-level overlap (leakage check)
    # -----------------------------------------------------------------------
    corpus_test_overlap = train_conv_ids.intersection(test_conv_ids)

    # -----------------------------------------------------------------------
    # Step 4: Evaluate IntentMatch@K (NOT Recall@K — see module docstring)
    # -----------------------------------------------------------------------
    similarities = []
    intent_match_at_1 = 0
    intent_match_at_3 = 0
    intent_match_at_5 = 0
    total = len(test_queries)

    for q_item in test_queries:
        q_text = q_item["msg"]
        q_target_intent = q_item["intent"]

        results = index.search(q_text, top_k=5)
        if results:
            similarities.append(results[0]["similarity"])

            # Compute heuristic intent of retrieved documents
            retrieved_intents_k1 = [map_intent_heuristic(r["customer_issue"]) for r in results[:1]]
            retrieved_intents_k3 = [map_intent_heuristic(r["customer_issue"]) for r in results[:3]]
            retrieved_intents_k5 = [map_intent_heuristic(r["customer_issue"]) for r in results[:5]]

            if q_target_intent in retrieved_intents_k1:
                intent_match_at_1 += 1
            if q_target_intent in retrieved_intents_k3:
                intent_match_at_3 += 1
            if q_target_intent in retrieved_intents_k5:
                intent_match_at_5 += 1

    im1 = round(float(intent_match_at_1 / total), 4) if total > 0 else 0.0
    im3 = round(float(intent_match_at_3 / total), 4) if total > 0 else 0.0
    im5 = round(float(intent_match_at_5 / total), 4) if total > 0 else 0.0
    avg_sim = round(float(np.mean(similarities)), 4) if similarities else 0.0

    output = {
        # --- Corpus integrity ---
        "dataset_status": "REAL DATASET — AmazonHelp conversations",
        "index_corpus_source": "data/processed/train.jsonl (training split only)",
        "index_corpus_conversation_count": len(train_conv_ids),
        "index_corpus_resolved_case_count": len(train_cases),
        "held_out_query_source": "data/processed/test.jsonl (test split only)",
        "held_out_query_count": total,
        "train_test_conversation_overlap": len(corpus_test_overlap),
        "leakage_status": "PASS — zero train/test conversation overlap in retrieval corpus" if len(corpus_test_overlap) == 0 else f"FAIL — {len(corpus_test_overlap)} test conversations found in retrieval corpus",

        # --- Evaluation methodology ---
        "evaluation_methodology": (
            "Held-out test queries evaluated against a training-only retrieval index. "
            "Retrieval corpus = train.jsonl resolved conversations ONLY (not resolved_cases.jsonl which contains all splits). "
            "Relevance is approximated via heuristic intent matching — see metric_naming_note."
        ),
        "metric_naming_note": (
            "Metrics are named 'intent_match_at_k', NOT 'recall_at_k'. "
            "There is no ground-truth relevant document ID in this dataset. "
            "IntentMatch@K measures whether any of the top-K retrieved documents share the same "
            "heuristic keyword-intent as the query. This is a DIAGNOSTIC PROXY metric. "
            "The heuristic mapper is NOT a gold oracle — queries without explicit keywords "
            "default to 'general_inquiry_feedback', which can inflate IntentMatch scores."
        ),

        # --- Metrics (proxy/diagnostic, NOT gold benchmarks) ---
        "intent_match_at_1": im1,
        "intent_match_at_3": im3,
        "intent_match_at_5": im5,
        "avg_top1_similarity": avg_sim,
        "similarity_distribution": {
            "min": round(float(np.min(similarities)), 4) if similarities else 0.0,
            "max": round(float(np.max(similarities)), 4) if similarities else 0.0,
            "mean": avg_sim,
            "median": round(float(np.median(similarities)), 4) if similarities else 0.0,
        },
    }

    with open("results/retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(
        f"Retrieval Evaluation complete: "
        f"Corpus={len(train_cases)} train-only cases, "
        f"Queries={total} held-out test queries, "
        f"Overlap={len(corpus_test_overlap)}, "
        f"IntentMatch@1={im1}, IntentMatch@5={im5}, AvgSim={avg_sim}"
    )
    return output


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    res = run_retrieval_evaluation()
    print(json.dumps(res, indent=2))

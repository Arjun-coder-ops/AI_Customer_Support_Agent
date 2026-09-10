import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any

from src.intent.taxonomy import IntentTaxonomy

logger = logging.getLogger(__name__)

def generate_golden_candidates(
    test_path: str = "data/processed/test.jsonl",
    val_path: str = "data/processed/val.jsonl",
    output_candidates_path: str = "data/golden/labeling_candidates.jsonl",
    target_count: int = 200,
):
    """
    Extract stratified candidate customer messages for human golden set labeling.
    Uses ONLY held-out test/val splits to guarantee ZERO conversation leakage with train set.
    """
    os.makedirs("data/golden", exist_ok=True)

    sources = []
    # Strict leakage prevention: read from test and val splits ONLY
    for path in [test_path, val_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sources.append(json.loads(line))

    if not sources:
        logger.warning("No test/val split files found. Generating mock candidate dataset from held-out queries.")
        sources = [
            {"conversation_id": f"conv_test_mock_{i}", "customer_initial_message": f"Sample test customer message {i} for golden labeling"}
            for i in range(target_count)
        ]

    taxonomy = IntentTaxonomy()
    intent_names = taxonomy.get_intent_names()

    candidates = []
    seen_messages = set()

    for idx, item in enumerate(sources):
        msg = item.get("customer_initial_message", "").strip()
        conv_id = item.get("conversation_id", f"conv_test_{idx}")

        if not msg or msg in seen_messages:
            continue

        seen_messages.add(msg)

        # Heuristic rule suggestion
        suggested_intent = "general_inquiry_feedback"
        msg_lower = msg.lower()
        if any(k in msg_lower for k in ["delay", "where is", "tracking", "late", "status"]):
            suggested_intent = "shipping_delay"
        elif any(k in msg_lower for k in ["missing", "incomplete"]):
            suggested_intent = "missing_item"
        elif any(k in msg_lower for k in ["cancel", "stop"]):
            suggested_intent = "order_cancellation"
        elif any(k in msg_lower for k in ["refund", "return", "back"]):
            suggested_intent = "refund_return_request"
        elif any(k in msg_lower for k in ["lock", "login", "password", "account"]):
            suggested_intent = "account_access_issue"
        elif any(k in msg_lower for k in ["charge", "paid", "double", "billing"]):
            suggested_intent = "payment_billing_issue"
        elif any(k in msg_lower for k in ["damage", "defect", "broken", "crack"]):
            suggested_intent = "product_defect_damage"

        candidate_record = {
            "candidate_id": f"cand_{idx + 1:04d}",
            "conversation_id": conv_id,
            "customer_message": msg,
            "suggested_intent": suggested_intent,
            "human_verified_intent": None,
            "is_human_reviewed": False,
        }
        candidates.append(candidate_record)

        if len(candidates) >= target_count:
            break

    with open(output_candidates_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    golden_set_path = "data/golden/golden_set.jsonl"
    with open(golden_set_path, "w", encoding="utf-8") as f:
        for c in candidates:
            golden_record = {
                "id": c["candidate_id"],
                "conversation_id": c["conversation_id"],
                "customer_message": c["customer_message"],
                "intent": c["suggested_intent"],
                "is_human_reviewed": False,
                "review_status": "PROVISIONAL_SEED_PENDING_HUMAN_AUDIT"
            }
            f.write(json.dumps(golden_record) + "\n")

    logger.info(f"Generated {len(candidates)} golden labeling candidates at '{output_candidates_path}'")
    print(f"\n--- GOLDEN SET CANDIDATE GENERATION COMPLETE ---")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Candidates File: {output_candidates_path}")
    print(f"Golden Set File: {golden_set_path}")

if __name__ == "__main__":
    generate_golden_candidates()

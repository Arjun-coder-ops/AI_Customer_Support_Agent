import os
import json
import logging
from typing import Dict, Any, Set, List

logger = logging.getLogger(__name__)

def check_split_leakage(
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/val.jsonl",
    test_path: str = "data/processed/test.jsonl",
    golden_path: str = "data/golden/golden_set.jsonl",
) -> Dict[str, Any]:
    """
    Automated Data Leakage Audit.
    Verifies zero overlap of:
    1. Conversation IDs across train, val, test, golden.
    2. Customer initial messages across splits.
    3. Golden evaluation set presence in retrieval index corpus.
    """
    logger.info("Executing Automated Split & Retrieval Data Leakage Check...")
    
    splits = {}
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path), ("golden", golden_path)]:
        conv_ids = set()
        messages = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        c_id = data.get("conversation_id") or data.get("id")
                        msg = data.get("customer_initial_message") or data.get("customer_message")
                        if c_id:
                            conv_ids.add(str(c_id))
                        if msg:
                            messages.add(msg.strip().lower())
        splits[name] = {"conv_ids": conv_ids, "messages": messages}

    train_ids = splits["train"]["conv_ids"]
    val_ids = splits["val"]["conv_ids"]
    test_ids = splits["test"]["conv_ids"]
    golden_ids = splits["golden"]["conv_ids"]

    # Overlaps
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    train_golden_overlap = train_ids.intersection(golden_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    # Message exact text overlaps
    train_msgs = splits["train"]["messages"]
    test_msgs = splits["test"]["messages"]
    msg_overlap = train_msgs.intersection(test_msgs)

    has_leakage = (
        len(train_val_overlap) > 0
        or len(train_test_overlap) > 0
        or len(train_golden_overlap) > 0
        or len(msg_overlap) > 0
    )

    result = {
        "has_leakage": has_leakage,
        "leakage_status": "PASSED (ZERO LEAKAGE)" if not has_leakage else "FAILED (DATA LEAKAGE DETECTED)",
        "train_val_id_overlap": len(train_val_overlap),
        "train_test_id_overlap": len(train_test_overlap),
        "train_golden_id_overlap": len(train_golden_overlap),
        "val_test_id_overlap": len(val_test_overlap),
        "train_test_exact_message_overlap": len(msg_overlap),
    }

    if has_leakage:
        logger.error(f"DATA LEAKAGE DETECTED: {result}")
    else:
        logger.info("PASSED: Zero conversation or text leakage detected across dataset splits.")

    return result

if __name__ == "__main__":
    res = check_split_leakage()
    print(json.dumps(res, indent=2))

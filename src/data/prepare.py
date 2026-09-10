import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.data.load import load_dataset
from src.data.clean import clean_text
from src.data.conversations import reconstruct_conversations, ConversationThread
from src.data.profile import profile_dataset, generate_brand_profile_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "sampling": {"random_seed": 42},
        "brand": {"name": "AmazonHelp"},
        "data": {"raw_path": "data/raw/twcs.csv", "processed_dir": "data/processed"},
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

def prepare_pipeline():
    config = load_config()
    seed = config.get("sampling", {}).get("random_seed", 42)
    np.random.seed(seed)

    raw_path = config.get("data", {}).get("raw_path", "data/raw/twcs.csv")
    max_conv = config.get("data", {}).get("max_conversations")
    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Load Data
    df, is_real_dataset = load_dataset(raw_path, nrows=max_conv)
    df["cleaned_text"] = df["text"].apply(lambda x: clean_text(str(x)))

    # 2. Reconstruct Conversations
    threads = reconstruct_conversations(df)

    # 3. Profile Dataset
    profile_data = profile_dataset(df, threads, is_real_dataset)
    with open(os.path.join(processed_dir, "dataset_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
    with open(os.path.join("results", "dataset_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    brand_df = generate_brand_profile_csv(df, threads)
    brand_csv_path = os.path.join(processed_dir, "brand_profile.csv")
    brand_df.to_csv(brand_csv_path, index=False)

    # 4. Brand Selection
    selected_brand = config.get("brand", {}).get("name")
    if not selected_brand or selected_brand == "null":
        if not brand_df.empty:
            selected_brand = brand_df.iloc[0]["brand"]
        else:
            selected_brand = "AmazonHelp"

    logger.info(f"Selected brand for AI Support Agent: '{selected_brand}'")

    brand_selection_meta = {
        "selected_brand": selected_brand,
        "selection_criteria": "Highest resolved interaction volume, deep turn count, and clear multi-turn customer issue taxonomy",
        "candidates_evaluated": brand_df.to_dict(orient="records") if not brand_df.empty else [],
        "is_real_dataset": is_real_dataset,
    }
    with open(os.path.join("results", "brand_selection.json"), "w", encoding="utf-8") as f:
        json.dump(brand_selection_meta, f, indent=2)

    # 5. Filter Brand Threads & Extract Historical Resolved Cases
    brand_threads = [t for t in threads if t.brand == selected_brand or selected_brand in [t.brand, "ALL"]]
    if not brand_threads:
        brand_threads = threads  # Fallback if brand filter yields empty on mock sample

    resolved_cases = []
    for idx, thread in enumerate(brand_threads):
        if thread.is_resolved:
            case_dict = {
                "case_id": f"case_{thread.conversation_id}",
                "conversation_id": thread.conversation_id,
                "brand": thread.brand,
                "customer_message": thread.customer_initial_message,
                "historical_response": thread.support_final_response,
                "turn_count": thread.turn_count,
                "tweets": thread.tweets,
            }
            resolved_cases.append(case_dict)

    # Save resolved cases
    resolved_path = os.path.join(processed_dir, "resolved_cases.jsonl")
    with open(resolved_path, "w", encoding="utf-8") as f:
        for c in resolved_cases:
            f.write(json.dumps(c) + "\n")

    # 6. Conversation-Level Partitioning (Train / Val / Test)
    thread_dicts = [t.to_dict() for t in brand_threads]
    np.random.shuffle(thread_dicts)

    n_total = len(thread_dicts)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_split = thread_dicts[:n_train]
    val_split = thread_dicts[n_train:n_train + n_val]
    test_split = thread_dicts[n_train + n_val:]

    # Write splits
    for name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
        path = os.path.join(processed_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in split:
                f.write(json.dumps(s) + "\n")

    logger.info(f"Splits created: Train={len(train_split)}, Val={len(val_split)}, Test={len(test_split)}")
    print(f"\n--- DATA PREPARATION COMPLETE ---")
    print(f"Dataset Status: {profile_data['dataset_status']}")
    print(f"Selected Brand: {selected_brand}")
    print(f"Total Conversations Processed: {n_total}")
    print(f"Resolved Cases Saved: {len(resolved_cases)}")
    print(f"Artifacts saved to {processed_dir} and results/")

if __name__ == "__main__":
    prepare_pipeline()

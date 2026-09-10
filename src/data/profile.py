import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List
from src.data.conversations import reconstruct_conversations, ConversationThread

logger = logging.getLogger(__name__)

def profile_dataset(df: pd.DataFrame, threads: List[ConversationThread], is_real_dataset: bool) -> Dict[str, Any]:
    """
    Programmatically profile the dataset metrics and save output JSON.
    """
    total_tweets = len(df)
    inbound_tweets = int(df["inbound"].sum())
    outbound_tweets = total_tweets - inbound_tweets
    unique_authors = int(df["author_id"].nunique())

    # Brands (outbound author_ids)
    outbound_df = df[~df["inbound"]]
    brand_counts = outbound_df["author_id"].value_counts().to_dict()
    brands_list = list(brand_counts.keys())

    # Conversation thread profiling
    total_conversations = len(threads)
    resolved_interactions = sum(1 for t in threads if t.is_resolved)
    incomplete_threads = total_conversations - resolved_interactions

    turn_lengths = [t.turn_count for t in threads]
    avg_turn_length = float(pd.Series(turn_lengths).mean()) if turn_lengths else 0.0
    max_turn_length = int(pd.Series(turn_lengths).max()) if turn_lengths else 0

    # Text length stats
    text_lengths = df["text"].fillna("").apply(len)
    avg_text_length = float(text_lengths.mean()) if not text_lengths.empty else 0.0

    # Missing parent references
    has_parent = df["in_response_to_tweet_id"].notna() & (df["in_response_to_tweet_id"] != "")
    missing_parent_count = int((has_parent & (~df["in_response_to_tweet_id"].isin(df["tweet_id"]))).sum())

    # Duplicate tweets
    duplicate_text_count = int(df.duplicated(subset=["text"]).sum())
    duplicate_rate = float(duplicate_text_count / total_tweets) if total_tweets > 0 else 0.0

    profile = {
        "is_real_dataset": is_real_dataset,
        "dataset_status": "REAL DATASET" if is_real_dataset else "TEST SAMPLE (GENUINE DATASET PENDING INGESTION)",
        "total_tweets": total_tweets,
        "inbound_tweets": inbound_tweets,
        "outbound_tweets": outbound_tweets,
        "unique_authors": unique_authors,
        "unique_brands_count": len(brands_list),
        "brands": brand_counts,
        "total_conversations": total_conversations,
        "resolved_interactions": resolved_interactions,
        "incomplete_threads": incomplete_threads,
        "avg_conversation_length": round(avg_turn_length, 2),
        "max_conversation_length": max_turn_length,
        "avg_text_length_chars": round(avg_text_length, 2),
        "missing_parent_references": missing_parent_count,
        "duplicate_text_count": duplicate_text_count,
        "duplicate_rate": round(duplicate_rate, 4),
    }

    return profile

def generate_brand_profile_csv(df: pd.DataFrame, threads: List[ConversationThread]) -> pd.DataFrame:
    """
    Generate brand candidate comparison DataFrame and save to brand_profile.csv.
    """
    outbound_df = df[~df["inbound"]]
    brand_names = outbound_df["author_id"].value_counts().index.tolist()

    brand_stats = []
    for b in brand_names:
        b_outbound = len(df[df["author_id"] == b])
        b_threads = [t for t in threads if t.brand == b]
        b_thread_count = len(b_threads)
        b_resolved = sum(1 for t in b_threads if t.is_resolved)
        b_avg_turns = float(pd.Series([t.turn_count for t in b_threads]).mean()) if b_threads else 0.0

        brand_stats.append({
            "brand": b,
            "outbound_tweets": b_outbound,
            "total_conversations": b_thread_count,
            "resolved_conversations": b_resolved,
            "avg_turn_length": round(b_avg_turns, 2),
        })

    brand_df = pd.DataFrame(brand_stats)
    if not brand_df.empty:
        brand_df.sort_values(by="total_conversations", ascending=False, inplace=True)

    return brand_df

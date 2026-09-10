import pytest
import pandas as pd
from src.data.conversations import reconstruct_conversations, ConversationThread

def test_normal_thread_reconstruction():
    data = [
        {"tweet_id": "1", "author_id": "cust1", "inbound": True, "text": "Help me", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "text": "How can I help?", "response_tweet_id": None, "in_response_to_tweet_id": "1"},
    ]
    df = pd.DataFrame(data)
    threads = reconstruct_conversations(df)
    assert len(threads) == 1
    assert threads[0].turn_count == 2
    assert threads[0].is_resolved is True

def test_cycle_prevention():
    # Cyclical graph 1 -> 2 -> 1 must not cause infinite loop
    data = [
        {"tweet_id": "1", "author_id": "cust1", "inbound": True, "text": "Msg 1", "response_tweet_id": "2", "in_response_to_tweet_id": "2"},
        {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "text": "Msg 2", "response_tweet_id": "1", "in_response_to_tweet_id": "1"},
    ]
    df = pd.DataFrame(data)
    threads = reconstruct_conversations(df)
    assert len(threads) >= 1

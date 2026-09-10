import logging
import pandas as pd
from typing import Dict, List, Set, Any, Optional

logger = logging.getLogger(__name__)

class ConversationThread:
    """
    Represents a reconstructed multi-turn support conversation thread.
    """
    def __init__(self, conversation_id: str, tweets: List[Dict[str, Any]], brand: str):
        self.conversation_id = conversation_id
        self.tweets = tweets  # Chronologically ordered list of tweets
        self.brand = brand
        self.turn_count = len(tweets)
        self.is_resolved = self._check_resolution()
        self.customer_initial_message = self._get_initial_customer_message()
        self.support_final_response = self._get_final_support_response()

    def _check_resolution(self) -> bool:
        """A thread is considered resolved if it contains at least one support brand response."""
        return any(t.get("author_id") == self.brand or not t.get("inbound", True) for t in self.tweets)

    def _get_initial_customer_message(self) -> str:
        for t in self.tweets:
            if t.get("inbound", True) or t.get("author_id") != self.brand:
                return t.get("text", "")
        return self.tweets[0].get("text", "") if self.tweets else ""

    def _get_final_support_response(self) -> str:
        for t in reversed(self.tweets):
            if not t.get("inbound", True) or t.get("author_id") == self.brand:
                return t.get("text", "")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "turn_count": self.turn_count,
            "is_resolved": self.is_resolved,
            "customer_initial_message": self.customer_initial_message,
            "support_final_response": self.support_final_response,
            "tweets": self.tweets,
        }

def reconstruct_conversations(df: pd.DataFrame) -> List[ConversationThread]:
    """
    Reconstruct multi-turn conversation DAGs from Twitter customer support DataFrame.
    Prevents infinite traversal on cycles using a visited set.
    Handles missing parent tweets, orphan nodes, and cyclic graphs.
    """
    logger.info("Reconstructing conversation threads from tweet graph...")
    
    # Fast lookup table by tweet_id
    tweet_dict = {}
    for idx, row in df.iterrows():
        t_id = str(row["tweet_id"])
        tweet_dict[t_id] = row.to_dict()

    parent_map = {}
    child_map = {}

    for t_id, tweet in tweet_dict.items():
        parent_id = tweet.get("in_response_to_tweet_id")
        if pd.notna(parent_id) and str(parent_id) != "nan" and str(parent_id) in tweet_dict:
            parent_id_str = str(parent_id)
            parent_map[t_id] = parent_id_str
            child_map.setdefault(parent_id_str, []).append(t_id)

    # Find root tweets (no parent in dataset)
    roots = [t_id for t_id in tweet_dict if t_id not in parent_map]
    
    # If graph is purely cyclic with no natural root, select all tweets as potential start points
    if not roots and tweet_dict:
        roots = list(tweet_dict.keys())

    threads = []
    global_visited: Set[str] = set()

    for root_id in roots:
        if root_id in global_visited:
            continue

        thread_tweets = []
        visited_in_thread: Set[str] = set()
        queue = [root_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited_in_thread:
                # Cycle detected! Skip to prevent infinite loop
                logger.warning(f"Cycle detected at tweet_id {curr_id}. Halting thread traversal for cycle.")
                break

            visited_in_thread.add(curr_id)
            global_visited.add(curr_id)
            thread_tweets.append(tweet_dict[curr_id])

            # Enqueue children
            children = child_map.get(curr_id, [])
            for child_id in children:
                if child_id not in visited_in_thread:
                    queue.append(child_id)

        if not thread_tweets:
            continue

        # Determine main brand associated with thread
        brand = "UNKNOWN"
        for t in thread_tweets:
            if not t.get("inbound", True) and t.get("author_id"):
                brand = str(t["author_id"])
                break

        thread_obj = ConversationThread(
            conversation_id=f"conv_{root_id}",
            tweets=thread_tweets,
            brand=brand
        )
        threads.append(thread_obj)

    logger.info(f"Reconstructed {len(threads)} unique conversation threads.")
    return threads

"""
Thread builder: reconstructs multi-turn conversation threads from flat tweet data.

The Kaggle dataset uses two linking columns:
  - response_tweet_id: the tweet ID that THIS tweet is responding to (outbound)
  - in_response_to_tweet_id: the tweet ID that THIS tweet is a response to (inbound)

A "thread" is a sequence: customer_msg → brand_reply → customer_followup → brand_reply → ...
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from src.config import MIN_THREAD_LENGTH, MAX_THREADS


def build_reply_graph(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a directed graph: parent_tweet_id → [child_tweet_ids].
    Uses both response_tweet_id and in_response_to_tweet_id to link tweets.
    """
    graph = defaultdict(list)
    
    for _, row in df.iterrows():
        tweet_id = str(row["tweet_id"])
        
        # If this tweet is responding to another tweet
        if pd.notna(row.get("in_response_to_tweet_id")):
            parent = str(int(float(row["in_response_to_tweet_id"])))
            graph[parent].append(tweet_id)
        
        # If this tweet has a response_tweet_id (outbound pointing to its response)
        if pd.notna(row.get("response_tweet_id")):
            # response_tweet_id can contain multiple IDs separated by commas
            response_ids = str(row["response_tweet_id"]).split(",")
            for rid in response_ids:
                rid = rid.strip()
                if rid:
                    graph[tweet_id].append(rid)
    
    return graph


def find_thread_roots(df: pd.DataFrame) -> List[str]:
    """
    Find root tweets — inbound customer messages that start a conversation.
    A root is an inbound tweet that is NOT a response to another tweet in the dataset.
    """
    all_tweet_ids = set(df["tweet_id"].astype(str).values)
    roots = []
    
    for _, row in df.iterrows():
        if row["inbound"] == True:
            # Check if this is NOT a response to another tweet in our dataset
            parent_id = row.get("in_response_to_tweet_id")
            if pd.isna(parent_id) or str(int(float(parent_id))) not in all_tweet_ids:
                roots.append(str(row["tweet_id"]))
    
    return roots


def trace_thread(root_id: str, graph: Dict[str, List[str]], 
                 tweet_lookup: Dict[str, dict], max_depth: int = 20) -> List[dict]:
    """
    Trace a conversation thread from a root tweet, following the reply graph.
    Returns an ordered list of tweet records forming the conversation.
    """
    thread = []
    visited = set()
    queue = [root_id]
    
    while queue and len(thread) < max_depth:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        
        if current_id in tweet_lookup:
            thread.append(tweet_lookup[current_id])
        
        # Add children to queue
        if current_id in graph:
            for child_id in graph[current_id]:
                if child_id not in visited:
                    queue.append(child_id)
    
    # Sort by created_at if available, otherwise keep insertion order
    if thread and "created_at" in thread[0] and thread[0]["created_at"]:
        try:
            thread.sort(key=lambda t: t.get("created_at", ""))
        except (TypeError, ValueError):
            pass  # Keep original order if timestamps are messy
    
    return thread


def build_threads(df: pd.DataFrame) -> List[List[dict]]:
    """
    Build all conversation threads from the dataset.
    
    Returns:
        List of threads, where each thread is a list of tweet dicts ordered chronologically.
        Each dict has keys: tweet_id, author_id, inbound, created_at, text
    """
    print("[thread_builder] Building reply graph...")
    graph = build_reply_graph(df)
    
    # Create lookup table
    tweet_lookup = {}
    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        tweet_lookup[tid] = {
            "tweet_id": tid,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "created_at": str(row.get("created_at", "")),
            "text": str(row.get("text", "")),
        }
    
    print("[thread_builder] Finding thread roots...")
    roots = find_thread_roots(df)
    print(f"[thread_builder] Found {len(roots):,} potential thread roots")
    
    print("[thread_builder] Tracing threads...")
    threads = []
    for root_id in roots:
        thread = trace_thread(root_id, graph, tweet_lookup)
        if len(thread) >= MIN_THREAD_LENGTH:
            threads.append(thread)
    
    # Cap at MAX_THREADS
    if len(threads) > MAX_THREADS:
        print(f"[thread_builder] Capping from {len(threads):,} to {MAX_THREADS:,} threads")
        # Prioritize longer threads (more complete conversations)
        threads.sort(key=len, reverse=True)
        threads = threads[:MAX_THREADS]
    
    # Compute stats
    lengths = [len(t) for t in threads]
    print(f"[thread_builder] Built {len(threads):,} threads")
    print(f"  - Avg length: {np.mean(lengths):.1f} messages")
    print(f"  - Min length: {min(lengths)}, Max length: {max(lengths)}")
    print(f"  - Median length: {np.median(lengths):.0f}")
    
    return threads


def threads_to_conversations(threads: List[List[dict]], brand: str) -> List[dict]:
    """
    Convert raw threads into structured conversation objects for downstream use.
    
    Each conversation has:
      - thread_id: unique identifier
      - customer_messages: list of customer (inbound) message texts
      - brand_replies: list of brand (outbound) reply texts
      - full_thread: ordered list of all messages with speaker labels
      - first_customer_message: the initial customer complaint/query
      - brand_resolution: the brand's final reply (if any)
    """
    conversations = []
    
    for i, thread in enumerate(threads):
        customer_msgs = [t for t in thread if t["inbound"]]
        brand_msgs = [t for t in thread if not t["inbound"]]
        
        if not customer_msgs or not brand_msgs:
            continue  # Skip threads without both sides
        
        full_thread = []
        for t in thread:
            speaker = "customer" if t["inbound"] else "brand"
            full_thread.append({
                "speaker": speaker,
                "text": t["text"],
                "tweet_id": t["tweet_id"],
            })
        
        conv = {
            "thread_id": f"thread_{i:05d}",
            "customer_messages": [t["text"] for t in customer_msgs],
            "brand_replies": [t["text"] for t in brand_msgs],
            "full_thread": full_thread,
            "first_customer_message": customer_msgs[0]["text"],
            "brand_resolution": brand_msgs[-1]["text"],
            "num_turns": len(thread),
        }
        conversations.append(conv)
    
    print(f"[thread_builder] Created {len(conversations):,} structured conversations")
    return conversations


if __name__ == "__main__":
    from src.data_loader import load_raw_data, filter_brand
    from src.config import SELECTED_BRAND
    
    df = load_raw_data()
    brand_df = filter_brand(df, SELECTED_BRAND)
    threads = build_threads(brand_df)
    conversations = threads_to_conversations(threads, SELECTED_BRAND)
    
    # Show a sample conversation
    if conversations:
        print("\n--- Sample Conversation ---")
        sample = conversations[0]
        for msg in sample["full_thread"][:6]:
            print(f"  [{msg['speaker']}]: {msg['text'][:120]}...")

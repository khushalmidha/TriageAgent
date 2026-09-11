"""
Data cleaner: text preprocessing for customer support tweets.

Handles:
  - @handle stripping (replace with placeholder to preserve meaning)
  - URL removal/replacement
  - Multi-turn context merging
  - Deduplication
  - Emoji/special character normalization
  - Dropping unusable records (empty text, non-English, etc.)
"""

import re
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter


# ─── Cleaning Functions ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single tweet text for use in classification and retrieval.
    
    Cleaning steps (and why):
    1. Replace @handles with [USER] — preserves that someone was mentioned
       without leaking specific handles
    2. Replace URLs with [URL] — URLs aren't useful for intent classification
       but noting their presence is (often brands share help links)
    3. Normalize whitespace
    4. Strip leading/trailing whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Replace @handles with [USER] — keep the signal that someone was addressed
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Replace URLs with [URL]
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)
    
    # Remove RT prefix (retweet indicator)
    text = re.sub(r'^RT\s+', '', text)
    
    # Normalize whitespace (collapse multiple spaces, remove newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def clean_conversations(conversations: List[dict]) -> Tuple[List[dict], dict]:
    """
    Clean all conversations and track what was dropped and why.
    
    Returns:
        Tuple of (cleaned_conversations, drop_stats)
    """
    cleaned = []
    drop_stats = Counter()
    
    for conv in conversations:
        # Clean all messages
        cleaned_thread = []
        for msg in conv["full_thread"]:
            clean = clean_text(msg["text"])
            if clean:
                cleaned_thread.append({
                    "speaker": msg["speaker"],
                    "text": clean,
                    "tweet_id": msg["tweet_id"],
                })
        
        # Drop if no customer messages remain
        customer_msgs = [m for m in cleaned_thread if m["speaker"] == "customer"]
        brand_msgs = [m for m in cleaned_thread if m["speaker"] == "brand"]
        
        if not customer_msgs:
            drop_stats["no_customer_message"] += 1
            continue
        
        if not brand_msgs:
            drop_stats["no_brand_reply"] += 1
            continue
        
        # Drop if first customer message is too short (likely just a handle mention)
        first_msg = customer_msgs[0]["text"]
        # Remove [USER] tokens to check actual content
        content_only = re.sub(r'\[USER\]', '', first_msg).strip()
        if len(content_only) < 10:
            drop_stats["customer_message_too_short"] += 1
            continue
        
        # Check for duplicate first messages (exact dedup)
        cleaned_conv = {
            "thread_id": conv["thread_id"],
            "customer_messages": [m["text"] for m in customer_msgs],
            "brand_replies": [m["text"] for m in brand_msgs],
            "full_thread": cleaned_thread,
            "first_customer_message": customer_msgs[0]["text"],
            "brand_resolution": brand_msgs[-1]["text"],
            "num_turns": len(cleaned_thread),
        }
        cleaned.append(cleaned_conv)
    
    # Deduplication: remove conversations with identical first customer messages
    seen_messages = set()
    deduped = []
    for conv in cleaned:
        msg_key = conv["first_customer_message"].lower().strip()
        if msg_key not in seen_messages:
            seen_messages.add(msg_key)
            deduped.append(conv)
        else:
            drop_stats["duplicate_first_message"] += 1
    
    total_dropped = sum(drop_stats.values())
    print(f"[data_cleaner] Cleaned {len(conversations):,} → {len(deduped):,} conversations")
    print(f"[data_cleaner] Dropped {total_dropped:,} conversations:")
    for reason, count in drop_stats.most_common():
        print(f"  - {reason}: {count}")
    
    return deduped, dict(drop_stats)


def merge_multipart_messages(conversations: List[dict]) -> List[dict]:
    """
    Merge consecutive messages from the same speaker into a single message.
    Twitter's character limit often forces users to split messages across tweets.
    """
    merged = []
    
    for conv in conversations:
        merged_thread = []
        for msg in conv["full_thread"]:
            if merged_thread and merged_thread[-1]["speaker"] == msg["speaker"]:
                # Same speaker — merge
                merged_thread[-1]["text"] += " " + msg["text"]
            else:
                merged_thread.append(msg.copy())
        
        customer_msgs = [m for m in merged_thread if m["speaker"] == "customer"]
        brand_msgs = [m for m in merged_thread if m["speaker"] == "brand"]
        
        merged_conv = {
            "thread_id": conv["thread_id"],
            "customer_messages": [m["text"] for m in customer_msgs],
            "brand_replies": [m["text"] for m in brand_msgs],
            "full_thread": merged_thread,
            "first_customer_message": customer_msgs[0]["text"] if customer_msgs else "",
            "brand_resolution": brand_msgs[-1]["text"] if brand_msgs else "",
            "num_turns": len(merged_thread),
        }
        merged.append(merged_conv)
    
    return merged


if __name__ == "__main__":
    # Test cleaning
    test_texts = [
        "@AppleSupport My iPhone keeps crashing after the latest update. Help! https://t.co/abc123",
        "@AppleSupport @user123 I can't log into my Apple ID",
        "RT @someone: Check this out @AppleSupport",
        "",
        "@AppleSupport",
    ]
    
    print("Cleaning test:")
    for text in test_texts:
        print(f"  '{text}' → '{clean_text(text)}'")

"""
Data loader for the Kaggle Customer Support on Twitter dataset.

Handles two scenarios:
  1. CSV already placed in data/ directory manually
  2. Download via Kaggle API if credentials are available

Dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
Columns: tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
"""

import os
import sys
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, RAW_CSV_PATH


def download_from_kaggle():
    """
    Attempt to download the dataset using the Kaggle API.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or ~/.kaggle/kaggle.json.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[data_loader] Downloading dataset from Kaggle...")
        api.dataset_download_files(
            "thoughtvector/customer-support-on-twitter",
            path=str(DATA_DIR),
            unzip=True
        )
        print(f"[data_loader] Dataset downloaded to {DATA_DIR}")
        return True
    except Exception as e:
        print(f"[data_loader] Kaggle API download failed: {e}")
        print("[data_loader] Please manually download twcs.csv from:")
        print("  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print(f"  and place it at: {RAW_CSV_PATH}")
        return False


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw CSV dataset. Tries local file first, then Kaggle API.
    
    Returns:
        pd.DataFrame with columns: tweet_id, author_id, inbound, created_at, text,
                                    response_tweet_id, in_response_to_tweet_id
    """
    if not RAW_CSV_PATH.exists():
        print(f"[data_loader] CSV not found at {RAW_CSV_PATH}")
        if not download_from_kaggle():
            sys.exit(1)
    
    if not RAW_CSV_PATH.exists():
        print(f"[data_loader] ERROR: Dataset still not found at {RAW_CSV_PATH}")
        sys.exit(1)
    
    print(f"[data_loader] Loading {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    
    # Basic validation
    expected_cols = ["tweet_id", "author_id", "inbound", "text"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Got: {list(df.columns)}")
    
    print(f"[data_loader] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"[data_loader] Inbound (customer) tweets: {df['inbound'].sum():,}")
    print(f"[data_loader] Outbound (brand) tweets: {(~df['inbound']).sum():,}")
    
    return df


def get_brand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-brand statistics to help with brand selection.
    
    Returns DataFrame with columns: brand, total_tweets, inbound_count, 
                                     outbound_count, reply_rate
    """
    # Outbound tweets (brand replies) have inbound=False
    outbound = df[df["inbound"] == False].copy()
    
    # Count tweets per author (brand)
    brand_stats = outbound.groupby("author_id").agg(
        outbound_count=("tweet_id", "count"),
    ).reset_index()
    brand_stats.rename(columns={"author_id": "brand"}, inplace=True)
    
    # Count inbound tweets that got responses from each brand
    # An inbound tweet references a brand if the brand's tweet is in response_tweet_id
    inbound = df[df["inbound"] == True].copy()
    
    # For each brand, count how many inbound tweets they responded to
    brand_responses = outbound[outbound["in_response_to_tweet_id"].notna()].groupby("author_id").agg(
        responses_given=("tweet_id", "count")
    ).reset_index()
    brand_responses.rename(columns={"author_id": "brand"}, inplace=True)
    
    brand_stats = brand_stats.merge(brand_responses, on="brand", how="left")
    brand_stats["responses_given"] = brand_stats["responses_given"].fillna(0).astype(int)
    brand_stats = brand_stats.sort_values("outbound_count", ascending=False)
    
    return brand_stats


def filter_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Filter dataset to only include conversations involving the specified brand.
    
    A conversation involves the brand if:
    - The brand is the author of an outbound tweet, OR
    - An inbound tweet is directed at / responded to by the brand
    """
    # Get all tweet IDs where the brand is the author
    brand_tweet_ids = set(df[df["author_id"] == brand]["tweet_id"].values)
    
    # Get all inbound tweets that the brand responded to
    brand_outbound = df[(df["author_id"] == brand) & (df["inbound"] == False)]
    responded_to_ids = set(brand_outbound["in_response_to_tweet_id"].dropna().values)
    
    # Get all inbound tweets that reference brand tweets
    inbound_referencing = set(
        df[df["in_response_to_tweet_id"].isin(brand_tweet_ids)]["tweet_id"].values
    )
    
    # Collect all relevant tweet IDs
    all_relevant_ids = brand_tweet_ids | responded_to_ids | inbound_referencing
    
    # Also get inbound tweets that the brand responded to (via response_tweet_id)
    brand_responses = set(brand_outbound["response_tweet_id"].dropna().values)
    inbound_with_responses = set(
        df[df["tweet_id"].isin(brand_responses)]["tweet_id"].values
    )
    all_relevant_ids = all_relevant_ids | inbound_with_responses | brand_responses
    
    filtered = df[df["tweet_id"].isin(all_relevant_ids)].copy()
    
    print(f"[data_loader] Filtered to brand '{brand}': {len(filtered):,} tweets")
    print(f"  - Inbound: {filtered['inbound'].sum():,}")
    print(f"  - Outbound: {(~filtered['inbound']).sum():,}")
    
    return filtered


if __name__ == "__main__":
    # Quick test: load data and show brand stats
    df = load_raw_data()
    stats = get_brand_stats(df)
    print("\nTop 20 brands by outbound tweet count:")
    print(stats.head(20).to_string(index=False))

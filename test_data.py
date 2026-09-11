"""Quick data verification before pipeline run."""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/twcs.csv")
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Inbound: {df['inbound'].sum():,}")
print(f"Outbound: {(~df['inbound']).sum():,}")

# Brand stats
outbound = df[df["inbound"] == False]
brand_counts = outbound["author_id"].value_counts().head(20)
print("\nTop 20 brands by outbound tweets:")
for brand, count in brand_counts.items():
    print(f"  {brand:30s} {count:>8,}")

# Check AppleSupport specifically
apple = df[df["author_id"] == "AppleSupport"]
print(f"\nAppleSupport tweets: {len(apple):,}")
print(f"  Inbound to Apple: {apple['inbound'].sum():,}")
print(f"  Outbound (replies): {(~apple['inbound']).sum():,}")

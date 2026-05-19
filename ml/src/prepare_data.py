"""
PropWise — Data Preparation
============================
Reads raw Mumbai property dataset, cleans it, and saves a processed CSV
ready for ML training.

Expected input files in ml/data/raw/:
  - Mumbai1.csv (primary dataset with amenities)
  - House_Rent_Dataset.csv (optional, for rent context)

Output: ml/data/processed/mumbai_clean.csv

Run: python ml/src/prepare_data.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"


def normalize_location(value):
    """Standardize Mumbai neighborhood names."""
    if pd.isna(value):
        return "unknown"
    s = str(value).lower().strip()
    s = re.sub(r"\s*,?\s*mumbai\s*$", "", s)
    s = re.sub(r"\s*\(.*?\)", "", s)
    # Replace spaces and special chars with underscores
    s = re.sub(r"[\s/\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def load_mumbai_dataset():
    """Load and clean the Mumbai1.csv dataset."""
    csv_path = RAW_DIR / "Mumbai1.csv"
    if not csv_path.exists():
        print(f"❌ Mumbai1.csv not found at {csv_path}")
        return None

    print(f"📂 Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Columns: {list(df.columns)}")

    # The first column appears to be an unnamed index
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])

    # Rename columns to match our schema
    rename_map = {
        "Price": "price",
        "Area": "area_sqft",
        "Location": "location",
        "No. of Bedrooms": "bhk",
        "New/Resale": "is_resale",
        "Gymnasium": "amenity_gym",
        "Lift Available": "lift",
        "Car Parking": "parking",
        "Maintenance Staff": "amenity_maintenance",
        "24x7 Security": "amenity_security",
        "Children's Play Area": "amenity_kids_play",
        "Clubhouse": "amenity_clubhouse",
        "Intercom": "amenity_intercom",
        "Landscaped Gardens": "amenity_gardens",
        "Indoor Games": "amenity_indoor_games",
        "Gas Connection": "amenity_gas",
        "Jogging Track": "amenity_jogging",
        "Swimming Pool": "amenity_pool",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Normalize location
    df["location"] = df["location"].apply(normalize_location)

    # Convert numeric columns
    for col in ["price", "area_sqft", "bhk"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=["price", "area_sqft", "location", "bhk"])
    print(f"🧹 Dropped {before - len(df)} rows with missing critical fields")

    # Remove obvious outliers
    df = df[(df["price"] >= 1e6) & (df["price"] <= 2e9)]   # ₹10L to ₹200Cr
    df = df[(df["area_sqft"] >= 100) & (df["area_sqft"] <= 20000)]
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 10)]

    # Compute price per sqft + remove outliers
    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    df = df[(df["price_per_sqft"] >= 3000) & (df["price_per_sqft"] <= 250000)]

    # Count amenities (engineered feature)
    amenity_cols = [c for c in df.columns if c.startswith("amenity_")]
    if amenity_cols:
        # Make sure they're 0/1
        for c in amenity_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        df["amenity_count"] = df[amenity_cols].sum(axis=1)

    # is_resale is a useful signal
    if "is_resale" in df.columns:
        df["is_resale"] = pd.to_numeric(df["is_resale"], errors="coerce").fillna(0).astype(int)

    # Lift and parking already exist as 0/1
    for col in ["lift", "parking"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    print(f"✅ Cleaned data: {len(df)} rows")
    print(f"   Unique locations: {df['location'].nunique()}")
    print(f"   Top 10 locations by count:")
    for loc, n in df["location"].value_counts().head(10).items():
        print(f"      {loc:30s} {n}")
    print(f"\n   Price range: ₹{df['price'].min():,.0f} – ₹{df['price'].max():,.0f}")
    print(f"   Median price: ₹{df['price'].median():,.0f}")
    print(f"   Median price/sqft: ₹{df['price_per_sqft'].median():,.0f}")

    return df


def main():
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = load_mumbai_dataset()
    if df is None:
        print("\n💡 Place Mumbai1.csv in:", RAW_DIR)
        return

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n✅ Saved cleaned data → {PROCESSED_PATH}")


if __name__ == "__main__":
    main()

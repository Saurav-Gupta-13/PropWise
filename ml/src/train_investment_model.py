"""
PropWise — Investment Score Model
==================================
Trains a classifier that rates a property's investment potential (0-100).

Investment score logic (derived from data):
  - High score = price is below market median for that area + BHK + amenity profile
  - Low score = price is above market median (overpriced)

This is a FAIR VALUE detector. If a 2BHK in Andheri is listed cheaper than
similar 2BHKs in Andheri, it gets a high investment score.

Output: ml/models/investment_model.joblib
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_investment_label(df):
    """
    Classify each property into 4 investment tiers based on its
    price-per-sqft RELATIVE to similar properties in the same area+BHK group.

    Tiers:
      0 = Overpriced (price > 1.15× group median)
      1 = Fair value (within ±15% of group median)
      2 = Good deal (price 0.85× to 0.95× of group median)
      3 = Excellent deal (price < 0.85× of group median)
    """
    # Group by location + BHK to find local median price/sqft
    df["group_median_psf"] = df.groupby(["location", "bhk"])["price_per_sqft"].transform("median")
    df["price_ratio"] = df["price_per_sqft"] / df["group_median_psf"]

    def label(ratio):
        if ratio > 1.15:
            return 0  # Overpriced
        elif ratio > 0.95:
            return 1  # Fair
        elif ratio > 0.85:
            return 2  # Good deal
        else:
            return 3  # Excellent deal

    df["investment_label"] = df["price_ratio"].apply(label)
    return df


def main():
    if not DATA_PATH.exists():
        print(f"❌ Cleaned data not found at {DATA_PATH}")
        return

    print("📂 Loading cleaned data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df)}")

    # Compute investment labels from data
    df = compute_investment_label(df)
    print("\n📊 Investment label distribution:")
    label_names = {0: "Overpriced", 1: "Fair", 2: "Good deal", 3: "Excellent deal"}
    for lbl, name in label_names.items():
        count = (df["investment_label"] == lbl).sum()
        pct = count / len(df) * 100
        print(f"   {name:18s} {count:5d}  ({pct:.1f}%)")

    # Features (same as price model + price itself)
    numeric_features = ["bhk", "area_sqft", "price"]
    binary_features = []
    for col in df.columns:
        if col in ["lift", "parking", "is_resale", "amenity_count"]:
            binary_features.append(col)
        elif col.startswith("amenity_"):
            binary_features.append(col)
    numeric_features += binary_features

    categorical_features = ["location"]

    # Encode categoricals
    encoders = {}
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc

    feature_cols = numeric_features + [c for c in categorical_features if c in df.columns]
    print(f"\n📋 Features: {feature_cols}")

    X = df[feature_cols]
    y = df["investment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🧠 Training XGBoost classifier on {len(X_train)} samples...")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 Investment Model Evaluation:")
    print(f"   Accuracy: {accuracy * 100:.1f}%")
    print(f"\n   Classification report:")
    target_names = ["Overpriced", "Fair", "Good deal", "Excellent deal"]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Save
    joblib.dump(model, MODELS_DIR / "investment_model.joblib")
    joblib.dump(encoders, MODELS_DIR / "investment_encoders.joblib")
    joblib.dump({
        "feature_cols": feature_cols,
        "label_names": label_names,
        "metrics": {"accuracy": float(accuracy * 100)},
    }, MODELS_DIR / "investment_metadata.joblib")

    print(f"\n✅ Saved investment model → {MODELS_DIR / 'investment_model.joblib'}")


if __name__ == "__main__":
    main()

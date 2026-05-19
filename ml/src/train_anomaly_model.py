"""
PropWise — Anomaly Detection Model
====================================
Trains an Isolation Forest to detect suspicious/fraudulent listings.

Anomalies = listings whose features don't match their price profile.
Examples:
  - "5 BHK with 200 sqft area" (impossible)
  - "₹50L for Bandra West 3BHK" (price way too low → likely fake)
  - "₹10 Cr for Kurla 1BHK" (price way too high)

Output: ml/models/anomaly_model.joblib
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"❌ Cleaned data not found at {DATA_PATH}")
        return

    print("📂 Loading cleaned data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df)}")

    # Features for anomaly detection
    numeric_features = ["bhk", "area_sqft", "price", "price_per_sqft"]
    for col in df.columns:
        if col in ["lift", "parking", "is_resale", "amenity_count"]:
            numeric_features.append(col)

    categorical_features = ["location"]
    encoders = {}
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc

    feature_cols = numeric_features + [c for c in categorical_features if c in df.columns]
    print(f"📋 Features: {feature_cols}")

    X = df[feature_cols]

    # Scale features (Isolation Forest works better with scaled data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n🧠 Training Isolation Forest on {len(X)} samples...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # Assume 5% of training data are anomalies
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Score the training data to find anomalies
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)
    n_anomalies = (predictions == -1).sum()

    print(f"\n📊 Anomaly Detection Results:")
    print(f"   Total samples:     {len(X)}")
    print(f"   Anomalies found:   {n_anomalies} ({n_anomalies/len(X)*100:.1f}%)")
    print(f"   Score range:       {scores.min():.3f} to {scores.max():.3f}")

    # Show some example anomalies
    df_with_scores = df.copy()
    df_with_scores["anomaly_score"] = scores
    df_with_scores["is_anomaly"] = predictions == -1

    anomalies = df_with_scores[df_with_scores["is_anomaly"]].head(5)
    print(f"\n🚨 Example anomalies detected:")
    for _, row in anomalies.iterrows():
        loc = encoders["location"].inverse_transform([int(row["location"])])[0] if "location" in encoders else "?"
        print(f"   {loc:20s} BHK={int(row['bhk'])} {row['area_sqft']:.0f}sqft "
              f"₹{row['price']/1e7:.2f}Cr  score={row['anomaly_score']:.3f}")

    # Save artifacts
    joblib.dump(model, MODELS_DIR / "anomaly_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "anomaly_scaler.joblib")
    joblib.dump(encoders, MODELS_DIR / "anomaly_encoders.joblib")
    joblib.dump({
        "feature_cols": feature_cols,
        "anomaly_threshold": 0,  # negative score = anomaly
        "metrics": {"anomaly_rate": float(n_anomalies / len(X) * 100)},
    }, MODELS_DIR / "anomaly_metadata.joblib")

    print(f"\n✅ Saved anomaly model → {MODELS_DIR / 'anomaly_model.joblib'}")


if __name__ == "__main__":
    main()

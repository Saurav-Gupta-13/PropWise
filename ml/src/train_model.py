"""
PropWise — Model Training
==========================
Trains XGBoost regressor on cleaned Mumbai property data.
Uses log-transformed price + amenity features.

Input:  ml/data/processed/mumbai_clean.csv
Output: ml/models/price_model.joblib
        ml/models/encoders.joblib
        ml/models/feature_metadata.joblib

Run: python ml/src/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import xgboost as xgb

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def encode_categorical(df, columns):
    """Label-encode categorical columns. Returns df + dict of encoders."""
    encoders = {}
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("unknown").astype(str)
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc
    return df, encoders


def main():
    if not DATA_PATH.exists():
        print(f"❌ Cleaned data not found at {DATA_PATH}")
        print("Run: python ml/src/prepare_data.py")
        return

    print("📂 Loading cleaned data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df)}")

    # Define features
    numeric_features = ["bhk", "area_sqft"]
    
    # Add all 0/1 binary features that exist (lift, parking, amenities, is_resale)
    binary_features = []
    for col in df.columns:
        if col in ["lift", "parking", "is_resale", "amenity_count"]:
            binary_features.append(col)
        elif col.startswith("amenity_"):
            binary_features.append(col)
    
    numeric_features += binary_features

    categorical_features = ["location"]

    # Encode categoricals
    df, encoders = encode_categorical(df, categorical_features)
    feature_cols = numeric_features + [c for c in categorical_features if c in df.columns]

    print(f"📋 Features ({len(feature_cols)}):")
    print(f"   Numeric/Binary: {numeric_features}")
    print(f"   Categorical:    {[c for c in categorical_features if c in df.columns]}")

    X = df[feature_cols]
    y = df["price"]

    # Log-transform target
    y_log = np.log1p(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print(f"\n🧠 Training XGBoost on {len(X_train)} samples...")

    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # Metrics
    r2 = r2_score(y_test_actual, y_pred)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    mae = mean_absolute_error(y_test_actual, y_pred)

    print("\n📊 Evaluation on held-out test set:")
    print(f"   Test samples:       {len(X_test)}")
    print(f"   R² Score:           {r2:.4f}")
    print(f"   MAPE:               {mape:.2f}%  (lower is better)")
    print(f"   Mean Abs Error:     ₹{mae:,.0f}")
    print(f"   Accuracy estimate:  {(100 - mape):.1f}%")

    # Feature importance
    print("\n🎯 Top 15 features by importance:")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importance.head(15).items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:25s} {imp:.4f}  {bar}")

    # Save artifacts
    joblib.dump(model, MODELS_DIR / "price_model.joblib")
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump({
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": [c for c in categorical_features if c in df.columns],
        "metrics": {
            "r2": float(r2),
            "mape": float(mape),
            "mae": float(mae),
            "accuracy": float(100 - mape),
            "test_samples": int(len(X_test)),
            "train_samples": int(len(X_train)),
        },
    }, MODELS_DIR / "feature_metadata.joblib")

    print(f"\n✅ Saved model → {MODELS_DIR / 'price_model.joblib'}")
    print(f"✅ Saved encoders → {MODELS_DIR / 'encoders.joblib'}")
    print(f"✅ Saved metadata → {MODELS_DIR / 'feature_metadata.joblib'}")
    print(f"\n🚀 Next: python ml/service/main.py (or use uvicorn)")


if __name__ == "__main__":
    main()

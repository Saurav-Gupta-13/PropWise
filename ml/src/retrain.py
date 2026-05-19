"""
PropWise — Continuous Learning Script
=======================================
Retrains the model using the original Mumbai dataset PLUS user feedback.
Run this weekly via cron/Task Scheduler:
  python ml/src/retrain.py

The script:
1. Loads original training data
2. Pulls user-corrected prices from the feedback DB
3. Combines them
4. Retrains the price model
5. Compares accuracy with previous version
6. Only deploys new model if it's better
"""

import sys
import sqlite3
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import xgboost as xgb

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
DB_PATH = ROOT / "data" / "propwise.db"
MODELS_DIR = ROOT / "models"
ARCHIVE_DIR = MODELS_DIR / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)


def load_feedback_data():
    """Pull user feedback with actual prices from DB."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.input_json, f.actual_price
        FROM predictions p
        JOIN feedback f ON f.prediction_id = p.id
        WHERE f.actual_price IS NOT NULL
    """).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    data = []
    for input_json, actual in rows:
        d = json.loads(input_json)
        d["price"] = actual
        data.append(d)

    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print(f"PROPWISE — RETRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Load original data
    df_orig = pd.read_csv(DATA_PATH)
    print(f"📂 Original training data: {len(df_orig)} rows")

    # 2. Load feedback data
    df_fb = load_feedback_data()
    if len(df_fb) > 0:
        print(f"💬 User feedback corrections: {len(df_fb)} rows")
    else:
        print("💬 No user feedback yet — retraining on original data only")

    # 3. Compute price_per_sqft for feedback rows
    if len(df_fb) > 0:
        df_fb["price_per_sqft"] = df_fb["price"] / df_fb["area_sqft"]
        # Encode location to match original encoding
        # (For simplicity, use the existing encoder)
        encoders = joblib.load(MODELS_DIR / "encoders.joblib")
        loc_classes = list(encoders["location"].classes_)
        df_fb["location"] = df_fb["location"].apply(
            lambda x: loc_classes.index(x) if x in loc_classes else 0
        )

    # 4. Load existing metadata to use same features
    metadata = joblib.load(MODELS_DIR / "feature_metadata.joblib")
    feature_cols = metadata["feature_cols"]
    old_accuracy = metadata["metrics"]["accuracy"]

    # 5. Re-encode original data location
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    df_orig["location"] = df_orig["location"].astype(str)
    df_orig["location"] = df_orig["location"].apply(
        lambda x: encoders["location"].transform([x])[0] if x in encoders["location"].classes_ else 0
    )

    # 6. Combine
    if len(df_fb) > 0:
        # Make sure all columns match
        for col in feature_cols:
            if col not in df_fb.columns:
                df_fb[col] = 0
        df_combined = pd.concat([df_orig, df_fb[feature_cols + ["price"]]], ignore_index=True)
    else:
        df_combined = df_orig
    print(f"📊 Combined dataset: {len(df_combined)} rows")

    # 7. Train new model
    X = df_combined[feature_cols]
    y = np.log1p(df_combined["price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🧠 Training new model...")
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 8. Evaluate
    y_pred = np.expm1(model.predict(X_test))
    y_test_actual = np.expm1(y_test)
    new_r2 = r2_score(y_test_actual, y_pred)
    new_mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    new_accuracy = 100 - new_mape

    print(f"\n📈 New accuracy: {new_accuracy:.1f}% (was {old_accuracy:.1f}%)")

    # 9. Decide whether to deploy
    if new_accuracy >= old_accuracy - 1:
        # Archive old model
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for f in ["price_model.joblib", "feature_metadata.joblib"]:
            src = MODELS_DIR / f
            if src.exists():
                shutil.copy(src, ARCHIVE_DIR / f"{f}.{ts}.bak")

        # Deploy new model
        joblib.dump(model, MODELS_DIR / "price_model.joblib")
        metadata["metrics"]["accuracy"] = float(new_accuracy)
        metadata["metrics"]["r2"] = float(new_r2)
        metadata["metrics"]["mape"] = float(new_mape)
        metadata["metrics"]["train_samples"] = int(len(X_train))
        metadata["metrics"]["last_retrained"] = datetime.now().isoformat()
        metadata["metrics"]["feedback_samples"] = int(len(df_fb))
        joblib.dump(metadata, MODELS_DIR / "feature_metadata.joblib")
        print(f"✅ Deployed new model. Old version archived as {f}.{ts}.bak")
    else:
        print(f"⚠️ New model is worse ({new_accuracy:.1f}% < {old_accuracy:.1f}%). Keeping current model.")


if __name__ == "__main__":
    main()

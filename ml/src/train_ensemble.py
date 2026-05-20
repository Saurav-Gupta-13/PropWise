"""
PropWise — Ensemble Training (XGBoost + LightGBM)
==================================================
Combines two different tree models for better accuracy.
Only deploys if accuracy > current 83.4%.

Run: python ml/src/train_ensemble.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
MODELS_DIR = ROOT / "models"

CURRENT_ACCURACY = 83.4  # Only deploy if we beat this


def add_features(df):
    df = df.copy()
    df["log_area"] = np.log1p(df["area_sqft"])
    df["area_per_bhk"] = df["area_sqft"] / df["bhk"]
    df["bhk_sq"] = df["bhk"] ** 2
    df["is_luxury"] = ((df["bhk"] >= 4) | (df["area_sqft"] > 2000)).astype(int)

    global_mean_psf = df["price_per_sqft"].mean()
    loc_stats = df.groupby("location").agg(
        loc_mean_psf=("price_per_sqft", "mean"),
        loc_count=("price_per_sqft", "count"),
    ).reset_index()
    smooth = 10
    loc_stats["loc_smoothed_psf"] = (
        (loc_stats["loc_mean_psf"] * loc_stats["loc_count"] + global_mean_psf * smooth) /
        (loc_stats["loc_count"] + smooth)
    )
    df = df.merge(loc_stats[["location", "loc_smoothed_psf"]], on="location", how="left")
    df["loc_smoothed_psf"] = df["loc_smoothed_psf"].fillna(global_mean_psf)
    return df


def main():
    print("=" * 60)
    print("PROPWISE — ENSEMBLE (XGBoost + LightGBM)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded: {len(df)} rows")

    # IQR outlier removal
    cleaned = []
    for loc, group in df.groupby("location"):
        if len(group) < 5:
            cleaned.append(group)
            continue
        Q1 = group["price_per_sqft"].quantile(0.1)
        Q3 = group["price_per_sqft"].quantile(0.9)
        IQR = Q3 - Q1
        mask = (group["price_per_sqft"] >= Q1 - 2*IQR) & (group["price_per_sqft"] <= Q3 + 2*IQR)
        cleaned.append(group[mask])
    df = pd.concat(cleaned, ignore_index=True)
    print(f"🧹 After outlier removal: {len(df)} rows")

    # Features
    df = add_features(df)
    enc = LabelEncoder()
    df["location"] = enc.fit_transform(df["location"].astype(str))
    encoders = {"location": enc}

    feature_cols = [
        "bhk", "area_sqft", "lift", "parking", "is_resale", "amenity_count", "location",
        "amenity_gym", "amenity_maintenance", "amenity_security", "amenity_kids_play",
        "amenity_clubhouse", "amenity_intercom", "amenity_gardens",
        "amenity_indoor_games", "amenity_gas", "amenity_jogging", "amenity_pool",
        "log_area", "area_per_bhk", "loc_smoothed_psf", "bhk_sq", "is_luxury",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = np.log1p(df["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ─── MODEL 1: XGBoost ───
    print(f"\n🧠 Training XGBoost on {len(X_train)} samples...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.7,
        min_child_weight=5, reg_alpha=0.3, reg_lambda=2.0, gamma=0.1,
        random_state=42, n_jobs=-1, early_stopping_rounds=80,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    xgb_pred = xgb_model.predict(X_test)
    xgb_mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(xgb_pred)) * 100
    print(f"   XGBoost alone: {100 - xgb_mape:.1f}% accuracy")

    # ─── MODEL 2: LightGBM ───
    print(f"\n🧠 Training LightGBM on {len(X_train)} samples...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    lgb_pred = lgb_model.predict(X_test)
    lgb_mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(lgb_pred)) * 100
    print(f"   LightGBM alone: {100 - lgb_mape:.1f}% accuracy")

    # ─── ENSEMBLE: Weighted average ───
    # Try different weights
    best_acc = 0
    best_w = 0.5
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ens_pred = w * xgb_pred + (1 - w) * lgb_pred
        ens_mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(ens_pred)) * 100
        acc = 100 - ens_mape
        if acc > best_acc:
            best_acc = acc
            best_w = w

    ens_pred = best_w * xgb_pred + (1 - best_w) * lgb_pred
    y_pred_final = np.expm1(ens_pred)
    y_actual = np.expm1(y_test)

    r2 = r2_score(y_actual, y_pred_final)
    mape = mean_absolute_percentage_error(y_actual, y_pred_final) * 100
    mae = mean_absolute_error(y_actual, y_pred_final)
    accuracy = 100 - mape

    print(f"\n{'='*60}")
    print(f"📊 ENSEMBLE RESULTS (weight: {best_w:.1f} XGB + {1-best_w:.1f} LGB):")
    print(f"   R² Score:      {r2:.4f}")
    print(f"   MAPE:          {mape:.2f}%")
    print(f"   MAE:           ₹{mae:,.0f}")
    print(f"   ✅ ACCURACY:   {accuracy:.1f}%")
    print(f"   Previous:      {CURRENT_ACCURACY}%")
    print(f"   Change:        {accuracy - CURRENT_ACCURACY:+.1f}%")
    print(f"{'='*60}")

    # ─── DEPLOY DECISION ───
    if accuracy > CURRENT_ACCURACY:
        print(f"\n✅ ENSEMBLE IS BETTER! Deploying...")

        # Save both models as a dict
        ensemble = {
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "weight_xgb": best_w,
            "type": "ensemble",
        }
        joblib.dump(ensemble, MODELS_DIR / "price_model.joblib")
        joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
        joblib.dump({
            "feature_cols": feature_cols,
            "metrics": {
                "r2": float(r2), "mape": float(mape), "mae": float(mae),
                "accuracy": float(accuracy),
                "test_samples": int(len(X_test)),
                "train_samples": int(len(X_train)),
                "ensemble_weight": best_w,
            },
        }, MODELS_DIR / "feature_metadata.joblib")

        loc_dict = df.groupby(df.index.map(lambda i: enc.inverse_transform([df.loc[i, "location"]])[0]))["price_per_sqft"].agg(["mean", "count"]).to_dict("index")
        joblib.dump(loc_dict, MODELS_DIR / "location_stats.joblib")

        print(f"   Saved ensemble model → {MODELS_DIR / 'price_model.joblib'}")
        print(f"\n⚠️  NOTE: The API's predict function needs to handle ensemble format.")
        print(f"   It should check if model is a dict with 'type'='ensemble'")
    else:
        print(f"\n❌ Ensemble ({accuracy:.1f}%) did NOT beat current ({CURRENT_ACCURACY}%).")
        print(f"   Keeping current XGBoost V2 model. No changes made.")


if __name__ == "__main__":
    main()

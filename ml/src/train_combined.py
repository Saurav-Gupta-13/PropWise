"""
PropWise — Combined Dataset Training (V3)
===========================================
Merges Mumbai1.csv (6,317 rows) + Mumbai.csv (7,719 rows) 
= ~12,000+ unique Mumbai listings for maximum accuracy.

Run: python ml/src/train_combined.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import xgboost as xgb
import re

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_location(value):
    if pd.isna(value):
        return "unknown"
    s = str(value).lower().strip()
    s = re.sub(r"\s*,?\s*mumbai\s*$", "", s)
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"[\s/\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def load_mumbai1():
    """Load original Mumbai1.csv (18 features, 6317 rows)."""
    path = RAW_DIR / "Mumbai1.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    rename = {
        "Price": "price", "Area": "area_sqft", "Location": "location",
        "No. of Bedrooms": "bhk", "New/Resale": "is_resale",
        "Gymnasium": "amenity_gym", "Lift Available": "lift",
        "Car Parking": "parking", "Maintenance Staff": "amenity_maintenance",
        "24x7 Security": "amenity_security", "Children's Play Area": "amenity_kids_play",
        "Clubhouse": "amenity_clubhouse", "Intercom": "amenity_intercom",
        "Landscaped Gardens": "amenity_gardens", "Indoor Games": "amenity_indoor_games",
        "Gas Connection": "amenity_gas", "Jogging Track": "amenity_jogging",
        "Swimming Pool": "amenity_pool",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["source"] = "mumbai1"
    return df


def load_mumbai_new():
    """Load new Mumbai.csv (40+ features, 7719 rows)."""
    path = RAW_DIR / "Mumbai.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    rename = {
        "Price": "price", "Area": "area_sqft", "Location": "location",
        "No. of Bedrooms": "bhk", "Resale": "is_resale",
        "Gymnasium": "amenity_gym", "LiftAvailable": "lift",
        "CarParking": "parking", "MaintenanceStaff": "amenity_maintenance",
        "24X7Security": "amenity_security", "Children'splayarea": "amenity_kids_play",
        "ClubHouse": "amenity_clubhouse", "Intercom": "amenity_intercom",
        "LandscapedGardens": "amenity_gardens", "IndoorGames": "amenity_indoor_games",
        "Gasconnection": "amenity_gas", "JoggingTrack": "amenity_jogging",
        "SwimmingPool": "amenity_pool",
        "PowerBackup": "amenity_power_backup",
        "RainWaterHarvesting": "amenity_rainwater",
        "SportsFacility": "amenity_sports",
        "MultipurposeRoom": "amenity_multipurpose",
        "Hospital": "amenity_hospital",
        "School": "amenity_school",
        "Wifi": "amenity_wifi",
        "AC": "amenity_ac",
        "VaastuCompliant": "vastu_compliant",
        "BED": "bed_count",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["source"] = "mumbai_new"
    return df


def main():
    print("=" * 60)
    print("PROPWISE V3 — COMBINED DATASET TRAINING")
    print("=" * 60)

    # Load both datasets
    df1 = load_mumbai1()
    df2 = load_mumbai_new()

    if df1 is not None:
        print(f"📂 Mumbai1.csv: {len(df1)} rows")
    if df2 is not None:
        print(f"📂 Mumbai.csv:  {len(df2)} rows")

    # Find common columns
    common_cols = ["price", "area_sqft", "location", "bhk", "is_resale",
                   "lift", "parking", "amenity_gym", "amenity_security",
                   "amenity_clubhouse", "amenity_pool", "amenity_gardens",
                   "amenity_kids_play", "amenity_indoor_games", "amenity_jogging",
                   "amenity_intercom", "amenity_maintenance", "amenity_gas", "source"]

    # Keep only common columns + extras from df2
    extra_from_df2 = ["amenity_power_backup", "amenity_rainwater", "amenity_sports",
                      "amenity_wifi", "amenity_ac", "vastu_compliant"]

    all_cols = common_cols + extra_from_df2

    # Ensure columns exist
    for df in [df1, df2]:
        if df is not None:
            for col in all_cols:
                if col not in df.columns:
                    df[col] = 0

    # Combine
    frames = [x[all_cols] for x in [df1, df2] if x is not None]
    df = pd.concat(frames, ignore_index=True)
    print(f"\n📊 Combined: {len(df)} rows")

    # Normalize location
    df["location"] = df["location"].apply(normalize_location)

    # Convert numeric
    for col in ["price", "area_sqft", "bhk"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop nulls
    before = len(df)
    df = df.dropna(subset=["price", "area_sqft", "location", "bhk"])
    print(f"🧹 Dropped {before - len(df)} rows with missing values")

    # Remove outliers
    df = df[(df["price"] >= 1e6) & (df["price"] <= 2e9)]
    df = df[(df["area_sqft"] >= 100) & (df["area_sqft"] <= 20000)]
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 10)]
    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    df = df[(df["price_per_sqft"] >= 3000) & (df["price_per_sqft"] <= 200000)]

    # Remove duplicate listings (same price + area + location + bhk)
    before = len(df)
    df = df.drop_duplicates(subset=["price", "area_sqft", "location", "bhk"], keep="first")
    print(f"🔁 Removed {before - len(df)} duplicates")
    print(f"✅ Final dataset: {len(df)} unique rows")
    print(f"   Locations: {df['location'].nunique()}")

    # Make binary columns clean
    binary_cols = [c for c in df.columns if c.startswith("amenity_") or c in ["lift", "parking", "is_resale", "vastu_compliant"]]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Amenity count
    amenity_cols = [c for c in df.columns if c.startswith("amenity_")]
    df["amenity_count"] = df[amenity_cols].sum(axis=1)

    # Feature engineering
    df["log_area"] = np.log1p(df["area_sqft"])
    df["area_per_bhk"] = df["area_sqft"] / df["bhk"]
    df["bhk_sq"] = df["bhk"] ** 2
    df["is_luxury"] = ((df["bhk"] >= 4) | (df["area_sqft"] > 2000)).astype(int)

    # Location target encoding (smoothed)
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

    # IQR outlier removal per location
    cleaned = []
    for loc, group in df.groupby("location"):
        if len(group) < 5:
            cleaned.append(group)
            continue
        Q1 = group["price_per_sqft"].quantile(0.1)
        Q3 = group["price_per_sqft"].quantile(0.9)
        IQR = Q3 - Q1
        mask = (group["price_per_sqft"] >= Q1 - 2 * IQR) & (group["price_per_sqft"] <= Q3 + 2 * IQR)
        cleaned.append(group[mask])
    df = pd.concat(cleaned, ignore_index=True)
    print(f"🧹 After IQR cleanup: {len(df)} rows")

    # Encode location
    enc = LabelEncoder()
    df["location_str"] = df["location"].copy()
    df["location"] = enc.fit_transform(df["location"].astype(str))
    encoders = {"location": enc}

    # Define features
    feature_cols = (
        ["bhk", "area_sqft", "lift", "parking", "is_resale", "amenity_count", "location"]
        + amenity_cols
        + ["log_area", "area_per_bhk", "loc_smoothed_psf", "bhk_sq", "is_luxury", "vastu_compliant"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    print(f"\n📋 Features: {len(feature_cols)}")

    X = df[feature_cols]
    y = np.log1p(df["price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n🧠 Training XGBoost V3 on {len(X_train)} samples...")

    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=0.15,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = np.expm1(model.predict(X_test))
    y_actual = np.expm1(y_test)
    r2 = r2_score(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    mae = mean_absolute_error(y_actual, y_pred)
    accuracy = 100 - mape

    print(f"\n📊 Results:")
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   R² Score:      {r2:.4f}")
    print(f"   MAPE:          {mape:.2f}%")
    print(f"   MAE:           ₹{mae:,.0f}")
    print(f"   ✅ ACCURACY:   {accuracy:.1f}%")

    # Save
    joblib.dump(model, MODELS_DIR / "price_model.joblib")
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump({
        "feature_cols": feature_cols,
        "metrics": {
            "r2": float(r2), "mape": float(mape), "mae": float(mae),
            "accuracy": float(accuracy),
            "test_samples": int(len(X_test)),
            "train_samples": int(len(X_train)),
        },
    }, MODELS_DIR / "feature_metadata.joblib")

    # Save location stats
    loc_dict = df.groupby("location_str")["price_per_sqft"].agg(["mean", "count"]).to_dict("index")
    joblib.dump(loc_dict, MODELS_DIR / "location_stats.joblib")

    # Save updated clean CSV
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"\n✅ All saved. Previous: 83.4% → New: {accuracy:.1f}%")


if __name__ == "__main__":
    main()

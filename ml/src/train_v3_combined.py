"""
PropWise V3 — Combined Training with New Features
===================================================
Merges:
  - Mumbai1.csv (6,317 rows, 18 features)
  - data_for_model.csv (7,483 rows, with floor/age/bathrooms/furnishing/balcony)

New features: bathrooms, balcony, age_of_property, furnishing
Expected accuracy: 87-90%

Run: python ml/src/train_v3_combined.py
"""

import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import xgboost as xgb

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"
MODELS_DIR = ROOT / "models"


def normalize_location(value):
    if pd.isna(value):
        return "unknown"
    s = str(value).lower().strip()
    # Remove "mumbai" suffix
    s = re.sub(r"\s*,?\s*mumbai\s*$", "", s)
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"[\s/\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s if s else "unknown"


def load_mumbai1():
    """Original dataset: 6,317 rows."""
    path = RAW_DIR / "Mumbai1.csv"
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
    df["location"] = df["location"].apply(normalize_location)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area_sqft"] = pd.to_numeric(df["area_sqft"], errors="coerce")
    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")

    # These features don't exist in Mumbai1 — fill with median/unknown
    df["bathrooms"] = df["bhk"]  # Reasonable default: bathrooms ≈ bhk
    df["balcony"] = 1  # Most flats have at least 1
    df["age_years"] = 5  # Unknown, use median
    df["furnishing_encoded"] = 1  # Semi-furnished (middle value)

    return df


def load_new_dataset():
    """New housing.com dataset: 7,483 rows with extra features."""
    path = RAW_DIR / "data_for_model.csv"
    df = pd.read_csv(path)

    # Rename
    df = df.rename(columns={
        "flat_price": "price_cr",
        "location1": "location",
        "buildupArea_sqft": "area_sqft",
        "bedrooms": "bhk",
    })

    # Convert price from Crores to INR
    df["price"] = df["price_cr"] * 1e7

    # Normalize location
    df["location"] = df["location"].apply(normalize_location)

    # Parse age_of_property (e.g., "3 Year Old", "New", "Under Construction")
    def parse_age(val):
        if pd.isna(val):
            return 5
        s = str(val).lower()
        if "new" in s or "under" in s or "0" in s:
            return 0
        nums = re.findall(r"(\d+)", s)
        return int(nums[0]) if nums else 5
    df["age_years"] = df["age_of_property"].apply(parse_age)

    # Encode furnishing: Unfurnished=0, Semi=1, Fully=2
    def encode_furnishing(val):
        if pd.isna(val):
            return 1
        s = str(val).lower()
        if "unfurnish" in s:
            return 0
        elif "semi" in s:
            return 1
        elif "full" in s or "furnish" in s:
            return 2
        return 1
    df["furnishing_encoded"] = df["furnishing"].apply(encode_furnishing)

    # Parse parking (e.g., "2 Covered Parking", "No Parking", "1 Open")
    def parse_parking(val):
        if pd.isna(val):
            return 0
        s = str(val).lower()
        if "no" in s:
            return 0
        return 1
    df["parking"] = df["parking"].apply(parse_parking)

    # Fill amenity columns with 0 (not available in this dataset)
    amenity_cols = ["amenity_gym", "amenity_security", "amenity_clubhouse",
                    "amenity_pool", "amenity_gardens", "amenity_kids_play",
                    "amenity_indoor_games", "amenity_jogging", "amenity_intercom",
                    "amenity_maintenance", "amenity_gas"]
    for col in amenity_cols:
        df[col] = 0

    df["lift"] = 1  # Most multi-floor buildings have lifts
    df["is_resale"] = 0  # Unknown

    return df


def main():
    print("=" * 60)
    print("PROPWISE V3 — COMBINED TRAINING (with new features)")
    print("=" * 60)

    df1 = load_mumbai1()
    df2 = load_new_dataset()
    print(f"📂 Mumbai1.csv:        {len(df1)} rows (original)")
    print(f"📂 data_for_model.csv: {len(df2)} rows (new — with floor/age/bath)")

    # Common columns for merging
    common_cols = [
        "price", "area_sqft", "location", "bhk", "is_resale",
        "lift", "parking", "bathrooms", "balcony", "age_years", "furnishing_encoded",
        "amenity_gym", "amenity_security", "amenity_clubhouse", "amenity_pool",
        "amenity_gardens", "amenity_kids_play", "amenity_indoor_games",
        "amenity_jogging", "amenity_intercom", "amenity_maintenance", "amenity_gas",
    ]

    for df in [df1, df2]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = 0

    df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
    print(f"\n📊 Combined: {len(df)} rows")

    # Clean
    for col in ["price", "area_sqft", "bhk", "bathrooms", "balcony", "age_years"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "area_sqft", "bhk"])
    df = df[(df["price"] >= 1e6) & (df["price"] <= 3e9)]
    df = df[(df["area_sqft"] >= 150) & (df["area_sqft"] <= 15000)]
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 8)]

    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    df = df[(df["price_per_sqft"] >= 2000) & (df["price_per_sqft"] <= 200000)]

    # De-duplicate
    before = len(df)
    df = df.drop_duplicates(subset=["price", "area_sqft", "location", "bhk"], keep="first")
    print(f"🔁 Removed {before - len(df)} duplicates")

    # Make binary cols clean
    for col in ["lift", "parking", "is_resale"] + [c for c in df.columns if c.startswith("amenity_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["amenity_count"] = df[[c for c in df.columns if c.startswith("amenity_")]].sum(axis=1)

    # Feature engineering
    df["log_area"] = np.log1p(df["area_sqft"])
    df["area_per_bhk"] = df["area_sqft"] / df["bhk"]
    df["bhk_sq"] = df["bhk"] ** 2
    df["is_luxury"] = ((df["bhk"] >= 4) | (df["area_sqft"] > 2000)).astype(int)
    df["bath_per_bhk"] = df["bathrooms"] / df["bhk"]

    # Location target encoding
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
    print(f"✅ Final dataset: {len(df)} rows, {df['location'].nunique()} locations")

    # Encode location
    df["location_str"] = df["location"].copy()
    enc = LabelEncoder()
    df["location"] = enc.fit_transform(df["location"].astype(str))
    encoders = {"location": enc}

    # Feature columns (including new ones!)
    feature_cols = [
        "bhk", "area_sqft", "lift", "parking", "is_resale", "amenity_count", "location",
        "bathrooms", "balcony", "age_years", "furnishing_encoded",
        "amenity_gym", "amenity_security", "amenity_clubhouse", "amenity_pool",
        "amenity_gardens", "amenity_kids_play", "amenity_indoor_games",
        "amenity_jogging", "amenity_intercom", "amenity_maintenance", "amenity_gas",
        "log_area", "area_per_bhk", "loc_smoothed_psf", "bhk_sq", "is_luxury", "bath_per_bhk",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"\n📋 Features: {len(feature_cols)}")

    X = df[feature_cols]
    y = np.log1p(df["price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🧠 Training XGBoost V3 on {len(X_train)} samples...")

    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.75,
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

    print(f"\n{'='*60}")
    print(f"📊 RESULTS:")
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   R² Score:      {r2:.4f}")
    print(f"   MAPE:          {mape:.2f}%")
    print(f"   MAE:           ₹{mae:,.0f}")
    print(f"   ✅ ACCURACY:   {accuracy:.1f}%")
    print(f"{'='*60}")

    # Feature importance
    print("\n🎯 Top 10 features:")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 60)
        print(f"   {feat:25s} {imp:.4f}  {bar}")

    # Save all artifacts
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

    loc_dict = df.groupby("location_str")["price_per_sqft"].agg(["mean", "count"]).to_dict("index")
    joblib.dump(loc_dict, MODELS_DIR / "location_stats.joblib")

    # Save processed CSV
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"\n📈 Previous: 83.4% → New: {accuracy:.1f}%")
    print(f"✅ All models saved. Push to deploy.")


if __name__ == "__main__":
    main()

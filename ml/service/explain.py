"""
PropWise — SHAP Explainability
================================
Explains WHY the model predicted a specific price.
Returns top features that increased/decreased the price.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

# Lazy load — SHAP is expensive
_explainer = None
_model = None
_metadata = None


def _load():
    global _explainer, _model, _metadata
    if _explainer is None:
        try:
            import shap
            _model = joblib.load(MODELS_DIR / "price_model.joblib")
            _metadata = joblib.load(MODELS_DIR / "feature_metadata.joblib")
            _explainer = shap.TreeExplainer(_model)
            print("✅ SHAP explainer loaded")
        except Exception as e:
            print(f"⚠️ SHAP not available: {e}")
            return False
    return True


def explain_prediction(feature_row):
    """
    Generate SHAP-based explanation for a single prediction.
    Returns top features that pushed the price up/down.
    """
    if not _load():
        return None

    feature_cols = _metadata["feature_cols"]
    X = pd.DataFrame([{c: feature_row.get(c, 0) for c in feature_cols}])

    # Get SHAP values
    shap_values = _explainer.shap_values(X)
    base_value = float(_explainer.expected_value)

    # SHAP values are in log-space (we trained on log price)
    contributions = []
    for feat, val in zip(feature_cols, shap_values[0]):
        contributions.append({
            "feature": feat,
            "shap_value": float(val),  # log-space
            "input_value": float(X[feat].iloc[0]),
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Convert log-space to approximate % impact on price
    # If shap_value=0.1 in log space, that's ~10.5% increase in price
    for c in contributions:
        c["price_impact_pct"] = round((np.exp(c["shap_value"]) - 1) * 100, 1)

    # Pretty names
    pretty = {
        "bhk": "Bedrooms (BHK)",
        "area_sqft": "Carpet Area",
        "location": "Location",
        "lift": "Lift Available",
        "parking": "Car Parking",
        "is_resale": "Resale vs New",
        "amenity_count": "Total Amenities",
        "amenity_gym": "Gym",
        "amenity_security": "24×7 Security",
        "amenity_clubhouse": "Clubhouse",
        "amenity_pool": "Swimming Pool",
        "amenity_gardens": "Gardens",
        "amenity_kids_play": "Kids Play Area",
        "amenity_indoor_games": "Indoor Games",
        "amenity_jogging": "Jogging Track",
        "amenity_intercom": "Intercom",
        "amenity_maintenance": "Maintenance Staff",
        "amenity_gas": "Gas Connection",
    }

    for c in contributions:
        c["pretty_name"] = pretty.get(c["feature"], c["feature"].replace("_", " ").title())

    return {
        "base_log_value": base_value,
        "base_price": round(float(np.expm1(base_value))),
        "top_increases": [c for c in contributions if c["shap_value"] > 0][:5],
        "top_decreases": [c for c in contributions if c["shap_value"] < 0][:5],
        "all_contributions": contributions[:10],
    }

"""
PropWise — ML Service (FastAPI)
================================
Serves predictions from 3 ML models:
  - Price Predictor (XGBoost regressor)
  - Investment Score (XGBoost classifier)
  - Anomaly Detector (Isolation Forest)

Plus user feedback collection for continuous learning.

Run from propwise/ml/:
  uvicorn service.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from service.database import init_db, log_prediction, log_feedback, get_stats

# ─── Load Models ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def safe_load(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"⚠️  Missing: {path.name}")
        return None


# Price model (always required)
price_model = safe_load(MODELS_DIR / "price_model.joblib")
price_encoders = safe_load(MODELS_DIR / "encoders.joblib") or {}
price_metadata = safe_load(MODELS_DIR / "feature_metadata.joblib") or {
    "feature_cols": [],
    "metrics": {"accuracy": 0, "mape": 30, "train_samples": 0},
}

# Investment model (optional)
investment_model = safe_load(MODELS_DIR / "investment_model.joblib")
investment_encoders = safe_load(MODELS_DIR / "investment_encoders.joblib") or {}
investment_metadata = safe_load(MODELS_DIR / "investment_metadata.joblib") or {
    "feature_cols": [],
    "label_names": {0: "Overpriced", 1: "Fair", 2: "Good deal", 3: "Excellent deal"},
    "metrics": {"accuracy": 0},
}

# Anomaly model (optional)
anomaly_model = safe_load(MODELS_DIR / "anomaly_model.joblib")
anomaly_scaler = safe_load(MODELS_DIR / "anomaly_scaler.joblib")
anomaly_encoders = safe_load(MODELS_DIR / "anomaly_encoders.joblib") or {}
anomaly_metadata = safe_load(MODELS_DIR / "anomaly_metadata.joblib") or {
    "feature_cols": [],
    "metrics": {"anomaly_rate": 0},
}

if price_model:
    print(f"✅ Price model:      {price_metadata['metrics']['accuracy']:.1f}% accuracy on {price_metadata['metrics'].get('train_samples', 0)} samples")
if investment_model:
    print(f"✅ Investment model: {investment_metadata['metrics']['accuracy']:.1f}% accuracy")
if anomaly_model:
    print(f"✅ Anomaly model:    detects {anomaly_metadata['metrics']['anomaly_rate']:.1f}% as outliers")

# Initialize database
init_db()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class PropertyInput(BaseModel):
    location: str = Field(..., description="Mumbai neighborhood")
    bhk: int = Field(..., ge=1, le=10)
    area_sqft: float = Field(..., gt=100, lt=20000)
    is_resale: int = Field(0, ge=0, le=1)
    lift: int = Field(1, ge=0, le=1)
    parking: int = Field(1, ge=0, le=1)
    amenity_gym: int = Field(0, ge=0, le=1)
    amenity_security: int = Field(0, ge=0, le=1)
    amenity_clubhouse: int = Field(0, ge=0, le=1)
    amenity_pool: int = Field(0, ge=0, le=1)
    amenity_gardens: int = Field(0, ge=0, le=1)
    amenity_kids_play: int = Field(0, ge=0, le=1)
    amenity_indoor_games: int = Field(0, ge=0, le=1)
    amenity_jogging: int = Field(0, ge=0, le=1)
    amenity_intercom: int = Field(0, ge=0, le=1)
    amenity_maintenance: int = Field(0, ge=0, le=1)
    amenity_gas: int = Field(0, ge=0, le=1)


class FullAnalysisInput(PropertyInput):
    actual_listed_price: Optional[int] = Field(None, description="If provided, compare against listing")


class FeedbackInput(BaseModel):
    prediction_id: int
    actual_price: Optional[int] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="PropWise ML Service", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "PropWise ML",
        "version": "0.2.0",
        "status": "ready" if price_model else "model not loaded",
        "models": {
            "price_predictor": {
                "loaded": price_model is not None,
                "accuracy": price_metadata["metrics"]["accuracy"],
                "trained_on": price_metadata["metrics"].get("train_samples", 0),
            },
            "investment_score": {
                "loaded": investment_model is not None,
                "accuracy": investment_metadata["metrics"]["accuracy"],
            },
            "anomaly_detector": {
                "loaded": anomaly_model is not None,
                "anomaly_rate": anomaly_metadata["metrics"]["anomaly_rate"],
            },
        },
        "stats": get_stats(),
    }


@app.get("/locations")
def locations():
    if "location" not in price_encoders:
        return {"locations": []}
    return {"locations": sorted(list(price_encoders["location"].classes_))}


@app.get("/stats")
def stats():
    """Usage statistics."""
    return get_stats()


def encode_location(location, encoders_dict):
    """Encode a location, fall back to mode (0) if unknown."""
    location_normalized = location.lower().replace(" ", "_").replace("-", "_")
    if "location" in encoders_dict and location_normalized in encoders_dict["location"].classes_:
        return int(encoders_dict["location"].transform([location_normalized])[0]), True
    return 0, False


def build_feature_row(input: PropertyInput, location_encoded, predicted_price=None):
    """Build feature dict for model input."""
    amenity_fields = [
        "amenity_gym", "amenity_security", "amenity_clubhouse",
        "amenity_pool", "amenity_gardens", "amenity_kids_play",
        "amenity_indoor_games", "amenity_jogging", "amenity_intercom",
        "amenity_maintenance", "amenity_gas",
    ]
    amenity_count = sum(getattr(input, f) for f in amenity_fields)

    row = {
        "bhk": input.bhk,
        "area_sqft": input.area_sqft,
        "lift": input.lift,
        "parking": input.parking,
        "is_resale": input.is_resale,
        "amenity_count": amenity_count,
        "location": location_encoded,
    }
    for f in amenity_fields:
        row[f] = getattr(input, f)
    if predicted_price is not None:
        row["price"] = predicted_price
        row["price_per_sqft"] = predicted_price / input.area_sqft
    return row


@app.post("/predict")
def predict(input: PropertyInput):
    """Single price prediction."""
    if price_model is None:
        raise HTTPException(503, "Price model not loaded.")

    location_encoded, location_known = encode_location(input.location, price_encoders)
    row = build_feature_row(input, location_encoded)

    feature_cols = price_metadata["feature_cols"]
    X = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])

    pred_log = price_model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))

    mape = price_metadata["metrics"]["mape"] / 100
    range_low = pred_price * (1 - mape)
    range_high = pred_price * (1 + mape)

    location_normalized = input.location.lower().replace(" ", "_").replace("-", "_")

    # Log to database
    pred_id = log_prediction(
        input_dict=input.dict(),
        predicted_price=int(pred_price),
        price_low=int(range_low),
        price_high=int(range_high),
    )

    return {
        "prediction_id": pred_id,
        "predicted_price": int(pred_price),
        "price_range_low": int(range_low),
        "price_range_high": int(range_high),
        "price_per_sqft": int(pred_price / input.area_sqft),
        "confidence": round(price_metadata["metrics"]["accuracy"] / 100, 3),
        "location_used": location_normalized,
        "location_known": location_known,
        "model_accuracy": round(price_metadata["metrics"]["accuracy"], 1),
        "insights": {
            "location_known": location_known,
            "model_confidence": "high" if price_metadata["metrics"]["accuracy"] >= 80 else "medium",
            "amenity_count": row.get("amenity_count", 0),
            "recommendation": (
                f"Property is in our high-confidence zone. Estimate is reliable."
                if location_known
                else "Location not in training data. Estimate uses similar Mumbai areas — validate locally."
            ),
        },
    }


@app.post("/analyze")
def analyze(input: FullAnalysisInput):
    """Full analysis: price + investment score + anomaly check."""
    if price_model is None:
        raise HTTPException(503, "Price model not loaded.")

    # 1. Price prediction
    location_encoded, location_known = encode_location(input.location, price_encoders)
    row = build_feature_row(input, location_encoded)

    feature_cols = price_metadata["feature_cols"]
    X = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])

    pred_log = price_model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))

    mape = price_metadata["metrics"]["mape"] / 100
    range_low = pred_price * (1 - mape)
    range_high = pred_price * (1 + mape)

    # 2. Investment score (uses price as a feature)
    investment_result = None
    if investment_model is not None:
        # Use listed price if provided, else predicted price
        analysis_price = input.actual_listed_price if input.actual_listed_price else pred_price
        inv_loc_encoded, _ = encode_location(input.location, investment_encoders)
        inv_row = build_feature_row(input, inv_loc_encoded, predicted_price=analysis_price)
        inv_feature_cols = investment_metadata["feature_cols"]
        X_inv = pd.DataFrame([{c: inv_row.get(c, 0) for c in inv_feature_cols}])

        label_pred = int(investment_model.predict(X_inv)[0])
        label_proba = investment_model.predict_proba(X_inv)[0]

        label_names = investment_metadata["label_names"]
        # Investment score from 0-100
        # Excellent deal=100, Good deal=75, Fair=50, Overpriced=25
        score_map = {0: 25, 1: 50, 2: 75, 3: 100}
        investment_score = score_map.get(label_pred, 50)

        investment_result = {
            "verdict": label_names[label_pred],
            "score": investment_score,
            "confidence": round(float(label_proba[label_pred]) * 100, 1),
            "probabilities": {
                label_names[i]: round(float(p) * 100, 1)
                for i, p in enumerate(label_proba)
            },
        }

    # 3. Anomaly check
    anomaly_result = None
    if anomaly_model is not None and anomaly_scaler is not None:
        anom_loc_encoded, _ = encode_location(input.location, anomaly_encoders)
        anom_row = build_feature_row(input, anom_loc_encoded, predicted_price=pred_price)
        anom_feature_cols = anomaly_metadata["feature_cols"]
        X_anom = pd.DataFrame([{c: anom_row.get(c, 0) for c in anom_feature_cols}])
        X_anom_scaled = anomaly_scaler.transform(X_anom)

        score = float(anomaly_model.decision_function(X_anom_scaled)[0])
        is_anomaly = score < 0

        anomaly_result = {
            "is_anomalous": is_anomaly,
            "score": round(score, 3),
            "verdict": "⚠️ Suspicious listing — features don't match price profile" if is_anomaly else "✅ Normal listing — features match expected profile",
        }

    location_normalized = input.location.lower().replace(" ", "_").replace("-", "_")

    # Compute additional insights
    insights = {
        "location_known": location_known,
        "amenity_count": row.get("amenity_count", 0),
        "model_confidence": "high" if price_metadata["metrics"]["accuracy"] >= 80 else "medium",
    }

    # If user provided actual listed price, compute negotiation insight
    negotiation = None
    if input.actual_listed_price:
        diff = input.actual_listed_price - pred_price
        diff_pct = (diff / pred_price) * 100
        if abs(diff_pct) < 5:
            verdict = "Listed price matches our prediction — fair value."
        elif diff_pct > 5:
            verdict = f"⚠️ Listed price is {diff_pct:.1f}% higher than fair value. Negotiate down by ~₹{abs(diff)/100000:.1f}L."
        else:
            verdict = f"💰 Listed price is {abs(diff_pct):.1f}% lower than fair value. Possible underpricing — investigate."
        negotiation = {
            "listed_price": input.actual_listed_price,
            "predicted_fair_price": int(pred_price),
            "difference": int(diff),
            "difference_pct": round(diff_pct, 1),
            "verdict": verdict,
        }

    # Log prediction
    pred_id = log_prediction(
        input_dict=input.dict(),
        predicted_price=int(pred_price),
        price_low=int(range_low),
        price_high=int(range_high),
    )

    return {
        "prediction_id": pred_id,
        "price_prediction": {
            "predicted_price": int(pred_price),
            "price_range_low": int(range_low),
            "price_range_high": int(range_high),
            "price_per_sqft": int(pred_price / input.area_sqft),
            "model_accuracy": round(price_metadata["metrics"]["accuracy"], 1),
        },
        "investment_analysis": investment_result,
        "anomaly_check": anomaly_result,
        "negotiation": negotiation,
        "location_used": location_normalized,
        "location_known": location_known,
        "insights": insights,
    }


@app.post("/feedback")
def feedback(input: FeedbackInput):
    """Submit feedback on a prediction. Used for continuous learning."""
    log_feedback(
        prediction_id=input.prediction_id,
        actual_price=input.actual_price,
        rating=input.rating,
        comment=input.comment,
    )
    return {"ok": True, "message": "Feedback recorded. Thanks for helping improve PropWise!"}


# ─── Additional Feature Endpoints ────────────────────────────────────────────

from service.features import calculate_emi, get_similar_properties, get_neighborhood_stats, compare_neighborhoods, get_top_neighborhoods
from pydantic import BaseModel as BM
from typing import List


class EMIRequest(BM):
    principal: int = Field(..., description="Loan amount in INR")
    annual_rate: float = Field(8.5, description="Annual interest rate in %")
    tenure_years: int = Field(20, description="Loan tenure in years")


class CompareRequest(BM):
    locations: List[str] = Field(..., description="List of neighborhood names to compare")


@app.post("/emi")
def emi(input: EMIRequest):
    """Calculate EMI for a home loan."""
    return calculate_emi(input.principal, input.annual_rate, input.tenure_years)


@app.get("/similar/{location}/{bhk}/{area_sqft}")
def similar(location: str, bhk: int, area_sqft: int):
    """Find similar properties in the dataset."""
    props = get_similar_properties(location, bhk, area_sqft)
    return {"similar_properties": props, "count": len(props)}


@app.get("/neighborhood/{location}")
def neighborhood(location: str):
    """Get statistics for a specific neighborhood."""
    stats = get_neighborhood_stats(location)
    if not stats:
        raise HTTPException(404, f"No data for location: {location}")
    return stats


@app.post("/compare")
def compare(input: CompareRequest):
    """Compare multiple neighborhoods side by side."""
    results = compare_neighborhoods(input.locations)
    return {"comparison": results, "count": len(results)}


@app.get("/top-neighborhoods")
def top_neighborhoods(n: int = 10, sort: str = "expensive"):
    """Get top neighborhoods by price. sort=expensive or sort=cheapest"""
    return {"neighborhoods": get_top_neighborhoods(n, sort)}


# ─── SHAP / Rent vs Buy / Price Trends Endpoints ──────────────────────────────

from service.rent_buy import rent_vs_buy_analysis, estimate_monthly_rent

try:
    from service.explain import explain_prediction
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class RentBuyRequest(BM):
    property_price: int
    bhk: int
    area_sqft: int
    horizon_years: int = 10
    down_payment_pct: float = 20
    loan_rate: float = 8.5
    loan_tenure: int = 20


class ExplainRequest(BM):
    location: str
    bhk: int
    area_sqft: float
    is_resale: int = 0
    lift: int = 1
    parking: int = 1
    amenity_gym: int = 0
    amenity_security: int = 0
    amenity_clubhouse: int = 0
    amenity_pool: int = 0
    amenity_gardens: int = 0
    amenity_kids_play: int = 0
    amenity_indoor_games: int = 0
    amenity_jogging: int = 0
    amenity_intercom: int = 0
    amenity_maintenance: int = 0
    amenity_gas: int = 0


@app.post("/rent-vs-buy")
def rent_vs_buy(input: RentBuyRequest):
    """Compare rent vs buy financial impact over a time horizon."""
    result = rent_vs_buy_analysis(
        property_price=input.property_price,
        bhk=input.bhk,
        area_sqft=input.area_sqft,
        years_horizon=input.horizon_years,
        down_payment_pct=input.down_payment_pct,
        loan_rate=input.loan_rate,
        loan_tenure=input.loan_tenure,
    )
    if not result:
        raise HTTPException(404, "Insufficient rental data for this BHK")
    return result


@app.post("/explain")
def explain(input: ExplainRequest):
    """Get SHAP-based explanation for why the model predicted a specific price."""
    if not SHAP_AVAILABLE or price_model is None:
        raise HTTPException(503, "SHAP explainer not loaded. Run: pip install shap")

    location_encoded, _ = encode_location(input.location, price_encoders)

    amenity_fields = [
        "amenity_gym", "amenity_security", "amenity_clubhouse",
        "amenity_pool", "amenity_gardens", "amenity_kids_play",
        "amenity_indoor_games", "amenity_jogging", "amenity_intercom",
        "amenity_maintenance", "amenity_gas",
    ]
    amenity_count = sum(getattr(input, f) for f in amenity_fields)

    feature_row = {
        "bhk": input.bhk,
        "area_sqft": input.area_sqft,
        "lift": input.lift,
        "parking": input.parking,
        "is_resale": input.is_resale,
        "amenity_count": amenity_count,
        "location": location_encoded,
    }
    for f in amenity_fields:
        feature_row[f] = getattr(input, f)

    explanation = explain_prediction(feature_row)
    if not explanation:
        raise HTTPException(503, "Explainer failed.")
    return explanation


@app.get("/rent-estimate/{bhk}/{area_sqft}")
def rent_estimate(bhk: int, area_sqft: int):
    """Get rent estimate for a BHK + area in Mumbai."""
    result = estimate_monthly_rent(bhk, area_sqft)
    if not result:
        raise HTTPException(404, "Rental data not available")
    return result


@app.get("/price-trend/{location}")
def price_trend(location: str):
    """
    Generate a 5-year price trend for a neighborhood.
    Uses Mumbai market growth rates (8% avg) to project backwards.
    """
    from service.features import get_neighborhood_stats
    stats = get_neighborhood_stats(location)
    if not stats:
        raise HTTPException(404, f"No data for {location}")

    current_avg = stats["avg_price_per_sqft"]
    # Mumbai avg appreciation: 8% CAGR over last 5 years
    trend = []
    for years_back in range(5, -1, -1):
        psf = round(current_avg / ((1.08) ** years_back))
        trend.append({"year": 2025 - years_back, "price_per_sqft": psf})

    return {
        "location": location,
        "current_avg_psf": current_avg,
        "five_year_growth_pct": round(((1.08 ** 5) - 1) * 100, 1),
        "trend": trend,
    }


# ─── Admin / Database Viewer ──────────────────────────────────────────────────

@app.get("/admin/predictions")
def admin_predictions(limit: int = 50):
    """View recent predictions from the database."""
    from service.database import get_db
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, location, bhk, area_sqft, predicted_price, price_low, price_high "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"count": len(rows), "predictions": [dict(r) for r in rows]}


@app.get("/admin/feedback")
def admin_feedback(limit: int = 50):
    """View user feedback from the database."""
    from service.database import get_db
    with get_db() as conn:
        rows = conn.execute(
            "SELECT f.*, p.location, p.predicted_price FROM feedback f "
            "LEFT JOIN predictions p ON p.id = f.prediction_id "
            "ORDER BY f.id DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"count": len(rows), "feedback": [dict(r) for r in rows]}


@app.post("/admin/retrain")
def auto_retrain():
    """
    Auto-retrain the price model using accumulated user feedback.
    Called by a cron job (e.g., every Sunday at midnight).
    Only deploys new model if accuracy improves or stays within 1%.
    """
    global price_model, price_metadata

    from service.database import get_training_supplements
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error, r2_score

    feedback_data = get_training_supplements()
    if len(feedback_data) < 5:
        return {"status": "skipped", "reason": f"Only {len(feedback_data)} feedback entries with actual price. Need at least 5."}

    # Load original training data
    data_path = ROOT / "data" / "processed" / "mumbai_clean.csv"
    df_orig = pd.read_csv(data_path)

    # Encode locations in original data
    df_orig["location"] = df_orig["location"].astype(str).apply(
        lambda x: int(price_encoders["location"].transform([x])[0]) if x in price_encoders["location"].classes_ else 0
    )

    # Build feedback dataframe
    fb_rows = []
    for item in feedback_data:
        inp = item["input"]
        loc = inp.get("location", "unknown").lower().replace(" ", "_").replace("-", "_")
        loc_encoded = int(price_encoders["location"].transform([loc])[0]) if loc in price_encoders["location"].classes_ else 0

        amenity_fields = [
            "amenity_gym", "amenity_security", "amenity_clubhouse", "amenity_pool",
            "amenity_gardens", "amenity_kids_play", "amenity_indoor_games",
            "amenity_jogging", "amenity_intercom", "amenity_maintenance", "amenity_gas",
        ]
        amenity_count = sum(inp.get(f, 0) for f in amenity_fields)

        row = {
            "bhk": inp.get("bhk", 2),
            "area_sqft": inp.get("area_sqft", 800),
            "lift": inp.get("lift", 1),
            "parking": inp.get("parking", 1),
            "is_resale": inp.get("is_resale", 0),
            "amenity_count": amenity_count,
            "location": loc_encoded,
            "price": item["price"],
        }
        for f in amenity_fields:
            row[f] = inp.get(f, 0)
        row["price_per_sqft"] = row["price"] / row["area_sqft"]
        fb_rows.append(row)

    df_fb = pd.DataFrame(fb_rows)

    # Combine
    feature_cols = price_metadata["feature_cols"]
    for col in feature_cols:
        if col not in df_fb.columns:
            df_fb[col] = 0
    df_combined = pd.concat([df_orig, df_fb[feature_cols + ["price"]]], ignore_index=True)

    # Train
    X = df_combined[feature_cols]
    y = np.log1p(df_combined["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = np.expm1(model.predict(X_test))
    y_actual = np.expm1(y_test)
    new_mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    new_accuracy = 100 - new_mape
    old_accuracy = price_metadata["metrics"]["accuracy"]

    if new_accuracy >= old_accuracy - 1:
        # Deploy in-memory (hot swap)
        price_model = model
        price_metadata["metrics"]["accuracy"] = float(new_accuracy)
        price_metadata["metrics"]["mape"] = float(new_mape)
        price_metadata["metrics"]["train_samples"] = int(len(X_train))
        price_metadata["metrics"]["feedback_samples"] = int(len(df_fb))
        price_metadata["metrics"]["last_retrained"] = pd.Timestamp.now().isoformat()

        # Also save to disk
        joblib.dump(model, MODELS_DIR / "price_model.joblib")
        joblib.dump(price_metadata, MODELS_DIR / "feature_metadata.joblib")

        return {
            "status": "retrained",
            "old_accuracy": round(old_accuracy, 1),
            "new_accuracy": round(new_accuracy, 1),
            "training_samples": len(df_combined),
            "feedback_used": len(df_fb),
        }
    else:
        return {
            "status": "rejected",
            "reason": f"New model ({new_accuracy:.1f}%) worse than current ({old_accuracy:.1f}%). Keeping current.",
            "feedback_available": len(df_fb),
        }


@app.get("/admin/dashboard", response_class=__import__("fastapi.responses", fromlist=["HTMLResponse"]).HTMLResponse)
def admin_dashboard():
    """HTML dashboard to view database contents."""
    from service.database import get_db, get_stats
    stats = get_stats()
    with get_db() as conn:
        recent = conn.execute(
            "SELECT id, timestamp, location, bhk, area_sqft, predicted_price "
            "FROM predictions ORDER BY id DESC LIMIT 20"
        ).fetchall()
        feedbacks = conn.execute(
            "SELECT f.*, p.location, p.predicted_price as predicted "
            "FROM feedback f LEFT JOIN predictions p ON p.id = f.prediction_id "
            "ORDER BY f.id DESC LIMIT 10"
        ).fetchall()

    rows_html = "".join([
        f"<tr><td>{r['id']}</td><td>{r['timestamp'][:19]}</td><td>{r['location']}</td>"
        f"<td>{r['bhk']}</td><td>{r['area_sqft']:.0f}</td>"
        f"<td>₹{r['predicted_price']:,}</td></tr>"
        for r in recent
    ])

    fb_html = "".join([
        f"<tr><td>{r['id']}</td><td>{r['timestamp'][:19]}</td><td>{r.get('location','-')}</td>"
        f"<td>₹{r.get('predicted',0):,}</td>"
        f"<td>{('₹' + format(r['actual_price'],',')) if r['actual_price'] else '-'}</td>"
        f"<td>{r.get('rating','-')}</td></tr>"
        for r in feedbacks
    ])

    top_loc_html = "".join([
        f"<tr><td>{l['location']}</td><td>{l['count']}</td></tr>"
        for l in stats.get('top_locations', [])
    ])

    return f"""
    <html><head><title>PropWise Admin</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
      h1 {{ color: #3b82f6; }}
      h2 {{ color: #cbd5e1; margin-top: 32px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
      table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
      th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
      th {{ background: #334155; color: #93c5fd; font-size: 12px; text-transform: uppercase; }}
      tr:hover {{ background: #334155; }}
      .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
      .stat {{ background: #1e293b; padding: 18px; border-radius: 10px; border-left: 4px solid #3b82f6; }}
      .stat-num {{ font-size: 28px; font-weight: 800; color: #fff; }}
      .stat-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
    </style></head>
    <body>
      <h1>📊 PropWise Admin Dashboard</h1>
      <div class="stats">
        <div class="stat"><div class="stat-label">Total Predictions</div><div class="stat-num">{stats['total_predictions']}</div></div>
        <div class="stat"><div class="stat-label">Total Feedback</div><div class="stat-num">{stats['total_feedback']}</div></div>
        <div class="stat"><div class="stat-label">With Actual Price</div><div class="stat-num">{stats['feedback_with_actual_price']}</div></div>
      </div>
      <h2>Recent Predictions (last 20)</h2>
      <table>
        <tr><th>ID</th><th>Time</th><th>Location</th><th>BHK</th><th>Area</th><th>Predicted</th></tr>
        {rows_html or '<tr><td colspan=6>No predictions yet</td></tr>'}
      </table>
      <h2>Top Searched Locations</h2>
      <table>
        <tr><th>Location</th><th>Count</th></tr>
        {top_loc_html or '<tr><td colspan=2>No data</td></tr>'}
      </table>
      <h2>Recent Feedback</h2>
      <table>
        <tr><th>ID</th><th>Time</th><th>Location</th><th>Predicted</th><th>Actual</th><th>Rating</th></tr>
        {fb_html or '<tr><td colspan=6>No feedback yet</td></tr>'}
      </table>
    </body></html>
    """

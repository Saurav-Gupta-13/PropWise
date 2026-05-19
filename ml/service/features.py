"""
PropWise — Additional Feature Endpoints
=========================================
- EMI Calculator
- Similar Properties (from training data)
- Neighborhood Comparison
- Price per sqft statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mumbai_clean.csv"

# Load dataset for similar properties + stats
try:
    df_all = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df_all)} rows for feature queries")
except:
    df_all = pd.DataFrame()


def calculate_emi(principal, annual_rate_pct, tenure_years):
    """Calculate EMI for a home loan."""
    r = annual_rate_pct / 100 / 12  # monthly rate
    n = tenure_years * 12  # total months
    if r == 0:
        return principal / n
    emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
    total_payment = emi * n
    total_interest = total_payment - principal
    return {
        "emi": round(emi),
        "total_payment": round(total_payment),
        "total_interest": round(total_interest),
        "principal": principal,
        "rate": annual_rate_pct,
        "tenure_years": tenure_years,
        "tenure_months": n,
    }


def get_similar_properties(location, bhk, area_sqft, limit=5):
    """Find similar properties from the dataset.
    
    Strategy:
    1. Try same location + same BHK → take 2-3 closest by area
    2. Add 2-3 from NEARBY areas with similar price profile
    3. Show diversity — different neighborhoods, similar value
    """
    if df_all.empty:
        return []

    same_loc = df_all[(df_all["location"] == location) & (df_all["bhk"] == bhk)].copy()
    
    results = []
    
    if len(same_loc) > 0:
        # Get 2-3 from same location closest by area
        same_loc["area_diff"] = abs(same_loc["area_sqft"] - area_sqft)
        same_loc = same_loc.sort_values("area_diff").head(3)
        results.extend(same_loc[["location", "bhk", "area_sqft", "price", "price_per_sqft"]].to_dict("records"))
    
    # Find similar-priced properties in OTHER neighborhoods
    if same_loc.empty:
        # No data for this location — find similar BHK + area anywhere
        other = df_all[df_all["bhk"] == bhk].copy()
    else:
        # Find properties with similar price profile in other neighborhoods
        target_psf = same_loc["price_per_sqft"].median()
        other = df_all[
            (df_all["location"] != location) &
            (df_all["bhk"] == bhk) &
            (df_all["price_per_sqft"].between(target_psf * 0.7, target_psf * 1.3))
        ].copy()
    
    if len(other) > 0:
        other["area_diff"] = abs(other["area_sqft"] - area_sqft)
        other = other.sort_values("area_diff").head(limit - len(results))
        # Pick from DIFFERENT locations for diversity
        seen_locs = set()
        diverse_other = []
        for _, row in other.iterrows():
            if row["location"] not in seen_locs:
                diverse_other.append(row)
                seen_locs.add(row["location"])
            if len(diverse_other) >= (limit - len(results)):
                break
        results.extend([dict(r) for r in [pd.Series(d).to_dict() for d in diverse_other]])
    
    # Clean up: remove area_diff column if present
    return [{k: v for k, v in p.items() if k != "area_diff"} for p in results[:limit]]


def get_neighborhood_stats(location):
    """Get price statistics for a neighborhood."""
    if df_all.empty:
        return None

    subset = df_all[df_all["location"] == location]
    if len(subset) == 0:
        return None

    return {
        "location": location,
        "total_listings": len(subset),
        "avg_price": round(subset["price"].mean()),
        "median_price": round(subset["price"].median()),
        "min_price": round(subset["price"].min()),
        "max_price": round(subset["price"].max()),
        "avg_price_per_sqft": round(subset["price_per_sqft"].mean()),
        "avg_area": round(subset["area_sqft"].mean()),
        "bhk_distribution": subset["bhk"].value_counts().to_dict(),
    }


def compare_neighborhoods(locations):
    """Compare multiple neighborhoods side by side."""
    results = []
    for loc in locations:
        stats = get_neighborhood_stats(loc)
        if stats:
            results.append(stats)
    return results


def get_top_neighborhoods(n=10, sort_by="avg_price_per_sqft"):
    """Get top neighborhoods by a metric."""
    if df_all.empty:
        return []

    stats = df_all.groupby("location").agg(
        avg_price=("price", "mean"),
        avg_psf=("price_per_sqft", "mean"),
        count=("price", "count"),
        median_price=("price", "median"),
    ).reset_index()

    # Only include locations with enough data
    stats = stats[stats["count"] >= 5]

    if sort_by == "cheapest":
        stats = stats.sort_values("avg_psf", ascending=True)
    else:
        stats = stats.sort_values("avg_psf", ascending=False)

    return stats.head(n).to_dict("records")

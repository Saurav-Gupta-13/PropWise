"""
PropWise — Rent vs Buy Calculator
====================================
Compares the financial impact of renting vs buying a property.
Uses real Mumbai rental data + EMI calculation.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RENT_DATA_PATH = ROOT / "data" / "raw" / "House_Rent_Dataset.csv"

# Load rent dataset (filter to Mumbai)
try:
    df_rent = pd.read_csv(RENT_DATA_PATH)
    df_rent = df_rent[df_rent["City"].str.lower() == "mumbai"].copy()
    print(f"✅ Loaded {len(df_rent)} Mumbai rental listings")
except Exception as e:
    print(f"⚠️ Rent data not loaded: {e}")
    df_rent = pd.DataFrame()


def estimate_monthly_rent(bhk, area_sqft, location=None):
    """Estimate monthly rent based on real Mumbai rental data."""
    if df_rent.empty:
        # Rough fallback: ~3.5% annual yield → 0.29% monthly
        return None

    # Filter rentals matching BHK
    matches = df_rent[df_rent["BHK"] == bhk].copy()

    if len(matches) == 0:
        return None

    # Sort by size closeness
    matches["size_diff"] = abs(matches["Size"] - area_sqft)
    matches = matches.sort_values("size_diff").head(20)

    # Median rent of similar properties
    median_rent = matches["Rent"].median()
    min_rent = matches["Rent"].quantile(0.25)
    max_rent = matches["Rent"].quantile(0.75)

    return {
        "estimated_monthly_rent": round(median_rent),
        "rent_range_low": round(min_rent),
        "rent_range_high": round(max_rent),
        "based_on_listings": len(matches),
    }


def rent_vs_buy_analysis(property_price, bhk, area_sqft, years_horizon=10,
                         down_payment_pct=20, loan_rate=8.5, loan_tenure=20,
                         rent_increase_pct=5, property_appreciation_pct=8,
                         alternate_investment_return=10):
    """
    Comprehensive rent vs buy comparison over a time horizon.
    """
    rent_estimate = estimate_monthly_rent(bhk, area_sqft)
    if not rent_estimate or rent_estimate.get("based_on_listings", 0) < 5:
        return None

    monthly_rent = rent_estimate["estimated_monthly_rent"]

    # Sanity check — rent should be 0.2-0.6% of property value monthly
    expected_min_rent = property_price * 0.002
    expected_max_rent = property_price * 0.006
    if monthly_rent < expected_min_rent or monthly_rent > expected_max_rent:
        # Use estimated rent based on yield instead
        monthly_rent = property_price * 0.0035  # 3.5% annual yield
        rent_estimate["estimated_monthly_rent"] = round(monthly_rent)
        rent_estimate["adjusted"] = True
        rent_estimate["note"] = "Rent estimated using 3.5% rental yield (insufficient comparable rentals)"

    down_payment = property_price * (down_payment_pct / 100)
    loan_amount = property_price - down_payment

    # EMI calculation
    r = loan_rate / 100 / 12
    n = loan_tenure * 12
    emi = loan_amount * r * ((1 + r) ** n) / (((1 + r) ** n) - 1) if r > 0 else loan_amount / n

    months_in_horizon = min(years_horizon * 12, n)
    total_emi_paid = emi * months_in_horizon

    # Maintenance & taxes (~1.5% of property value/year)
    maintenance_annual = property_price * 0.015
    total_maintenance = maintenance_annual * years_horizon

    # Property value after horizon
    future_property_value = property_price * ((1 + property_appreciation_pct / 100) ** years_horizon)

    # Remaining loan principal at end of horizon (linear approximation)
    months_remaining = max(0, n - months_in_horizon)
    if months_remaining > 0 and r > 0:
        remaining_principal = loan_amount * (((1 + r) ** n - (1 + r) ** months_in_horizon) / ((1 + r) ** n - 1))
    else:
        remaining_principal = 0

    sale_proceeds = future_property_value - remaining_principal

    # BUYING — total money out, money back when sold
    buy_money_out = down_payment + total_emi_paid + total_maintenance
    buy_net_cost = buy_money_out - sale_proceeds

    # RENTING — cumulative rent with annual increases
    total_rent = 0
    current_rent = monthly_rent
    for year in range(years_horizon):
        total_rent += current_rent * 12
        current_rent *= (1 + rent_increase_pct / 100)

    # Renter invests their down payment elsewhere
    investment_value = down_payment * ((1 + alternate_investment_return / 100) ** years_horizon)
    investment_gain = investment_value - down_payment

    # RENTING net cost = rent paid - gain from investing what would have been down payment
    rent_net_cost = total_rent - investment_gain

    # Verdict
    savings = rent_net_cost - buy_net_cost
    if savings > 0:
        verdict = f"Buying saves you ₹{abs(savings)/100000:.1f}L over {years_horizon} years"
        recommendation = "buy"
    else:
        verdict = f"Renting saves you ₹{abs(savings)/100000:.1f}L over {years_horizon} years"
        recommendation = "rent"

    # Break-even year — when cumulative rent costs exceed cumulative buy costs
    breakeven_year = None
    for yr in range(1, years_horizon + 1):
        # Buying cumulative cost up to year yr
        emi_paid_yr = emi * min(yr * 12, n)
        maint_yr = maintenance_annual * yr
        property_val_yr = property_price * ((1 + property_appreciation_pct / 100) ** yr)
        # If sold at year yr
        months_left_at_yr = max(0, n - yr * 12)
        remaining_principal_yr = loan_amount * (months_left_at_yr / n) if months_left_at_yr > 0 else 0
        sale_at_yr = property_val_yr - remaining_principal_yr
        buy_cost_yr = down_payment + emi_paid_yr + maint_yr - sale_at_yr

        # Renting cumulative cost up to year yr
        rent_paid_yr = 0
        cur = monthly_rent
        for i in range(yr):
            rent_paid_yr += cur * 12
            cur *= (1 + rent_increase_pct / 100)
        invest_yr = down_payment * ((1 + alternate_investment_return / 100) ** yr) - down_payment
        rent_cost_yr = rent_paid_yr - invest_yr

        if buy_cost_yr < rent_cost_yr and breakeven_year is None:
            breakeven_year = yr

    return {
        "scenario_horizon_years": years_horizon,
        "rent_estimate": rent_estimate,
        "buying": {
            "down_payment": round(down_payment),
            "monthly_emi": round(emi),
            "loan_amount": round(loan_amount),
            "total_emi_paid": round(total_emi_paid),
            "total_maintenance": round(total_maintenance),
            "future_property_value": round(future_property_value),
            "sale_proceeds": round(sale_proceeds),
            "net_cost": round(buy_net_cost),
        },
        "renting": {
            "starting_monthly_rent": round(monthly_rent),
            "total_rent_paid": round(total_rent),
            "down_payment_invested_value": round(investment_value),
            "investment_gain": round(investment_gain),
            "net_cost": round(rent_net_cost),
        },
        "savings_by_choosing_better": round(abs(savings)),
        "verdict": verdict,
        "recommendation": recommendation,
        "breakeven_year": breakeven_year,
        "assumptions": {
            "loan_rate": loan_rate,
            "loan_tenure": loan_tenure,
            "down_payment_pct": down_payment_pct,
            "rent_increase_pct": rent_increase_pct,
            "property_appreciation_pct": property_appreciation_pct,
            "alternate_investment_return": alternate_investment_return,
        },
    }

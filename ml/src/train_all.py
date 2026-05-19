"""
PropWise — Train All Models
=============================
Runs the full training pipeline:
  1. Prepare data (clean Mumbai1.csv)
  2. Train price predictor (XGBoost regressor)
  3. Train investment score classifier
  4. Train anomaly detector (Isolation Forest)

Run: python ml/src/train_all.py
"""

import sys
from pathlib import Path

# Add src dir to path so we can import other modules
sys.path.insert(0, str(Path(__file__).parent))

import prepare_data
import train_model
import train_investment_model
import train_anomaly_model


def main():
    print("=" * 60)
    print("PROPWISE — TRAINING ALL MODELS")
    print("=" * 60)

    print("\n[1/4] Preparing data...")
    print("-" * 60)
    prepare_data.main()

    print("\n[2/4] Training price predictor...")
    print("-" * 60)
    train_model.main()

    print("\n[3/4] Training investment score model...")
    print("-" * 60)
    train_investment_model.main()

    print("\n[4/4] Training anomaly detector...")
    print("-" * 60)
    train_anomaly_model.main()

    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext: cd ml && uvicorn service.main:app --reload --port 8000")


if __name__ == "__main__":
    main()

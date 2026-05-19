# 🏠 PropWise — AI Real Estate Intelligence for Mumbai

Predict fair market price, investment quality, and anomalies on Mumbai properties using **3 ML models trained on real Kaggle data**.

[Live demo](https://propwise.vercel.app) (after deploy)

## ✨ What it does

Enter a Mumbai property's details (BHK, sqft, location, amenities) and instantly get:
- 🎯 **Fair price prediction** with confidence range
- 📊 **Investment score** — Overpriced / Fair / Good deal / Excellent deal
- ✅ **Anomaly detection** — flags suspicious listings
- 💡 **SHAP explanation** — why the model predicted that price
- 🔑 **Rent vs Buy** — 10-year financial comparison
- 💰 **EMI calculator** + Similar properties + Neighborhood stats + Top areas

## 🧠 The 3 ML models

| Model | Type | Accuracy | Trained on |
|---|---|---|---|
| Price Predictor | XGBoost regressor | **81.7%** | 5,053 real Mumbai listings |
| Investment Score | XGBoost classifier | 67% | Same dataset, 4 tiers |
| Anomaly Detector | Isolation Forest | 5% flag rate | Unsupervised |

**No synthetic data, no third-party APIs, no LLMs in the pipeline.** 100% real ML.

## 🏗️ Architecture

```
Browser  →  Next.js (Vercel)  →  FastAPI (Render)  →  XGBoost models  →  SQLite
                                                                          ↓
                                                              user feedback for retraining
```

## 🚀 Deploy

See [DEPLOY.md](./DEPLOY.md) — full step-by-step guide for Render (backend) + Vercel (frontend). Free tiers, ~25 min total.

## 💻 Run locally

**Backend:**
```bash
pip install -r requirements.txt
cd ml
uvicorn service.main:app --reload --port 8000
```

**Frontend (separate terminal):**
```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000

## 🔁 Continuous learning

Users can submit actual purchase prices via the feedback widget. Run `python ml/src/retrain.py` to retrain on combined data. New model only deploys if accuracy improves.

## 📂 Project structure

```
propwise/
├── ml/
│   ├── data/raw/          # Mumbai1.csv, House_Rent_Dataset.csv (Kaggle)
│   ├── data/processed/    # mumbai_clean.csv
│   ├── models/            # .joblib files (trained models)
│   ├── src/               # train_model.py, retrain.py, prepare_data.py
│   └── service/           # FastAPI: main.py, features.py, rent_buy.py, explain.py
├── web/app/page.jsx       # Next.js single-page UI
├── requirements.txt
├── render.yaml            # One-click Render deploy config
└── DEPLOY.md              # Step-by-step deploy guide
```

## 📊 API endpoints (15 total)

`GET /` `GET /locations` `POST /predict` `POST /analyze` `POST /feedback` `POST /emi` `GET /similar/{loc}/{bhk}/{area}` `GET /neighborhood/{loc}` `POST /compare` `GET /top-neighborhoods` `POST /rent-vs-buy` `POST /explain` `GET /price-trend/{loc}` `GET /admin/dashboard` `GET /stats`

## 📜 License

MIT

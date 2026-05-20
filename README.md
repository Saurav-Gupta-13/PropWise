# 🏠 PropWise — AI Real Estate Intelligence for Mumbai

**Live Demo:** [prop-wise-one.vercel.app](https://prop-wise-one.vercel.app)  
**Backend API:** [propwise-f6pz.onrender.com](https://propwise-f6pz.onrender.com)  
**GitHub:** [Saurav-Gupta-13/PropWise](https://github.com/Saurav-Gupta-13/PropWise)

---

## 📌 What is PropWise?

PropWise is a full-stack ML-powered web application that predicts fair market prices for Mumbai properties. Users enter property details (location, BHK, area, amenities) and get instant AI-driven analysis including price prediction, deal quality assessment, fraud detection, rent vs buy comparison, and neighborhood insights.

**No external APIs. No LLMs. Pure machine learning trained on real data.**

---

## 🧠 Models & Algorithms

### Model 1: Price Predictor (Ensemble — XGBoost + LightGBM)

| Metric | Value |
|---|---|
| **Accuracy (MAPE-based)** | **84.5%** |
| R² Score | 0.8846 |
| Mean Absolute Error | ₹26.4 Lakhs |
| Training samples | 5,029 |
| Test samples | 1,258 |
| Ensemble weights | 30% XGBoost + 70% LightGBM |

**Why XGBoost + LightGBM?**
- Both are gradient-boosted tree algorithms — the gold standard for tabular/structured data
- XGBoost: Better at capturing complex feature interactions, more regularized
- LightGBM: Faster training, handles categorical features natively, better on this dataset (84.7% alone)
- Ensemble averaging reduces overfitting — when one model makes an error, the other often compensates
- Trees handle non-linear relationships (e.g., price doesn't scale linearly with area)

**Why NOT these alternatives?**

| Algorithm | Why rejected |
|---|---|
| Linear Regression | Assumes linear relationship. Mumbai prices are highly non-linear (Bandra vs Virar) |
| Random Forest | Tested — 78% accuracy. Less precise than boosted trees on this data |
| Neural Network / Deep Learning | Needs 50K+ samples to outperform trees. We have 5K — would overfit |
| SVR (Support Vector Regression) | Too slow on 5K rows × 23 features. No interpretability |
| CatBoost | Tested — similar to LightGBM (~84%). Added complexity for no gain |
| LLM (GPT/Gemini) | Not ML — just text generation. Can't learn from structured data |

### Model 2: Investment Score (XGBoost Classifier)

| Metric | Value |
|---|---|
| Accuracy | 67% |
| Classes | Overpriced, Fair, Good Deal, Excellent Deal |
| Method | Compares predicted fair price to area median |

### Model 3: Anomaly Detector (Isolation Forest)

| Metric | Value |
|---|---|
| Anomaly rate | 5% |
| Method | Unsupervised — flags listings where features don't match price profile |
| Use case | Catches fraudulent/fake listings |

---

## 📊 Dataset

### Source
- **Mumbai1.csv** — 6,348 real Mumbai property listings scraped from housing platforms
- **House_Rent_Dataset.csv** — Mumbai rental data (used for Rent vs Buy feature)

### Raw features (18)
| Feature | Type | Example |
|---|---|---|
| Price | Target | ₹1.5 Cr |
| Area (sqft) | Numeric | 850 |
| Location | Categorical (409 areas) | Andheri West |
| BHK | Numeric | 2 |
| New/Resale | Binary | 0 or 1 |
| Lift | Binary | 1 |
| Parking | Binary | 1 |
| 11 Amenities | Binary each | Gym, Pool, Security, etc. |

### Engineered features (5 additional)
| Feature | Formula | Why |
|---|---|---|
| `log_area` | log(area_sqft) | Captures diminishing returns on size |
| `area_per_bhk` | area / bhk | Measures spaciousness per room |
| `bhk_sq` | bhk² | Non-linear bedroom premium |
| `is_luxury` | bhk≥4 OR area>2000 | Luxury segment flag |
| `loc_smoothed_psf` | Smoothed location avg ₹/sqft | Target-encoded location signal |

**Total features used: 23**

---

## 🔧 Data Cleaning & Preprocessing

| Step | What | Result |
|---|---|---|
| 1. Load raw CSV | Read Mumbai1.csv | 6,348 rows |
| 2. Drop missing | Remove rows with null price/area/location/bhk | 6,317 rows |
| 3. Price filter | Keep ₹10L – ₹200Cr only | Removes unrealistic listings |
| 4. Area filter | Keep 100 – 20,000 sqft | Removes data entry errors |
| 5. BHK filter | Keep 1-10 BHK | Removes outliers |
| 6. Price/sqft filter | Keep ₹3,000 – ₹2,50,000/sqft | Removes extreme outliers |
| 7. Location normalize | "Andheri West, Mumbai" → "andheri_west" | 409 unique areas |
| 8. IQR outlier removal | Per-location, remove if outside 2×IQR from 10th-90th percentile | 6,287 rows final |
| 9. Binary encoding | All amenities → 0/1 | Clean boolean features |
| 10. Amenity count | Sum of all 11 amenities | Single aggregate feature |
| 11. Log-transform target | y = log(price) | Normalizes skewed distribution |
| 12. Label encoding | Location string → integer | For tree model input |
| 13. Feature engineering | Add 5 derived features | Richer signal |

---

## 📈 Accuracy Journey

| Version | Accuracy | What changed |
|---|---|---|
| V1 — Basic XGBoost | 81.7% | Label-encoded location, raw features |
| V2 — Feature engineering | 83.4% | +log_area, area_per_bhk, loc_smoothed_psf, bhk_sq, is_luxury |
| **V3 — Ensemble** | **84.5%** | XGBoost + LightGBM weighted average |

### What we tried that DIDN'T work
| Attempt | Result | Why it failed |
|---|---|---|
| Merging Mumbai.csv (7,719 rows) | 62.9% ⬇️ | Different price methodology (builder-quoted vs actual) |
| Merging data_for_model.csv (7,483 rows) | 81.7% | Different location naming conventions diluted signal |
| More trees (3000 estimators) | Same 83.4% | Overfitting — early stopping kicks in |
| Deep hyperparameter tuning | +0.1% | Diminishing returns after initial tuning |

---

## 🏗️ Architecture

```
┌──────────────────┐       ┌─────────────────────────────────┐
│   User Browser   │       │   Vercel (Frontend)             │
│   (any device)   │◄─────►│   Next.js 14 + React            │
└──────────────────┘       └───────────────┬─────────────────┘
                                           │ HTTPS
                                           ▼
                           ┌─────────────────────────────────┐
                           │   Render (Backend)              │
                           │   FastAPI + Python 3.11         │
                           │                                 │
                           │   ┌───────────────────────┐     │
                           │   │ XGBoost + LightGBM    │     │
                           │   │ (Ensemble Model)      │     │
                           │   └───────────────────────┘     │
                           │   ┌───────────────────────┐     │
                           │   │ Investment Classifier  │     │
                           │   └───────────────────────┘     │
                           │   ┌───────────────────────┐     │
                           │   │ Isolation Forest       │     │
                           │   └───────────────────────┘     │
                           │   ┌───────────────────────┐     │
                           │   │ SQLite (feedback DB)   │     │
                           │   └───────────────────────┘     │
                           └─────────────────────────────────┘
                                           │
                                           ▼ (weekly cron)
                           ┌─────────────────────────────────┐
                           │   Auto-Retrain (Background)     │
                           │   Feedback → Retrain → Deploy   │
                           └─────────────────────────────────┘
```

---

## 🌐 API Endpoints (16 total)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Health check + model status |
| `/locations` | GET | 409 Mumbai neighborhoods |
| `/predict` | POST | Single price prediction |
| `/analyze` | POST | Full analysis (price + investment + anomaly + negotiation) |
| `/feedback` | POST | User submits actual price for retraining |
| `/emi` | POST | EMI calculator |
| `/similar/{loc}/{bhk}/{area}` | GET | Comparable properties |
| `/neighborhood/{loc}` | GET | Area statistics |
| `/compare` | POST | Multi-area comparison |
| `/top-neighborhoods` | GET | Rankings by price |
| `/rent-vs-buy` | POST | Financial comparison (10-year) |
| `/explain` | POST | SHAP feature importance |
| `/price-trend/{loc}` | GET | 5-year projected trend |
| `/admin/retrain` | POST | Auto-retrain from feedback |
| `/admin/dashboard` | GET | HTML admin panel |
| `/stats` | GET | Usage statistics |

---

## 🔄 Continuous Learning Pipeline

```
User uses app → Submits actual price via feedback widget
                        ↓
              Saved to SQLite (feedback table)
                        ↓
         Every Sunday 2 AM (cron-job.org → POST /admin/retrain)
                        ↓
    Load original 5,029 rows + feedback rows → Retrain XGBoost+LightGBM
                        ↓
    If new accuracy ≥ old accuracy - 1% → Hot-swap model (zero downtime)
    If worse → Reject, keep current model
```

---

## ✅ Applications (Real-world use cases)

1. **Home buyers** — Check if listed price is fair before negotiating
2. **Sellers** — Price your property competitively based on market data
3. **Real estate agents** — Quick valuation tool for client meetings
4. **Investors** — Identify underpriced deals across Mumbai neighborhoods
5. **Banks/NBFCs** — Automated property valuation for loan approvals
6. **Fraud detection** — Flag suspicious listings on property portals

---

## ⚠️ Limitations & Drawbacks

| Limitation | Why | Impact |
|---|---|---|
| 84.5% accuracy (not 95%+) | Real estate is inherently noisy — same flat sells for ±15% depending on negotiation | Predictions have ₹20-30L margin of error |
| Mumbai only | Trained exclusively on Mumbai data | Won't work for Bangalore/Delhi/Pune |
| No floor/age/bathroom features | Primary dataset lacks these | Misses key price signals |
| Location granularity | 409 areas, but some have <5 listings | Low-data areas are less reliable |
| Price trend is projected | Uses 8% CAGR, not actual historical data | Directional only, not precise |
| Free-tier hosting | Render sleeps after 15 min inactivity | First load after sleep = 30-60s delay |
| SQLite on ephemeral disk | Feedback DB wipes on Render redeploy | Long-term feedback needs Postgres migration |

---

## 💡 Benefits

- **Zero API costs** — no Gemini/OpenAI dependency, runs entirely on own models
- **Real ML** — actual trained models, not LLM text generation
- **Fast** — 17-100ms response time (vs 2-5s for API-based solutions)
- **Transparent** — shows confidence, SHAP explanations, assumptions
- **Self-improving** — gets better with user feedback over time
- **Mobile responsive** — works on all phones and browsers
- **Open source** — full code visible on GitHub

---

## 🚀 How to Make it More Scalable

| Level | What to do | Impact |
|---|---|---|
| **Data** | Scrape 50K+ listings from MagicBricks/99acres with floor/age/builder features | 90%+ accuracy |
| **Data** | Add geo features (distance to metro, airport, sea) via Google Maps API | +2-3% accuracy |
| **Model** | Switch to CatBoost with native categorical handling | Slightly better location handling |
| **Infra** | Migrate from Render free → AWS Lambda or Railway ($7/mo) | No sleep, always fast |
| **DB** | Migrate from SQLite → Postgres (Render free tier) | Persistent feedback across deploys |
| **Scale** | Add Redis caching for repeated predictions | 10x faster for same inputs |
| **Coverage** | Train separate models per city (Mumbai, Pune, Bangalore) | Pan-India support |
| **Features** | Add image analysis (property photos → quality score) | Richer predictions |
| **Auth** | Add user accounts + saved properties | Personalized experience |
| **Notifications** | Price drop alerts for watched areas | User retention |

---

## 📂 Project Structure

```
propwise/
├── ml/
│   ├── data/
│   │   ├── raw/              # Mumbai1.csv, House_Rent_Dataset.csv, mumbai_flats.csv
│   │   └── processed/        # mumbai_clean.csv (after preprocessing)
│   ├── models/               # .joblib files (trained models)
│   ├── service/              # FastAPI application
│   │   ├── main.py           # All 16 endpoints
│   │   ├── features.py       # EMI, similar properties, neighborhoods
│   │   ├── rent_buy.py       # Rent vs Buy calculator
│   │   ├── explain.py        # SHAP explainability
│   │   └── database.py       # SQLite operations
│   └── src/                  # Training scripts
│       ├── prepare_data.py   # Data cleaning pipeline
│       ├── train_model.py    # V1 XGBoost training
│       ├── train_model_v2.py # V2 with feature engineering
│       ├── train_ensemble.py # V3 XGBoost + LightGBM ensemble
│       └── retrain.py        # Continuous learning script
├── web/
│   └── app/
│       ├── page.jsx          # Entire frontend (single-page app)
│       └── layout.jsx        # HTML head + viewport
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deploy config
├── DEPLOY.md                 # Step-by-step deploy guide
└── README.md                 # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML Training | XGBoost, LightGBM, scikit-learn | Industry standard for tabular data |
| Data | pandas, numpy | Fast dataframe operations |
| API | FastAPI, Uvicorn | Fastest Python web framework, async support |
| Frontend | Next.js 14, React 18 | Server-side rendering, fast deploys on Vercel |
| Database | SQLite | Lightweight, zero-config, embedded |
| Hosting (Backend) | Render (free tier) | Free Python hosting with auto-deploy |
| Hosting (Frontend) | Vercel (free tier) | Free Next.js hosting, global CDN |
| Monitoring | cron-job.org | Free keep-alive pinger + auto-retrain scheduler |

---

## 💻 Run Locally

```bash
# Backend
pip install -r requirements.txt
cd ml
uvicorn service.main:app --reload --port 8000

# Frontend (separate terminal)
cd web
npm install
npm run dev
```

Open http://localhost:3000

---

## 📜 License

MIT — free to use, modify, and distribute.

---

Built by [Saurav Gupta](https://github.com/Saurav-Gupta-13)

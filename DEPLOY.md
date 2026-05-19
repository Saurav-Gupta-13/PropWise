# PropWise — Deployment Guide

Deploy in two parts: **ML backend on Render** + **Frontend on Vercel**. Both have free tiers. Total time: ~25 minutes.

---

## Part 1: Push to GitHub (5 min)

```bash
cd c:\Users\Lenovo\Downloads\ARBITER_v8\propwise
git init
git add .
git commit -m "PropWise v1 — initial release"
```

Create a new repo on https://github.com/new (name it `propwise`, public or private).

```bash
git remote add origin https://github.com/YOUR_USERNAME/propwise.git
git branch -M main
git push -u origin main
```

---

## Part 2: Deploy ML Backend on Render (10 min)

1. Go to https://render.com → Sign in with GitHub
2. Click **New +** → **Web Service**
3. Connect your `propwise` repo
4. Fill in:
   - **Name:** `propwise-ml`
   - **Root Directory:** `ml`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r ../requirements.txt`
   - **Start Command:** `uvicorn service.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. Click **Create Web Service**

Wait 5-8 minutes for first deploy. You'll get a URL like:
**`https://propwise-ml.onrender.com`**

Verify by opening: `https://propwise-ml.onrender.com/` — you should see JSON with `"status": "ready"`.

> ⚠️ **Free-tier note:** Render free services sleep after 15 min of inactivity. The first request after sleep takes ~30 sec to wake up. For real users, upgrade to Starter ($7/mo) or use Railway.

---

## Part 3: Deploy Frontend on Vercel (10 min)

1. Go to https://vercel.com → Sign in with GitHub
2. Click **Add New** → **Project**
3. Import your `propwise` repo
4. Configure:
   - **Root Directory:** `web`
   - **Framework Preset:** Next.js (auto-detected)
   - **Environment Variables:** Add one:
     - Name: `NEXT_PUBLIC_ML_URL`
     - Value: `https://propwise-ml.onrender.com` (from Part 2)
5. Click **Deploy**

Wait ~2 minutes. You'll get a URL like:
**`https://propwise.vercel.app`**

Open it. Try a prediction. **This is the URL you share with friends.**

---

## After deploy — share with testers

Send them: `https://propwise.vercel.app`

What to ask them:
- "Try a property you know the price of — does our prediction match?"
- "Try locations you know — does the ranking feel right?"
- Use the in-app feedback widget (⭐ rating + comment) — saved to your SQLite DB

To see what testers entered:
- Visit: `https://propwise-ml.onrender.com/admin/dashboard`

---

## When to update production

After collecting 10+ feedback entries:
```bash
cd ml
python src/retrain.py
git add models/
git commit -m "Retrain with user feedback"
git push
```
Render and Vercel auto-redeploy on push. New model live in ~5 minutes.

---

## Quick rollback

In Render/Vercel dashboards, every deploy keeps history. One click reverts to the previous version.

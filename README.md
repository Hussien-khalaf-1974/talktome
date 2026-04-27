# 📡 Talk to Me — Streamlit Churn App

## Free Deployment: Streamlit Community Cloud ✅

**100% free, no credit card, no Visa required.**
Built specifically for Streamlit apps — it's the official free hosting platform made by the Streamlit team.

---

## Project structure

```
talktome_streamlit/
├── app.py                  ← Main Streamlit app (only file you run)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Dark theme config
└── models/                 ← PUT YOUR 4 MODEL FILES HERE
    ├── tuned_churn_model.pkl
    ├── churn_scaler.pkl
    ├── selected_features.pkl
    └── model_config.json
```

---

## Step-by-step deployment (≈ 10 minutes total)

### STEP 1 — Add your model files

Copy your 4 files into the `models/` folder:
- `tuned_churn_model.pkl`
- `churn_scaler.pkl`
- `selected_features.pkl`
- `model_config.json` (already provided — update with your real values)

---

### STEP 2 — Push to GitHub

1. Go to https://github.com → sign in → New repository
2. Name it `talktome-churn` → Private → Create
3. In your terminal (inside the `talktome_streamlit/` folder):

```bash
git init
git add .
git commit -m "Talk to Me — churn app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/talktome-churn.git
git push -u origin main
```

---

### STEP 3 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Click **Sign in with GitHub** (free, no card needed)
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/talktome-churn`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. Wait ~2 minutes → your app is live at:
   `https://YOUR_USERNAME-talktome-churn-app-XXXXX.streamlit.app`

---

### STEP 4 — Update the app

Any time you push to GitHub, the app redeploys automatically:

```bash
git add .
git commit -m "Update: describe your change"
git push
```

---

## Run locally (for testing)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

---

## Notes

- Streamlit Community Cloud is **permanently free** for public and private repos
- Apps sleep after 7 days of no traffic — wake them by visiting the URL
- No size limit issues — sklearn/xgboost/plotly all work fine
- You can invite collaborators from the Streamlit dashboard

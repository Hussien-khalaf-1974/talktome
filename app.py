"""
Talk to Me — Telecom Churn Intelligence
Streamlit application: Dashboard · Single Prediction · Batch Upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import io
import os

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talk to Me — Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --primary:   #0C5CAB;
    --primary-l: #1a6dc2;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --surface:   #09090b;
    --s1:        #111113;
    --s2:        #18181b;
    --s3:        #27272a;
    --border:    rgba(255,255,255,.07);
    --text:      #fafafa;
    --t2:        #a1a1aa;
    --t3:        #71717a;
    --mono:      'IBM Plex Mono', monospace;
}

/* ── App background ── */
html, body, .stApp {
    background-color: #09090b !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #fafafa !important;
}

/* ── Main block padding ── */
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111113 !important;
    border-right: 1px solid rgba(255,255,255,.07) !important;
    min-width: 240px !important;
    max-width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.2rem 1rem !important;
}

/* Hide default sidebar header decoration */
[data-testid="stSidebarUserContent"] { padding-top: 0 !important; }

/* ── Sidebar nav button ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    color: #a1a1aa !important;
    text-align: left !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 140ms ease !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,.06) !important;
    color: #fafafa !important;
}

/* ── Active nav button (via class injected by JS workaround via st.session_state) ── */
.nav-active button {
    background: rgba(12,92,171,.2) !important;
    color: #60a5fa !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #18181b !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] > div {
    color: #71717a !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] > div {
    color: #fafafa !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -1px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #18181b !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid rgba(255,255,255,.07) !important;
    gap: 2px !important;
    padding: 4px 6px 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: #71717a !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    border-radius: 7px 7px 0 0 !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    background: rgba(12,92,171,.15) !important;
    border-bottom: 2px solid #0C5CAB !important;
}
[data-testid="stTabPanel"] {
    background: #18181b !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.4rem !important;
}

/* ── Form fields ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: #27272a !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 7px !important;
    color: #fafafa !important;
    font-size: 13px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: rgba(12,92,171,.6) !important;
    box-shadow: 0 0 0 3px rgba(12,92,171,.12) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    color: #71717a !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: .8px !important;
    font-weight: 600 !important;
}

/* ── Primary submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background: #0C5CAB !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    box-shadow: 0 0 20px rgba(12,92,171,.3) !important;
    transition: all 140ms ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #1a6dc2 !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #18181b !important;
    border: 1.5px dashed rgba(255,255,255,.12) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(12,92,171,.5) !important;
    background: rgba(12,92,171,.04) !important;
}
[data-testid="stFileUploader"] label { color: #a1a1aa !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #a1a1aa !important;
    font-size: 13px !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,.07) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,.07) !important; }

/* ── Info / warning / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 12.5px !important;
    border: 1px solid !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 99px; }

/* ── Custom card HTML ── */
.ttm-card {
    background: #18181b;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: border-color 140ms ease;
    margin-bottom: 0;
}
.ttm-card:hover { border-color: rgba(255,255,255,.12); }

.ttm-card-glow-blue::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(12,92,171,.7), transparent);
}
.ttm-card-glow-green::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(16,185,129,.6), transparent);
}
.ttm-card-glow-red::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(239,68,68,.6), transparent);
}
.ttm-card-glow-yellow::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(245,158,11,.6), transparent);
}

.ttm-lbl {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-weight: 600;
    color: #71717a;
    margin-bottom: 8px;
}
.ttm-val {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -1.2px;
    line-height: 1;
    font-family: 'IBM Plex Mono', monospace;
}
.ttm-sub { font-size: 11px; color: #71717a; margin-top: 4px; }

.ttm-delta {
    display: inline-flex; align-items: center; gap: 3px;
    font-size: 10px; font-family: 'IBM Plex Mono', monospace;
    padding: 2px 7px; border-radius: 4px; margin-top: 6px;
    border: 1px solid;
}
.d-up   { background: rgba(16,185,129,.12); color:#10b981; border-color:rgba(16,185,129,.25); }
.d-down { background: rgba(239,68,68,.10);  color:#ef4444; border-color:rgba(239,68,68,.25);  }
.d-flat { background: rgba(245,158,11,.10); color:#f59e0b; border-color:rgba(245,158,11,.25); }

.ttm-section-title {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    color: #71717a;
    padding: 0.7rem 0 0.4rem;
    border-top: 1px solid rgba(255,255,255,.07);
    margin-top: 0.4rem;
}

.verdict-box {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    border: 1px solid;
    margin-bottom: 0.8rem;
}
.vbox-churn  { background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.3); }
.vbox-retain { background: rgba(16,185,129,.07); border-color: rgba(16,185,129,.3); }

.verdict-icon { font-size: 28px; flex-shrink: 0; margin-top: 2px; }
.verdict-title { font-size: 18px; font-weight: 700; letter-spacing: -.4px; margin-bottom: 3px; }
.vt-churn  { color: #ef4444; }
.vt-retain { color: #10b981; }
.verdict-prob { font-size: 10px; color: #71717a; font-family: 'IBM Plex Mono', monospace; }

.meter-wrap {
    height: 5px; background: #27272a; border-radius: 99px;
    overflow: hidden; margin-top: 8px;
}
.meter-fill { height: 100%; border-radius: 99px; }

.metric-row {
    display: flex; border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px; overflow: hidden; margin-top: 8px;
}
.metric-cell {
    flex: 1; padding: 9px 12px;
    border-right: 1px solid rgba(255,255,255,.07);
    background: #18181b;
}
.metric-cell:last-child { border-right: none; }
.mc-label { font-size: 9px; color: #71717a; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 3px; }
.mc-value  { font-size: 17px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

.tip-row {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 8px 10px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 7px;
    font-size: 11.5px;
    color: #a1a1aa;
    margin-bottom: 6px;
}
.tip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    flex-shrink: 0; margin-top: 5px;
}

.prog-row { margin-bottom: 8px; }
.prog-head { display: flex; justify-content: space-between; font-size: 10px; color: #a1a1aa; margin-bottom: 4px; }
.prog-val  { font-family: 'IBM Plex Mono', monospace; color: #fafafa; }
.prog-track { height: 4px; background: #27272a; border-radius: 99px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 99px; }

.badge {
    display: inline-flex; align-items: center;
    padding: 2px 7px; border-radius: 4px;
    font-size: 9px; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid;
}
.badge-r { background:rgba(239,68,68,.1);  color:#ef4444; border-color:rgba(239,68,68,.3);  }
.badge-g { background:rgba(16,185,129,.1); color:#10b981; border-color:rgba(16,185,129,.3); }
.badge-y { background:rgba(245,158,11,.1); color:#f59e0b; border-color:rgba(245,158,11,.3); }
.badge-b { background:rgba(12,92,171,.15); color:#60a5fa; border-color:rgba(12,92,171,.4);  }

.sidebar-brand {
    font-size: 17px; font-weight: 700;
    color: #fafafa; letter-spacing: -.3px;
    margin-bottom: 2px;
}
.sidebar-sub {
    font-size: 9.5px; color: #71717a;
    text-transform: uppercase; letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 18px;
}
.sidebar-section {
    font-size: 9px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #52525b; font-weight: 600;
    padding: 12px 4px 5px;
}
.model-pill {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 8px;
    padding: 9px 11px;
    margin-top: 8px;
}
.mp-label { font-size: 9px; color: #52525b; text-transform: uppercase; letter-spacing:.8px; margin-bottom:3px; }
.mp-name  { font-family: 'IBM Plex Mono', monospace; color: #60a5fa; font-size: 11px; font-weight: 600; }
.mp-auc   { font-family: 'IBM Plex Mono', monospace; color: #10b981; font-size: 10px; margin-top: 2px; }

.feat-pill {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 7px; padding: 8px 10px; margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px; color: #60a5fa; line-height: 1.9;
}

.page-header {
    font-size: 20px; font-weight: 700; letter-spacing: -.5px;
    color: #fafafa; margin-bottom: 4px;
}
.page-sub {
    font-size: 12px; color: #71717a; margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL ARTEFACTS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artefacts():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "models")
    model    = joblib.load(os.path.join(models_dir, "tuned_churn_model.pkl"))
    scaler   = joblib.load(os.path.join(models_dir, "churn_scaler.pkl"))
    features = joblib.load(os.path.join(models_dir, "selected_features.pkl"))
    with open(os.path.join(models_dir, "model_config.json")) as f:
        config = json.load(f)
    return model, scaler, features, config

model, scaler, selected_features, config = load_artefacts()
THRESHOLD  = config.get("optimal_threshold", 0.55)
MODEL_NAME = config.get("model_name", "Logistic Regression")
VAL_AUC    = config.get("val_auc", 0.837)

# ─────────────────────────────────────────────────────────────────
#  ENCODING HELPER
# ─────────────────────────────────────────────────────────────────
def encode_row(raw: dict) -> dict:
    r = {}
    for col in ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
                "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]:
        val = str(raw.get(col, "No"))
        r[col] = 1 if val in ("Yes","Male") else 0
    r["SeniorCitizen"]  = int(raw.get("SeniorCitizen", 0))
    r["tenure"]         = float(raw.get("tenure", 0))
    r["MonthlyCharges"] = float(raw.get("MonthlyCharges", 0))
    r["TotalCharges"]   = float(raw.get("TotalCharges", 0))
    internet = str(raw.get("InternetService","DSL"))
    r["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    r["InternetService_No"]          = 1 if internet == "No" else 0
    contract = str(raw.get("Contract","Month-to-month"))
    r["Contract_One year"] = 1 if contract == "One year" else 0
    r["Contract_Two year"] = 1 if contract == "Two year" else 0
    payment = str(raw.get("PaymentMethod","Bank transfer (automatic)"))
    r["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
    r["PaymentMethod_Electronic check"]        = 1 if payment == "Electronic check" else 0
    r["PaymentMethod_Mailed check"]            = 1 if payment == "Mailed check" else 0
    return r

def predict(encoded_df: pd.DataFrame):
    X = encoded_df[selected_features].values
    X_sc = scaler.transform(X)
    probas = model.predict_proba(X_sc)[:, 1]
    preds  = (probas >= THRESHOLD).astype(int)
    return probas, preds

def risk_level(p):
    return "High Risk" if p >= 0.65 else "Medium Risk" if p >= 0.40 else "Low Risk"

# ─────────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────────────────────────
PLOT_BG   = "#18181b"
PAPER_BG  = "#18181b"
GRID_COL  = "rgba(255,255,255,.04)"
TEXT_COL  = "#71717a"
FONT_MONO = "IBM Plex Mono"

def base_layout(**kwargs):
    return dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL, size=11),
        margin=dict(l=8, r=8, t=28, b=8),
        showlegend=False,
        **kwargs
    )

def axis_style(title=""):
    return dict(
        title=title,
        gridcolor=GRID_COL,
        linecolor="rgba(255,255,255,.07)",
        tickfont=dict(family=FONT_MONO, size=9, color=TEXT_COL),
        titlefont=dict(size=10, color=TEXT_COL),
    )

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">📡 Talk to Me</div>
    <div class="sidebar-sub">Churn Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Workspace</div>', unsafe_allow_html=True)

    pages = {
        "Dashboard":          "◈  Dashboard",
        "Single Prediction":  "◎  Single Prediction",
        "Batch Upload":       "≡  Batch Upload",
    }
    for key, label in pages.items():
        is_active = st.session_state.page == key
        btn_style = """
        <style>
        div[data-testid="stButton"]:last-of-type > button {
            background: rgba(12,92,171,.2) !important;
            color: #60a5fa !important;
        }
        </style>""" if is_active else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="sidebar-section" style="margin-top:12px">Selected Features (9)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-pill">
    TotalCharges · tenure<br>
    Contract_Two year<br>
    MonthlyCharges<br>
    InternetService_Fiber<br>
    PaymentMethod_Echeck<br>
    InternetService_No<br>
    OnlineSecurity<br>
    PaperlessBilling
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section" style="margin-top:12px">Active Model</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-pill">
        <div class="mp-label">Best model</div>
        <div class="mp-name">{MODEL_NAME}</div>
        <div class="mp-auc">AUC {VAL_AUC:.3f} · Threshold {THRESHOLD}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  REAL DATA (from your notebook results)
# ─────────────────────────────────────────────────────────────────
MODEL_RESULTS = {
    "Model":      ["LR", "RF", "XGB", "DT", "SVM"],
    "Accuracy":   [0.73, 0.74, 0.73, 0.71, 0.73],
    "F1 (Churn)": [0.61, 0.60, 0.59, 0.58, 0.61],
    "ROC-AUC":    [0.82, 0.81, 0.81, 0.80, 0.78],
}
FEATURES = {
    "name":  ["TotalCharges","tenure","Contract_2yr","MonthlyChg","Fiber optic",
               "Echeck","Internet_No","OnlineSecurity","Paperless"],
    "score": [0.883, 0.631, 0.535, 0.483, 0.430, 0.291, 0.230, 0.160, 0.122],
    "color": ["#1a6dc2","#1a6dc2","#f59e0b","#f59e0b","#ef4444",
               "#ef4444","#71717a","#71717a","#71717a"],
}
THRESH_DATA = {
    "t":   [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70],
    "f1":  [0.55,0.58,0.60,0.614,0.616,0.630,0.621,0.607,0.580],
    "acc": [0.68,0.70,0.72,0.735,0.747,0.769,0.775,0.779,0.782],
}

# ═════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="page-header">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Model performance overview · Real data from your training pipeline</div>',
                unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")

    with k1:
        st.markdown("""
        <div class="ttm-card ttm-card-glow-blue">
            <div class="ttm-lbl">Dataset</div>
            <div class="ttm-val">6,043</div>
            <div class="ttm-sub">Total customers</div>
            <div class="ttm-delta d-flat">70 / 15 / 15 stratified split</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="ttm-card ttm-card-glow-red">
            <div class="ttm-lbl">Churn Rate</div>
            <div class="ttm-val" style="color:#ef4444">26.5%</div>
            <div class="ttm-sub">1,604 of 6,043 churned</div>
            <div class="ttm-delta d-down">Handled via SMOTE balancing</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="ttm-card ttm-card-glow-green">
            <div class="ttm-lbl">Best ROC-AUC</div>
            <div class="ttm-val" style="color:#10b981">{VAL_AUC:.3f}</div>
            <div class="ttm-sub">{MODEL_NAME} · test set</div>
            <div class="ttm-delta d-up">Threshold optimised → {THRESHOLD}</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="ttm-card ttm-card-glow-yellow">
            <div class="ttm-lbl">Best F1 (Churn)</div>
            <div class="ttm-val" style="color:#f59e0b">0.630</div>
            <div class="ttm-sub">vs 0.616 at default threshold 0.50</div>
            <div class="ttm-delta d-up">+0.014 from threshold sweep</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Row 2: Model comparison + Feature importance ──────────────
    col_left, col_right = st.columns([3, 2], gap="small")

    with col_left:
        df_models = pd.DataFrame(MODEL_RESULTS)
        fig = go.Figure()
        bar_cfg = [
            ("Accuracy",   MODEL_RESULTS["Accuracy"],   "rgba(12,92,171,.8)"),
            ("F1 (Churn)", MODEL_RESULTS["F1 (Churn)"], "rgba(245,158,11,.8)"),
            ("ROC-AUC",    MODEL_RESULTS["ROC-AUC"],    "rgba(16,185,129,.8)"),
        ]
        for name, vals, color in bar_cfg:
            fig.add_trace(go.Bar(
                name=name, x=MODEL_RESULTS["Model"], y=vals,
                marker_color=color, marker_line_width=0,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                textfont=dict(family=FONT_MONO, size=9, color=TEXT_COL),
            ))
        fig.update_layout(
            **base_layout(barmode="group", title="Model Comparison — Validation Set",
                          title_font=dict(size=12, color="#a1a1aa"),
                          showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="right", x=1,
                                      font=dict(family=FONT_MONO, size=9, color=TEXT_COL),
                                      bgcolor="transparent")),
            xaxis=axis_style(), yaxis=dict(**axis_style(), range=[0.5, 0.92]),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        fig2 = go.Figure(go.Bar(
            x=FEATURES["score"], y=FEATURES["name"],
            orientation="h",
            marker_color=FEATURES["color"],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in FEATURES["score"]],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=9, color=TEXT_COL),
        ))
        fig2.update_layout(
            **base_layout(title="Feature Importance (Avg Score)",
                          title_font=dict(size=12, color="#a1a1aa")),
            xaxis=dict(**axis_style(), range=[0, 1.05]),
            yaxis=dict(**axis_style(), autorange="reversed"),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ── Row 3: Threshold + Metrics table + Donut ─────────────────
    c1, c2, c3 = st.columns([1.6, 2.4, 1.4], gap="small")

    with c1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=THRESH_DATA["t"], y=THRESH_DATA["f1"],
            name="F1 Churn", mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(245,158,11,.07)",
        ))
        fig3.add_trace(go.Scatter(
            x=THRESH_DATA["t"], y=THRESH_DATA["acc"],
            name="Accuracy", mode="lines+markers",
            line=dict(color="#1a6dc2", width=2, dash="dot"),
            marker=dict(size=5),
        ))
        fig3.add_vline(x=0.55, line_color="#10b981", line_dash="dash", line_width=1.5,
                       annotation_text="0.55", annotation_font=dict(color="#10b981", size=9))
        fig3.update_layout(
            **base_layout(title="Threshold Sweep",
                          title_font=dict(size=12, color="#a1a1aa"),
                          showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      font=dict(family=FONT_MONO, size=9, color=TEXT_COL),
                                      bgcolor="transparent")),
            xaxis=dict(**axis_style("Threshold")),
            yaxis=dict(**axis_style("Score"), range=[0.5, 0.85]),
            height=240,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        # Mini metric row below chart
        st.markdown("""
        <div class="metric-row" style="margin-top:6px">
          <div class="metric-cell">
            <div class="mc-label">Default (0.50)</div>
            <div class="mc-value" style="color:#f59e0b">F1 0.616</div>
          </div>
          <div class="metric-cell">
            <div class="mc-label">Optimal (0.55)</div>
            <div class="mc-value" style="color:#10b981">F1 0.630</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="ttm-card" style="padding:0;overflow:hidden">
          <div style="padding:10px 14px 8px;border-bottom:1px solid rgba(255,255,255,.07)">
            <span style="font-size:12px;font-weight:600;color:#fafafa">Test Set Classification Report</span>
            <span style="font-size:10px;color:#71717a;margin-left:8px">LR · threshold 0.55 · 906 samples</span>
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:11.5px">
            <thead>
              <tr style="background:#27272a">
                <th style="padding:7px 12px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:#71717a;font-weight:600;border-bottom:1px solid rgba(255,255,255,.07)">Class</th>
                <th style="padding:7px 12px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:#71717a;font-weight:600;border-bottom:1px solid rgba(255,255,255,.07)">Precision</th>
                <th style="padding:7px 12px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:#71717a;font-weight:600;border-bottom:1px solid rgba(255,255,255,.07)">Recall</th>
                <th style="padding:7px 12px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:#71717a;font-weight:600;border-bottom:1px solid rgba(255,255,255,.07)">F1</th>
                <th style="padding:7px 12px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:#71717a;font-weight:600;border-bottom:1px solid rgba(255,255,255,.07)">Support</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07)"><span class="badge badge-g">No Churn</span></td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.89</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.78</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.83</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#71717a">665</td>
              </tr>
              <tr>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07)"><span class="badge badge-r">Churn</span></td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.55</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.74</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.63</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#71717a">241</td>
              </tr>
              <tr style="background:rgba(255,255,255,.02)">
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07)"><span class="badge badge-b">Macro avg</span></td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.72</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.76</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#fafafa">0.73</td>
                <td style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);font-family:'IBM Plex Mono',monospace;color:#71717a">906</td>
              </tr>
              <tr style="background:rgba(255,255,255,.02)">
                <td style="padding:8px 12px"><span class="badge badge-b">Weighted</span></td>
                <td style="padding:8px 12px;font-family:'IBM Plex Mono',monospace;color:#fafafa">0.80</td>
                <td style="padding:8px 12px;font-family:'IBM Plex Mono',monospace;color:#fafafa">0.77</td>
                <td style="padding:8px 12px;font-family:'IBM Plex Mono',monospace;color:#fafafa">0.78</td>
                <td style="padding:8px 12px;font-family:'IBM Plex Mono',monospace;color:#71717a">906</td>
              </tr>
            </tbody>
          </table>
          <div style="padding:9px 14px;border-top:1px solid rgba(255,255,255,.07)">
            <div class="metric-row">
              <div class="metric-cell"><div class="mc-label">Accuracy</div><div class="mc-value">76.9%</div></div>
              <div class="metric-cell"><div class="mc-label">ROC-AUC</div><div class="mc-value" style="color:#10b981">0.837</div></div>
              <div class="metric-cell"><div class="mc-label">F1 Churn</div><div class="mc-value" style="color:#f59e0b">0.630</div></div>
              <div class="metric-cell"><div class="mc-label">Recall</div><div class="mc-value" style="color:#60a5fa">74%</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        fig4 = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[4439, 1604],
            hole=0.66,
            marker=dict(
                colors=["rgba(16,185,129,.8)", "rgba(239,68,68,.8)"],
                line=dict(color=PLOT_BG, width=3),
            ),
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig4.add_annotation(
            text="6,043", x=0.5, y=0.57, showarrow=False,
            font=dict(family=FONT_MONO, size=18, color="#fafafa"),
        )
        fig4.add_annotation(
            text="customers", x=0.5, y=0.43, showarrow=False,
            font=dict(family="IBM Plex Sans", size=10, color=TEXT_COL),
        )
        fig4.update_layout(
            **base_layout(title="Churn Distribution",
                          title_font=dict(size=12, color="#a1a1aa"),
                          showlegend=True,
                          legend=dict(orientation="h", yanchor="top", y=-0.05,
                                      font=dict(family=FONT_MONO, size=9, color=TEXT_COL),
                                      bgcolor="transparent")),
            height=240,
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        # Churn by contract bar below donut
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Retained", x=["M-t-M","1yr","2yr"], y=[1655,1141,1380],
            marker_color="rgba(16,185,129,.75)", marker_line_width=0,
        ))
        fig5.add_trace(go.Bar(
            name="Churned", x=["M-t-M","1yr","2yr"], y=[1666,120,81],
            marker_color="rgba(239,68,68,.75)", marker_line_width=0,
        ))
        fig5.update_layout(
            **base_layout(barmode="stack", title="Churn by Contract",
                          title_font=dict(size=12, color="#a1a1aa"),
                          showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      font=dict(family=FONT_MONO, size=9, color=TEXT_COL),
                                      bgcolor="transparent")),
            xaxis=axis_style(), yaxis=axis_style(),
            height=210,
        )
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════
#  PAGE: SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="page-header">Single Customer Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Fill in the customer profile — the model uses 9 key features at threshold 0.55</div>',
                unsafe_allow_html=True)

    col_form, col_result = st.columns([2, 1.5], gap="medium")

    with col_form:
        with st.form("predict_form"):
            # ── Demographics ──────────────────────────────────────
            st.markdown('<div class="ttm-section-title">👤 Demographics</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1: gender  = st.selectbox("Gender", ["Male","Female"])
            with d2: senior  = st.selectbox("Senior Citizen", ["No","Yes"])
            with d3: partner = st.selectbox("Partner", ["No","Yes"])
            d4, d5 = st.columns(2)
            with d4: dependents = st.selectbox("Dependents", ["No","Yes"])
            with d5: tenure     = st.number_input("Tenure (months)", 0, 72, 12)

            # ── Services ─────────────────────────────────────────
            st.markdown('<div class="ttm-section-title">📡 Services</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1: phone    = st.selectbox("Phone Service", ["Yes","No"])
            with s2: multiline= st.selectbox("Multiple Lines", ["No","Yes"])
            with s3: internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            s4, s5, s6 = st.columns(3)
            with s4: online_sec  = st.selectbox("Online Security", ["No","Yes"])
            with s5: online_bk   = st.selectbox("Online Backup", ["No","Yes"])
            with s6: device_prot = st.selectbox("Device Protection", ["No","Yes"])
            s7, s8, s9 = st.columns(3)
            with s7: tech_sup    = st.selectbox("Tech Support", ["No","Yes"])
            with s8: streaming_tv= st.selectbox("Streaming TV", ["No","Yes"])
            with s9: streaming_mv= st.selectbox("Streaming Movies", ["No","Yes"])

            # ── Billing ───────────────────────────────────────────
            st.markdown('<div class="ttm-section-title">💳 Billing & Contract</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1: contract  = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            with b2: paperless = st.selectbox("Paperless Billing", ["Yes","No"])
            with b3: payment   = st.selectbox("Payment Method",
                                              ["Electronic check","Mailed check",
                                               "Bank transfer (automatic)","Credit card (automatic)"])
            b4, b5 = st.columns(2)
            with b4: monthly_chg = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.01)
            with b5: total_chg   = st.number_input("Total Charges ($)",   0.0, 10000.0, 780.0, step=0.01)

            submitted = st.form_submit_button("⚡  Predict Churn", use_container_width=True)

    with col_result:
        if submitted:
            raw = {
                "gender": gender, "SeniorCitizen": 1 if senior=="Yes" else 0,
                "Partner": partner, "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone, "MultipleLines": multiline,
                "InternetService": internet, "OnlineSecurity": online_sec,
                "OnlineBackup": online_bk, "DeviceProtection": device_prot,
                "TechSupport": tech_sup, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_mv, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly_chg, "TotalCharges": total_chg,
            }
            try:
                enc  = encode_row(raw)
                df_e = pd.DataFrame([enc])
                probas, preds = predict(df_e)
                prob = float(probas[0])
                pred = int(preds[0])
                rl   = risk_level(prob)
                pct  = round(prob * 100, 1)
                col  = "#ef4444" if pct >= 65 else "#f59e0b" if pct >= 40 else "#10b981"

                is_churn = pred == 1
                bclass   = "vbox-churn" if is_churn else "vbox-retain"
                tc       = "vt-churn"   if is_churn else "vt-retain"
                icon     = "⚠️" if is_churn else "✅"
                label    = "Will Churn" if is_churn else "Will Stay"

                # ── Verdict banner ───────────────────────────────
                meter_fill = f'<div class="meter-fill" style="width:{pct}%;background:{col}"></div>'
                st.markdown(f"""
                <div class="verdict-box {bclass}">
                  <div class="verdict-icon">{icon}</div>
                  <div style="flex:1">
                    <div class="verdict-title {tc}">{label}</div>
                    <div class="verdict-prob">Probability {pct}% · {rl} · threshold {THRESHOLD}</div>
                    <div class="meter-wrap">{meter_fill}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # ── Metric row ───────────────────────────────────
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-cell">
                    <div class="mc-label">Probability</div>
                    <div class="mc-value" style="color:{col}">{pct}%</div>
                  </div>
                  <div class="metric-cell">
                    <div class="mc-label">Risk Level</div>
                    <div class="mc-value" style="font-size:13px">{rl}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # ── Gauge chart ──────────────────────────────────
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={"suffix":"%","font":{"family":FONT_MONO,"size":24,"color":"#fafafa"}},
                    gauge=dict(
                        axis=dict(range=[0,100], tickfont=dict(family=FONT_MONO,size=8,color=TEXT_COL),
                                  tickcolor=TEXT_COL),
                        bar=dict(color=col, thickness=0.25),
                        bgcolor=PLOT_BG,
                        borderwidth=0,
                        steps=[
                            dict(range=[0,40],  color="rgba(16,185,129,.12)"),
                            dict(range=[40,65], color="rgba(245,158,11,.12)"),
                            dict(range=[65,100],color="rgba(239,68,68,.12)"),
                        ],
                        threshold=dict(line=dict(color=col,width=2), thickness=0.75, value=pct),
                    ),
                ))
                fig_g.update_layout(
                    **base_layout(height=200),
                    margin=dict(l=20,r=20,t=20,b=10),
                )
                st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

                # ── Retention tips ───────────────────────────────
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                tips_churn = [
                    ("#ef4444","Call within 48h — personalised retention offer"),
                    ("#f59e0b","Propose annual contract with 2 months free"),
                    ("#f59e0b","Add Online Security or Tech Support to plan"),
                    ("#60a5fa","Switch from Electronic check to automatic payment"),
                ]
                tips_retain = [
                    ("#10b981","Stable customer — consider upsell opportunity"),
                    ("#60a5fa","Invite to loyalty programme or referral scheme"),
                    ("#71717a","Monitor for upcoming contract renewal date"),
                ]
                tips = tips_churn if is_churn else tips_retain
                html = '<div style="margin-top:4px">'
                for dot_col, text in tips:
                    html += f'<div class="tip-row"><div class="tip-dot" style="background:{dot_col}"></div>{text}</div>'
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

            except Exception as err:
                st.error(f"Prediction failed: {err}")

        else:
            # Guide shown before first prediction
            st.markdown("""
            <div class="ttm-card">
                <div class="ttm-lbl">How it works</div>
                <div class="tip-row"><div class="tip-dot" style="background:#60a5fa"></div>Fill in the customer profile and click <strong>Predict Churn</strong></div>
                <div class="tip-row"><div class="tip-dot" style="background:#f59e0b"></div>Model uses <strong>9 consensus features</strong> (MI + Chi² + RF)</div>
                <div class="tip-row"><div class="tip-dot" style="background:#10b981"></div>Optimised threshold <strong>0.55</strong> maximises F1 for churn class</div>
                <div class="tip-row"><div class="tip-dot" style="background:#ef4444"></div>High ≥65% · Medium ≥40% · Low &lt;40%</div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  PAGE: BATCH UPLOAD
# ═════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown('<div class="page-header">Batch CSV Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a CSV file with customer records — all rows scored instantly</div>',
                unsafe_allow_html=True)

    col_up, col_sum = st.columns([1.8, 2.2], gap="medium")

    with col_up:
        uploaded = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="Must include the same columns as training data",
        )
        st.markdown("""
        <div class="ttm-card" style="margin-top:10px">
            <div class="ttm-lbl">Required columns</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#60a5fa;line-height:2">
            customerID · gender · SeniorCitizen<br>
            Partner · Dependents · tenure<br>
            PhoneService · MultipleLines<br>
            InternetService · OnlineSecurity<br>
            OnlineBackup · DeviceProtection<br>
            TechSupport · StreamingTV<br>
            StreamingMovies · Contract<br>
            PaperlessBilling · PaymentMethod<br>
            MonthlyCharges · TotalCharges
            </div>
        </div>""", unsafe_allow_html=True)

    with col_sum:
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                ids = df_raw["customerID"] if "customerID" in df_raw.columns \
                      else pd.Series(range(1, len(df_raw)+1)).astype(str)
                df = df_raw.drop(columns=["customerID","Churn"], errors="ignore").copy()
                if "TotalCharges" in df.columns:
                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

                enc_rows  = [encode_row(row.to_dict()) for _, row in df.iterrows()]
                df_enc    = pd.DataFrame(enc_rows)
                probas, preds = predict(df_enc)

                results = pd.DataFrame({
                    "Customer ID":  ids.values,
                    "Probability":  (probas * 100).round(1),
                    "Prediction":   ["Churn" if p else "Retain" for p in preds],
                    "Risk Level":   [risk_level(p) for p in probas],
                })

                total      = len(results)
                churners   = int(preds.sum())
                retained   = total - churners
                churn_rate = round(churners / total * 100, 1)
                high_r     = int((probas >= 0.65).sum())
                med_r      = int(((probas >= 0.40) & (probas < 0.65)).sum())
                low_r      = int((probas < 0.40).sum())

                # KPI row
                k1, k2, k3, k4 = st.columns(4, gap="small")
                with k1:
                    st.markdown(f"""<div class="ttm-card"><div class="ttm-lbl">Total</div>
                    <div class="ttm-val">{total:,}</div></div>""", unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""<div class="ttm-card ttm-card-glow-red"><div class="ttm-lbl">Will Churn</div>
                    <div class="ttm-val" style="color:#ef4444">{churners:,}</div></div>""", unsafe_allow_html=True)
                with k3:
                    st.markdown(f"""<div class="ttm-card ttm-card-glow-green"><div class="ttm-lbl">Retained</div>
                    <div class="ttm-val" style="color:#10b981">{retained:,}</div></div>""", unsafe_allow_html=True)
                with k4:
                    st.markdown(f"""<div class="ttm-card"><div class="ttm-lbl">Churn Rate</div>
                    <div class="ttm-val">{churn_rate}%</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                # Risk breakdown bars
                st.markdown(f"""
                <div class="ttm-card">
                    <div class="ttm-lbl">Risk Breakdown</div>
                    <div class="prog-row">
                        <div class="prog-head"><span style="color:#ef4444">● High risk ≥65%</span><span class="prog-val">{high_r}</span></div>
                        <div class="prog-track"><div class="prog-fill" style="width:{high_r/total*100:.1f}%;background:#ef4444"></div></div>
                    </div>
                    <div class="prog-row">
                        <div class="prog-head"><span style="color:#f59e0b">● Medium risk ≥40%</span><span class="prog-val">{med_r}</span></div>
                        <div class="prog-track"><div class="prog-fill" style="width:{med_r/total*100:.1f}%;background:#f59e0b"></div></div>
                    </div>
                    <div class="prog-row">
                        <div class="prog-head"><span style="color:#10b981">● Low risk &lt;40%</span><span class="prog-val">{low_r}</span></div>
                        <div class="prog-track"><div class="prog-fill" style="width:{low_r/total*100:.1f}%;background:#10b981"></div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Batch bar chart
                fig_b = go.Figure(go.Bar(
                    x=["High Risk","Medium Risk","Low Risk"],
                    y=[high_r, med_r, low_r],
                    marker_color=["rgba(239,68,68,.8)","rgba(245,158,11,.8)","rgba(16,185,129,.8)"],
                    marker_line_width=0,
                    text=[high_r, med_r, low_r],
                    textposition="outside",
                    textfont=dict(family=FONT_MONO, size=10, color=TEXT_COL),
                ))
                fig_b.update_layout(
                    **base_layout(title="Risk Distribution",
                                  title_font=dict(size=11, color="#a1a1aa")),
                    xaxis=axis_style(), yaxis=axis_style(),
                    height=200,
                )
                st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

            except Exception as err:
                st.error(f"Processing failed: {err}")
                return

            # ── Results table ─────────────────────────────────────
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("**Customer Results**")

            # Colour-code the dataframe
            def style_pred(val):
                c = "#ef4444" if val == "Churn" else "#10b981"
                return f"color: {c}; font-weight: 600"
            def style_risk(val):
                if val == "High Risk":   return "color:#ef4444"
                if val == "Medium Risk": return "color:#f59e0b"
                return "color:#10b981"

            styled = (results.style
                      .applymap(style_pred, subset=["Prediction"])
                      .applymap(style_risk, subset=["Risk Level"])
                      .format({"Probability": "{:.1f}%"})
                      .set_properties(**{"font-family":"IBM Plex Mono","font-size":"12px"}))

            st.dataframe(styled, use_container_width=True, height=360,
                         hide_index=True)

            # Download
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Download results CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.markdown("""
            <div class="ttm-card" style="text-align:center;padding:3rem 2rem">
                <div style="font-size:36px;margin-bottom:10px">📂</div>
                <div style="font-size:14px;font-weight:600;color:#fafafa;margin-bottom:6px">Upload a CSV file to get started</div>
                <div style="font-size:11px;color:#71717a">Drag and drop into the uploader on the left</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "Dashboard":
    page_dashboard()
elif page == "Single Prediction":
    page_predict()
elif page == "Batch Upload":
    page_batch()

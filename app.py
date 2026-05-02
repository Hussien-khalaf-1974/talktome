"""
Talk to Me — Telecom Churn Intelligence
Streamlit app  |  Dashboard · Single Prediction · Batch Upload
Plotly imported lazily so startup is fast.
"""

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page config (MUST be first) ─────────────────────────────────
st.set_page_config(
    page_title="Talk to Me — Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, .stApp                    { background:#09090b !important; font-family:'IBM Plex Sans',sans-serif !important; color:#fafafa !important; }
.block-container                      { padding:1.4rem 1.8rem 2rem !important; max-width:100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"]             { background:#111113 !important; border-right:1px solid rgba(255,255,255,.07) !important; min-width:230px !important; max-width:230px !important; }
[data-testid="stSidebar"] > div       { padding:1.1rem 0.9rem !important; }

/* ── All buttons → nav style ── */
div[data-testid="stButton"] > button  { width:100% !important; background:transparent !important; border:none !important; color:#a1a1aa !important; text-align:left !important; padding:0.42rem 0.75rem !important; border-radius:7px !important; font-size:12.5px !important; font-family:'IBM Plex Sans',sans-serif !important; font-weight:500 !important; box-shadow:none !important; transition:all 140ms ease !important; }
div[data-testid="stButton"] > button:hover { background:rgba(255,255,255,.06) !important; color:#fafafa !important; }

/* ── Form submit ── */
[data-testid="stFormSubmitButton"] > button { background:#0C5CAB !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:13px !important; width:100% !important; padding:0.55rem 1.4rem !important; box-shadow:0 0 20px rgba(12,92,171,.3) !important; }
[data-testid="stFormSubmitButton"] > button:hover { background:#1a6dc2 !important; }

/* ── Selectbox / number input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input   { background:#27272a !important; border:1px solid rgba(255,255,255,.1) !important; border-radius:7px !important; color:#fafafa !important; font-size:12.5px !important; }
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label   { color:#71717a !important; font-size:9.5px !important; text-transform:uppercase !important; letter-spacing:.8px !important; font-weight:600 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"]        { background:#18181b !important; border:1.5px dashed rgba(255,255,255,.12) !important; border-radius:12px !important; padding:1.2rem !important; }
[data-testid="stFileUploader"]:hover  { border-color:rgba(12,92,171,.5) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div { color:#a1a1aa !important; font-size:12.5px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"]           { border-radius:10px !important; border:1px solid rgba(255,255,255,.07) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button { background:#18181b !important; color:#a1a1aa !important; border:1px solid rgba(255,255,255,.1) !important; border-radius:7px !important; font-size:12px !important; width:100% !important; }
[data-testid="stDownloadButton"] > button:hover { background:#27272a !important; color:#fafafa !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-thumb { background:#27272a; border-radius:99px; }

/* ── Custom HTML components ── */
.ttm-card { background:#18181b; border:1px solid rgba(255,255,255,.07); border-radius:12px; padding:1rem 1.2rem; position:relative; overflow:hidden; }
.ttm-card:hover { border-color:rgba(255,255,255,.12); }
.glow-b::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(12,92,171,.7),transparent); }
.glow-g::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(16,185,129,.6),transparent); }
.glow-r::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(239,68,68,.6),transparent); }
.glow-y::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(245,158,11,.6),transparent); }

.kpi-label { font-size:9.5px; text-transform:uppercase; letter-spacing:1.1px; font-weight:600; color:#71717a; margin-bottom:7px; }
.kpi-val   { font-size:29px; font-weight:700; letter-spacing:-1.2px; line-height:1; font-family:'IBM Plex Mono',monospace; }
.kpi-sub   { font-size:10.5px; color:#71717a; margin-top:4px; }
.kpi-delta { display:inline-flex; align-items:center; gap:3px; font-size:9.5px; font-family:'IBM Plex Mono',monospace; padding:2px 7px; border-radius:4px; margin-top:6px; border:1px solid; }
.d-up  { background:rgba(16,185,129,.12); color:#10b981; border-color:rgba(16,185,129,.25); }
.d-dn  { background:rgba(239,68,68,.10);  color:#ef4444; border-color:rgba(239,68,68,.25); }
.d-fl  { background:rgba(245,158,11,.10); color:#f59e0b; border-color:rgba(245,158,11,.25); }

.sec-title { font-size:9.5px; text-transform:uppercase; letter-spacing:1.2px; font-weight:600; color:#71717a; padding:.65rem 0 .4rem; border-top:1px solid rgba(255,255,255,.07); margin-top:.3rem; }

.verdict  { border-radius:10px; padding:.95rem 1.1rem; display:flex; align-items:flex-start; gap:11px; border:1px solid; margin-bottom:.7rem; }
.v-churn  { background:rgba(239,68,68,.08); border-color:rgba(239,68,68,.3); }
.v-retain { background:rgba(16,185,129,.07); border-color:rgba(16,185,129,.3); }
.v-icon   { font-size:26px; flex-shrink:0; margin-top:2px; }
.v-title  { font-size:17px; font-weight:700; letter-spacing:-.4px; margin-bottom:2px; }
.v-c { color:#ef4444; } .v-r { color:#10b981; }
.v-prob   { font-size:9.5px; color:#71717a; font-family:'IBM Plex Mono',monospace; }
.meter    { height:5px; background:#27272a; border-radius:99px; overflow:hidden; margin-top:7px; }
.mfill    { height:100%; border-radius:99px; }

.mrow  { display:flex; border:1px solid rgba(255,255,255,.07); border-radius:9px; overflow:hidden; margin-top:7px; }
.mcell { flex:1; padding:8px 11px; border-right:1px solid rgba(255,255,255,.07); background:#18181b; }
.mcell:last-child { border-right:none; }
.mc-l  { font-size:8.5px; color:#71717a; text-transform:uppercase; letter-spacing:.8px; margin-bottom:2px; }
.mc-v  { font-size:16px; font-weight:700; font-family:'IBM Plex Mono',monospace; }

.tip   { display:flex; align-items:flex-start; gap:8px; padding:7px 9px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:7px; font-size:11px; color:#a1a1aa; margin-bottom:5px; }
.tdot  { width:5px; height:5px; border-radius:50%; flex-shrink:0; margin-top:4px; }

.prog       { margin-bottom:8px; }
.prog-head  { display:flex; justify-content:space-between; font-size:10px; color:#a1a1aa; margin-bottom:4px; }
.prog-val   { font-family:'IBM Plex Mono',monospace; color:#fafafa; }
.prog-track { height:4px; background:#27272a; border-radius:99px; overflow:hidden; }
.prog-fill  { height:100%; border-radius:99px; }

.badge   { display:inline-flex; align-items:center; padding:2px 6px; border-radius:4px; font-size:9px; font-weight:600; font-family:'IBM Plex Mono',monospace; border:1px solid; }
.b-r { background:rgba(239,68,68,.1);  color:#ef4444; border-color:rgba(239,68,68,.3); }
.b-g { background:rgba(16,185,129,.1); color:#10b981; border-color:rgba(16,185,129,.3); }
.b-y { background:rgba(245,158,11,.1); color:#f59e0b; border-color:rgba(245,158,11,.3); }
.b-b { background:rgba(12,92,171,.15); color:#60a5fa; border-color:rgba(12,92,171,.4); }

.sb-brand { font-size:17px; font-weight:700; color:#fafafa; letter-spacing:-.3px; margin-bottom:1px; }
.sb-sub   { font-size:9px; color:#71717a; text-transform:uppercase; letter-spacing:1px; font-family:'IBM Plex Mono',monospace; margin-bottom:16px; }
.sb-sec   { font-size:8.5px; text-transform:uppercase; letter-spacing:1.2px; color:#52525b; font-weight:600; padding:10px 3px 4px; }
.sb-feat  { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:7px; padding:7px 9px; margin-top:5px; font-family:'IBM Plex Mono',monospace; font-size:9.5px; color:#60a5fa; line-height:1.9; }
.sb-mpill { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:7px; padding:8px 10px; margin-top:5px; }
.sb-ml    { font-size:8.5px; color:#52525b; text-transform:uppercase; letter-spacing:.7px; margin-bottom:2px; }
.sb-mn    { font-family:'IBM Plex Mono',monospace; color:#60a5fa; font-size:11px; font-weight:600; }
.sb-ma    { font-family:'IBM Plex Mono',monospace; color:#10b981; font-size:9.5px; margin-top:1px; }

.pg-title { font-size:19px; font-weight:700; letter-spacing:-.5px; color:#fafafa; margin-bottom:3px; }
.pg-sub   { font-size:11.5px; color:#71717a; margin-bottom:1rem; }

/* Report table */
.rtbl { width:100%; border-collapse:collapse; font-size:11.5px; }
.rtbl thead th { background:#27272a; padding:7px 11px; text-align:left; font-size:8.5px; text-transform:uppercase; letter-spacing:.8px; color:#71717a; font-weight:600; border-bottom:1px solid rgba(255,255,255,.07); }
.rtbl tbody td { padding:7px 11px; border-bottom:1px solid rgba(255,255,255,.07); color:#a1a1aa; }
.rtbl tbody tr:last-child td { border-bottom:none; }
.rtbl tbody tr:hover td { background:rgba(255,255,255,.03); color:#fafafa; }
.mono { font-family:'IBM Plex Mono',monospace; color:#fafafa; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ─── Load model artefacts (cached — runs once) ────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artefacts():
    base   = os.path.dirname(os.path.abspath(__file__))
    mdir   = os.path.join(base, "models")
    model  = joblib.load(os.path.join(mdir, "tuned_churn_model.pkl"))
    scaler = joblib.load(os.path.join(mdir, "churn_scaler.pkl"))
    feats  = joblib.load(os.path.join(mdir, "selected_features.pkl"))
    with open(os.path.join(mdir, "model_config.json")) as f:
        cfg = json.load(f)
    return model, scaler, feats, cfg

model, scaler, SEL_FEATS, cfg = load_artefacts()
THRESHOLD  = cfg.get("optimal_threshold", 0.55)
MODEL_NAME = cfg.get("model_name", "Logistic Regression")
VAL_AUC    = cfg.get("val_auc", 0.837)

# ─── Encoding ────────────────────────────────────────────────────
def encode_row(raw: dict) -> dict:
    r = {}
    bin_cols = ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
                "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]
    for c in bin_cols:
        v = str(raw.get(c, "No"))
        r[c] = 1 if v in ("Yes", "Male") else 0
    r["SeniorCitizen"]  = int(raw.get("SeniorCitizen", 0))
    r["tenure"]         = float(raw.get("tenure", 0))
    r["MonthlyCharges"] = float(raw.get("MonthlyCharges", 0))
    r["TotalCharges"]   = float(raw.get("TotalCharges", 0))
    internet = str(raw.get("InternetService","DSL"))
    r["InternetService_Fiber optic"] = int(internet == "Fiber optic")
    r["InternetService_No"]          = int(internet == "No")
    contract = str(raw.get("Contract","Month-to-month"))
    r["Contract_One year"] = int(contract == "One year")
    r["Contract_Two year"] = int(contract == "Two year")
    payment = str(raw.get("PaymentMethod","Bank transfer (automatic)"))
    r["PaymentMethod_Credit card (automatic)"] = int(payment == "Credit card (automatic)")
    r["PaymentMethod_Electronic check"]        = int(payment == "Electronic check")
    r["PaymentMethod_Mailed check"]            = int(payment == "Mailed check")
    return r

def run_predict(df_enc: pd.DataFrame):
    X    = df_enc[SEL_FEATS].values
    X_sc = scaler.transform(X)
    prob = model.predict_proba(X_sc)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return prob, pred

def risk_label(p):
    return "High Risk" if p >= 0.65 else "Medium Risk" if p >= 0.40 else "Low Risk"

# ─── Lazy plotly import ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_plotly():
    import plotly.graph_objects as go
    return go

# ─── Plotly theme helpers ─────────────────────────────────────────
BG   = "#18181b"
GRID = "rgba(255,255,255,.04)"
TC   = "#71717a"
MONO = "IBM Plex Mono"

# def base_layout(**kw):
#     return dict(paper_bgcolor=BG, plot_bgcolor=BG,
#                 font=dict(family="IBM Plex Sans", color=TC, size=11),
#                 margin=dict(l=6,r=6,t=30,b=6), showlegend=False, **kw)

def base_layout(**kw):
    return dict(paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family="IBM Plex Sans", color=TC, size=11),
                margin=dict(l=6,r=6,t=30,b=6), **kw)

def ax(title=""):
    return dict(title=title, gridcolor=GRID,
                linecolor="rgba(255,255,255,.06)",
                tickfont=dict(family=MONO,size=9,color=TC),
                titlefont=dict(size=10,color=TC))

CFG = {"displayModeBar": False}

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">📡 Talk to Me</div>
    <div class="sb-sub">Churn Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Workspace</div>', unsafe_allow_html=True)
    for pg, lbl in [("Dashboard","◈  Dashboard"),
                    ("Single Prediction","◎  Single Prediction"),
                    ("Batch Upload","≡  Batch Upload")]:
        if st.button(lbl, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    st.markdown('<div class="sb-sec" style="margin-top:10px">Selected Features (9)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-feat">TotalCharges · tenure<br>Contract_Two year<br>MonthlyCharges<br>
    InternetService_Fiber<br>PaymentMethod_Echeck<br>InternetService_No<br>
    OnlineSecurity<br>PaperlessBilling</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec" style="margin-top:10px">Active Model</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-mpill">
      <div class="sb-ml">Best model</div>
      <div class="sb-mn">{MODEL_NAME}</div>
      <div class="sb-ma">AUC {VAL_AUC:.3f} · Threshold {THRESHOLD}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  REAL DATA
# ═══════════════════════════════════════════════════════════════
MODELS = dict(
    names  = ["LR",  "RF",  "XGB", "DT",  "SVM"],
    acc    = [0.73,  0.74,  0.73,  0.71,  0.73 ],
    f1     = [0.61,  0.60,  0.59,  0.58,  0.61 ],
    auc    = [0.82,  0.81,  0.81,  0.80,  0.78 ],
)
FEATS = dict(
    names  = ["TotalCharges","tenure","Contract_2yr","MonthlyChg","Fiber optic",
               "Echeck","Internet_No","OnlineSec","Paperless"],
    scores = [0.883,0.631,0.535,0.483,0.430,0.291,0.230,0.160,0.122],
    colors = ["#1a6dc2","#1a6dc2","#f59e0b","#f59e0b","#ef4444",
               "#ef4444","#71717a","#71717a","#71717a"],
)
THRESH = dict(
    t   = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70],
    f1  = [0.55,0.58,0.60,0.614,0.616,0.630,0.621,0.607,0.580],
    acc = [0.68,0.70,0.72,0.735,0.747,0.769,0.775,0.779,0.782],
)

# ═══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_dashboard():
    go = get_plotly()

    st.markdown('<div class="pg-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Model performance overview — real data from your training pipeline</div>',
                unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4, gap="small")
    with c1:
        st.markdown("""<div class="ttm-card glow-b">
          <div class="kpi-label">Dataset</div><div class="kpi-val">6,043</div>
          <div class="kpi-sub">Total customers</div>
          <div class="kpi-delta d-fl">70 / 15 / 15 stratified</div></div>""",
          unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="ttm-card glow-r">
          <div class="kpi-label">Churn Rate</div>
          <div class="kpi-val" style="color:#ef4444">26.5%</div>
          <div class="kpi-sub">1,604 of 6,043 churned</div>
          <div class="kpi-delta d-dn">Balanced via SMOTE</div></div>""",
          unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="ttm-card glow-g">
          <div class="kpi-label">Best ROC-AUC</div>
          <div class="kpi-val" style="color:#10b981">{VAL_AUC:.3f}</div>
          <div class="kpi-sub">{MODEL_NAME} · test set</div>
          <div class="kpi-delta d-up">Threshold → {THRESHOLD}</div></div>""",
          unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="ttm-card glow-y">
          <div class="kpi-label">Best F1 (Churn)</div>
          <div class="kpi-val" style="color:#f59e0b">0.630</div>
          <div class="kpi-sub">vs 0.616 at default 0.50</div>
          <div class="kpi-delta d-up">+0.014 from threshold sweep</div></div>""",
          unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Row 2: model bar + feature bar ────────────────────────
    cl, cr = st.columns([3,2], gap="small")

    with cl:
        fig = go.Figure()
        for name, vals, col in [
            ("Accuracy",   MODELS["acc"], "rgba(12,92,171,.8)"),
            ("F1 (Churn)", MODELS["f1"],  "rgba(245,158,11,.8)"),
            ("ROC-AUC",    MODELS["auc"], "rgba(16,185,129,.8)"),
        ]:
            fig.add_trace(go.Bar(
                name=name, x=MODELS["names"], y=vals,
                marker_color=col, marker_line_width=0,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                textfont=dict(family=MONO, size=9, color=TC),
            ))
        fig.update_layout(
            **base_layout(barmode="group",
                title="Model Comparison — Validation Set",
                title_font=dict(size=12,color="#a1a1aa"),
                showlegend=True,
                legend=dict(orientation="h", y=1.12,
                    font=dict(family=MONO,size=9,color=TC), bgcolor="transparent")),
            xaxis=ax(), yaxis=dict(**ax(), range=[0.5,0.93]),
            height=270,
        )
        st.plotly_chart(fig, use_container_width=True, config=CFG)

    with cr:
        fig2 = go.Figure(go.Bar(
            x=FEATS["scores"], y=FEATS["names"],
            orientation="h",
            marker_color=FEATS["colors"], marker_line_width=0,
            text=[f"{v:.3f}" for v in FEATS["scores"]],
            textposition="outside",
            textfont=dict(family=MONO, size=9, color=TC),
        ))
        fig2.update_layout(
            **base_layout(title="Feature Importance (Avg Score)",
                title_font=dict(size=12,color="#a1a1aa")),
            xaxis=dict(**ax(), range=[0,1.06]),
            yaxis=dict(**ax(), autorange="reversed"),
            height=270,
        )
        st.plotly_chart(fig2, use_container_width=True, config=CFG)

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ── Row 3: threshold + report table + donut ───────────────
    c1, c2, c3 = st.columns([1.6, 2.5, 1.4], gap="small")

    with c1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=THRESH["t"], y=THRESH["f1"], name="F1 Churn",
            mode="lines+markers",
            line=dict(color="#f59e0b",width=2),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(245,158,11,.06)",
        ))
        fig3.add_trace(go.Scatter(
            x=THRESH["t"], y=THRESH["acc"], name="Accuracy",
            mode="lines+markers",
            line=dict(color="#1a6dc2",width=2,dash="dot"),
            marker=dict(size=5),
        ))
        fig3.add_vline(x=0.55, line_color="#10b981", line_dash="dash",
                       line_width=1.5,
                       annotation_text="0.55",
                       annotation_font=dict(color="#10b981",size=9))
        fig3.update_layout(
            **base_layout(title="Threshold Sweep",
                title_font=dict(size=12,color="#a1a1aa"),
                showlegend=True,
                legend=dict(orientation="h", y=1.12,
                    font=dict(family=MONO,size=9,color=TC), bgcolor="transparent")),
            xaxis=dict(**ax("Threshold")),
            yaxis=dict(**ax("Score"), range=[0.5,0.83]),
            height=230,
        )
        st.plotly_chart(fig3, use_container_width=True, config=CFG)
        st.markdown("""
        <div class="mrow" style="margin-top:4px">
          <div class="mcell"><div class="mc-l">Default (0.50)</div><div class="mc-v" style="color:#f59e0b;font-size:15px">F1 0.616</div></div>
          <div class="mcell"><div class="mc-l">Optimised (0.55)</div><div class="mc-v" style="color:#10b981;font-size:15px">F1 0.630</div></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="ttm-card" style="padding:0;overflow:hidden">
          <div style="padding:10px 13px 8px;border-bottom:1px solid rgba(255,255,255,.07)">
            <span style="font-size:12px;font-weight:600;color:#fafafa">Classification Report</span>
            <span style="font-size:10px;color:#71717a;margin-left:8px">LR · threshold 0.55 · test set (906 samples)</span>
          </div>
          <table class="rtbl">
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>
              <tr><td><span class="badge b-g">No Churn</span></td><td class="mono">0.89</td><td class="mono">0.78</td><td class="mono">0.83</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">665</td></tr>
              <tr><td><span class="badge b-r">Churn</span></td><td class="mono">0.55</td><td class="mono">0.74</td><td class="mono">0.63</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">241</td></tr>
              <tr style="background:rgba(255,255,255,.02)"><td><span class="badge b-b">Macro avg</span></td><td class="mono">0.72</td><td class="mono">0.76</td><td class="mono">0.73</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">906</td></tr>
              <tr style="background:rgba(255,255,255,.02)"><td><span class="badge b-b">Weighted</span></td><td class="mono">0.80</td><td class="mono">0.77</td><td class="mono">0.78</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">906</td></tr>
            </tbody>
          </table>
          <div style="padding:8px 13px;border-top:1px solid rgba(255,255,255,.07)">
            <div class="mrow">
              <div class="mcell"><div class="mc-l">Accuracy</div><div class="mc-v">76.9%</div></div>
              <div class="mcell"><div class="mc-l">ROC-AUC</div><div class="mc-v" style="color:#10b981">0.837</div></div>
              <div class="mcell"><div class="mc-l">F1 Churn</div><div class="mc-v" style="color:#f59e0b">0.630</div></div>
              <div class="mcell"><div class="mc-l">Recall</div><div class="mc-v" style="color:#60a5fa">74%</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        fig4 = go.Figure(go.Pie(
            labels=["Retained","Churned"], values=[4439,1604],
            hole=0.66,
            marker=dict(colors=["rgba(16,185,129,.8)","rgba(239,68,68,.8)"],
                        line=dict(color=BG,width=3)),
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig4.add_annotation(text="6,043", x=0.5, y=0.58, showarrow=False,
            font=dict(family=MONO,size=16,color="#fafafa"))
        fig4.add_annotation(text="customers", x=0.5, y=0.42, showarrow=False,
            font=dict(family="IBM Plex Sans",size=10,color=TC))
        fig4.update_layout(
            **base_layout(title="Churn Distribution",
                title_font=dict(size=12,color="#a1a1aa"),
                showlegend=True,
                legend=dict(orientation="h", y=-0.06,
                    font=dict(family=MONO,size=9,color=TC), bgcolor="transparent")),
            height=220,
        )
        st.plotly_chart(fig4, use_container_width=True, config=CFG)

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Retained", x=["M-t-M","1 yr","2 yr"],
            y=[1655,1141,1380], marker_color="rgba(16,185,129,.75)", marker_line_width=0))
        fig5.add_trace(go.Bar(name="Churned",  x=["M-t-M","1 yr","2 yr"],
            y=[1666,120,81],   marker_color="rgba(239,68,68,.75)",  marker_line_width=0))
        fig5.update_layout(
            **base_layout(barmode="stack", title="Churn by Contract",
                title_font=dict(size=12,color="#a1a1aa"),
                showlegend=True,
                legend=dict(orientation="h", y=1.12,
                    font=dict(family=MONO,size=9,color=TC), bgcolor="transparent")),
            xaxis=ax(), yaxis=ax(), height=200,
        )
        st.plotly_chart(fig5, use_container_width=True, config=CFG)


# ═══════════════════════════════════════════════════════════════
#  PAGE: SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="pg-title">Single Customer Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Fill in the customer profile — model uses 9 key features at threshold 0.55</div>',
                unsafe_allow_html=True)

    col_form, col_res = st.columns([2, 1.6], gap="medium")

    with col_form:
        with st.form("predict_form"):

            st.markdown('<div class="sec-title">👤 Demographics</div>', unsafe_allow_html=True)
            d1,d2,d3 = st.columns(3)
            gender   = d1.selectbox("Gender",          ["Male","Female"])
            senior   = d2.selectbox("Senior Citizen",  ["No","Yes"])
            partner  = d3.selectbox("Partner",          ["No","Yes"])
            d4,d5    = st.columns(2)
            deps     = d4.selectbox("Dependents",       ["No","Yes"])
            tenure   = d5.number_input("Tenure (months)", 0, 72, 12)

            st.markdown('<div class="sec-title">📡 Services</div>', unsafe_allow_html=True)
            s1,s2,s3 = st.columns(3)
            phone    = s1.selectbox("Phone Service",    ["Yes","No"])
            multiln  = s2.selectbox("Multiple Lines",   ["No","Yes"])
            internet = s3.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            s4,s5,s6 = st.columns(3)
            onsec    = s4.selectbox("Online Security",  ["No","Yes"])
            onbk     = s5.selectbox("Online Backup",    ["No","Yes"])
            devprot  = s6.selectbox("Device Protection",["No","Yes"])
            s7,s8,s9 = st.columns(3)
            techsup  = s7.selectbox("Tech Support",     ["No","Yes"])
            stv      = s8.selectbox("Streaming TV",     ["No","Yes"])
            smv      = s9.selectbox("Streaming Movies", ["No","Yes"])

            st.markdown('<div class="sec-title">💳 Billing & Contract</div>', unsafe_allow_html=True)
            b1,b2,b3 = st.columns(3)
            contract = b1.selectbox("Contract",          ["Month-to-month","One year","Two year"])
            paperless= b2.selectbox("Paperless Billing", ["Yes","No"])
            payment  = b3.selectbox("Payment Method",
                                    ["Electronic check","Mailed check",
                                     "Bank transfer (automatic)","Credit card (automatic)"])
            b4,b5    = st.columns(2)
            monthly  = b4.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.01)
            total    = b5.number_input("Total Charges ($)",   0.0,10000.0,780.0, step=0.01)

            submitted = st.form_submit_button("⚡  Predict Churn", use_container_width=True)

    with col_res:
        if submitted:
            raw = dict(
                gender=gender, SeniorCitizen=1 if senior=="Yes" else 0,
                Partner=partner, Dependents=deps, tenure=tenure,
                PhoneService=phone, MultipleLines=multiln,
                InternetService=internet, OnlineSecurity=onsec,
                OnlineBackup=onbk, DeviceProtection=devprot,
                TechSupport=techsup, StreamingTV=stv, StreamingMovies=smv,
                Contract=contract, PaperlessBilling=paperless,
                PaymentMethod=payment, MonthlyCharges=monthly, TotalCharges=total,
            )
            try:
                enc  = encode_row(raw)
                prob, pred = run_predict(pd.DataFrame([enc]))
                pct  = round(float(prob[0]) * 100, 1)
                is_c = bool(pred[0])
                rl   = risk_label(float(prob[0]))
                col  = "#ef4444" if pct>=65 else "#f59e0b" if pct>=40 else "#10b981"

                # Verdict
                icon  = "⚠️" if is_c else "✅"
                label = "Will Churn" if is_c else "Will Stay"
                bc    = "v-churn" if is_c else "v-retain"
                tc    = "v-c"     if is_c else "v-r"
                st.markdown(f"""
                <div class="verdict {bc}">
                  <div class="v-icon">{icon}</div>
                  <div style="flex:1">
                    <div class="v-title {tc}">{label}</div>
                    <div class="v-prob">Probability {pct}% · {rl} · threshold {THRESHOLD}</div>
                    <div class="meter"><div class="mfill" style="width:{pct}%;background:{col}"></div></div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Metric row
                st.markdown(f"""
                <div class="mrow">
                  <div class="mcell"><div class="mc-l">Probability</div><div class="mc-v" style="color:{col}">{pct}%</div></div>
                  <div class="mcell"><div class="mc-l">Risk Level</div><div class="mc-v" style="font-size:13px">{rl}</div></div>
                </div>""", unsafe_allow_html=True)

                # Gauge (plotly loaded lazily)
                go = get_plotly()
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=pct,
                    number={"suffix":"%","font":{"family":MONO,"size":22,"color":"#fafafa"}},
                    gauge=dict(
                        axis=dict(range=[0,100],
                            tickfont=dict(family=MONO,size=8,color=TC),
                            tickcolor=TC),
                        bar=dict(color=col, thickness=0.25),
                        bgcolor=BG, borderwidth=0,
                        steps=[
                            dict(range=[0,40],   color="rgba(16,185,129,.1)"),
                            dict(range=[40,65],  color="rgba(245,158,11,.1)"),
                            dict(range=[65,100], color="rgba(239,68,68,.1)"),
                        ],
                    ),
                ))
                fig_g.update_layout(**base_layout(height=190),
                                    margin=dict(l=20,r=20,t=18,b=8))
                st.plotly_chart(fig_g, use_container_width=True, config=CFG)

                # Tips
                tips_c = [("#ef4444","Call within 48h — personalised retention offer"),
                           ("#f59e0b","Propose annual contract with 2 months free"),
                           ("#f59e0b","Add Online Security or Tech Support to plan"),
                           ("#60a5fa","Switch from Electronic check to automatic payment")]
                tips_r = [("#10b981","Stable customer — consider upsell opportunity"),
                           ("#60a5fa","Invite to loyalty programme or referral scheme"),
                           ("#71717a","Monitor for upcoming contract renewal date")]
                html = "<div style='margin-top:8px'>"
                for dc, txt in (tips_c if is_c else tips_r):
                    html += f'<div class="tip"><div class="tdot" style="background:{dc}"></div>{txt}</div>'
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.markdown("""
            <div class="ttm-card">
              <div class="kpi-label" style="margin-bottom:10px">How it works</div>
              <div class="tip"><div class="tdot" style="background:#60a5fa"></div>Fill in the customer profile and click <strong>Predict Churn</strong></div>
              <div class="tip"><div class="tdot" style="background:#f59e0b"></div>Model uses <strong>9 consensus features</strong> (MI + Chi² + RF)</div>
              <div class="tip"><div class="tdot" style="background:#10b981"></div>Optimised threshold <strong>0.55</strong> maximises F1 for churn class</div>
              <div class="tip"><div class="tdot" style="background:#ef4444"></div>High ≥65% · Medium ≥40% · Low &lt;40%</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════
def page_batch():
    st.markdown('<div class="pg-title">Batch CSV Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload a CSV — all rows scored instantly</div>',
                unsafe_allow_html=True)

    col_up, col_res = st.columns([1.6, 2.4], gap="medium")

    with col_up:
        uploaded = st.file_uploader("Drop your CSV here",
                                    type=["csv"],
                                    help="Same columns as training data")
        st.markdown("""
        <div class="ttm-card" style="margin-top:10px">
          <div class="kpi-label">Required columns</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#60a5fa;line-height:2">
          customerID · gender · SeniorCitizen<br>Partner · Dependents · tenure<br>
          PhoneService · MultipleLines<br>InternetService · OnlineSecurity<br>
          OnlineBackup · DeviceProtection<br>TechSupport · StreamingTV<br>
          StreamingMovies · Contract<br>PaperlessBilling · PaymentMethod<br>
          MonthlyCharges · TotalCharges
          </div>
        </div>""", unsafe_allow_html=True)

    with col_res:
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                ids    = df_raw["customerID"].astype(str) if "customerID" in df_raw.columns \
                         else pd.Series(range(1,len(df_raw)+1)).astype(str)
                df     = df_raw.drop(columns=["customerID","Churn"], errors="ignore").copy()
                if "TotalCharges" in df.columns:
                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

                enc_rows  = [encode_row(r.to_dict()) for _,r in df.iterrows()]
                df_enc    = pd.DataFrame(enc_rows)
                prob, pred = run_predict(df_enc)

                total    = len(pred)
                churners = int(pred.sum())
                retained = total - churners
                rate     = round(churners/total*100, 1)
                high_r   = int((prob>=0.65).sum())
                med_r    = int(((prob>=0.40)&(prob<0.65)).sum())
                low_r    = int((prob<0.40).sum())

                # KPI cards
                k1,k2,k3,k4 = st.columns(4, gap="small")
                k1.markdown(f"""<div class="ttm-card"><div class="kpi-label">Total</div>
                    <div class="kpi-val">{total:,}</div></div>""", unsafe_allow_html=True)
                k2.markdown(f"""<div class="ttm-card glow-r"><div class="kpi-label">Will Churn</div>
                    <div class="kpi-val" style="color:#ef4444">{churners:,}</div></div>""", unsafe_allow_html=True)
                k3.markdown(f"""<div class="ttm-card glow-g"><div class="kpi-label">Retained</div>
                    <div class="kpi-val" style="color:#10b981">{retained:,}</div></div>""", unsafe_allow_html=True)
                k4.markdown(f"""<div class="ttm-card"><div class="kpi-label">Churn Rate</div>
                    <div class="kpi-val">{rate}%</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                # Risk bars + bar chart side by side
                rb, rc = st.columns([1,1], gap="small")
                with rb:
                    st.markdown(f"""
                    <div class="ttm-card">
                      <div class="kpi-label" style="margin-bottom:10px">Risk Breakdown</div>
                      <div class="prog"><div class="prog-head"><span style="color:#ef4444">● High ≥65%</span><span class="prog-val">{high_r}</span></div>
                        <div class="prog-track"><div class="prog-fill" style="width:{high_r/total*100:.1f}%;background:#ef4444"></div></div></div>
                      <div class="prog"><div class="prog-head"><span style="color:#f59e0b">● Medium ≥40%</span><span class="prog-val">{med_r}</span></div>
                        <div class="prog-track"><div class="prog-fill" style="width:{med_r/total*100:.1f}%;background:#f59e0b"></div></div></div>
                      <div class="prog"><div class="prog-head"><span style="color:#10b981">● Low &lt;40%</span><span class="prog-val">{low_r}</span></div>
                        <div class="prog-track"><div class="prog-fill" style="width:{low_r/total*100:.1f}%;background:#10b981"></div></div></div>
                    </div>""", unsafe_allow_html=True)
                with rc:
                    go = get_plotly()
                    fig_b = go.Figure(go.Bar(
                        x=["High","Medium","Low"],
                        y=[high_r, med_r, low_r],
                        marker_color=["rgba(239,68,68,.8)","rgba(245,158,11,.8)","rgba(16,185,129,.8)"],
                        marker_line_width=0,
                        text=[high_r,med_r,low_r], textposition="outside",
                        textfont=dict(family=MONO,size=10,color=TC),
                    ))
                    fig_b.update_layout(**base_layout(title="Risk Distribution",
                        title_font=dict(size=11,color="#a1a1aa")),
                        xaxis=ax(), yaxis=ax(), height=200)
                    st.plotly_chart(fig_b, use_container_width=True, config=CFG)

                # Results table
                results = pd.DataFrame({
                    "Customer ID": ids.values,
                    "Probability %": (prob*100).round(1),
                    "Prediction":  ["Churn" if p else "Retain" for p in pred],
                    "Risk Level":  [risk_label(p) for p in prob],
                })

                def style_pred(v):
                    return "color:#ef4444;font-weight:600" if v=="Churn" else "color:#10b981;font-weight:600"
                def style_risk(v):
                    return ("color:#ef4444" if v=="High Risk" else
                            "color:#f59e0b" if v=="Medium Risk" else "color:#10b981")

                styled = (results.style
                    .applymap(style_pred, subset=["Prediction"])
                    .applymap(style_risk, subset=["Risk Level"])
                    .format({"Probability %": "{:.1f}%"})
                    .set_properties(**{"font-family":"IBM Plex Mono","font-size":"12px"}))

                st.dataframe(styled, use_container_width=True,
                             height=340, hide_index=True)

                csv_bytes = results.to_csv(index=False).encode()
                st.download_button("⬇  Download results CSV",
                                   data=csv_bytes,
                                   file_name="churn_predictions.csv",
                                   mime="text/csv",
                                   use_container_width=True)

            except Exception as e:
                st.error(f"Processing failed: {e}")
        else:
            st.markdown("""
            <div class="ttm-card" style="text-align:center;padding:3rem 2rem">
              <div style="font-size:34px;margin-bottom:10px">📂</div>
              <div style="font-size:14px;font-weight:600;color:#fafafa;margin-bottom:5px">Upload a CSV to get started</div>
              <div style="font-size:11px;color:#71717a">Use the file uploader on the left</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════
{
    "Dashboard":         page_dashboard,
    "Single Prediction": page_predict,
    "Batch Upload":      page_batch,
}[st.session_state.page]()

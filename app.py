"""
Talk to Me — Telecom Churn Intelligence
Streamlit app: Dashboard · Single Prediction · Batch Upload
Every chart written explicitly — no shared layout helpers.
"""

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Must be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Talk to Me — Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html,body,.stApp{background:#09090b!important;font-family:'IBM Plex Sans',sans-serif!important;color:#fafafa!important}
.block-container{padding:3.5rem 1.8rem 2rem!important;max-width:100%!important}

[data-testid="stSidebar"]{background:#111113!important;border-right:1px solid rgba(255,255,255,.07)!important;min-width:228px!important;max-width:228px!important}
[data-testid="stSidebar"]>div{padding:1rem 0.85rem!important}

div[data-testid="stButton"]>button{width:100%!important;background:transparent!important;border:none!important;color:#a1a1aa!important;text-align:left!important;padding:0.42rem 0.75rem!important;border-radius:7px!important;font-size:12.5px!important;font-family:'IBM Plex Sans',sans-serif!important;font-weight:500!important;box-shadow:none!important}
div[data-testid="stButton"]>button:hover{background:rgba(255,255,255,.06)!important;color:#fafafa!important}

[data-testid="stFormSubmitButton"]>button{background:#0C5CAB!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:13px!important;width:100%!important;padding:0.55rem 1.4rem!important;box-shadow:0 0 20px rgba(12,92,171,.3)!important}
[data-testid="stFormSubmitButton"]>button:hover{background:#1a6dc2!important}

[data-testid="stSelectbox"]>div>div,[data-testid="stNumberInput"] input{background:#27272a!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:7px!important;color:#fafafa!important;font-size:12.5px!important}
[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label{color:#71717a!important;font-size:9.5px!important;text-transform:uppercase!important;letter-spacing:.8px!important;font-weight:600!important}

[data-testid="stFileUploader"]{background:#18181b!important;border:1.5px dashed rgba(255,255,255,.12)!important;border-radius:12px!important;padding:1.2rem!important}
[data-testid="stFileUploaderDropzoneInstructions"] div{color:#a1a1aa!important;font-size:12.5px!important}

[data-testid="stDownloadButton"]>button{background:#18181b!important;color:#a1a1aa!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:7px!important;font-size:12px!important;width:100%!important}
[data-testid="stDownloadButton"]>button:hover{background:#27272a!important;color:#fafafa!important}

[data-testid="stDataFrame"]{border-radius:10px!important;border:1px solid rgba(255,255,255,.07)!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:#27272a;border-radius:99px}
hr{border-color:rgba(255,255,255,.07)!important}

/* Custom components */
.card{background:#18181b;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:1rem 1.2rem;position:relative;overflow:hidden}
.card:hover{border-color:rgba(255,255,255,.12)}
.gb::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(12,92,171,.7),transparent)}
.gg::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(16,185,129,.6),transparent)}
.gr::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(239,68,68,.6),transparent)}
.gy::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(245,158,11,.6),transparent)}

.lbl{font-size:9.5px;text-transform:uppercase;letter-spacing:1.1px;font-weight:600;color:#71717a;margin-bottom:7px}
.val{font-size:29px;font-weight:700;letter-spacing:-1.2px;line-height:1;font-family:'IBM Plex Mono',monospace}
.sub{font-size:10.5px;color:#71717a;margin-top:4px}
.delta{display:inline-flex;align-items:center;font-size:9.5px;font-family:'IBM Plex Mono',monospace;padding:2px 7px;border-radius:4px;margin-top:6px;border:1px solid}
.du{background:rgba(16,185,129,.12);color:#10b981;border-color:rgba(16,185,129,.25)}
.dd{background:rgba(239,68,68,.10);color:#ef4444;border-color:rgba(239,68,68,.25)}
.df{background:rgba(245,158,11,.10);color:#f59e0b;border-color:rgba(245,158,11,.25)}

.sec{font-size:9.5px;text-transform:uppercase;letter-spacing:1.2px;font-weight:600;color:#71717a;padding:.65rem 0 .4rem;border-top:1px solid rgba(255,255,255,.07);margin-top:.3rem}

.verdict{border-radius:10px;padding:.95rem 1.1rem;display:flex;align-items:flex-start;gap:11px;border:1px solid;margin-bottom:.7rem}
.vc{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.3)}
.vr{background:rgba(16,185,129,.07);border-color:rgba(16,185,129,.3)}
.vicon{font-size:26px;flex-shrink:0;margin-top:2px}
.vtitle{font-size:17px;font-weight:700;letter-spacing:-.4px;margin-bottom:2px}
.vtc{color:#ef4444}.vtr{color:#10b981}
.vprob{font-size:9.5px;color:#71717a;font-family:'IBM Plex Mono',monospace}
.meter{height:5px;background:#27272a;border-radius:99px;overflow:hidden;margin-top:7px}
.mfill{height:100%;border-radius:99px}

.mrow{display:flex;border:1px solid rgba(255,255,255,.07);border-radius:9px;overflow:hidden;margin-top:7px}
.mcell{flex:1;padding:8px 11px;border-right:1px solid rgba(255,255,255,.07);background:#18181b}
.mcell:last-child{border-right:none}
.ml{font-size:8.5px;color:#71717a;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px}
.mv{font-size:16px;font-weight:700;font-family:'IBM Plex Mono',monospace}

.tip{display:flex;align-items:flex-start;gap:8px;padding:7px 9px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:7px;font-size:11px;color:#a1a1aa;margin-bottom:5px}
.tdot{width:5px;height:5px;border-radius:50%;flex-shrink:0;margin-top:4px}

.prog{margin-bottom:8px}
.ph{display:flex;justify-content:space-between;font-size:10px;color:#a1a1aa;margin-bottom:4px}
.pv{font-family:'IBM Plex Mono',monospace;color:#fafafa}
.pt{height:4px;background:#27272a;border-radius:99px;overflow:hidden}
.pf{height:100%;border-radius:99px}

.badge{display:inline-flex;align-items:center;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600;font-family:'IBM Plex Mono',monospace;border:1px solid}
.br2{background:rgba(239,68,68,.1);color:#ef4444;border-color:rgba(239,68,68,.3)}
.bg2{background:rgba(16,185,129,.1);color:#10b981;border-color:rgba(16,185,129,.3)}
.by2{background:rgba(245,158,11,.1);color:#f59e0b;border-color:rgba(245,158,11,.3)}
.bb2{background:rgba(12,92,171,.15);color:#60a5fa;border-color:rgba(12,92,171,.4)}

.rtbl{width:100%;border-collapse:collapse;font-size:11.5px}
.rtbl thead th{background:#27272a;padding:7px 11px;text-align:left;font-size:8.5px;text-transform:uppercase;letter-spacing:.8px;color:#71717a;font-weight:600;border-bottom:1px solid rgba(255,255,255,.07)}
.rtbl tbody td{padding:7px 11px;border-bottom:1px solid rgba(255,255,255,.07);color:#a1a1aa}
.rtbl tbody tr:last-child td{border-bottom:none}
.rtbl tbody tr:hover td{background:rgba(255,255,255,.03);color:#fafafa}
.mono{font-family:'IBM Plex Mono',monospace;color:#fafafa}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ── Load artefacts ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artefacts():
    base  = os.path.dirname(os.path.abspath(__file__))
    mdir  = os.path.join(base, "models")
    mdl   = joblib.load(os.path.join(mdir, "tuned_churn_model.pkl"))
    scl   = joblib.load(os.path.join(mdir, "churn_scaler.pkl"))
    feats = joblib.load(os.path.join(mdir, "selected_features.pkl"))
    with open(os.path.join(mdir, "model_config.json")) as f:
        cfg = json.load(f)
    return mdl, scl, feats, cfg

model, scaler, SEL_FEATS, cfg = load_artefacts()
THRESHOLD  = cfg.get("optimal_threshold", 0.55)
MODEL_NAME = cfg.get("model_name", "Logistic Regression")
VAL_AUC    = cfg.get("val_auc", 0.837)

# ── Helpers ───────────────────────────────────────────────────────
def encode_row(raw):
    r = {}
    for c in ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
              "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
              "TechSupport","StreamingTV","StreamingMovies"]:
        r[c] = 1 if str(raw.get(c,"No")) in ("Yes","Male") else 0
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

def run_predict(df_enc):
    X    = df_enc[SEL_FEATS].values
    X_sc = scaler.transform(X)
    prob = model.predict_proba(X_sc)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return prob, pred

def risk_label(p):
    return "High Risk" if p >= 0.65 else "Medium Risk" if p >= 0.40 else "Low Risk"

# ── Chart constants ───────────────────────────────────────────────
BG   = "#18181b"
GRID = "rgba(255,255,255,.04)"
TC   = "#71717a"
MONO = "IBM Plex Mono"
SANS = "IBM Plex Sans"
CFG  = {"displayModeBar": False}

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    # st.markdown("""
    # <div style="font-size:17px;font-weight:700;color:#fafafa;letter-spacing:-.3px;margin-bottom:1px">📡 Talk to Me</div>
    # <div style="font-size:9px;color:#71717a;text-transform:uppercase;letter-spacing:1px;font-family:'IBM Plex Mono',monospace;margin-bottom:16px">Churn Intelligence</div>
    # """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:12px;padding-bottom:14px;border-bottom:1px solid rgba(255,255,255,.07);margin-bottom:4px">
      <div style="font-size:18px;font-weight:700;color:#fafafa;letter-spacing:-.4px;line-height:1.2">📡 Talk to Me</div>
      <div style="font-size:9px;color:#71717a;text-transform:uppercase;letter-spacing:1.2px;font-family:'IBM Plex Mono',monospace;margin-top:4px">Churn Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:8.5px;text-transform:uppercase;letter-spacing:1.2px;color:#52525b;font-weight:600;padding:10px 3px 4px">Workspace</div>', unsafe_allow_html=True)
    for pg, lbl in [("Dashboard","◈  Dashboard"),
                    ("Single Prediction","◎  Single Prediction"),
                    ("Batch Upload","≡  Batch Upload")]:
        if st.button(lbl, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    st.markdown('<div style="font-size:8.5px;text-transform:uppercase;letter-spacing:1.2px;color:#52525b;font-weight:600;padding:10px 3px 4px;margin-top:8px">Selected Features (9)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:7px;padding:7px 9px;font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#60a5fa;line-height:1.9">
    TotalCharges · tenure<br>Contract_Two year<br>MonthlyCharges<br>
    InternetService_Fiber<br>PaymentMethod_Echeck<br>InternetService_No<br>
    OnlineSecurity<br>PaperlessBilling</div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:8.5px;text-transform:uppercase;letter-spacing:1.2px;color:#52525b;font-weight:600;padding:10px 3px 4px;margin-top:8px">Active Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:7px;padding:8px 10px">
      <div style="font-size:8.5px;color:#52525b;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">Best model</div>
      <div style="font-family:'IBM Plex Mono',monospace;color:#60a5fa;font-size:11px;font-weight:600">{MODEL_NAME}</div>
      <div style="font-family:'IBM Plex Mono',monospace;color:#10b981;font-size:9.5px;margin-top:1px">AUC {VAL_AUC:.3f} · Threshold {THRESHOLD}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Real data ─────────────────────────────────────────────────────
MNAMES = ["LR", "RF", "XGB", "DT", "SVM"]
MACC   = [0.73, 0.74, 0.73, 0.71, 0.73]
MF1    = [0.61, 0.60, 0.59, 0.58, 0.61]
MAUC   = [0.82, 0.81, 0.81, 0.80, 0.78]

FNAMES  = ["TotalCharges","tenure","Contract_2yr","MonthlyChg","Fiber optic",
           "Echeck","Internet_No","OnlineSec","Paperless"]
FSCORES = [0.883, 0.631, 0.535, 0.483, 0.430, 0.291, 0.230, 0.160, 0.122]
FCOLORS = ["#1a6dc2","#1a6dc2","#f59e0b","#f59e0b","#ef4444",
           "#ef4444","#71717a","#71717a","#71717a"]

TH_X  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TH_F1 = [0.55, 0.58, 0.60, 0.614, 0.616, 0.630, 0.621, 0.607, 0.580]
TH_AC = [0.68, 0.70, 0.72, 0.735, 0.747, 0.769, 0.775, 0.779, 0.782]


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════
def page_dashboard():
    import plotly.graph_objects as go

    st.markdown('<div style="font-size:19px;font-weight:700;letter-spacing:-.5px;color:#fafafa;margin-bottom:3px">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11.5px;color:#71717a;margin-bottom:1rem">Model performance overview — real data from your training pipeline</div>', unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.markdown("""<div class="card gb"><div class="lbl">Dataset</div><div class="val">6,043</div><div class="sub">Total customers</div><div class="delta df">70 / 15 / 15 stratified</div></div>""", unsafe_allow_html=True)
    k2.markdown("""<div class="card gr"><div class="lbl">Churn Rate</div><div class="val" style="color:#ef4444">26.5%</div><div class="sub">1,604 of 6,043 churned</div><div class="delta dd">Balanced via SMOTE</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="card gg"><div class="lbl">Best ROC-AUC</div><div class="val" style="color:#10b981">{VAL_AUC:.3f}</div><div class="sub">{MODEL_NAME} · test set</div><div class="delta du">Threshold → {THRESHOLD}</div></div>""", unsafe_allow_html=True)
    k4.markdown("""<div class="card gy"><div class="lbl">Best F1 (Churn)</div><div class="val" style="color:#f59e0b">0.630</div><div class="sub">vs 0.616 at default 0.50</div><div class="delta du">+0.014 from threshold sweep</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Model comparison + Feature importance ──────────────────────
    cl, cr = st.columns([3, 2], gap="small")

    with cl:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy",   x=MNAMES, y=MACC, marker_color="rgba(12,92,171,.8)",  marker_line_width=0, text=[f"{v:.2f}" for v in MACC], textposition="outside", textfont=dict(family=MONO, size=9, color=TC)))
        fig.add_trace(go.Bar(name="F1 (Churn)", x=MNAMES, y=MF1,  marker_color="rgba(245,158,11,.8)", marker_line_width=0, text=[f"{v:.2f}" for v in MF1],  textposition="outside", textfont=dict(family=MONO, size=9, color=TC)))
        fig.add_trace(go.Bar(name="ROC-AUC",    x=MNAMES, y=MAUC, marker_color="rgba(16,185,129,.8)", marker_line_width=0, text=[f"{v:.2f}" for v in MAUC], textposition="outside", textfont=dict(family=MONO, size=9, color=TC)))
        fig.update_layout(
            barmode="group",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=SANS, color=TC, size=11),
            margin=dict(l=6, r=6, t=36, b=6),
            height=270,
            title=dict(text="Model Comparison — Validation Set", font=dict(size=12, color="#a1a1aa")),
            showlegend=True,
            legend=dict(orientation="h", y=1.15, font=dict(family=MONO, size=9, color=TC), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC)),
            yaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC), range=[0.5, 0.93]),
        )
        st.plotly_chart(fig, use_container_width=True, config=CFG)

    with cr:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=FSCORES, y=FNAMES, orientation="h", marker_color=FCOLORS, marker_line_width=0, text=[f"{v:.3f}" for v in FSCORES], textposition="outside", textfont=dict(family=MONO, size=9, color=TC)))
        fig2.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=SANS, color=TC, size=11),
            margin=dict(l=6, r=6, t=36, b=6),
            height=270,
            title=dict(text="Feature Importance (Avg Score)", font=dict(size=12, color="#a1a1aa")),
            showlegend=False,
            xaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC), range=[0, 1.06]),
            yaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC), autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True, config=CFG)

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ── Threshold chart + Report table + Donut ─────────────────────
    c1, c2, c3 = st.columns([1.6, 2.5, 1.4], gap="small")

    with c1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=TH_X, y=TH_F1, name="F1 Churn", mode="lines+markers", line=dict(color="#f59e0b", width=2), marker=dict(size=5), fill="tozeroy", fillcolor="rgba(245,158,11,.06)"))
        fig3.add_trace(go.Scatter(x=TH_X, y=TH_AC, name="Accuracy",  mode="lines+markers", line=dict(color="#1a6dc2",  width=2, dash="dot"), marker=dict(size=5)))
        fig3.add_vline(x=0.55, line_color="#10b981", line_dash="dash", line_width=1.5, annotation_text="0.55", annotation_font=dict(color="#10b981", size=9))
        fig3.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=SANS, color=TC, size=11),
            margin=dict(l=6, r=6, t=36, b=6),
            height=230,
            title=dict(text="Threshold Sweep", font=dict(size=12, color="#a1a1aa")),
            showlegend=True,
            legend=dict(orientation="h", y=1.18, font=dict(family=MONO, size=9, color=TC), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(title="Threshold", gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC)),
            yaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC), range=[0.5, 0.83]),
        )
        st.plotly_chart(fig3, use_container_width=True, config=CFG)
        st.markdown("""<div class="mrow" style="margin-top:4px">
          <div class="mcell"><div class="ml">Default (0.50)</div><div class="mv" style="color:#f59e0b;font-size:15px">F1 0.616</div></div>
          <div class="mcell"><div class="ml">Optimised (0.55)</div><div class="mv" style="color:#10b981;font-size:15px">F1 0.630</div></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card" style="padding:0;overflow:hidden">
          <div style="padding:10px 13px 8px;border-bottom:1px solid rgba(255,255,255,.07)">
            <span style="font-size:12px;font-weight:600;color:#fafafa">Classification Report</span>
            <span style="font-size:10px;color:#71717a;margin-left:8px">LR · threshold 0.55 · test set (906 samples)</span>
          </div>
          <table class="rtbl">
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>
              <tr><td><span class="badge bg2">No Churn</span></td><td class="mono">0.89</td><td class="mono">0.78</td><td class="mono">0.83</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">665</td></tr>
              <tr><td><span class="badge br2">Churn</span></td><td class="mono">0.55</td><td class="mono">0.74</td><td class="mono">0.63</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">241</td></tr>
              <tr style="background:rgba(255,255,255,.02)"><td><span class="badge bb2">Macro avg</span></td><td class="mono">0.72</td><td class="mono">0.76</td><td class="mono">0.73</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">906</td></tr>
              <tr style="background:rgba(255,255,255,.02)"><td><span class="badge bb2">Weighted</span></td><td class="mono">0.80</td><td class="mono">0.77</td><td class="mono">0.78</td><td style="color:#52525b;font-family:'IBM Plex Mono',monospace">906</td></tr>
            </tbody>
          </table>
          <div style="padding:8px 13px;border-top:1px solid rgba(255,255,255,.07)">
            <div class="mrow">
              <div class="mcell"><div class="ml">Accuracy</div><div class="mv">76.9%</div></div>
              <div class="mcell"><div class="ml">ROC-AUC</div><div class="mv" style="color:#10b981">0.837</div></div>
              <div class="mcell"><div class="ml">F1 Churn</div><div class="mv" style="color:#f59e0b">0.630</div></div>
              <div class="mcell"><div class="ml">Recall</div><div class="mv" style="color:#60a5fa">74%</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        fig4 = go.Figure()
        fig4.add_trace(go.Pie(
            labels=["Retained", "Churned"], values=[4439, 1604],
            hole=0.66,
            marker=dict(colors=["rgba(16,185,129,.8)", "rgba(239,68,68,.8)"], line=dict(color=BG, width=3)),
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig4.add_annotation(text="6,043",     x=0.5, y=0.58, showarrow=False, font=dict(family=MONO, size=16, color="#fafafa"))
        fig4.add_annotation(text="customers", x=0.5, y=0.42, showarrow=False, font=dict(family=SANS,  size=10, color=TC))
        fig4.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=SANS, color=TC, size=11),
            margin=dict(l=6, r=6, t=36, b=6),
            height=220,
            title=dict(text="Churn Distribution", font=dict(size=12, color="#a1a1aa")),
            showlegend=True,
            legend=dict(orientation="h", y=-0.08, font=dict(family=MONO, size=9, color=TC), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig4, use_container_width=True, config=CFG)

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Retained", x=["M-t-M","1 yr","2 yr"], y=[1655, 1141, 1380], marker_color="rgba(16,185,129,.75)", marker_line_width=0))
        fig5.add_trace(go.Bar(name="Churned",  x=["M-t-M","1 yr","2 yr"], y=[1666,  120,   81], marker_color="rgba(239,68,68,.75)",  marker_line_width=0))
        fig5.update_layout(
            barmode="stack",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=SANS, color=TC, size=11),
            margin=dict(l=6, r=6, t=36, b=6),
            height=200,
            title=dict(text="Churn by Contract", font=dict(size=12, color="#a1a1aa")),
            showlegend=True,
            legend=dict(orientation="h", y=1.18, font=dict(family=MONO, size=9, color=TC), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC)),
            yaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC)),
        )
        st.plotly_chart(fig5, use_container_width=True, config=CFG)


# ══════════════════════════════════════════════════════════════════
#  SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div style="font-size:19px;font-weight:700;letter-spacing:-.5px;color:#fafafa;margin-bottom:3px">Single Customer Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11.5px;color:#71717a;margin-bottom:1rem">Fill in the customer profile — model uses 9 key features at threshold 0.55</div>', unsafe_allow_html=True)

    col_form, col_res = st.columns([2, 1.6], gap="medium")

    with col_form:
        with st.form("predict_form"):
            st.markdown('<div class="sec">👤 Demographics</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            gender  = d1.selectbox("Gender",         ["Male","Female"])
            senior  = d2.selectbox("Senior Citizen",  ["No","Yes"])
            partner = d3.selectbox("Partner",          ["No","Yes"])
            d4, d5  = st.columns(2)
            deps    = d4.selectbox("Dependents",       ["No","Yes"])
            tenure  = d5.number_input("Tenure (months)", 0, 72, 12)

            st.markdown('<div class="sec">📡 Services</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            phone    = s1.selectbox("Phone Service",    ["Yes","No"])
            multiln  = s2.selectbox("Multiple Lines",   ["No","Yes"])
            internet = s3.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            s4, s5, s6 = st.columns(3)
            onsec    = s4.selectbox("Online Security",   ["No","Yes"])
            onbk     = s5.selectbox("Online Backup",     ["No","Yes"])
            devprot  = s6.selectbox("Device Protection", ["No","Yes"])
            s7, s8, s9 = st.columns(3)
            techsup  = s7.selectbox("Tech Support",      ["No","Yes"])
            stv      = s8.selectbox("Streaming TV",      ["No","Yes"])
            smv      = s9.selectbox("Streaming Movies",  ["No","Yes"])

            st.markdown('<div class="sec">💳 Billing & Contract</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            contract  = b1.selectbox("Contract",         ["Month-to-month","One year","Two year"])
            paperless = b2.selectbox("Paperless Billing", ["Yes","No"])
            payment   = b3.selectbox("Payment Method",
                                     ["Electronic check","Mailed check",
                                      "Bank transfer (automatic)","Credit card (automatic)"])
            b4, b5  = st.columns(2)
            monthly = b4.number_input("Monthly Charges ($)",  0.0, 200.0,  65.0, step=0.01)
            total   = b5.number_input("Total Charges ($)",    0.0,10000.0,780.0, step=0.01)

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
                col  = "#ef4444" if pct >= 65 else "#f59e0b" if pct >= 40 else "#10b981"

                st.markdown(f"""
                <div class="verdict {'vc' if is_c else 'vr'}">
                  <div class="vicon">{'⚠️' if is_c else '✅'}</div>
                  <div style="flex:1">
                    <div class="vtitle {'vtc' if is_c else 'vtr'}">{'Will Churn' if is_c else 'Will Stay'}</div>
                    <div class="vprob">Probability {pct}% · {rl} · threshold {THRESHOLD}</div>
                    <div class="meter"><div class="mfill" style="width:{pct}%;background:{col}"></div></div>
                  </div>
                </div>
                <div class="mrow">
                  <div class="mcell"><div class="ml">Probability</div><div class="mv" style="color:{col}">{pct}%</div></div>
                  <div class="mcell"><div class="ml">Risk Level</div><div class="mv" style="font-size:13px">{rl}</div></div>
                </div>""", unsafe_allow_html=True)

                # Gauge
                import plotly.graph_objects as go
                fig_g = go.Figure()
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number", value=pct,
                    number=dict(suffix="%", font=dict(family=MONO, size=22, color="#fafafa")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(family=MONO, size=8, color=TC)),
                        bar=dict(color=col, thickness=0.25),
                        bgcolor=BG,
                        borderwidth=0,
                        steps=[
                            dict(range=[0,   40], color="rgba(16,185,129,.1)"),
                            dict(range=[40,  65], color="rgba(245,158,11,.1)"),
                            dict(range=[65, 100], color="rgba(239,68,68,.1)"),
                        ],
                    ),
                ))
                fig_g.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG,
                    font=dict(family=SANS, color=TC, size=11),
                    margin=dict(l=20, r=20, t=18, b=8),
                    height=190,
                )
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
            <div class="card">
              <div class="lbl" style="margin-bottom:10px">How it works</div>
              <div class="tip"><div class="tdot" style="background:#60a5fa"></div>Fill in the customer profile and click <strong>Predict Churn</strong></div>
              <div class="tip"><div class="tdot" style="background:#f59e0b"></div>Model uses <strong>9 consensus features</strong> (MI + Chi² + RF)</div>
              <div class="tip"><div class="tdot" style="background:#10b981"></div>Optimised threshold <strong>0.55</strong> maximises F1 for churn</div>
              <div class="tip"><div class="tdot" style="background:#ef4444"></div>High ≥65% · Medium ≥40% · Low &lt;40%</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown('<div style="font-size:19px;font-weight:700;letter-spacing:-.5px;color:#fafafa;margin-bottom:3px">Batch CSV Upload</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11.5px;color:#71717a;margin-bottom:1rem">Upload a CSV file — all rows scored instantly</div>', unsafe_allow_html=True)

    col_up, col_res = st.columns([1.6, 2.4], gap="medium")

    with col_up:
        uploaded = st.file_uploader("Drop your CSV here", type=["csv"],
                                    help="Same columns as training data")
        st.markdown("""
        <div class="card" style="margin-top:10px">
          <div class="lbl">Required columns</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#60a5fa;line-height:2">
          customerID · gender · SeniorCitizen<br>Partner · Dependents · tenure<br>
          PhoneService · MultipleLines<br>InternetService · OnlineSecurity<br>
          OnlineBackup · DeviceProtection<br>TechSupport · StreamingTV<br>
          StreamingMovies · Contract<br>PaperlessBilling · PaymentMethod<br>
          MonthlyCharges · TotalCharges</div>
        </div>""", unsafe_allow_html=True)

    with col_res:
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                ids    = df_raw["customerID"].astype(str) if "customerID" in df_raw.columns \
                         else pd.Series(range(1, len(df_raw)+1)).astype(str)
                df     = df_raw.drop(columns=["customerID","Churn"], errors="ignore").copy()
                if "TotalCharges" in df.columns:
                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

                enc_rows   = [encode_row(r.to_dict()) for _, r in df.iterrows()]
                df_enc     = pd.DataFrame(enc_rows)
                prob, pred = run_predict(df_enc)

                total    = len(pred)
                churners = int(pred.sum())
                retained = total - churners
                rate     = round(churners / total * 100, 1)
                high_r   = int((prob >= 0.65).sum())
                med_r    = int(((prob >= 0.40) & (prob < 0.65)).sum())
                low_r    = int((prob <  0.40).sum())

                # KPIs
                k1,k2,k3,k4 = st.columns(4, gap="small")
                k1.markdown(f"""<div class="card"><div class="lbl">Total</div><div class="val">{total:,}</div></div>""", unsafe_allow_html=True)
                k2.markdown(f"""<div class="card gr"><div class="lbl">Will Churn</div><div class="val" style="color:#ef4444">{churners:,}</div></div>""", unsafe_allow_html=True)
                k3.markdown(f"""<div class="card gg"><div class="lbl">Retained</div><div class="val" style="color:#10b981">{retained:,}</div></div>""", unsafe_allow_html=True)
                k4.markdown(f"""<div class="card"><div class="lbl">Churn Rate</div><div class="val">{rate}%</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                rb, rc = st.columns(2, gap="small")
                with rb:
                    st.markdown(f"""
                    <div class="card">
                      <div class="lbl" style="margin-bottom:10px">Risk Breakdown</div>
                      <div class="prog"><div class="ph"><span style="color:#ef4444">● High ≥65%</span><span class="pv">{high_r}</span></div><div class="pt"><div class="pf" style="width:{high_r/total*100:.1f}%;background:#ef4444"></div></div></div>
                      <div class="prog"><div class="ph"><span style="color:#f59e0b">● Medium ≥40%</span><span class="pv">{med_r}</span></div><div class="pt"><div class="pf" style="width:{med_r/total*100:.1f}%;background:#f59e0b"></div></div></div>
                      <div class="prog"><div class="ph"><span style="color:#10b981">● Low &lt;40%</span><span class="pv">{low_r}</span></div><div class="pt"><div class="pf" style="width:{low_r/total*100:.1f}%;background:#10b981"></div></div></div>
                    </div>""", unsafe_allow_html=True)

                with rc:
                    import plotly.graph_objects as go
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Bar(
                        x=["High", "Medium", "Low"],
                        y=[high_r, med_r, low_r],
                        marker_color=["rgba(239,68,68,.8)", "rgba(245,158,11,.8)", "rgba(16,185,129,.8)"],
                        marker_line_width=0,
                        text=[high_r, med_r, low_r],
                        textposition="outside",
                        textfont=dict(family=MONO, size=10, color=TC),
                    ))
                    fig_b.update_layout(
                        paper_bgcolor=BG, plot_bgcolor=BG,
                        font=dict(family=SANS, color=TC, size=11),
                        margin=dict(l=6, r=6, t=36, b=6),
                        height=200,
                        title=dict(text="Risk Distribution", font=dict(size=11, color="#a1a1aa")),
                        showlegend=False,
                        xaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC)),
                        yaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,.06)", tickfont=dict(family=MONO, size=9, color=TC)),
                    )
                    st.plotly_chart(fig_b, use_container_width=True, config=CFG)

                # Results table
                results = pd.DataFrame({
                    "Customer ID":   ids.values,
                    "Probability %": (prob * 100).round(1),
                    "Prediction":    ["Churn" if p else "Retain" for p in pred],
                    "Risk Level":    [risk_label(p) for p in prob],
                })

                def style_pred(v):
                    return "color:#ef4444;font-weight:600" if v == "Churn" else "color:#10b981;font-weight:600"
                def style_risk(v):
                    return ("color:#ef4444" if v == "High Risk" else
                            "color:#f59e0b" if v == "Medium Risk" else "color:#10b981")

                styled = (results.style
                          .map(style_pred, subset=["Prediction"])
                          .map(style_risk, subset=["Risk Level"])
                          .format({"Probability %": "{:.1f}%"})
                          .set_properties(**{"font-family": "IBM Plex Mono", "font-size": "12px"}))

                st.dataframe(styled, use_container_width=True, height=340, hide_index=True)

                st.download_button(
                    "⬇  Download results CSV",
                    data=results.to_csv(index=False).encode(),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Processing failed: {e}")
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 2rem">
              <div style="font-size:34px;margin-bottom:10px">📂</div>
              <div style="font-size:14px;font-weight:600;color:#fafafa;margin-bottom:5px">Upload a CSV to get started</div>
              <div style="font-size:11px;color:#71717a">Use the file uploader on the left</div>
            </div>""", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────
PAGE_MAP = {
    "Dashboard":         page_dashboard,
    "Single Prediction": page_predict,
    "Batch Upload":      page_batch,
}
PAGE_MAP[st.session_state.page]()

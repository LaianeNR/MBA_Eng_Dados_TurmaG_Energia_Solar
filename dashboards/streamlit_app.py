
import io
import base64
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Previsão de Bandeiras | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# ENERGY INTELLIGENCE — DARK EXECUTIVE DASHBOARD
# Layout intentionally mirrors the approved visual reference.
# No real prediction probabilities are fabricated.
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #061522;
        --bg-2: #081d2e;
        --panel: #0a2133;
        --panel-2: #0c273b;
        --line: #16405d;
        --line-soft: #12334b;
        --white: #f4f8fc;
        --muted: #8fa8bb;
        --blue: #087cff;
        --blue-2: #16a0ff;
        --green: #00b86b;
        --yellow: #ffc400;
        --red: #ff3b4e;
        --orange: #ff7a00;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(8,124,255,.08), transparent 28%),
            radial-gradient(circle at 8% 35%, rgba(0,184,107,.035), transparent 25%),
            var(--bg);
        color: var(--white);
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        background: transparent;
    }

    /* Top brand */
    .brandbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:30px;
        padding: 4px 2px 20px 2px;
        border-bottom: 1px solid var(--line-soft);
    }

    .brand-left {
        display:flex;
        align-items:center;
        gap:14px;
    }

    .brand-mark {
        width:42px;
        height:42px;
        border:1px solid #1b6da4;
        border-radius:11px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:var(--blue-2);
        font-size:24px;
        font-weight:800;
        background:#071b2b;
    }

    .brand-name {
        font-size:18px;
        font-weight:800;
        letter-spacing:.4px;
        color:#f6fbff;
    }

    .brand-sub {
        margin-top:2px;
        color:#87a2b6;
        font-size:10px;
        letter-spacing:.2px;
    }

    .brand-meta {
        display:flex;
        align-items:center;
        gap:28px;
        color:#b5c7d5;
        font-size:11px;
        text-align:left;
    }

    .brand-meta-item {
        padding-left:20px;
        border-left:1px solid #1a3b53;
        line-height:1.45;
    }

    .brand-meta-dot {
        color:var(--blue);
        font-size:16px;
        vertical-align:middle;
        margin-right:7px;
    }

    /* Hero */
    .hero-right-cluster {
        position:absolute;
        right:34px;
        bottom:28px;
        display:flex;
        align-items:center;
        gap:24px;
        z-index:3;
    }

    .hero-qr {
        display:flex;
        flex-direction:column;
        align-items:center;
        gap:7px;
    }

    .hero-qr img {
        width:148px;
        height:148px;
        display:block;
        background:#ffffff;
        padding:5px;
        border-radius:7px;
    }

    .hero-qr-label {
        color:#b5c7d5;
        font-size:8px;
        line-height:1.25;
        text-transform:uppercase;
        letter-spacing:1px;
        text-align:center;
    }

    .hero {
        position:relative;
        overflow:hidden;
        min-height:285px;
        margin-top:26px;
        padding:42px 38px 34px 38px;
        border:1px solid #145077;
        border-radius:18px;
        background:
            linear-gradient(90deg, rgba(6,21,34,.99) 0%, rgba(7,28,45,.96) 57%, rgba(5,30,48,.86) 100%);
        box-shadow:0 18px 50px rgba(0,0,0,.22);
    }

    .hero::after {
        content:"";
        position:absolute;
        inset:0;
        pointer-events:none;
        background:
            linear-gradient(90deg, transparent 62%, rgba(8,124,255,.04)),
            repeating-linear-gradient(120deg, transparent 0 80px, rgba(33,105,145,.025) 80px 81px);
    }

    .hero-kicker {
        position:relative;
        z-index:2;
        color:#1595ff;
        font-size:11px;
        font-weight:800;
        letter-spacing:2.2px;
        text-transform:uppercase;
        margin-bottom:12px;
    }

    .hero-title {
        position:relative;
        z-index:2;
        margin:0;
        max-width:760px;
        font-size:48px;
        line-height:1.02;
        font-weight:800;
        letter-spacing:-1.7px;
        color:#f5f8fb;
    }

    .hero-title .accent {
        color:#087cff;
    }

    .hero-sub {
        position:relative;
        z-index:2;
        max-width:690px;
        margin-top:15px;
        color:#bfd0dc;
        font-size:16px;
        line-height:1.5;
    }

    .hero-benefits {
        position:relative;
        z-index:2;
        display:flex;
        gap:28px;
        margin-top:28px;
        flex-wrap:wrap;
    }

    .benefit {
        display:flex;
        align-items:center;
        gap:10px;
        color:#dce8f0;
        font-size:13px;
        line-height:1.35;
        padding-right:26px;
        border-right:1px solid #15374e;
    }

    .benefit:last-child { border-right:0; }

    .benefit-icon {
        width:31px;
        height:31px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#087cff;
        color:white;
        font-size:16px;
        flex:0 0 auto;
    }
    .benefit-icon.lightning {
        font-family: "Segoe UI Symbol", "Arial Unicode MS", sans-serif;
        font-size: 24px;
        line-height: 1;
        transform: rotate(0deg);
    }
    .benefit-icon.lightning-shape {
        width: 14px;
        height: 22px;
        display: inline-block;
        flex: 0 0 14px;
        background: #087cff;
        clip-path: polygon(58% 0%, 100% 0%, 67% 40%, 94% 40%, 35% 100%, 47% 54%, 18% 54%);
        transform: none;
    }


    .hero-side {
        width:150px;
        border-left:3px solid #0d8dff;
        padding-left:22px;
        color:#b5c7d5;
        font-size:15px;
        line-height:1.85;
        letter-spacing:4px;
        text-transform:uppercase;
    }

    /* Business question */
    .question {
        display:flex;
        align-items:center;
        gap:24px;
        margin-top:20px;
        padding:23px 25px;
        border:1px solid #154563;
        border-radius:15px;
        background:#071c2c;
    }

    .question-bar {
        width:7px;
        min-height:64px;
        border-radius:7px;
        background:#087cff;
        flex:0 0 auto;
    }

    .eyebrow {
        color:#1195ff;
        font-size:9px;
        font-weight:800;
        letter-spacing:2px;
        text-transform:uppercase;
    }

    .question-text {
        margin-top:6px;
        color:#f1f6fa;
        font-size:17px;
        line-height:1.4;
        font-weight:700;
    }

    .question-action {
        margin-left:auto;
        min-width:130px;
        padding-left:24px;
        border-left:1px solid #1a415b;
        color:#e3edf4;
        font-size:12px;
        line-height:1.4;
    }

    .question-action span {
        color:#087cff;
        font-size:28px;
        vertical-align:middle;
        margin-right:8px;
    }

    /* Section */
    .section-head {
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        margin-top:27px;
        margin-bottom:13px;
    }

    .section-title {
        border-left:4px solid #087cff;
        padding-left:16px;
    }

    .section-title h2 {
        margin:0;
        color:#eef5fa;
        font-size:17px;
        font-weight:800;
        letter-spacing:.3px;
        text-transform:uppercase;
    }

    .section-title p {
        margin:5px 0 0;
        color:#829bad;
        font-size:11px;
    }

    .update {
        color:#93aabb;
        font-size:11px;
        text-align:right;
    }

    /* Cards */
    .cards {
        display:grid;
        grid-template-columns:repeat(4, 1fr);
        gap:14px;
    }

    .risk-card {
        position:relative;
        min-height:155px;
        overflow:hidden;
        border:1px solid #19425b;
        border-radius:14px;
        background:linear-gradient(180deg, #0b2639 0%, #081d2e 100%);
        box-shadow:0 10px 25px rgba(0,0,0,.14);
    }

    .risk-card::before {
        content:"";
        display:block;
        height:6px;
        background:var(--card-color);
    }

    .risk-inner {
        padding:19px 20px 17px;
    }

    .risk-label {
        color:#dce8ef;
        font-size:11px;
        font-weight:800;
        letter-spacing:1.3px;
        text-transform:uppercase;
    }

    .risk-value {
        margin-top:30px;
        color:#f6fbff;
        font-size:30px;
        line-height:1;
        font-weight:800;
    }

    .risk-note {
        margin-top:12px;
        color:#8da7b8;
        font-size:11px;
        line-height:1.4;
    }

    .risk-icon {
        position:absolute;
        right:18px;
        bottom:18px;
        color:#6b92ae;
        font-size:25px;
    }

    /* Impact */
    .impact {
        display:grid;
        grid-template-columns:1.05fr 2fr;
        gap:18px;
        margin-top:18px;
        padding:25px;
        border:1px solid #16405a;
        border-radius:15px;
        background:#071c2c;
    }

    .impact-intro h3 {
        margin:7px 0 10px;
        color:#f4f8fb;
        font-size:21px;
        line-height:1.18;
    }

    .impact-intro p {
        margin:0;
        color:#a8bccb;
        font-size:12px;
        line-height:1.65;
        max-width:330px;
    }

    .impact-button {
        display:inline-block;
        margin-top:17px;
        padding:9px 15px;
        border-radius:7px;
        background:#087cff;
        color:white;
        font-size:11px;
        font-weight:700;
    }

    .impact-grid {
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:12px;
    }

    .impact-card {
        min-height:150px;
        padding:18px;
        border:1px solid #19435c;
        border-radius:12px;
        background:#092337;
    }

    .impact-icon {
        color:#087cff;
        font-size:22px;
    }

    .impact-card h4 {
        margin:19px 0 8px;
        color:#f2f7fa;
        font-size:12px;
        letter-spacing:.5px;
    }

    .impact-card p {
        margin:0;
        color:#92aabb;
        font-size:11px;
        line-height:1.55;
    }

    /* Evidence cards */
    .evidence-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:18px;
    }

    .chart-card {
        min-height:290px;
        padding:20px;
        border:1px solid #16405a;
        border-radius:15px;
        background:#071c2c;
    }

    .chart-title {
        color:#edf4f8;
        font-size:14px;
        font-weight:800;
    }

    .chart-sub {
        margin-top:4px;
        color:#819bad;
        font-size:10px;
    }

    .chart-placeholder {
        height:185px;
        margin-top:18px;
        border-radius:9px;
        border:1px dashed #214a63;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#5f7f95;
        font-size:11px;
        background:
            repeating-linear-gradient(0deg, transparent 0 35px, rgba(42,86,112,.14) 35px 36px);
    }

    .legend {
        display:flex;
        gap:16px;
        margin-top:12px;
        color:#93aabb;
        font-size:10px;
    }

    .dot {
        display:inline-block;
        width:8px;
        height:8px;
        border-radius:2px;
        margin-right:5px;
    }

    /* Closing */
    .closing {
        display:flex;
        align-items:center;
        gap:22px;
        margin-top:18px;
        padding:25px;
        border:1px solid #16405a;
        border-radius:15px;
        background:#071c2c;
    }

    .quote {
        color:#087cff;
        font-size:45px;
        font-weight:800;
        line-height:1;
    }

    .closing-text {
        color:#e7eef3;
        font-size:14px;
        line-height:1.55;
        font-weight:600;
    }

    .closing-text small {
        display:block;
        margin-top:5px;
        color:#718da2;
        font-size:10px;
        font-weight:400;
    }

    .closing-side {
        margin-left:auto;
        padding-left:25px;
        border-left:1px solid #1a415a;
        color:#7893a6;
        font-size:9px;
        line-height:1.8;
        letter-spacing:2px;
        text-transform:uppercase;
    }

    .footer {
        display:flex;
        justify-content:space-between;
        margin-top:30px;
        padding:0 3px;
        color:#5f7d91;
        font-size:9px;
        letter-spacing:1px;
        text-transform:uppercase;
    }

    /* Streamlit tabs */
    div[data-baseweb="tab-list"] {
        gap:4px;
        background:transparent;
        border-bottom:1px solid #153b54;
        margin-top:12px;
    }

    button[data-baseweb="tab"] {
        color:#7793a7 !important;
        background:transparent !important;
        border-radius:7px 7px 0 0 !important;
        padding:10px 15px !important;
        font-size:11px !important;
        font-weight:600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color:#ffffff !important;
        background:#087cff !important;
    }

    div[data-baseweb="tab-highlight"] {
        display:none !important;
    }

    /* Native Streamlit text inside dark theme */
    .stMarkdown, .stText, label, p {
        color:inherit;
    }

    @media (max-width: 900px) {
        .hero-title { font-size:36px; }
        .hero-side { display:none; }
        .cards { grid-template-columns:repeat(2,1fr); }
        .impact, .evidence-grid { grid-template-columns:1fr; }
        .brand-meta { display:none; }
    }

    @media (max-width: 600px) {
        .cards { grid-template-columns:1fr; }
        .impact-grid { grid-template-columns:1fr; }
        .hero { padding:28px 22px; }
        .question-action { display:none; }
    }

    /* Navegação de apresentação: permanece acessível durante a rolagem */
    .brandbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(6,21,34,.96);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    [data-testid="stRadio"] {
        position: sticky;
        top: 82px;
        z-index: 999;
        background: rgba(6,21,34,.97);
        padding: 8px 0 10px 0;
        border-bottom: 1px solid #16405d;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    [data-testid="stRadio"] > label {
        display: none;
    }

    [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        gap: 6px;
        width: 100%;
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1 1 0;
        justify-content: center;
        min-height: 36px;
        padding: 7px 8px;
        border: 1px solid #16405d;
        border-radius: 8px;
        background: #0a2133;
        color: #9fb5c6;
        font-size: 11px;
        font-weight: 700;
        cursor: pointer;
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #16a0ff;
        color: #f4f8fc;
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #10334d;
        border-color: #16a0ff;
        color: #ffffff;
        box-shadow: inset 0 -2px 0 #16a0ff;
    }

    [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 12px !important;
    }

    .section-title p {
        font-size: 13px !important;
    }

    .impact-card p {
        font-size: 13px !important;
        line-height: 1.55 !important;
    }

    .presentation-bottom {
        margin-top: 34px;
        padding-top: 18px;
        border-top: 1px solid #16405d;
    }

    @media (max-width: 900px) {
        [data-testid="stRadio"] {
            top: 76px;
            overflow-x: auto;
        }
        [data-testid="stRadio"] div[role="radiogroup"] {
            min-width: 760px;
        }
    }

    .flag-definition-row {
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin:8px 0 22px 0;
    }
    .flag-pill {
        display:inline-block;
        padding:8px 12px;
        border:1px solid #28506b;
        border-radius:999px;
        background:#0a2133;
        color:#d7e4ed;
        font-size:11px;
        font-weight:800;
        letter-spacing:.5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABRICKS CONNECTION
# ============================================================

from databricks import sql

CATALOG = "mba"
REFINED = f"{CATALOG}.refined"
TRUSTED = f"{CATALOG}.trusted"


def get_connection():
    """Open a read-only SQL connection using Streamlit Secrets."""
    return sql.connect(
        server_hostname=st.secrets["DATABRICKS_SERVER_HOSTNAME"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"],
    )


@st.cache_data(ttl=300)
def query_df(query: str) -> pd.DataFrame:
    """Execute a SELECT query in the Databricks SQL Warehouse."""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()



@st.cache_data(ttl=300)
def load_model_source_data():
    """Load the same real sources used by the final notebook 07."""
    clima = query_df(f"""
        SELECT MesCompetencia, MesReferenciaClima, PrecipitacaoMediaMm,
               PrecipitacaoAcumuladaMm, PrecipitacaoPctNormal,
               TemperaturaMediaC, UmidadeMediaPct
        FROM {REFINED}.f_modelo_bandeira_clima
        ORDER BY MesCompetencia
    """)
    band = query_df(f"""
        SELECT MesCompetencia, IsVermelha
        FROM {TRUSTED}.f_bandeira
        ORDER BY MesCompetencia
    """)
    ear = query_df(f"""
        SELECT ear_data AS Data, id_subsistema,
               CAST(ear_verif_subsistema_percentual AS DOUBLE) AS Valor
        FROM {TRUSTED}.f_ear_subsistema
        WHERE id_subsistema = 'SE' AND ear_data IS NOT NULL
        ORDER BY ear_data
    """)
    return clima, band, ear


def periodo_yyyymm(value):
    """Normalize date/int/string values to pandas Period('YYYY-MM')."""
    if pd.isna(value):
        return pd.NaT
    text = str(value)[:10]
    if len(text) == 6 and text.isdigit():
        return pd.Period(text, freq="M")
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed.to_period("M")


SIM_MARCOS = {
    "regime_gsf_pld": ("2018-12", None),
    "regime_faixas_2019": ("2019-06", None),
    "intervencao_pandemia": ("2020-05", "2020-11"),
    "intervencao_escassez": ("2021-09", "2022-04"),
}
SIM_FEATURES = [
    "mes_clima", "chuva_acum", "chuva_media", "chuva_pct_normal_ok",
    "temperatura", "umidade", "bandeira_origem", "ear_pct", *SIM_MARCOS.keys()
]


def marca_regime(meses, inicio, fim=None):
    v = meses >= pd.Period(inicio, "M")
    if fim is not None:
        v = v & (meses <= pd.Period(fim, "M"))
    return v.astype(int)


def montar_base_simulador(clima, band, ear):
    """Reproduce the feature construction of notebook 07 using real data."""
    c = clima.copy()
    b = band.copy()
    e = ear.copy()

    c["origem"] = c["MesReferenciaClima"].map(periodo_yyyymm)
    c["competencia"] = c["MesCompetencia"].map(periodo_yyyymm)
    b["periodo"] = b["MesCompetencia"].map(periodo_yyyymm)
    e["mes"] = pd.to_datetime(e["Data"], errors="coerce").dt.to_period("M")

    serie_band = (
        b.dropna(subset=["periodo"])
        .set_index("periodo")["IsVermelha"]
        .astype(float)
        .sort_index()
    )
    ear_se = e.dropna(subset=["mes"]).groupby("mes")["Valor"].mean()

    c["bandeira_origem"] = [
        float(serie_band.get(o - 1, np.nan)) for o in c["origem"]
    ]
    c = c.dropna(subset=["origem", "competencia", "bandeira_origem"]).copy()
    c["mes_clima"] = [p.month for p in c["origem"]]
    c["chuva_media"] = pd.to_numeric(c["PrecipitacaoMediaMm"], errors="coerce")
    c["chuva_acum"] = pd.to_numeric(c["PrecipitacaoAcumuladaMm"], errors="coerce")
    c["temperatura"] = pd.to_numeric(c["TemperaturaMediaC"], errors="coerce")
    c["umidade"] = pd.to_numeric(c["UmidadeMediaPct"], errors="coerce")
    c["ear_pct"] = [float(ear_se.get(o, np.nan)) for o in c["origem"]]

    # Same expanding rainfall-normal logic used by notebook 07:
    # only previous occurrences of the same calendar month.
    c = c.sort_values("origem").reset_index(drop=True)
    normals = []
    for i, row in c.iterrows():
        previous = c.loc[:i - 1]
        same_month = previous[
            previous["mes_clima"] == row["mes_clima"]
        ]["chuva_acum"].dropna()
        normals.append(float(same_month.mean()) if len(same_month) else np.nan)

    c["normal_expansiva"] = normals
    reported_pct = pd.to_numeric(c["PrecipitacaoPctNormal"], errors="coerce")
    c["chuva_pct_normal_ok"] = np.where(
        c["normal_expansiva"] > 0,
        c["chuva_acum"] / c["normal_expansiva"] * 100,
        reported_pct,
    )
    c["IsVermelha"] = [
        int(serie_band.get(p, np.nan)) if p in serie_band.index else np.nan
        for p in c["competencia"]
    ]
    c = c.dropna(subset=["IsVermelha"]).copy()
    c["IsVermelha"] = c["IsVermelha"].astype(int)
    return c, serie_band


def peso_amostra_sim(y, alvos):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    class_weights = {
        cls: n / (len(classes) * count)
        for cls, count in zip(classes, counts)
    }
    class_w = np.array([class_weights[v] for v in y])
    recent_w = np.where(
        pd.PeriodIndex(alvos) >= pd.Period("2024-04", "M"), 5.0, 1.0
    )
    return class_w * recent_w


def fit_model_sim(X, y, alvos):
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    model.fit(X, y, clf__sample_weight=peso_amostra_sim(y, alvos))
    return model


def build_scenario_row(reference_month, scenario, horizon):
    """Build the full feature row expected by the notebook-07 recipe."""
    alvo = reference_month + horizon
    row = dict(scenario)
    for name in SIM_MARCOS:
        ini, fim = SIM_MARCOS[name]
        active = alvo >= pd.Period(ini, "M")
        if fim is not None:
            active = active and alvo <= pd.Period(fim, "M")
        row[name] = int(active)
    return row


def predict_scenario(sim_base, serie_band, reference_month, scenario, horizon):
    """Train the same notebook-07 recipe up to the reference month, then score a scenario."""
    eligible = []
    for _, row in sim_base.iterrows():
        origem = row["origem"]
        alvo = origem + horizon
        if alvo in serie_band.index and alvo <= reference_month:
            item = {c: row[c] for c in SIM_FEATURES if c not in SIM_MARCOS}
            item["alvo_mes"] = alvo
            item["y"] = int(serie_band.loc[alvo])
            eligible.append(item)

    train = pd.DataFrame(eligible)
    if len(train) < 36 or train["y"].nunique() < 2:
        raise ValueError(f"Base histórica insuficiente para M+{horizon}.")

    for name, (ini, fim) in SIM_MARCOS.items():
        train[name] = marca_regime(
            pd.PeriodIndex(train["alvo_mes"]), ini, fim
        )

    model = fit_model_sim(
        train[SIM_FEATURES],
        train["y"].values,
        train["alvo_mes"].values,
    )
    scenario_row = build_scenario_row(reference_month, scenario, horizon)
    x_scenario = pd.DataFrame([scenario_row])[SIM_FEATURES]
    probability = float(model.predict_proba(x_scenario)[0][1])
    return probability, reference_month + horizon, len(train)


GITHUB_PROJECT_URL = "https://github.com/FabioFumioWada/MACK_MBA_Eng_Dados_TurmaG_Energia_Solar"


def get_qr_data_uri():
    """Generate the project QR as an embeddable PNG data URI."""
    try:
        import qrcode

        qr = qrcode.QRCode(version=None, box_size=7, border=2)
        qr.add_data(GITHUB_PROJECT_URL)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def render_qr():
    """QR Code for the project GitHub link requested for the presentation."""
    url = GITHUB_PROJECT_URL
    if not url:
        st.info("O QR Code aparecerá quando o app estiver publicado.")
        return

    try:
        import qrcode

        qr = qrcode.QRCode(version=None, box_size=8, border=3)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        st.image(buf.getvalue(), width=180)
        st.caption(
            f"Aponte a câmera do celular para abrir o dashboard.\n{url}"
        )
    except Exception as exc:
        st.warning(f"Não foi possível gerar o QR Code: {exc}")


@st.cache_data(ttl=300)
def load_bandeiras():
    return query_df(f"""
        SELECT
            MesCompetencia,
            AnoMes,
            NivelBandeira,
            NomBandeiraAcionada,
            ValorAdicionalBandeira,
            EarPercentualNacional,
            EnaPercentualMltNacional,
            CmoMedioNacional,
            CargaTotalNacional,
            DatCarga
        FROM {REFINED}.painel_mensal_bandeira_hidrologia
        ORDER BY MesCompetencia
    """)


@st.cache_data(ttl=300)
def load_model_features():
    return query_df(f"""
        SELECT *
        FROM {REFINED}.modelo_previsao_bandeira
        WHERE ChuvaMedia_M2 IS NOT NULL
        ORDER BY MesCompetencia
    """)


@st.cache_data(ttl=300)
def load_clima():
    return query_df(f"""
        SELECT
            MesCompetencia,
            MesReferenciaClima,
            Mes,
            QtdEstacoesUsadas,
            PrecipitacaoMediaMm,
            PrecipitacaoAcumuladaMm,
            PrecipitacaoNormalMm,
            PrecipitacaoPctNormal,
            TemperaturaMediaC,
            UmidadeMediaPct,
            IsVermelha,
            DatCarga
        FROM {REFINED}.f_modelo_bandeira_clima
        ORDER BY MesCompetencia
    """)


@st.cache_data(ttl=300)
def load_training():
    return query_df(f"""
        SELECT *
        FROM {REFINED}.modelo_treino_m1
        ORDER BY MesCompetencia
    """)


def safe_date(df, column):
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column].astype(str), errors="coerce")
    return df


def flag_name(level):
    try:
        level = int(level)
    except Exception:
        return "—"
    return {
        0: "VERDE",
        1: "AMARELA",
        2: "VERMELHA P1",
        3: "VERMELHA P2",
        4: "ESCASSEZ HÍDRICA",
    }.get(level, f"NÍVEL {level}")


def pct(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def html_cards(current_flag, current_date, model_status):
    return f"""
    <div class="cards">
      <div class="risk-card" style="--card-color:#00b86b;">
        <div class="risk-inner">
          <div class="risk-label">Bandeira atual</div>
          <div class="risk-value">{current_flag}</div>
          <div class="risk-note">{current_date}</div>
        </div>
        <div class="risk-icon">⚑</div>
      </div>

      <div class="risk-card" style="--card-color:#ff3b4e;">
        <div class="risk-inner">
          <div class="risk-label">Risco M+1</div>
          <div class="risk-value">—</div>
          <div class="risk-note">{model_status}</div>
        </div>
        <div class="risk-icon">▥</div>
      </div>

      <div class="risk-card" style="--card-color:#ffc400;">
        <div class="risk-inner">
          <div class="risk-label">Risco M+2</div>
          <div class="risk-value">—</div>
          <div class="risk-note">{model_status}</div>
        </div>
        <div class="risk-icon">▥</div>
      </div>

      <div class="risk-card" style="--card-color:#087cff;">
        <div class="risk-inner">
          <div class="risk-label">Risco M+3</div>
          <div class="risk-value">—</div>
          <div class="risk-note">{model_status}</div>
        </div>
        <div class="risk-icon">▥</div>
      </div>
    </div>
    """


def impact_section():
    return """
    <div class="impact">
      <div class="impact-intro">
        <div class="eyebrow">Por que isso importa?</div>
        <h3>A bandeira tarifária impacta diretamente o custo da energia.</h3>
        <p>
          Antecipar o risco permite transformar informação em planejamento,
          eficiência e maior resiliência para o setor elétrico.
        </p>
        <div class="impact-button">Entenda o problema&nbsp; →</div>
      </div>

      <div class="impact-grid">
        <div class="impact-card">
          <div class="impact-icon">●</div>
          <h4>CONSUMIDORES</h4>
          <p>Mais previsibilidade no planejamento financeiro.</p>
        </div>
        <div class="impact-card">
          <div class="impact-icon">▣</div>
          <h4>EMPRESAS</h4>
          <p>Maior eficiência operacional e redução de riscos.</p>
        </div>
        <div class="impact-card">
          <div class="impact-icon">◆</div>
          <h4>SISTEMA ELÉTRICO</h4>
          <p>Contribuição para um setor mais estável e sustentável.</p>
        </div>
      </div>
    </div>
    """


def closing_section():
    return """
    <div class="closing">
      <div class="quote">“</div>
      <div class="closing-text">
        O objetivo não é prever uma certeza.<br>
        É transformar dados disponíveis hoje em um sinal antecipado de risco para os próximos meses.
        <small>DADOS · ANÁLISE · PREVISÃO · DECISÃO</small>
      </div>
      <div class="closing-side">
        Dados<br>
        Análise<br>
        Previsão<br>
        Decisão
      </div>
    </div>

    <div class="footer">
      <div>Energy Intelligence · Mackenzie MBA · Engenharia de Dados</div>
      <div>Setor elétrico mais inteligente, decisões mais sustentáveis.</div>
    </div>
    """


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------


# ------------------------------------------------------------
# STORYTELLING DASHBOARD

# Imagem oficial da arquitetura fornecida pelo grupo, embutida no próprio .py.
ARCHITECTURE_IMAGE_B64 = "/9j/4AAQSkZJRgABAQAASABIAAD/4QEMRXhpZgAATU0AKgAAAAgAAgEyAAIAAAAUAAAAJodpAAQAAAABAAAAOgAAAAAyMDI2OjA5OjE2IDIxOjM0OjU1AAALkAMAAgAAABQAAADEkAQAAgAAABQAAADYkBAAAgAAAAcAAADskBEAAgAAAAcAAAD0kBIAAgAAAAcAAAD8kpAAAgAAAAQwMDAAkpEAAgAAAAQwMDAAkpIAAgAAAAQwMDAAoAEAAwAAAAEAAQAAoAIABAAAAAEAAAZAoAMABAAAAAEAAAMSAAAAADIwMjY6MDk6MTYgMjE6MzQ6NTUAMjAyNjowOToxNiAyMTozNDo1NQAtMDM6MDAAAC0wMzowMAAALTAzOjAwAAD/7QB8UGhvdG9zaG9wIDMuMAA4QklNBAQAAAAAAEQcAVoAAxslRxwCAAACAAIcAj8ABjIxMzQ1NRwCPgAIMjAyNjA5MTYcAjcACDIwMjYwOTE2HAI8AAsyMTM0NTUtMDMwMDhCSU0EJQAAAAAAEFnkVB3pO/hW8F5C9AE6co3/wgARCAMSBkADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAwIEAQUABgcICQoL/8QAwxAAAQMDAgQDBAYEBwYECAZzAQIAAxEEEiEFMRMiEAZBUTIUYXEjB4EgkUIVoVIzsSRiMBbBctFDkjSCCOFTQCVjFzXwk3OiUESyg/EmVDZklHTCYNKEoxhw4idFN2WzVXWklcOF8tNGdoDjR1ZmtAkKGRooKSo4OTpISUpXWFlaZ2hpand4eXqGh4iJipCWl5iZmqClpqeoqaqwtba3uLm6wMTFxsfIycrQ1NXW19jZ2uDk5ebn6Onq8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAABAgADBAUGBwgJCgv/xADDEQACAgEDAwMCAwUCBQIEBIcBAAIRAxASIQQgMUETBTAiMlEUQAYzI2FCFXFSNIFQJJGhQ7EWB2I1U/DRJWDBROFy8ReCYzZwJkVUkiei0ggJChgZGigpKjc4OTpGR0hJSlVWV1hZWmRlZmdoaWpzdHV2d3h5eoCDhIWGh4iJipCTlJWWl5iZmqCjpKWmp6ipqrCys7S1tre4ubrAwsPExcbHyMnK0NPU1dbX2Nna4OLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAgICAgICAwICAwUDAwMFBgUFBQUGCAYGBgYGCAoICAgICAgKCgoKCgoKCgwMDAwMDA4ODg4ODw8PDw8PDw8PD//bAEMBAgMDBAQEBwQEBxALCQsQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEP/aAAwDAQACEQMRAAAB+K9t9B4O21J3Yk7fP4rK3B6Cd0Dvr4+U3S4XNbowhqLddL48grqQq/Oa9sGuS3R8zjuuO3rNuXnM76PLbkddSak3TOtMuP3SVmetclWx6ttmttpVbattqyk6lZWlTlJmmU6VSdq2ypk5SZcpOpSdq2yq22a22rbattpdtmlbaslSa22rKTq22W2yqSrZrbaXbZpW2rbatsqk5WpOyqTlattqyVakq2rbattq22rbattq22rbattpdtpttq22rbattq22rbattq22pOVq22rbattqyVak5WrbaslWpOVqTlakq2pOVqTsqk7KpOVqTlakpVlk5WpOUmslWpOUlW22rbattpslWpKVZWSrattq22oe2Vttq9TesnP3XwPkbhvvhfvfYKfzRfreF6wx87sdcvTOM5QWO3X9x4ktNOvNxWx6fTn3lAenh9a8+o3XP1+rl8f3TwXPbebxj2+qG8fJvxd8/wDMg5b+p8tyRc9tO3D6u0TW20u2U0nE1DVtSttWSrUnKTW21KyVQ22hk7RVtoKyVVttW21ZKtWydSslVbbMu21bbVttW21bRppUlUu2zLttSslVbbNZWy221bbNbbVttW21bbVttW21bbVttLttNttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVkqTW2y221bKTScrUnKTWSrKycrUnbVtsrbbVttWSpM221D2ytttXqbpq8+5+B8gRKvgv0H9E7P138keLv8Aqn47/Yn4FZPnIn374Lrj8/I/Q3Fvzyj23r2z+Yu59TkN9Cfnj9reY46fOxP0d+Vtc+m9h4H2rHp+Q/EPcX3Vx/Pqv0q8Bx2+VYC/6eRt7D9cuuPu8/8AiH7r8XYfPO/R/wCZ9MvnbtuJ9c1z+y9Z/LPB6FR4N+jXzF08fgcfpR4AR8rJ95+iWX8/d+lv56GofbvqmcdeO+GPvXxQP84o/Sv5mbL0fxH0Dy9NPJ1foHyeufxVvrr3Vh+aEe3eE68/0UTz77z5O380Puv59+iFPwQL6n976MfzeV7x2oX5S36D+IMPmdX371wb81E/d1ZD4kV2P3QyfnN6V9V/LIb61879X814u/z350/Sthtz/nHrz706uX870/ov+e9MlDV0cqttWUnNKydSsnUrZNTKdSttW21bbVttLttW20221bbVttW21bbVttW2TSsnUrbVttW21bbVttW21bbVttW21bbVttW21baKnJ1KydW21bbUrJVW21bbVttWSrLJUlVZKtSdtW21bZNbbKycpNbbLbbTbbUnbTbbUPJytsrV6k9ZPfufgfHhFn4H9D/ZXxn0z8geD0P0X8d+TfqLow9F8W9M8Fr9BKO+8K5Orkvo782f0P2zafGH3H8Fsn3z5v6N85o3uXyJ9QHJ8B969t+Zs36D1Wh5yvh39PvnD3x0/LP0XzPpezh++flr7r8T8/0Y93+f/QDeR+7fPdK6/J/qvmfpHVxfpt4l6t8/ef6G+efuCo2z8a+nfmv6jzf5HeMhb4+0+Fe0fOWWn018s/ePi1NfdPEeuDec++/MdG6+leFe2+Jw+8fyb/Vf8ojn9E+1+4fL417z8+f01/NDp5Q/q7+Uv3ms8+av0V/IPPT9hPlX2752y2+g/O/FPZGXwyx4fs9svRvRfLfSOff47/Rv83f0m1X5r9C+efoFD33yF6V846ZfZ3mf0L84Y7c79N/LH1Dpn+fH0vyP0cy9l+d33r8fBvnpad6fkKydKrbNK21bbVts1ttSsnUrJ1K21bbVttW21bbVslVbbVttW21bbVttW21bbVttW21bbVttW21bJVW21bbVttW21bbVttScpNbbLbbVtlUnZVbbVts1ttWSrLJUnUpKk1ttW21bJ1bbKydtW2y220ydlUnKTNtk0PbKyttXqT1m8+5+B8eiJ+E/QPVPLnDYGbyjxXpecStT6HceOkB3Y8c81TouWbTlr6DzlE+r2Pxto7h3EeeQ6+rectx17b5NUwmiFSvXH0zeavkfpeZaDI9z8ibV4M2LDFfWy+RHVrCx5Fzonp9DxMo/RK5wjjueVqUy+mNPPsmnc81V4r7l5RUHU9ZU0CtE9E8zXOb+hbgJdfS/MCaA+l5zS+k+dGWWc+teHkzLvpeaaaC1f83Dp2JuFUj2HoHl6aR7J5DIvQ/O4mX1fkaNma86Tz+Ke+weIug3QI5mWCFws5pxhsNnjOVWTmlJ2pW2rbZpSVJrbaspOpW2rZKq22rbattq22rbattq2yaVkqrbattq22rbattq2ia22rbaslSaykqrbattq22rbak5WWTlasnaspKmttlttmslSVttq22rbattq22rbak7attltsmbbaslSVbbKmTlJpOUmh5WVttq9Te8i6+u+M4PquXe/I/Ze+0HOu8+fkO14DrG0uuv8te2fqXFcjZJryPsfj/ZuvRbl2kGXonknXx6uz87cWdo85pmmlB2vGXVt2PRo5N+a6nze7nntKfnFa05W35B76BoeerRiw6jzq0tfRa3njKfTIf+Py9pU8fau3D+9+Del2Zk8YZNPRfPOi5J19T5iOaKd4F/zSi+uPLH83pHnLGwJ6ajYtq7Z3w+JuCsA109rxVosnqvJ3xn3BddxY19kjls2fpFH2/nIWi6bzvtp6w/Ktnz75zz1UB6ow4wVdH0vkvXi6EfMtZus4K65oVr0fH3jXe0VENV7zla4DC28k9I89LByczKUNUqslTSk7VttW21ZSdSts1ttSVJ1KyVVtk0rJVWydW21ZSdSsnUrJ1KydSsnUrZNKydSslVbbVttW21bbVsnUrJVW2y221bJ1KTtSsnVtk0rJ1KydW21K21bZNKycspO1ZSdSk5NKydNttW2ysnbVttNttSdkrbJUrbbNZNmbTCkVc5qo1vpabXWql11FU2udVNrmapdcpmqNcRVRrnVSJvtVJNxFU6rXVUjuy1QkvMUo9drqh19paHX2qhVeRVIm6I1S63pMtcrXEadN9tcqFV4YLzqekQw57dAkNQqvNVDr7VQ68mqTXmqj19oUKeg1c/ugxuf3Qrrn90CIc8S9VGii+hl59V7K1Dr6KotfJmodfYVCnopNQT0Ey8/F/mFCnoJrnd0a65rdBo0MdDoc9r7TUSug1nzu6JNc/r5U3P7oNDn90CY0Kr5cvPbocbntfJjR6+XXPa/RCj15o0evNGj18mFHrwkOf19qodfaqFN/AahVfY1Dr7VQ6+1UOvtVCq9mWh15NUGvtVDrvBqRT1hnurKsWSsTZqYVWs5qrTbTVRrTKavWU1Wa01VcWmNVzaaqvWkVWRZJWr9YyDW6y0K3WmNV61S1WRaStVJtl1Ta6mqTXeqkTeJVqfXCaqdth0ZKk0lW022xressq3Tl22TbKSqslWpOsLNG5zdLXBqtV1fG4VPbMRcvrC1rmlWdnXM65b1Va3ex5tTy/YcsrpKAoLbEbbS7bVttW21ZSdVxznS83rmvoud6Blptth0ZSdLttWzhuw2jKZ0TNtGqdE1tlUnYcSZBQuyhETtE06CQRto7ZFLwyAbbE7ZUu21ZScyq21ZSdSsnUrbVk5VJytSVZNKyVVttWTtW21bbUrJ1K21bbVtk0rJ1KTtWytWTtW21KyVVttWSpNbbVttW21bbVlbVtk1ttV1Q31D0c+uKa0nrslXPvttLiDdsGw15SgntR8erw4f045R/lontHVkfNm9Y65l+d0/TboH5Y30f4ktzi/oDzibht7l0BX5t3sfGMvGq+oKVNfnne1eL6c+yc+ak7VtlUnbVrinut+fn9th2bbLbbTbbS2tdZ1m+G22G+21KTtXt3a+F03P1fVfb+Cecpt7xXeY8019T8LylQml30/j/Z6c/0d84NqVG72w8y6Cb6o8G816GvfPPvH+lU994zZ8Vrnz226uDbattq22rbattqu+Z6bl9cS9FzfQ1UbbLbbaXban7B8x1z9E5u49A4/S9I5+04/HfoDVVRN2HY+F+gCpmCOlK+h8DxvUI998me8+Jbc1d9YfLH0sGdgHza6di85lwLqrXy/q7Udx451xTlfevAu5ZT+cPGYHjWQvr83KTiqttW21bbVlJ0qslTW21KTtSslVZKtW21ZO1bbVttW21bbVttSk5VJytW21ZO1bbVtlVttW21ZKk1ttW21bZVJVk1ttW21bbVdUN9Q7YxaVdrNVqSrn222azto6fNqpOy0+l1/MBcOz64qPmFsr/ad/wDBrkX0v1/x0h0+lfUfhUSP9beFcARk+4fipsEj7j5z5CWH+4fMPnBsp+wI+QF19LfNYlbc05W0xTtq22rbatcU9xvz0O2w7NtlttpttqtayyrdeZW2y1Vtq2TqlHotqreTk9mcsPEk+nIA80D6r04bwNfZ9O15Mr0ntYfP4/aUqfGF+0Na8k3S9eR5TvY7SPhG9urmXyLdBz5ttpdtpttq22lveT6zkts19LzXTg0u2y122lykqp6wsWG+aM5985en59T7A4h4yr1Nw48gj3XngfL1+y0kPL0+7BQ+J73nkzeaJtfQWHk6vXireOE9mk3is+pEZfKhey4N4wT13ocn8Cj2Hm9c+CJ1/oat4WL1rqofPivWQNeX7vrSXy1Pr3bK3zWr3/j2vMN7JdV4DnTV8ttq22rbatsqkq2rbaslWpO2rZWpOVq22rbattq22pO2rbalZOpW2rJyqTlak5SaVtq22rJUmttq22q85/oOf6OdFvUW66VeVufbbZrOmrx82iF5NLKp+gW2Wni773G5h88Nvc+hr5RX6d3FfO3W+1vw3zJvcepfL5hvPY7RNPmZf0555Xk2+qvMiPI1/R5TfNbz2jqjeBc99U0ovmrXFOw22lyk6tcU91vz89MTh3bbKu2022irWvfMdebbbLXKTqVkqroeg89mPqDbzhNegteIVDu3vm+j0Ljlsw9IH53lb1ifJtD1yv8AM8K96HgcL1XpvAtXolfxuYdPzGzLttNttW21bbS3vJ9VymqL6fl+nqm22T7bVlJ1P2Ck6J0nSeb7LTubHzXT+ys/Jsl63Xea5h7TwXJ4XsXL8LmvUVeWYXott5IoH0K58kSy+rj8u1en3Xi2U+gW3lOa7+z8t0vqvMcjq6Lv/HtXtLHyTV6dHmWrvbTy7R9DV51ivtvN+bYN6PYeVarmmTjmrbNbbVttW21KydSsnVttSsnUrJ1KydSk7VttW21KTtW2TSsnUrbVlJ1KTtW21bJ1KydSsnUrJ0ysnSqydV9zvRc3tjrintrSt2Vz6bOM6tyLllaS6ys3UbGb+gcNClhn+Vjsy587WlLgWiHygbOmNirRZtQrqq0W+dTBjLrTNJdaLfONBvnGM3zpIVvcVNxplz07Y922yrttNomKs2b4PTx6GCseh4pjpXmZ6Z5meE8hpjO801PJZYTyWWp5meM8zPU9hnqeQ0wneaana2OKvsx1Psx1Psx0H8MVRethuiLTleq5VlX0/MdORS7bLTZ3fb83Lbqc2fLbqcblt1Ka5jdPq5jdPq5jdPq5jdOuXld1Om5bdXEvK7qtXKq6VtPR7J5ezK6595/fwae+ytwKu8kXBbvdXBbvMbg93iTcLu7lbg93ma4Hd5luF3eTXCbu9XBbquV7uNWn0DXPz7eoQmvmG9PmvL96hq8vT6jq8u3qOrzDen43mG9PwvMN6fpfME+oxN5jvTYrzPemlZfLdMvinXnVeV1+b70bcuvnKfSEg+e70RRHnO9G1ec70SDeeJ9GwPnO9G1ec70bV5zvRorzrejavOd6MoXm+WL3uC/5ro+c3w1vUW9pXO2jhGZp2y1Vk6lbaspOqchVTGVScpNZJEiTtqVsmO20NtqmNq22rbattq22Nii0H79g76eKlyVc/obRK221aJgVqEzfq5GKkq5uvO7V/wBXnchg9PzdvPbZNNtq22rb2CyTbw3execQpN6o4j5Hr3pWTz3dBiOenoOuVvMd6Jzwuc3pHOkczr5cOeUlWme21bbVLls5fOx5jp+YcK6fmemIqMnZN1/H9byXVwTtuX0YnatEKqNOqNOqNOqNOqJ2rKTpVZOrdZyfU9PByrxlbcPrdDw3W8hx9asndvCrJ1KydSslVbbUrJ1KTtSsnUrJ1bbV6L593nBef6Fl3XHX/ocfBynNmrbVth0uRqpWTqVEppWTqVkqrbattmlOWiY+qeZereUZn1Lyf1Dy35/tVtvoODTGpWTq22rbattq22rbalZOrZOrGDkf1Hzb0zzHyejoOZ6PnPpPJi4p7hdK5w3chmG2x3UnrOWx70PDe8eZ9b865yP1/h0bp+Wz61bbbzvUvU/AvReT0OzpGaKZ3RAK7y54Z4buPk72/wAN055223JttW21bbVttW2zW21P3Ddx18dMpKuTu22rbattmrMDgO/HX6eqy1F2C/JfV8H1BhccPtz0O9ar+Xv8123ke9rGu1fcnnXy/PP1/RPoXx5mz9s9D+R0zfe1B8Yor7UqPjsiP9h2HxlDp9k9h8B+kpp7qD5s5kj7o5T5BWRttvxbbNbbVnTV1rnY8z03MkL6Tm+lIpttlp13I9byHV5yslXL6O21bbUr0bguu9TxX9HV9908XlyPW+gj4FrSr8T6PbZX22rbat1PLdR0cHK29RbcHrWfJdbyXP0ZW3XxZ+z/AEA+e+g+ULD663x32HxP5/8Ao74z63lfIWTvvPhFZKq23abcvF67pBpttltttXoXA9zw3n+hddDz196HHweymz2SgHqPafevRtm+Wub+y8x/Muj+6PhXJVbKl22rbq0dHn8vkq5/Q22pQ1Jh675R6v5Rm/pHl/qXlvg9qtt73Bts1lJJXt1X9kcD+QfcfC+Sr9d+H2yWlJyq29F5jr82hTtxelttW2yt6p5j6b5l4/Xd890fOfS+TFxT3CtXOWxgzVOJhr63d+de1/C/03zPVwTx/vPP+ivRa8HnXkN9SfefzGlWR6XyZd13bhvGk+u0gfzhXptgj+SovvSSnjE+lVDLxm7PslbxrelUrLx+62/rzPenAW843sWm8d3oPnzJtsy2Llo66+KkUlXJ37bVttW2xWzEsXRyspy+bq7zrPO+g+l+OpeUlHgfVZ20Nlr6X5n2nL+p47AgbjyPc7xy/fHSvo7VYFU7vuDB5x/a88t0PU8t1pNZNB27XLBs3KnyAg82KttLts1ttW21Z4zda52PMdPzJC+l5royKjbZadZyPXcj1edlJ3L6KslVbbU77/gfdPd+Y4/kubFwen0ZOcjDq9f8j9C887/NnbeT7m21bbVuo5fp+jg5a2qbng9aw5LsOP5ujbK7OJ59P/MH2Z8h9dx9h61vivsvGKT6L+eu7i+aoRv1T8unYjKP6J8+6n6b4iy+e/XuYy6OL2T899erbV6DwXfcD53feXtBf+jycFlZs8QcEfadWz9j128oee8chHy/5A+lfmXJZTlHNKld9tyelt23NfZfmvkyu/4D439KVkqx68lSYes+WepeV5a+leX+n+YeF2q2V73nJVaGxf6SqPb6H8c+77Xje24jw/R+FU7f0L+YK21Y7b1Ts8/uQ8uT63898c3ecH8d+kbbc/VtsunqPmnpXmvkdN9zPTcz9J4821PcLswMFwkx22epfZvFN531PuQfE9wfT+31/kGfPbb3fzXEHpTWNR02b9I346yZumq556a66fz1uo64XN2seqdeeh0z7GmLy2WvVdT53XuneNuHxX1B95Dheied7Q22ZX7hs66eKlVE83fttLttNts0/QYO/GzyVcnVu/4DdPL6KrzjdXD6OjzvNeg+eK3L3YiT8nUhPsXGVySPXOXW4iOtLNyiPW658/MlekBTTgR+iiZfPXHXcXWSrVtk0rJzSsnUrZNKdNHWudlzPSc0QTo+c6Uik22WnWcn1fKdXBtlcvfttW21O/Y/FPR/a+c80R6Fubs86u+oculVx5RcXo7bYdO21bbVun5jp+jg5a3qLfg9a05DreSw6Mrbq4nX298O/U3x32IdZO/n/eqrav5nbH5vy0fpP5wrdD2k96w5XofX+a3o3krz0vI8rV6vxPy33nPJ2bL0Pgu94LzvQuOg5/oPR4+D2zZ4RcL9EaXmme+1hfcMojrvhH60+Us1God9kKz2/nKfu8roK59SdPF7H8w+nAe82T13I+N9BkqST675R6z5Mmnpfl3q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PH9jzvegr35fO96Jq873os15yr0Ka88T6LFed70TV53vQczefb0LS+e70TK3ne9F1A4nv/PWW46HifXfQ4vGiesCj5YT0/R4Jp6Ohbz0Xo664m7u9Nx/N+qZh5XvVZl8p3qsV5XvVJrypXqia8tT6xYxYeUX9HJ6R5f6I6+X9LzFfpyO3Cgp+1VNwFp1qIcaPttXnrrutXG13omjwbT0dFeZb01MPNN6WqvMt6WWvL96gRG3mN3SdnPd8t0vNe1w68or2di4buVVEpzrMp1bbVO2hpTNbaIrQlSyFpxKtKiBKWisnZaJypttmXZKq0zFSmYrbalEHEEWNc/KUakq5e/baXK2a0TFWaFI6eNjtuTqytqyk5pO2W2yqV3vAJYe+2XzfYZv7OTx+rZfamvjqa9LeeUJTT2a38CIV9ZuvDE17r4UlVbKzKnKTW2zW21bZVQ5AZ8rDnei5xwboub6QrS7bLRSdqVkqrJUmttq22rKTq2yqTtq22rbattq2VqTlattq22rbatsmttq22rbalZOpW2pKtq22pKtq22pKtq22rJVqTlJrbalbattq22rbakq2rJVqSrattqVk6lZOrbattq22rbattq22rJVqTtq22m22q45nqOZ1zi8o7wMwcN3ChjtsdVbLaQnu7X0fI8vV6fzYflEqV5/qDUodEnq/pTHo+Q0/ob86pp4C29kYFPKN9LZX+a99WcPovhafo64zb5bV9R8Zrn4u2+pOQDeD76c5yHgypjTm22ayVanrps56eKk23J6OUlUu2zW2irNm8Y7czpLfKziW2pwprqcZvpnEttTrNdK6S31OFNdTuGup0lvqdZrqcZvqdZqqnEtNTrN9R821Oc21OobanbTKBtuX6fmDF6XmeldKjbZa7bS7bVsrUlW1ZKtScrVtk0rZNbbVttWytW21bbVttW21bbVttW2TW21bZVZKtW21bbVttW2zW2y221bbVttW21bbVttW21bbVttW21bbVttW21bbS7ZVJyk1ttNttW21bZNKyVVttW21bbUnKTV1zPS81pnF5R3doxMEyhmpKs9XXoq/KPX8HvuGtScvXTegck7cVDX1XzBNEZKuL0OwsOA6fPVbvm6pH9FoeZWy+uX/j9UG9bruHqSPTrbxa6j2g/N7Nbquk8ZRXrYuBM1TqEt8VZOZVbanzhu66+KhUlXF37bVsrNbbVaV9jXbc2UnY6qyVUnK1JytSVZNKyVVttW21J21bbVttWytSVbVts1kqTW2VW21JytSVbVb8v1HLsi+m5rpHWo22Wu20uUlVbbVttW21bbVttWSrUnK1JVtW21bbVttW21bbVttW21ZKtWSrVttW21bbVttW21bbVttSsnS7bTbbVttW21bbVttW21KydW21bbVttW21ZSVS7bNbbVtkzbbVttWSrLJytW21bbVttW21XfLdPzGucXdJdjSvcN3Siv6vle26OZvTqXpl7BxXMOfS8S0c+fW/D6Yew4TuNBwKInxfc6PsPKzpt9Ju/mOwN9Ag+b0zfS1f8APC2vp8fzQNb6EsvnlhXvth85RD6H43y7aL9d858zgU/RFV4UtWylJ058nan7ls66eKkUNXN6Ktkqqts1ttVrWWddtzJVtjvttLttW21KydW21KyVVttScrUnbUrbVkq1bbVttW2zW21bbVkq1JytVvzPTczoi+j5zoSKrJVlpttKrJ1KydSslVbbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVttW21bbVlJ0u21bbTbbVttW21bbVttW21bbUrJ0qttW2zSdtNttW21bbLbbVttW21KydShqTVzzXS806xd0l2Xr3AXCqxUnZ6+oNvOI9PxvTC+XJY+lI86yn1LzdtsOjbE4+/q+98Yf56etOPHugtPQF8GPVOtXz9GjWfVebmZPTOR5cUfZd5Llf0WfOUS+k0NPTlPX7PxUgb0+s8/aR9Z8Xvuag9ctnPX5tGpKub0dtlVW2a0TFWtfY1mvMrbZb7bVttW21bbVttLlJ1KyVVttWydW21bbUrbVttWydSk5TSVbVsnUrJVV1y3T81plPR890RFKpOy6FbaXbaXbaspOpWSqsnaspOpW2rbJrbattq22pWTqVtq2TqVkqrbatk6lbatsmlZOpWTqVtq22rZOpWSqttq22rbaspOl22m22rZOrbattqVkqrbJpWTqVk6lZKq22rbatsmlZOpWTqVtq2yaVkqrbatk6lJ2q65np+Y0zi7pbu1YOWjiVnsrHVOUmttq22rbaspOWce0eJ98r30MJZ3HMOauN8Fw3A9A4DmumDXjWGLK5MxDDlK27pGy2Sqtlak7attNPXTZz1cNHkq4/Q22pW2a22WeOKVW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qWIc3tF3G25qTddUmp9Z85ls+zHK73MsZ63E7NVbbk7dtqIlSWXbanrR6yfHrOy6Uf3fwHgvr/N+qcXZ82R1HM/KfYK9v8MVz9X2RZ/EgcOn7Ze/CxpvoD174pE+X2kD4rXlr9G+s/DxNcvrxr8oIr7YoPkJCP8AWPW/EK6+13XxDLr7X6f8jJZfttp8V7LT7jYfFS6+nu2+L0lftlt8WCD/AFXvlYmuf2a6+LBqfuXnPlekTT7Un4rUU+8A/DIK+4WnxMQF59vfCoXT7g8O8YWR9pj+KIz0+sbj46Sb7Wn4n1e7+y/E2ZPublvkLK32W6+Ks19m2vw5q7bjhq25FZOZVZOrbalZKq22rbJrbalbaspOrbaXbattpttq22rK2l22rbattmk5Wrbattq22rbak7attlttmspOpW2rbattq22rJUmttpttqytpdtq22pOVq22rbattq22rJVqSrattq22rbattq22rbattq22rJVqfM3jPTEK4Jnt1pH5evy6Zh0YE1pee7YqPwWUnDu22rPWD/XKnmJ5uvbaiJUllSran7Ry21x9x7z5SL9X8d3vvHyK7z37fz1O+d+im2qOo4+u0bvoZaM/SFNQ1d70IehrumiSlp+i01HV9zyMLU1gYrq2yaxY1Vldx443RdCLz1l07Oq3oXNEtXv57bQc0KxdR4Doa3rF0o7GGcHlVcCk5G1YdRFlW+i8+wqbBAlal6clitVVHrnmro2rb14G8sxgw22rbZrbZbbZrbKrbKpOVqTtq2TqVtqytqTlJrbKrbattq2VqTlaVOVqTlak7KpO2pWSqttq22a22rbattq22W22rbZrbattq2yaVtq22rbattqyVZWTlakqSqttmXbattq22rbattq22rbattq22rbattq22rbattq22rbZbbaslWafNHbTTEQyDw36B52c93k8E57OvhyKLe9z6POd6PC3nLf1Ly8bJfMXy6007c/XttREqSy7bU/bN3b5DhxiG8ngEGNERwVMEYuoWLqTiaDfHiIlF0B5cmFBtQYNAIsVRgKJNCxZlDjYluouAQk2oONohxtQVE0BSTQClzEzchYgPExIcfCBjam6jTQFHzK3h1qaqcKprnGpqoyqBLhTI1h3AmmeaZlL1FNc6wm2cpoGcy01ztNNs5SsCTKpvDrMrfPNTOHaRNc5UWa5yqVpn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1pfNkomWRKrK890vF3M0i81hhciWJZgRJMHkokyLUVLUs82UhU0ymqSRTE0wWVEn7udYu6JFxKTuqC7m+ubr7hJP89X7yOlZ1V/M8f5ur4/eUvJP3LPxb4gsYZ/GfiS4h4/cyWE0p/M07JUpH3AqiPu7fuV5tVzunjrxHu1tvfihe2bVuvivfN7Ts/iPdtiXL418QzTXu97jf225b7uW7RTeJN1udvJqfuBdI/v2O63dgx4oBa/E5Iurq4vZP9QWmk38wgplR7pch+6XT9zun7ndv3O8fud2/dLt+6Xb9zvH7leP3K8fuV4/c7x+53j9yvH7neP3O7fuV4/cr1+5Xj9zvH7neP3O7fud2/c7t+53b9zvH7ndv3S7fud4/c7x+53j9zvH7neP3O8fud2/c7x+5Xj9zvH7neP3O8fuV4/crx+5Xj9yvX7neP3K8fud4/dLt+6Xb9zu37ndv3O8fud2/c7x+53j9yvH7leP3G9fud4/c7t+53b9zu37ndv3O7fud2/crx+5Xj9zvH7neP3O8fud4/c7x+53j9yvH7neP3O8fud4/c7t+53j9zvH7neP3K8fud4/c7t+6Xb9zu37leP3K8fud4/c7x+53j9zvH7neP3O8fud4/c7x+53j9zvH7ndv3O8fud2/c7t+53b9zu37pdv3S7fud2/crx+5Xj9zvH7ndv3O7fuV4/c7t+53j9zvH7neP3O8fud4/c7x+53b9zvH7neP3K8fud2/dLt+53b9zu37ndv3O7ful2/c7t+53gful2/c7t+53b9zvH7ndv3O7fud4/c7x+53j9yvH7jeP3G8fuV4/c7x+53b9zu37ldv3S7fud4/c7t+53j9zvH7neP3G9fuV4/crx+5Xj9yvH7leP3K8fuV4/c7x+53j9zvH7leP3K8fuV4/c7t+6Xb90u37ndv3O8fud2/c7x+53b9zvH7neP3O8fud4/crx+53j9zvH7ndv3O7fud2/c7stYTbRfzIWtD50z58z50z50z50750750z50z50z58750z50z50z50z58758750750z50z50z58750z50z58z50758758750z50z50758750750z50z50z50z50z50z50z58750z50758758758758758758750z50z50z58z58z50z50750758z58z50z50z50z50z50750z50z58z50750750z50z50750z50z50z50750750750750750z58z58z50z5sz50z50z50758750z50z5sz50750z5sz5sz50z50750750z50750z50750750z50750z58z58z50z50750750758758758758750z50z50758z58750750758z58z50758758750758z50750z505ZluUnnzvnzPnzvnzvnTvnTvnzvnTvnTvnzvnzPnTvmzPnTvnTvnTvnzvnzPnzPnzPnTPnTPnzvnzvnTPnzvnzv3id8+Z86d86d8+Z8+Z8+d86Z86Z8+Z8+Z8+d8+d8+d86Z86Z86Z8+d86Z86Z8+d8+Z8+Z86Z8+Z8+Z8+d82U/zUUaSM7cPmQvmQvmQvmQPmQvmQPmQPmQvmQvmQvmQvmQPmwPmQvmQvmwvmwPmQPmQvmQvmQvmQvmQvmQvmQPOF5wvmQvmQvmQvmwvmwvmQvmQvOB5wPmQPmQvmQvmQvmwPmQPmQvmQvmQvmQvmQvOF5wvmQvmQPOB8yF8yF82B82B86F82F82B82B82F8yF5wvmQPmwvmwvmwvmwvnQPnQPmwPmwPmwPmwvmwPmwPnQPmwPmwPmwvmwvmwvmwvmwPmwvmwPmwPmQvmQvmQPmQvmQvmwvmwvmwvmwvmwvmwvmwvmwvmwPmwvmQvmQvmQvmQvmQvmwPmQvmQPmwvmQvmQPmwPmwPmQvmwvmwvmQvmQvmQvmwvmwvmwvmwvmwvmwvmwvnQPmwPmwPmwOC7RbzX+5/pO85sD5sD5sD5sD5sD5sD51u+dA+dbvnW750D5sD50D59u+dbvmwvnQPnQPmwvmQvmwvnQPn2751u+dbvn2z59s+fbPn2z94tnz7Z863fOtnzrd863fNgfNgfNgfNgfNgfNgfNhfNhfNgfNhfNhfNhfMhfMhfMhfMgfMgfMgfMhfMha40FH8xNpF9yrr9+v3OP85R8GkLWqrr9+j4Orr2ShS/5yhH3Layu7t/0c3ilxYXlp/qWhP+rcSPvpSVGa3mtl/wCp8SR96Gxu7hr2rcIhw/3y2uq/5i49jv4P2603Ddbs7jvWwbnc2/hi93fabKK0sLe0t96v135vb2zULibarLcPCV/tFteeON6s7W43O82Pbp/E3Ni8O7Ltfu3iHxT4uggkV4UgWrwx4hVdKv8AcruHwu77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L96uX7zcP3u5fvdy/erl+9XD97uX75cv3y5fvly/e7l++XL98uX75cv3u5fvdy/e7l+93L97uX75dP3y6fvly/fLl+93L97uX75cv3y6fvdy/e7l++XT98un75cv3u5fvdy/e7l+93L98un75dP3u5fvly/fLl++XL98uX71cv3u5fvVy/erl+93L97uX73cv3u5fvly/e7l+93L97uX75cv3y5fvdy/e7l++XL97uX75cv3y5fvl0/fbt++XT98uX73cv3u5fvly/e7l+93L97uX73cv3u5fvly/fLp+93JfvVy/erl+93L98uX73cv3u5fvdy/e7l+93L98uX75dP3y6fvl0/fLp++XT98un73cv3u5D98un73cv3u5fvly/fbt++XT98uX75dP3y6fvl0/fLp++XT98un77dP327fvt2/frt+/Xj9+vH79dv326fvl0/fbt++3T99u379dv327fvl0/fLp++XT98uX75dP326fvM74/8ALTf/2gAIAQMRAT8B0/XYff8A0xP3aHq8IG6/6P6vCaAl5f1WLjnyjrcB5B/2Bf1eLcY3yE9ZhBolhlhORjE8hHW4JDcJM8+OAEpHynqcQsE+BaetwCO62HU45S2xPP8Ava8/91of7x6J/II+LiIiO7j/AH5T+hmCIg/bxf8AmDh6CUTEyl4r/YM/j7EaPi/9iXL0m7dIHzX+wf7tuQlKX+8Xbg6M4chnuu/P+u/3ZeMQlLxTPp8h2bTVJ+MoGp+eP9hX+8n9Dk3e7u+7/Bx4cPQ+1kEoHj/fnr9M6V9Win/Tk/8AdaH+8ejFntjVhnjHFM8Jjyjp5PtcbmEZDlyQ3SplgrlhGQFJjIyossH5O0sMRHJcuPdLhlhI5YfiostoNU5MVzqLLpyOWOEnlniMWGAgglzQJlwnBXKRIwAD7EkYJHllAjgsYgwYQAiSxwSPJfYPhOCQ4L+lk/p5VbDCZcoxSjIWkfe5MF8hhhMnJiMf9NT/AN1of7x6MPLknVMJmU7LOdZaKfNgMebDOW2ost3uWGUbjZ4RGqfE7ccaslh+KymNytHEik1ZpxG5cst98JAM6RyCKRH+XSTURbIS3WEH73xdBHEA5jSIAU9TwXppf2S5eAAy5ogIszst/wA1/tsPMmHMKckqoUkffbD+KXHGwSXMLh/pqf8AutD/AHj00JkfKJUn7uS7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8MWTXgD0B/8AAP0vzPA7rF04nB1fYQ+IJPzb/fw+/JqwM0tWPlZ+tg/AXIZTidmPe3BmKoq6fHMkprchzYrsPi4Pw/0P+/U7sh/ZG0g8z6f0+0HaEH5AHoH1FjXUmmvxdn8n+3uGFod7M4CIrcsfKBnEiLtc+iPHpMPgOT8AAvzB4eydX1PxD8E/qz8P+v8At6dHhweatfgHXHj54eOj1HM1/wDB+l+Z40Z3oFE/a+BPoVXB+aX8IfQfjwcR6TB4d+Ug7RjFHE524Ezt5+S+g/GAPdA+S5HfUT5+z9JMDA4A6A+D8AOy4P4nGv5PoPnj8w2EKByAF8B7nX5Hn19Bv4QHb4fxdH4f6s/DiMcnyX5t+bfYb8y/MvzL+dv335l+Zfzv5W/ffm2/fb99v338rfvv5W/ffmX8r+V/O/nfm35t+bfyv5X8782+638787fm352/nfzv538787fnb+d/O/nfzv5352/O352/nfzv5X52/O352/nfnb+d/O/nfzv5387+d/O/O352/nfzv5387+d/O/O352/O38r+d/O/nfnb87fyv5352/Nvzb87fnb87fyvgnfq/hTPB1zfdfvfdfvfe/vfLP3gNH73ByfvcnL4vvv3vkH733X733PgPv8A9/DyT9zffX31914PyS+49CH3HorkBfceB+5vvr7m+5vub7m+5vub7m+5vub7j977j977j977m+4/e+4/e+4/e+5vu77u+7vuP3v12+7vub7m+4vub7i+7vu77j977m+4vub7j977j977j977j977m+4/e+5vu77q+7vu77u+7vu77u+7vu77m+5vub+TZfcbkacd83/QfSSdf3vub7/95O438P8Ap/tCjQ1PePwR5npro+rBdh7sEn1tQOz48NLruxrm39fD81wPsX5rgsYavuXyXzcjmTb7ng/NaDmDWviS+F0c3Q+i7d3w5vl/+run83zka+7/ADJH1fb63Yvf6w/D/p/tdIQDtjnKonHLdi/pwLt4fpcDwB6MTt+Iz/8AYIB0/m2oO48/5Lkov2Tp/taH0cWn5v6k/Dwv6i/O/a/O/tfnf2vzv7X537X537X537X3X7X537X537X537X3X7X537X537X537X537X537X537X537X3X7X3X7X537X3X7X537WfU/tfdf2vzv2vzv2vzv7X3X7H+b7r9j/N+d+x/m/O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"
# ------------------------------------------------------------

PROJECT_REFERENCE = pd.Period("2026-08", "M")


def period_label(p):
    if p is None or pd.isna(p):
        return "—"
    return pd.Period(p, "M").strftime("%m/%Y")


def month_label_pt(p):
    names = {
        1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril",
        5: "maio", 6: "junho", 7: "julho", 8: "agosto",
        9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro",
    }
    try:
        p = pd.Period(p, "M")
        return f"{names[p.month]} de {p.year}"
    except Exception:
        return "—"


def month_index(df, primary="MesCompetencia", fallback="AnoMes"):
    """Usa MesCompetencia e, quando vier nulo, cai para AnoMes."""
    out = df.copy()
    primary_dt = (
        pd.to_datetime(out[primary].astype("string"), errors="coerce")
        if primary in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    fallback_dt = (
        pd.to_datetime(out[fallback].astype("string"), errors="coerce")
        if fallback in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    out["_mes_dashboard"] = primary_dt.fillna(fallback_dt)
    return out


def get_reference_row(sim_base, ref):
    rows = sim_base[sim_base["origem"] == ref].sort_values("competencia")
    if rows.empty:
        raise ValueError(f"Não há dados climáticos para a referência {period_label(ref)}.")
    return rows.iloc[0]


def calculate_real_forecast(sim_base, sim_series, ref):
    """Executive forecast using the same feature recipe as notebook 07."""
    r = get_reference_row(sim_base, ref)
    scenario = {
        "mes_clima": int(ref.month),
        "chuva_acum": float(r["chuva_acum"]) if pd.notna(r["chuva_acum"]) else np.nan,
        "chuva_media": float(r["chuva_media"]) if pd.notna(r["chuva_media"]) else np.nan,
        "chuva_pct_normal_ok": float(r["chuva_pct_normal_ok"]) if pd.notna(r["chuva_pct_normal_ok"]) else np.nan,
        "temperatura": float(r["temperatura"]) if pd.notna(r["temperatura"]) else np.nan,
        "umidade": float(r["umidade"]) if pd.notna(r["umidade"]) else np.nan,
        "bandeira_origem": float(r["bandeira_origem"]) if pd.notna(r["bandeira_origem"]) else 0.0,
        "ear_pct": float(r["ear_pct"]) if pd.notna(r["ear_pct"]) else np.nan,
    }
    out = []
    for h in (1, 2, 3):
        try:
            p, target, n_train = predict_scenario(sim_base, sim_series, ref, scenario, h)
            out.append({"h": h, "prob": p, "target": target, "n_train": n_train, "ok": True})
        except Exception as exc:
            out.append({"h": h, "prob": None, "target": ref + h, "n_train": None, "ok": False, "error": str(exc)})
    return scenario, out


def render_forecast_cards(results, title="Risco estimado de bandeira vermelha"):
    cols = st.columns(3)
    for col, item in zip(cols, results):
        with col:
            target = period_label(item["target"])
            if item.get("prob") is None:
                st.metric(f"M+{item['h']} · {target}", "—")
                if item.get("error"):
                    st.caption("Resultado indisponível para este horizonte.")
            else:
                st.metric(f"M+{item['h']} · {target}", f"{item['prob']:.1%}")
                st.caption(title)
                st.progress(min(max(float(item["prob"]), 0.0), 1.0))
                st.caption(f"Base de treino até {period_label(PROJECT_REFERENCE)} · {item['n_train']} observações")


def render_kpi_card(label, value, note, accent="#087cff"):
    return f"""
    <div class="risk-card" style="--card-color:{accent};">
      <div class="risk-inner">
        <div class="risk-label">{label}</div>
        <div class="risk-value">{value}</div>
        <div class="risk-note">{note}</div>
      </div>
    </div>
    """


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
qr_uri = get_qr_data_uri()

st.markdown(
    f"""
    <div class="brandbar">
      <div class="brand-left">
        <div class="brand-mark">⌁</div>
        <div>
          <div class="brand-name">ENERGY INTELLIGENCE</div>
          <div class="brand-sub">Dados hoje. Decisões melhores amanhã.</div>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:22px;">
        <div class="brand-qr">
          <img src="{qr_uri}" alt="QR Code do projeto">
          <div class="brand-qr-label">Acesse<br>o projeto</div>
        </div>
        <div class="brand-meta">
          <div class="brand-meta-item">
            <span class="brand-meta-dot">●</span>
            MACKENZIE MBA<br>
            Engenharia de Dados
          </div>
          <div class="brand-meta-item">
            Setor Elétrico Brasileiro<br>
            Bandeiras Tarifárias
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load Databricks data
# ------------------------------------------------------------
try:
    df_bandeiras = load_bandeiras()
    df_model = load_model_features()
    df_clima = load_clima()
    df_training = load_training()
    df_sim_clima, df_sim_band, df_sim_ear = load_model_source_data()
    db_ok = True
    db_error = None
except Exception as exc:
    df_bandeiras = pd.DataFrame()
    df_model = pd.DataFrame()
    df_clima = pd.DataFrame()
    df_training = pd.DataFrame()
    df_sim_clima = pd.DataFrame()
    df_sim_band = pd.DataFrame()
    df_sim_ear = pd.DataFrame()
    db_ok = False
    db_error = str(exc)

if db_ok and not df_bandeiras.empty:
    df_bandeiras = month_index(df_bandeiras)
    latest = df_bandeiras.dropna(subset=["_mes_dashboard"]).sort_values("_mes_dashboard").iloc[-1]
    current_flag = flag_name(latest.get("NivelBandeira"))
    current_date = period_label(latest.get("_mes_dashboard"))
else:
    current_flag = "—"
    current_date = "—"

# The presentation reference is explicitly August/2026.
reference_month = PROJECT_REFERENCE
reference_available = False
sim_base = pd.DataFrame()
sim_series = pd.Series(dtype=float)
forecast_results = []
forecast_scenario = {}

if db_ok and not df_sim_clima.empty and not df_sim_band.empty and not df_sim_ear.empty:
    try:
        sim_base, sim_series = montar_base_simulador(df_sim_clima, df_sim_band, df_sim_ear)
        reference_available = reference_month in set(sim_base["origem"].dropna().unique())
        if reference_available:
            forecast_scenario, forecast_results = calculate_real_forecast(sim_base, sim_series, reference_month)
    except Exception as exc:
        reference_available = False
        db_error = str(exc)

# ------------------------------------------------------------
# Tabs — ordered as the presentation story
# ------------------------------------------------------------
PAGE_NAMES = [
    "01  Contexto",
    "02  Histórico",
    "03  Sinais",
    "04  Modelos",
    "05  Previsão",
    "06  Arquitetura",
]

if "presentation_page" not in st.session_state:
    st.session_state.presentation_page = 0

def render_presentation_nav():
    selected = st.radio(
        "Navegação da apresentação",
        PAGE_NAMES,
        index=int(st.session_state.presentation_page),
        horizontal=True,
        label_visibility="collapsed",
        key="presentation_nav",
    )
    st.session_state.presentation_page = PAGE_NAMES.index(selected)
    return st.session_state.presentation_page

current_page = render_presentation_nav()

def flag_history_chart(df):
    """Evolução histórica com a linha segmentada pela cor da bandeira oficial."""
    hist = month_index(df)
    hist["NivelBandeira"] = pd.to_numeric(hist["NivelBandeira"], errors="coerce")
    hist = hist.dropna(subset=["_mes_dashboard", "NivelBandeira"]).sort_values("_mes_dashboard")
    if hist.empty:
        st.warning("Histórico sem dados válidos para visualização.")
        return

    colors = {0: "#00b86b", 1: "#ffc400", 2: "#ff3b4e", 3: "#ff3b4e", 4: "#ff7a00"}
    names = {0: "Verde", 1: "Amarela", 2: "Vermelha P1", 3: "Vermelha P2", 4: "Escassez Hídrica"}

    fig = go.Figure()
    x = hist["_mes_dashboard"].tolist()
    y = hist["NivelBandeira"].astype(float).tolist()

    # Segmentos coloridos: cada trecho assume a cor da bandeira observada no mês de origem.
    for i in range(len(x) - 1):
        level = int(round(y[i]))
        fig.add_trace(go.Scatter(
            x=[x[i], x[i + 1]],
            y=[y[i], y[i + 1]],
            mode="lines",
            line=dict(color=colors.get(level, "#087cff"), width=3),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Pontos coloridos e legenda oficial.
    for level in sorted(colors):
        mask = hist["NivelBandeira"].round().astype(int) == level
        if mask.any():
            fig.add_trace(go.Scatter(
                x=hist.loc[mask, "_mes_dashboard"],
                y=hist.loc[mask, "NivelBandeira"],
                mode="markers",
                name=names[level],
                marker=dict(color=colors[level], size=7, line=dict(width=1, color="#061522")),
                hovertemplate="<b>%{x|%m/%Y}</b><br>" + names[level] + "<extra></extra>",
            ))

    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="#07131f",
        plot_bgcolor="#07131f",
        font=dict(color="#cfe0eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, color="#8fa8bb"),
        yaxis=dict(
            title="Nível da bandeira",
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Verde", "Amarela", "Vermelha P1", "Vermelha P2", "Escassez"],
            gridcolor="#183042",
            zeroline=False,
            color="#8fa8bb",
            range=[-0.2, 4.35],
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------
# TAB 1 — CONTEXTO
# ------------------------------------------------------------
if current_page == 0:
    qr_uri = get_qr_data_uri()

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Inteligência preditiva para o setor elétrico</div>
          <h1 class="hero-title">PREVISÃO DE<br><span class="accent">BANDEIRAS TARIFÁRIAS</span></h1>
          <div class="hero-sub">
            Transformando sinais climáticos, hidrológicos e históricos em uma estimativa
            antecipada de risco para os próximos meses.
          </div>
          <div class="hero-benefits">
            <div class="benefit"><span class="benefit-icon lightning-shape" aria-hidden="true"></span>Antecipação<br>de risco</div>
            <div class="benefit"><span class="benefit-icon">▥</span>Decisões<br>mais informadas</div>
            <div class="benefit"><span class="benefit-icon">◆</span>Leitura integrada<br>do sistema elétrico</div>
          </div>
          <div class="hero-side">ENERGIA<br>DADOS<br>RISCO<br>DECISÃO</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="question">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">Pergunta de negócio</div>
            <div class="question-text">
              Com as informações disponíveis ao fim de agosto de 2026, conseguimos estimar
              o risco de bandeira vermelha em setembro, outubro e novembro?
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Por que esse problema importa?</h2>
            <p>A bandeira tarifária transforma condições do sistema elétrico em impacto econômico.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(impact_section(), unsafe_allow_html=True)

    st.markdown("### Como a bandeira é classificada?")
    st.markdown(
        """
        <div class="flag-definition-row">
          <span class="flag-pill">VERDE</span>
          <span class="flag-pill">AMARELA</span>
          <span class="flag-pill">VERMELHA P1</span>
          <span class="flag-pill">VERMELHA P2</span>
          <span class="flag-pill">ESCASSEZ HÍDRICA</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">
            <h2>Ponto de partida</h2>
            <p>O projeto começa observando o comportamento que queremos antecipar.</p>
          </div>
          <div class="update">Referência do modelo<br><strong>{period_label(reference_month)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4, gap="small")
    kpi_data = [
        ("Última bandeira observada", current_flag, f"Competência {current_date}", "#00b86b"),
        ("Horizonte 1", period_label(reference_month + 1), "Primeiro mês à frente", "#ff3b4e"),
        ("Horizonte 2", period_label(reference_month + 2), "Segundo mês à frente", "#ffc400"),
        ("Horizonte 3", period_label(reference_month + 3), "Terceiro mês à frente", "#087cff"),
    ]
    for col, (label, value, note, accent) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(render_kpi_card(label, value, note, accent), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>O que vamos investigar?</h2>
            <p>Antes de chegar à previsão, entendemos o comportamento histórico e os sinais disponíveis.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>COMPORTAMENTO</h4><p>Como as bandeiras variaram ao longo do tempo e quais períodos se destacam?</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>SINAIS</h4><p>Quais informações estavam disponíveis antes de cada mudança de bandeira?</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>PREVISÃO</h4><p>Quanto esses sinais conseguem antecipar a probabilidade de bandeira vermelha?</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# TAB 2 — HISTÓRICO
# ------------------------------------------------------------
if current_page == 1:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>O passado mostra o comportamento do risco</h2>
            <p>Antes de prever o futuro, observamos frequência, intensidade e mudanças de regime.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if db_ok and not df_bandeiras.empty:
        hist = month_index(df_bandeiras)
        hist["NivelBandeira"] = pd.to_numeric(hist["NivelBandeira"], errors="coerce")
        hist = hist.dropna(subset=["_mes_dashboard", "NivelBandeira"]).sort_values("_mes_dashboard")

        st.markdown("### Evolução das bandeiras")
        st.caption("A evolução preserva os níveis oficiais: verde, amarela, vermelha P1, vermelha P2 e escassez hídrica.")
        flag_history_chart(df_bandeiras)

        dist = hist.copy()
        dist["Bandeira"] = dist["NivelBandeira"].map(flag_name)
        dist = dist["Bandeira"].value_counts().rename_axis("Bandeira").reset_index(name="Meses")
        dist["Percentual"] = dist["Meses"] / dist["Meses"].sum() * 100
        st.markdown("### Frequência das bandeiras")
        st.caption("Meses = quantidade de meses observados · Percentual = participação na série histórica.")
        st.dataframe(dist, use_container_width=True, hide_index=True,
                     column_config={"Percentual": st.column_config.NumberColumn("Percentual", format="%.1f%%")})

        st.markdown("### Um período que mudou o comportamento da série")
        st.info("Entre maio e novembro de 2020, a série atravessou um período excepcional associado à pandemia. No protocolo de modelagem, esse intervalo é tratado como uma marca de regime para evitar que um comportamento fora do padrão seja confundido com a dinâmica estrutural.")

        st.markdown("### Indicadores do sistema elétrico")
        cols = [c for c in ["EarPercentualNacional", "EnaPercentualMltNacional", "CmoMedioNacional", "CargaTotalNacional"] if c in hist.columns]
        if cols:
            numeric = hist[["_mes_dashboard"] + cols].copy()
            for c in cols:
                numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
            st.line_chart(numeric.dropna(subset=["_mes_dashboard"]).set_index("_mes_dashboard")[cols], use_container_width=True)

        st.markdown("### Últimos registros observados")
        latest_view = df_bandeiras.drop(columns=["_mes_dashboard"], errors="ignore").tail(12)
        st.dataframe(latest_view, use_container_width=True, hide_index=True)
    else:
        st.warning("Histórico indisponível.")

# ------------------------------------------------------------
# TAB 3 — SINAIS
# ------------------------------------------------------------
if current_page == 2:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Quais sinais estavam disponíveis antes da previsão?</h2>
            <p>O modelo transforma observações do mês de referência em atributos comparáveis.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">≈</div><h4>CLIMA</h4><p>Chuva média, chuva acumulada, percentual da normal, temperatura e umidade.</p></div>
          <div class="impact-card"><div class="impact-icon">◆</div><h4>HIDROLOGIA</h4><p>EAR — Energia Armazenada — indica o nível de energia disponível nos reservatórios.</p></div>
          <div class="impact-card"><div class="impact-icon">⚑</div><h4>PERSISTÊNCIA</h4><p>Indicador de bandeira vermelha do mês anterior, capturando a memória de curto prazo da série.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_model.empty:
        feature_cols = [c for c in [
            "EAR_M0", "EAR_M1", "EAR_M2", "ENA_M0", "ENA_M1", "ENA_M2",
            "CMO_M0", "CMO_M1", "Carga_M0", "Carga_M1",
            "ChuvaMedia_M0", "ChuvaMedia_M1", "ChuvaMedia_M2",
            "ChuvaAcumulada_M0", "ChuvaAcumulada_M1",
            "ChuvaPctNormal_M0", "ChuvaPctNormal_M1",
            "Temperatura_M0", "Umidade_M0", "ChuvaMedia_3M"
        ] if c in df_model.columns]
        if feature_cols:
            st.markdown("### Base refinada usada na modelagem")
            st.dataframe(df_model[["MesCompetencia"] + feature_cols].tail(12), use_container_width=True, hide_index=True)

    if reference_available:
        r = get_reference_row(sim_base, reference_month)
        st.markdown(f"### Retrato de {month_label_pt(reference_month)}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Chuva acumulada", f"{float(r['chuva_acum']):.1f} mm" if pd.notna(r['chuva_acum']) else "—")
        k2.metric("Chuva vs. normal", f"{float(r['chuva_pct_normal_ok']):.1f}%" if pd.notna(r['chuva_pct_normal_ok']) else "—")
        k3.metric("EAR SE", f"{float(r['ear_pct']):.1f}%" if pd.notna(r['ear_pct']) else "—")
        k4.metric("Bandeira anterior", "Vermelha" if int(r['bandeira_origem']) == 1 else "Não vermelha")

        st.caption("Esses valores são os sinais observados no mês de referência. O modelo não utiliza informação do alvo futuro para construir a previsão.")

# ------------------------------------------------------------
# ------------------------------------------------------------
# TAB 4 — MODELOS
# ------------------------------------------------------------
if current_page == 3:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Modelos testados e avaliação</h2>
            <p>O objetivo é identificar corretamente a classe de interesse: bandeira vermelha.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Problema de classificação</div>
            <div class="question-text" style="font-size:16px;">
              Vermelha = 1 · Não vermelha = 0
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Abordagens avaliadas")
    model_cards = """
    <div class="impact-grid">
      <div class="impact-card">
        <div class="impact-icon">01</div>
        <h4>ÁRVORE DE DECISÃO</h4>
        <p>Primeira abordagem de classificação, utilizando variáveis climáticas do mês anterior.</p>
      </div>
      <div class="impact-card">
        <div class="impact-icon">02</div>
        <h4>REGRESSÃO LOGÍSTICA</h4>
        <p>Classificação binária com variáveis climáticas e persistência da bandeira.</p>
      </div>
      <div class="impact-card">
        <div class="impact-icon">03</div>
        <h4>ESPECIFICAÇÃO DO NOTEBOOK 07</h4>
        <p>Regressão logística com clima, persistência, EAR e régua de marcos regulatórios.</p>
      </div>
    </div>
    """
    st.markdown(model_cards, unsafe_allow_html=True)

    st.markdown("### Teste final documentado")
    st.caption("Resultados registrados pelo grupo no teste final dos últimos 24 meses nunca vistos no treino.")
    comparison = pd.DataFrame([
        {
            "Modelo": "Árvore de decisão · clima",
            "Acurácia": "75,00%",
            "F1-score": "62,50%",
        },
        {
            "Modelo": "Regressão logística · + persistência",
            "Acurácia": "91,67%",
            "F1-score": "88,89%",
        },
        {
            "Modelo": "Regressão logística · + persistência + ONI",
            "Acurácia": "91,67%",
            "F1-score": "88,89%",
        },
    ])
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Modelo": st.column_config.TextColumn("Modelo"),
            "Acurácia": st.column_config.TextColumn("Acurácia"),
            "F1-score": st.column_config.TextColumn("F1-score"),
        },
    )

    st.markdown("### Por que o F1-score?")
    st.info(
        "A acurácia foi considerada, mas não foi usada isoladamente. Como a bandeira vermelha é a classe de interesse "
        "e é menos frequente, o F1-score ajuda a avaliar conjuntamente precisão e recall."
    )

    st.markdown("### O protocolo da previsão")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>SEM VAZAMENTO</h4><p>Cada previsão usa somente informações disponíveis até o mês de referência.</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>VALIDAÇÃO TEMPORAL</h4><p>A ordem cronológica é preservada, com janela expansiva e mínimo de 36 meses de treino.</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>RECÊNCIA</h4><p>O protocolo aplica peso 5x às observações a partir de abril de 2024.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_training.empty:
        st.markdown("### Base de treinamento")
        st.caption(
            f"{len(df_training):,} observações na camada refined. "
            "A visualização abaixo mostra o início e o fim da base para deixar explícita a janela temporal."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df_training.head(6), use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(df_training.tail(6), use_container_width=True, hide_index=True)

    st.markdown("### Limitações")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card">
            <div class="impact-icon">•</div>
            <h4>BASE HISTÓRICA</h4>
            <p>A quantidade de observações mensais e a menor frequência da classe vermelha limitam a complexidade da modelagem.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">•</div>
            <h4>CLIMA FUTURO</h4>
            <p>A previsão utiliza dados climáticos observados no mês de referência; não incorpora uma previsão meteorológica futura.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# TAB 5 — PREVISÃO
# ------------------------------------------------------------
if current_page == 4:
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">
            <h2>O que o modelo estima para frente?</h2>
            <p>Previsão calculada a partir da referência de {period_label(reference_month)}.</p>
          </div>
          <div class="update">Referência<br><strong>{period_label(reference_month)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reference_available and forecast_results:
        render_forecast_cards(forecast_results)
        st.success(f"Previsão executiva calculada com os dados reais disponíveis até {period_label(reference_month)} e a mesma receita do notebook 07.")
    else:
        st.warning(f"A referência {period_label(reference_month)} não está disponível nas fontes necessárias para calcular a previsão.")

    st.markdown("### Como interpretar")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">M+1</div><h4>MAIOR VISIBILIDADE</h4><p>É o horizonte mais próximo do mês observado e, em geral, o que possui maior quantidade de informação histórica comparável.</p></div>
          <div class="impact-card"><div class="impact-icon">M+2</div><h4>MAIOR INCERTEZA</h4><p>O horizonte aumenta a distância entre a informação observada e o evento que queremos antecipar.</p></div>
          <div class="impact-card"><div class="impact-icon">M+3</div><h4>SINAL DE TENDÊNCIA</h4><p>É útil como sinal de risco, mas deve ser interpretado com mais cautela do que M+1.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# ------------------------------------------------------------
# TAB 6 — ARQUITETURA
# ------------------------------------------------------------
if current_page == 5:
    architecture_bytes = base64.b64decode(ARCHITECTURE_IMAGE_B64)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Arquitetura de dados</h2>
            <p>Da coleta dos dados à geração de insights.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(architecture_bytes, use_container_width=True)



# ------------------------------------------------------------
# NAVEGAÇÃO INFERIOR — apresentação sem voltar ao topo
# ------------------------------------------------------------
st.markdown('<div class="presentation-bottom"></div>', unsafe_allow_html=True)
prev_col, mid_col, next_col = st.columns([1, 2, 1])
with prev_col:
    if current_page > 0:
        if st.button("← Anterior", use_container_width=True, key=f"prev_{current_page}"):
            st.session_state.presentation_page = current_page - 1
            st.session_state.presentation_nav = PAGE_NAMES[current_page - 1]
            st.rerun()
with mid_col:
    st.caption(f"Página {current_page + 1} de {len(PAGE_NAMES)} · {PAGE_NAMES[current_page]}")
with next_col:
    if current_page < len(PAGE_NAMES) - 1:
        if st.button("Próxima →", use_container_width=True, key=f"next_{current_page}"):
            st.session_state.presentation_page = current_page + 1
            st.session_state.presentation_nav = PAGE_NAMES[current_page + 1]
            st.rerun()

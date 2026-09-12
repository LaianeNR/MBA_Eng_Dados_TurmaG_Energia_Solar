
import io
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

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
    .hero {
        position:relative;
        overflow:hidden;
        min-height:320px;
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

    .hero-side {
        position:absolute;
        right:36px;
        top:100px;
        width:150px;
        z-index:2;
        border-left:2px solid #0d8dff;
        padding-left:16px;
        color:#9eb4c4;
        font-size:11px;
        line-height:1.75;
        letter-spacing:3px;
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
    /* Architecture visual */
    .arch-wrap {
        margin-top: 18px;
        padding: 22px;
        border: 1px solid #154563;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(10,33,51,.98), rgba(7,25,40,.98));
    }

    .arch-flow {
        display:grid;
        grid-template-columns: 1.05fr .95fr 1.55fr 1.05fr;
        gap:10px;
        align-items:stretch;
    }

    .arch-stage {
        position:relative;
        min-height:190px;
        padding:18px 16px;
        border:1px solid #1b4c69;
        border-radius:14px;
        background:#071b2b;
    }

    .arch-stage:not(:last-child)::after {
        content:"→";
        position:absolute;
        right:-16px;
        top:50%;
        transform:translateY(-50%);
        color:#1595ff;
        font-size:22px;
        font-weight:800;
        z-index:3;
    }

    .arch-icon {
        width:42px;
        height:42px;
        margin:2px 0 12px 0;
        display:flex;
        align-items:center;
        justify-content:center;
        border:1px solid #1b6da4;
        border-radius:12px;
        background:#09253a;
        font-size:22px;
    }

    .arch-number {
        color:#087cff;
        font-size:11px;
        font-weight:800;
        letter-spacing:1.5px;
        margin-bottom:8px;
    }

    .arch-stage h4 {
        margin:0 0 10px 0;
        color:#f5f8fb;
        font-size:15px;
        font-weight:800;
    }

    .arch-stage p {
        margin:0 0 9px 0;
        color:#a9bfce;
        font-size:11px;
        line-height:1.5;
    }

    .arch-pill {
        display:inline-block;
        margin:3px 4px 0 0;
        padding:4px 7px;
        border:1px solid #174762;
        border-radius:999px;
        color:#c9dbe7;
        font-size:9px;
        background:#0a2538;
    }

    .arch-layers {
        display:grid;
        gap:6px;
        margin-top:8px;
    }

    .arch-layer {
        padding:7px 9px;
        border-radius:8px;
        font-size:10px;
        font-weight:700;
        border-left:3px solid #087cff;
        background:#0b2a40;
        color:#dce8f0;
    }

    .arch-layer small {
        display:block;
        margin-top:2px;
        color:#8fa8bb;
        font-weight:400;
        font-size:9px;
    }

    .arch-bottom {
        display:grid;
        grid-template-columns:1fr 1fr 1fr;
        gap:10px;
        margin-top:12px;
    }

    .arch-note {
        padding:11px 13px;
        border:1px solid #123d58;
        border-radius:10px;
        background:#061925;
        color:#9fb6c5;
        font-size:10px;
        line-height:1.45;
    }

    .arch-note strong {
        display:block;
        color:#e9f2f8;
        margin-bottom:3px;
    }

    @media (max-width: 900px) {
        .arch-flow { grid-template-columns:1fr 1fr; }
        .arch-stage:not(:last-child)::after { display:none; }
    }

    @media (max-width: 600px) {
        .arch-flow, .arch-bottom { grid-template-columns:1fr; }
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
    "mes_clima", "chuva_media", "chuva_pct_normal_ok", "temperatura",
    "umidade", "bandeira_origem", "ear_pct", *SIM_MARCOS.keys()
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


def predict_scenario(sim_base, serie_band, reference_month, scenario, horizon):
    """Train exactly the notebook-07 recipe up to the reference month, then score a manual scenario."""
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
    scenario_h = dict(scenario)
    target_month = reference_month + horizon
    for name, (ini, fim) in SIM_MARCOS.items():
        scenario_h[name] = int(
            marca_regime(pd.PeriodIndex([target_month]), ini, fim)[0]
        )

    x_scenario = pd.DataFrame([scenario_h])[SIM_FEATURES]
    probability = float(model.predict_proba(x_scenario)[0][1])
    return probability, reference_month + horizon, len(train)


def forecast_current(sim_base, serie_band, reference_month):
    """Calculate M+1/M+2/M+3 using the same recipe exposed by the simulator."""
    ref_rows = sim_base[sim_base["origem"] == reference_month].copy()
    if ref_rows.empty:
        raise ValueError(f"Não há dados de referência para {reference_month.strftime('%m/%Y')}.")

    ref_row = ref_rows.sort_values("competencia").iloc[0]
    scenario = {
        "mes_clima": int(reference_month.month),
        "chuva_media": float(ref_row["chuva_media"]) if pd.notna(ref_row["chuva_media"]) else np.nan,
        "chuva_pct_normal_ok": float(ref_row["chuva_pct_normal_ok"]) if pd.notna(ref_row["chuva_pct_normal_ok"]) else np.nan,
        "temperatura": float(ref_row["temperatura"]) if pd.notna(ref_row["temperatura"]) else np.nan,
        "umidade": float(ref_row["umidade"]) if pd.notna(ref_row["umidade"]) else np.nan,
        "bandeira_origem": float(ref_row["bandeira_origem"]),
        "ear_pct": float(ref_row["ear_pct"]) if pd.notna(ref_row["ear_pct"]) else np.nan,
    }

    results = []
    for h in (1, 2, 3):
        p, target, n_train = predict_scenario(
            sim_base, serie_band, reference_month, scenario, h
        )
        results.append({"h": h, "target": target, "probability": p, "n_train": n_train})
    return results


def risk_label(probability):
    if probability is None or pd.isna(probability):
        return "INDISPONÍVEL", "#31516a"
    if probability >= 0.70:
        return "ALTO RISCO", "#ff3b4e"
    if probability >= 0.50:
        return "ATENÇÃO", "#ffc400"
    return "RISCO MODERADO", "#087cff"


def forecast_cards_html(forecasts, reference_month):
    cards = []
    for item in forecasts:
        p = item.get("probability")
        target = item.get("target")
        n_train = item.get("n_train")
        label, color = risk_label(p)
        value = "—" if p is None else f"{p:.1%}"
        target_txt = target.strftime("%m/%Y") if hasattr(target, "strftime") else "—"
        note = f"{label} · referência {reference_month.strftime('%m/%Y')}" if p is not None else "Previsão indisponível"
        train_note = f"{n_train} observações de treino" if n_train is not None else "Base de treino indisponível"
        cards.append(
            f"""
            <div class="risk-card" style="--card-color:{color};">
              <div class="risk-inner">
                <div class="risk-label">M+{item['h']} · {target_txt}</div>
                <div class="risk-value">{value}</div>
                <div class="risk-note">{note}<br>{train_note}</div>
              </div>
              <div class="risk-icon">⚡</div>
            </div>
            """
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def flag_history_chart(df):
    """History chart whose line changes color as the official tariff flag changes."""
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

    for i in range(len(x) - 1):
        level = int(round(y[i]))
        fig.add_trace(go.Scatter(
            x=[x[i], x[i + 1]], y=[y[i], y[i + 1]],
            mode="lines", line=dict(color=colors.get(level, "#087cff"), width=3),
            hoverinfo="skip", showlegend=False,
        ))

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
        margin=dict(l=10, r=10, t=25, b=10),
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


def render_architecture():
    """Architecture visual with symbols, using native Streamlit columns to avoid literal HTML."""
    st.markdown('<div class="arch-wrap"><div class="arch-flow">', unsafe_allow_html=True)

    cols = st.columns([1.0, 1.0, 1.55, 1.0])
    cards = [
        (cols[0], "01", "☁️", "FONTES", "Dados meteorológicos, operacionais e regulatórios.",
         ["INMET / CPTEC", "ONS / CCEE", "ANEEL"]),
        (cols[1], "02", "⚙️", "ORQUESTRAÇÃO & INGESTÃO", "Coleta, padronização e atualização dos dados.",
         ["Python", "Automação", "GitHub / Jobs"]),
        (cols[2], "03", "🗄️", "DATA LAKE", "Camadas que preservam, tratam e refinam os dados.",
         ["STAGE · Landing", "RAW · Estrutura", "TRUSTED · Qualidade + Star Schema", "REFINED · Negócio + Features"]),
        (cols[3], "04", "📊", "CONSUMO", "Analytics, modelo preditivo e apoio à decisão.",
         ["🧠 ML / IA", "📈 Analytics", "🖥️ Streamlit"]),
    ]

    for col, number, icon, title, desc, pills in cards:
        with col:
            pill_html = "".join(f'<span class="arch-pill">{p}</span>' for p in pills)
            st.markdown(
                f"""
                <div class="arch-stage">
                  <div class="arch-number">{number}</div>
                  <div class="arch-icon">{icon}</div>
                  <h4>{title}</h4>
                  <p>{desc}</p>
                  <div>{pill_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="arch-bottom">
          <div class="arch-note"><strong>🔧 Feature Engineering</strong>Clima + EAR + histórico de bandeira + transformações temporais.</div>
          <div class="arch-note"><strong>🧠 Modelo</strong>Regressão logística com balanceamento, padronização e backtest.</div>
          <div class="arch-note"><strong>🎯 Produto</strong>Probabilidade estimada de bandeira vermelha para M+1, M+2 e M+3.</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


GITHUB_PROJECT_URL = "https://github.com/FabioFumioWada/MACK_MBA_Eng_Dados_TurmaG_Energia_Solar"


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


def month_index(df, primary="MesCompetencia", fallback="AnoMes"):
    """Build a reliable monthly datetime index, using fallback when the primary is null."""
    out = df.copy()
    primary_dt = (
        pd.to_datetime(out[primary].astype("string"), errors="coerce")
        if primary in out.columns else pd.Series(pd.NaT, index=out.index)
    )
    fallback_dt = (
        pd.to_datetime(out[fallback].astype("string"), errors="coerce")
        if fallback in out.columns else pd.Series(pd.NaT, index=out.index)
    )
    out["_mes_dashboard"] = primary_dt.fillna(fallback_dt)
    return out


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

st.markdown(
    """
    <div class="brandbar">
      <div class="brand-left">
        <div class="brand-mark">⌁</div>
        <div>
          <div class="brand-name">ENERGY INTELLIGENCE</div>
          <div class="brand-sub">Dados hoje. Decisões melhores amanhã.</div>
        </div>
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
    valid_latest = df_bandeiras.dropna(subset=["_mes_dashboard"]).sort_values("_mes_dashboard")
    latest = valid_latest.iloc[-1] if not valid_latest.empty else df_bandeiras.iloc[-1]
    current_flag = flag_name(latest.get("NivelBandeira"))
    current_date = (
        latest["_mes_dashboard"].strftime("%m/%Y")
        if pd.notna(latest.get("_mes_dashboard")) else "—"
    )
    model_status = "Previsão calculada pelo mesmo pipeline do modelo 07"
else:
    current_flag = "—"
    current_date = "Databricks indisponível"
    model_status = "Conexão não disponível"

current_forecasts = []
forecast_reference = None
if db_ok and not df_sim_clima.empty and not df_sim_band.empty and not df_sim_ear.empty:
    try:
        _sim_base, _sim_series = montar_base_simulador(df_sim_clima, df_sim_band, df_sim_ear)
        _refs = sorted(_sim_base["origem"].dropna().unique())
        if _refs:
            forecast_reference = pd.Period("2026-07", "M") if pd.Period("2026-07", "M") in _refs else _refs[-1]
            current_forecasts = forecast_current(_sim_base, _sim_series, forecast_reference)
    except Exception:
        current_forecasts = []


tabs = st.tabs(
    [
        "⌂  Visão Geral",
        "▣  Previsão",
        "⌁  Variáveis",
        "↗  Histórico",
        "●  Modelo",
        "▣  Metodologia",
    ]
)

# ------------------------------------------------------------
# TAB 1 — Visão Geral
# ------------------------------------------------------------

with tabs[0]:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Inteligência preditiva para o setor elétrico</div>
          <h1 class="hero-title">
            PREVISÃO DE<br>
            <span class="accent">BANDEIRAS TARIFÁRIAS</span>
          </h1>
          <div class="hero-sub">
            Transformando dados climáticos, hidrológicos e do sistema elétrico
            em sinais antecipados de risco.
          </div>

          <div class="hero-benefits">
            <div class="benefit">
              <span class="benefit-icon">⚡</span>
              Antecipação<br>de risco
            </div>
            <div class="benefit">
              <span class="benefit-icon">▥</span>
              Decisões<br>mais informadas
            </div>
            <div class="benefit">
              <span class="benefit-icon">◆</span>
              Contribuição para um<br>setor elétrico mais estável
            </div>
          </div>

          <div class="hero-side">
            ENERGIA<br>
            DADOS<br>
            PESSOAS<br>
            UM FUTURO<br>
            MAIS ESTÁVEL
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">Pergunta de negócio</div>
            <div class="question-text">
              Com as informações disponíveis hoje, conseguimos estimar o risco
              de bandeira vermelha em M+1, M+2 e M+3?
            </div>
          </div>
          <div class="question-action">
            <span>›</span> Explorar<br>previsões
          </div>
        </div>

        <div class="section-head">
          <div class="section-title">
            <h2>Painel executivo</h2>
            <p>Dados históricos e modelo conectados diretamente ao Databricks.</p>
          </div>
          <div class="update">Última atualização<br><strong>Databricks</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not db_ok:
        st.error("Não foi possível conectar ao Databricks SQL Warehouse.")
        st.caption("Verifique os três Secrets do aplicativo. Nenhum token é exibido pelo dashboard.")
    elif current_forecasts and forecast_reference is not None:
        st.markdown(forecast_cards_html(current_forecasts, forecast_reference), unsafe_allow_html=True)
    else:
        st.markdown(html_cards(current_flag, current_date, model_status), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Por que isso importa?</h2>
            <p>O valor do modelo está em antecipar um sinal de risco antes da decisão.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(impact_section(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como os dados viram uma previsão?</h2>
            <p>A arquitetura conecta fontes, ingestão, camadas de dados, modelagem e consumo.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_architecture()

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Evidências no histórico</h2>
            <p>Os dados abaixo vêm diretamente das tabelas refinadas do Databricks.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_bandeiras.empty:
        chart = month_index(df_bandeiras)
        chart["NivelBandeira"] = pd.to_numeric(chart["NivelBandeira"], errors="coerce")
        chart = chart.dropna(subset=["_mes_dashboard", "NivelBandeira"])
        if not chart.empty:
            st.line_chart(
                chart.set_index("_mes_dashboard")["NivelBandeira"],
                use_container_width=True,
            )
        else:
            st.warning("Histórico sem datas válidas para visualização.")
    else:
        st.warning("Histórico indisponível.")

    st.markdown(closing_section(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Da pergunta ao modelo</h2>
            <p>A história do projeto antes da resposta final.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="question">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">PROBLEMA</div>
            <div class="question-text">
              A bandeira tarifária é observada no presente. Nosso desafio foi investigar
              se os sinais disponíveis hoje conseguem antecipar o risco de bandeira vermelha
              nos próximos meses.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="question">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">PERGUNTA DE NEGÓCIO</div>
            <div class="question-text">
              Com as informações disponíveis hoje, conseguimos estimar a probabilidade
              de bandeira vermelha em M+1, M+2 e M+3?
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Arquitetura da solução")
    st.caption(
        "Fluxo técnico que sustenta o dashboard: dados oficiais → camadas de dados → "
        "modelo → camada de consumo."
    )

    arch = st.columns(5)
    arch_data = [
        ("01", "FONTES", "ANEEL · ONS · INMET · outras fontes oficiais"),
        ("02", "RAW", "Dados brutos preservados para ingestão"),
        ("03", "TRUSTED", "Padronização, qualidade e relacionamentos"),
        ("04", "REFINED + ML", "Tabelas analíticas e modelo de previsão"),
        ("05", "STREAMLIT", "Visualização, previsão e simulador"),
    ]
    for col, (num, title, desc) in zip(arch, arch_data):
        with col:
            st.markdown(
                f"""
                <div class="impact-card" style="min-height:170px;">
                  <div class="impact-icon">{num}</div>
                  <h4>{title}</h4>
                  <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="closing">
          <div class="quote">→</div>
          <div class="closing-text">
            Problema → pergunta → dados → histórico → modelo → previsão → decisão
            <small>Essa é a sequência narrativa da apresentação.</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Acesse o projeto</h2>
            <p>QR Code para abrir o repositório GitHub durante a apresentação.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        qr1, qr2 = st.columns([1, 2])
        with qr1:
            render_qr()
        with qr2:
            st.markdown("### Código e arquitetura")
            st.write(
                "O QR Code leva ao repositório do projeto. A aplicação publicada "
                "é a camada de consumo; os dados e o processamento permanecem no Databricks."
            )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Próximo passo da narrativa</h2>
            <p>Antes de mostrar a previsão, entendemos o que aconteceu no histórico.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Começamos pelo histórico: quais bandeiras ocorreram, com que frequência e "
        "quais períodos mudaram o comportamento da série. Só depois mostramos o modelo "
        "e, por fim, a previsão."
    )

    # ------------------------------------------------------------
    # Existing QR/overview block replaced by the architecture story.
    # ------------------------------------------------------------


# ------------------------------------------------------------
# TAB 2 — Previsão
# ------------------------------------------------------------

with tabs[1]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Qual é o risco à frente?</h2>
            <p>A previsão transforma as variáveis disponíveis hoje em probabilidade de bandeira vermelha.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_forecasts and forecast_reference is not None:
        st.markdown(forecast_cards_html(current_forecasts, forecast_reference), unsafe_allow_html=True)
        st.success(
            f"Previsão calculada com dados reais do Databricks, usando referência de {forecast_reference.strftime('%m/%Y')} "
            "e a mesma lógica do modelo 07."
        )
    elif db_ok:
        st.warning("A previsão não pôde ser calculada com as fontes atuais do modelo.")
    else:
        st.error("Sem conexão com o Databricks.")

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Simulador de cenários</h2>
            <p>Altere as variáveis de entrada e observe como o modelo real responde.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_sim_clima.empty and not df_sim_band.empty and not df_sim_ear.empty:
        try:
            sim_base, sim_series = montar_base_simulador(
                df_sim_clima, df_sim_band, df_sim_ear
            )
            refs = sorted(sim_base["origem"].dropna().unique())
            default_ref = (
                pd.Period("2026-08", "M")
                if pd.Period("2026-08", "M") in refs
                else refs[-1]
            )
            ref = st.selectbox(
                "Mês de referência",
                refs,
                index=refs.index(default_ref),
                format_func=lambda p: p.strftime("%m/%Y"),
            )
            ref_row = (
                sim_base[sim_base["origem"] == ref]
                .sort_values("competencia")
                .iloc[0]
            )

            with st.form("form_simulador"):
                c1, c2, c3 = st.columns(3)

                with c1:
                    temperatura = st.number_input(
                        "Temperatura média (°C)",
                        value=float(ref_row["temperatura"])
                        if pd.notna(ref_row["temperatura"]) else 25.0,
                        step=0.5,
                    )
                    chuva_media = st.number_input(
                        "Chuva média (mm)",
                        value=float(ref_row["chuva_media"])
                        if pd.notna(ref_row["chuva_media"]) else 100.0,
                        step=5.0,
                    )

                with c2:
                    chuva_pct = st.number_input(
                        "Chuva (% da normal)",
                        value=float(ref_row["chuva_pct_normal_ok"])
                        if pd.notna(ref_row["chuva_pct_normal_ok"]) else 100.0,
                        step=5.0,
                    )
                    umidade = st.number_input(
                        "Umidade média (%)",
                        value=float(ref_row["umidade"])
                        if pd.notna(ref_row["umidade"]) else 70.0,
                        step=1.0,
                    )

                with c3:
                    ear = st.number_input(
                        "EAR SE (%)",
                        value=float(ref_row["ear_pct"])
                        if pd.notna(ref_row["ear_pct"]) else 50.0,
                        step=1.0,
                    )
                    band_ant = st.selectbox(
                        "Bandeira anterior",
                        [0, 1],
                        index=int(float(ref_row["bandeira_origem"]))
                        if pd.notna(ref_row["bandeira_origem"]) else 0,
                        format_func=lambda x: (
                            "Vermelha" if x == 1 else "Não vermelha"
                        ),
                    )

                submitted = st.form_submit_button(
                    "▶ SIMULAR CENÁRIO",
                    use_container_width=True,
                )

            if submitted:
                scenario = {
                    "mes_clima": ref.month,
                    "chuva_media": chuva_media,
                    "chuva_pct_normal_ok": chuva_pct,
                    "temperatura": temperatura,
                    "umidade": umidade,
                    "bandeira_origem": band_ant,
                    "ear_pct": ear,
                }

                # The four regulatory markers are model features.
                # Calculate them for each target horizon so the simulator
                # uses the same feature schema as notebook 07.
                for name, (ini, fim) in SIM_MARCOS.items():
                    scenario[name] = int(
                        marca_regime(pd.PeriodIndex([ref + 1]), ini, fim)[0]
                    )

                results = []
                for h in (1, 2, 3):
                    try:
                        p, target, n_train = predict_scenario(
                            sim_base, sim_series, ref, scenario, h
                        )
                        results.append((h, p, target, n_train))
                    except Exception as exc:
                        results.append((h, None, ref + h, None))
                        st.warning(f"M+{h}: não foi possível calcular o cenário ({type(exc).__name__}).")

                cards = st.columns(3)
                for col, (h, p, target, n_train) in zip(cards, results):
                    with col:
                        if p is None:
                            st.metric(
                                f"M+{h} · {target.strftime('%m/%Y')}",
                                "—",
                            )
                        else:
                            st.metric(
                                f"M+{h} · {target.strftime('%m/%Y')}",
                                f"{p:.1%}",
                            )
                            st.caption(
                                "Probabilidade estimada de bandeira vermelha"
                            )
                            st.progress(min(max(p, 0.0), 1.0))
                            st.caption(
                                f"Treino disponível até {ref.strftime('%m/%Y')}: "
                                f"{n_train} observações"
                            )

                st.info(
                    "Simulador de sensibilidade: os valores inseridos alteram as entradas "
                    "do mesmo algoritmo e da mesma engenharia de atributos do notebook 07. "
                    "A previsão executiva acima usa os dados reais disponíveis no Databricks."
                )
        except Exception as exc:
            st.error(
                f"Não foi possível preparar o simulador com os dados reais: {exc}"
            )
    else:
        st.info(
            "O simulador será habilitado quando as três fontes reais do modelo "
            "estiverem disponíveis no Databricks."
        )

# ------------------------------------------------------------
# TAB 3 — Variáveis
# ------------------------------------------------------------

with tabs[2]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Quais sinais entram na previsão?</h2>
            <p>As variáveis representam clima, hidrologia, sistema elétrico e histórico.</p>
          </div>
        </div>

        <div class="impact">
          <div class="impact-intro">
            <div class="eyebrow">Sinais do modelo</div>
            <h3>Dados disponíveis hoje para olhar o risco de amanhã.</h3>
            <p>
              O projeto combina variáveis observadas e transformações temporais
              para construir as features usadas na classificação.
            </p>
          </div>
          <div class="impact-grid">
            <div class="impact-card">
              <div class="impact-icon">≈</div>
              <h4>CLIMA</h4>
              <p>Precipitação, acumulado, normal climatológica, temperatura e umidade.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">◆</div>
              <h4>HIDROLOGIA</h4>
              <p>EAR, ENA e suas variações temporais.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">⚡</div>
              <h4>SISTEMA ELÉTRICO</h4>
              <p>CMO, carga e histórico de bandeira.</p>
            </div>
          </div>
        </div>

        <div class="section-head">
          <div class="section-title">
            <h2>Feature engineering</h2>
            <p>Features reais disponibilizadas pela camada refinada.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_model.empty:
        feature_cols = [
            c for c in [
                "EAR_M0", "EAR_M1", "EAR_M2",
                "ENA_M0", "ENA_M1", "ENA_M2",
                "CMO_M0", "CMO_M1",
                "Carga_M0", "Carga_M1",
                "ChuvaMedia_M0", "ChuvaMedia_M1", "ChuvaMedia_M2",
                "ChuvaAcumulada_M0", "ChuvaAcumulada_M1",
                "ChuvaPctNormal_M0", "ChuvaPctNormal_M1",
                "Temperatura_M0", "Umidade_M0",
                "ChuvaMedia_3M"
            ] if c in df_model.columns
        ]
        if feature_cols:
            st.dataframe(
                df_model[["MesCompetencia"] + feature_cols].tail(12),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("A tabela refinada foi encontrada, mas as colunas esperadas não estão disponíveis.")
    else:
        st.warning("Tabela de features indisponível.")

# ------------------------------------------------------------
# TAB 4 — Histórico
# ------------------------------------------------------------

with tabs[3]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Antes de prever o futuro, olhamos o passado</h2>
            <p>Evolução temporal das bandeiras e dos principais indicadores.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_bandeiras.empty:
        hist = month_index(df_bandeiras)
        hist["NivelBandeira"] = pd.to_numeric(hist["NivelBandeira"], errors="coerce")
        hist = hist.dropna(subset=["_mes_dashboard", "NivelBandeira"])

        st.markdown("### Evolução das bandeiras")
        st.caption(
            "A linha muda de cor conforme a bandeira oficial muda: verde, amarela, "
            "vermelha ou escassez hídrica. Quando MesCompetencia está vazio, o dashboard "
            "usa AnoMes como referência temporal."
        )
        if not hist.empty:
            flag_history_chart(hist)

        dist = hist.copy()
        dist["Bandeira"] = dist["NivelBandeira"].map(flag_name)
        dist = (
            dist["Bandeira"]
            .value_counts()
            .rename_axis("Bandeira")
            .reset_index(name="Meses")
        )
        dist["Percentual"] = dist["Meses"] / dist["Meses"].sum() * 100

        st.markdown("### Quanto tempo cada bandeira apareceu?")
        st.dataframe(
            dist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Percentual": st.column_config.NumberColumn(
                    "Percentual", format="%.1f%%"
                )
            },
        )

        st.markdown("### Um período histórico que merece atenção")
        st.info(
            "Entre maio e novembro de 2020, o modelo trata a pandemia como "
            "um período de intervenção histórica. Essa marca regulatória ajuda "
            "a separar um comportamento excepcional do padrão estrutural da série."
        )

        st.markdown("### Indicadores hidrológicos e do sistema")
        cols = [
            c for c in [
                "EarPercentualNacional",
                "EnaPercentualMltNacional",
                "CmoMedioNacional",
                "CargaTotalNacional",
            ] if c in hist.columns
        ]
        if cols:
            numeric = hist[["_mes_dashboard"] + cols].copy()
            for c in cols:
                numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
            numeric = numeric.dropna(subset=["_mes_dashboard"])
            if not numeric.empty:
                st.line_chart(
                    numeric.set_index("_mes_dashboard")[cols],
                    use_container_width=True,
                )

        st.dataframe(
            df_bandeiras.tail(12),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Histórico indisponível.")

# ------------------------------------------------------------
# TAB 5 — Modelo
# ------------------------------------------------------------

with tabs[4]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como transformamos sinais em previsão?</h2>
            <p>O modelo final utiliza clima, persistência da bandeira, EAR e marcos regulatórios.</p>
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Pipeline preditivo</div>
            <div class="question-text">
              Clima + Hidrologia + Histórico
              → Feature Engineering → Regressão Logística
              → Probabilidade M+1 / M+2 / M+3
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Como contar essa história na apresentação")
    st.info(
        "1) problema e pergunta → 2) arquitetura e fontes → 3) histórico das bandeiras "
        "→ 4) variáveis e insights → 5) modelos testados e métricas → 6) modelo final "
        "→ 7) previsão de M+1/M+2/M+3 → 8) simulador → 9) conclusão e próximos passos."
    )

    st.markdown("### Como chegamos ao modelo final")
    st.info(
        "A equipe comparou diferentes abordagens e variações de modelagem "
        "antes de definir a receita final. O notebook 07 usa regressão logística "
        "com balanceamento de classes, padronização e peso maior para observações recentes. "
        "As métricas oficiais devem ser apresentadas a partir do backtest executado no Databricks."
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card">
            <div class="impact-icon">01</div>
            <h4>COMPARAÇÃO</h4>
            <p>Diferentes abordagens e variações foram avaliadas antes da definição da receita final.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">02</div>
            <h4>PROTOCOLO</h4>
            <p>Backtest por janela expansiva: o modelo usa apenas informação disponível até cada mês de origem.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">03</div>
            <h4>ESCOLHA</h4>
            <p>Regressão logística com balanceamento de classes, padronização e peso maior para observações recentes.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_training.empty:
        st.markdown("### Base de treinamento disponível")
        st.dataframe(
            df_training.tail(12),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"{len(df_training):,} linhas disponíveis na tabela refined.modelo_treino_m1."
        )
    else:
        st.warning("Tabela de treinamento indisponível.")

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Avaliação</h2>
            <p>Os resultados do backtest devem ser apresentados a partir da execução oficial do notebook de modelagem.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "A conexão do dashboard está separada da execução do modelo: o Streamlit lê dados "
        "do SQL Warehouse, enquanto o notebook 07 continua sendo a referência do modelo final."
    )

# ------------------------------------------------------------
# TAB 6 — Metodologia
# ------------------------------------------------------------

with tabs[5]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Da informação à decisão</h2>
            <p>Arquitetura e narrativa do projeto.</p>
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Arquitetura de dados</div>
            <div class="question-text">
              Fontes → Orquestração & Ingestão → Stage / Raw / Trusted / Refined
              → Feature Engineering → Modelo → Probabilidade M+1 / M+2 / M+3
              → Streamlit
            </div>
          </div>
        </div>

        <div class="arch-wrap">
          <div class="arch-flow">
            <div class="arch-stage">
              <div class="arch-number">FONTES</div>
              <h4>Meteorologia + Sistema + Regulação</h4>
              <p>INMET/CPTEC, indicadores operacionais e histórico regulatório de bandeiras.</p>
            </div>
            <div class="arch-stage">
              <div class="arch-number">INGESTÃO</div>
              <h4>Python + automação</h4>
              <p>Coleta, padronização e disponibilização dos dados para as camadas do lake.</p>
            </div>
            <div class="arch-stage">
              <div class="arch-number">DATA LAKE</div>
              <h4>STAGE → RAW → TRUSTED → REFINED</h4>
              <div class="arch-layers">
                <div class="arch-layer">STAGE <small>landing / entrada</small></div>
                <div class="arch-layer">RAW <small>estrutura e validação</small></div>
                <div class="arch-layer">TRUSTED <small>qualidade + Star Schema</small></div>
                <div class="arch-layer">REFINED <small>negócio + features</small></div>
              </div>
            </div>
            <div class="arch-stage">
              <div class="arch-number">CONSUMO</div>
              <h4>Analytics + ML + Streamlit</h4>
              <p>Visualização, previsão e apoio à decisão.</p>
            </div>
          </div>
        </div>

        <div class="impact" style="margin-top:18px;">
          <div class="impact-intro">
            <div class="eyebrow">Princípio</div>
            <h3>O objetivo não é prever uma certeza.</h3>
            <p>
              É transformar dados disponíveis hoje em um sinal antecipado
              de risco para apoiar decisões sobre os próximos meses.
            </p>
          </div>
          <div class="impact-grid">
            <div class="impact-card">
              <div class="impact-icon">01</div>
              <h4>FONTES</h4>
              <p>ANEEL, INMET e indicadores do sistema elétrico.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">02</div>
              <h4>CAMADAS</h4>
              <p>Organização e tratamento em Raw, Trusted e Refined.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">03</div>
              <h4>PRODUTO</h4>
              <p>Probabilidade de bandeira vermelha em três horizontes.</p>
            </div>
          </div>
        </div>

        <div class="section-head">
          <div class="section-title">
            <h2>Transparência</h2>
            <p>Os dados históricos são consultados no Databricks. As probabilidades futuras serão lidas da saída oficial do modelo.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(closing_section(), unsafe_allow_html=True)

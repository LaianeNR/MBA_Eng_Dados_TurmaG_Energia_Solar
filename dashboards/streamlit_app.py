
import io
import base64
import math
import os

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

    .brand-qr {
        display:flex;
        align-items:center;
        gap:12px;
        padding-left:18px;
        margin-left:2px;
        border-left:1px solid var(--line-soft);
    }

    .brand-qr img {
        width:104px;
        height:104px;
        display:block;
        background:#ffffff;
        padding:5px;
        border-radius:7px;
    }

    .brand-qr-label {
        color:#b5c7d5;
        font-size:9px;
        line-height:1.35;
        text-transform:uppercase;
        letter-spacing:.7px;
        white-space:nowrap;
        align-self:center;
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

    /* Jornada do projeto — timeline, ferramentas e cartões de decisão */
    .timeline {
        position:relative;
        margin-top:8px;
        padding-left:30px;
    }

    .timeline::before {
        content:"";
        position:absolute;
        left:9px;
        top:6px;
        bottom:6px;
        width:2px;
        background:var(--line);
    }

    .tl-item {
        position:relative;
        padding:0 0 24px 22px;
    }

    .tl-item:last-child { padding-bottom:0; }

    .tl-dot {
        position:absolute;
        left:-30px;
        top:2px;
        width:16px;
        height:16px;
        border-radius:50%;
        background:var(--panel);
        border:2px solid var(--blue);
    }

    .tl-dot.abandoned { border-color:var(--red); }
    .tl-dot.done { border-color:var(--green); }
    .tl-dot.warn { border-color:var(--orange); }

    .tl-tag {
        font-size:9px;
        font-weight:800;
        letter-spacing:1.6px;
        text-transform:uppercase;
        color:var(--blue-2);
    }

    .tl-tag.abandoned { color:var(--red); }
    .tl-tag.done { color:var(--green); }
    .tl-tag.warn { color:var(--orange); }

    .tl-title {
        color:var(--white);
        font-size:14px;
        font-weight:800;
        margin-top:5px;
        line-height:1.35;
    }

    .tl-text {
        color:var(--muted);
        font-size:12px;
        line-height:1.65;
        margin-top:5px;
        max-width:880px;
    }

    .tool-pill {
        display:inline-flex;
        align-items:center;
        padding:7px 14px;
        margin:4px 6px 4px 0;
        border-radius:20px;
        border:1px solid var(--line);
        background:var(--panel-2);
        color:#cfe0eb;
        font-size:11px;
        font-weight:700;
    }

    .impact-card.abandoned { border-left:3px solid var(--red); }
    .impact-card.done { border-left:3px solid var(--green); }

    @media (max-width: 900px) {
        .timeline { padding-left:24px; }
        .tl-dot { left:-24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ARQUITETURA EMBUTIDA NO CÓDIGO
# A imagem é armazenada como Base64 para que o dashboard não
# dependa de um arquivo de imagem separado no Git.
# ============================================================
ARCHITECTURE_IMAGE_B64 = "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AAjIoILbxTTwabZhABKwQR45L7YJz4ziGP22M2UDG8yf+849/I64+nTz32rTxSgASylI87z7/teyj6IYJ6bY+aLzxTzwJrSoJdto46qb6555T5Q7lHgqIY4m0KCJ333iTzyEAADTy6Bb7ygAIKJpcN+sIBaijL+e+sOMP9pzxTzwLrDSRiiTQjTjgR45Txz6ELR70lM8IKHDXmBz7ygAC/uEotj/AM8oCOifDfHzf8CWgoAuPDjTaW2qe8c88mmSee+uGueO+e+KC8888pUEFUhtU3WixBzfJM18Mcvk4nl6/wDPKAgkz3/1zzzwloKAgAgAAAlOAPvPPPEusvvvvvvvvvvjhvvvHPKQjJEvqMIQaAiHfcQddvPtjvvrsl/PKAEGH85547yAvuOAgAAAgggAAvvPPPLgDsomssqtvqutvvvvHPITSTjjjjgggjjqwTa7iggggggjjF/PKAAkl9+/x0jgvrKAggAggggglvvNPPEkAOpovupovoskvvuvPPKWRmeO/OPzguuWupPe/PfPPPTDG/8AzygJbeuda6oPIL6igAJpIAQgIQD7xTSj5jZ7776L7J67Z776rzzym+mhy97Kh1lfPlhGW73/AN99988vX88oAAACCCAAAAC+ooAKmA2yZATIW8Uco+++++++ZRqC2+6+Gq888hz7I90T4c+kIYsAE0w89999988//wDvKAAAAAggAgAgPrPAApi5EKXpxkPGHKOvvvvuuuRjyhvvrjqvPPFV+gPVYSvKzbKzQ6Ud+NPPPPPPP/8AxygIIIL6IQIIAL6TwBQHKDZTmbzTxzyir77777aamIkLLb7qjzzRk1EhoRWAdghwk3CWEMyNG3DnTz//AMUsCCCOI8PECCCWo4Aa3qYEMVKAW88Uou+++++c5lTiWy++Gq8805BG+PWd8Mc884c8YMoQEMcs8c83/wDsJFssgqsnngAAlqOBBmnhshdpCFvPFKPvvvvvr1ouAhlutnqvPFFUWwIEAMAEMIAALGSiZHfPPLDDDN/FIAAgghoy9gAglgKC725w3+y959PPFKPvuttsusWuvusqgtqPPNARTkvvcZQQQGQEKB94wVfvvvg96U+gNAgggOrrkAgglqKJix84x8178lPNFKPviomjvirgkkisigrPPFAaojkE42MSCXRjzvzfMLDDKANPkPrMOAljk4/3+77gnqKEustw084x9PqFEIPqllmnjqivkokoghvsPFCP7V0hNpLBz9pXlMN0nrjjjjCAGl/AOAhAyy6405+glqKAjthzzzxmy1rHFKPvkx52xp9++z92wlqrEFLd6rhmcTl/MIeMgoqpvvvvvPKAL1fAKAlr6/505z6ylqKAAAAkksAsglvPFKPvvknt+q4y+vugglqvPNOKbzrOpFpV+F7wyhpNmPvvvvuHOX/qAAgsotrOEpvglgKAAAggggAgglvPHPPvvghtguovkvrhogivPFIOqUNVeQKZp/bKyIHZE/sPvuNAoX/KKAgggAggggAglgKAAADjigAgglvPNPPvqgutop35ojtvrlqnPFALs0a4YXA82idAbrNJx2PGF8txEv8AyigAAAAAAwAAIJaigAAbZIKIIABbzzzz774IbaMbrr7YJop6rzxQC5zc7QIgIJ7y2GwyxbjnZKfv9N//AMooAAAALN1gCCCWooAAWu+ukACC2880s+++C6uWG2OC+O+++680cACAxxhCCCCCyyVDuRlBdB5N97gDX8skKAEQqZ2mKACWooACWanXOSSmW8888+++ueuOuueO+++6+q8U8sS6CCCKC+CGSCKqegCiCyyGe9bVX88YSgUwsauWKCCWsoAAWOzUOiAA+888o+++++++++++++++Wu8E0BS3KGHHTEQOQ44wsXDDDCDDHLTHX88oAAAUiNSvOCCW8oAAGmYGyMiA88+8o+++++++++++++++++888NinD3DXLai2G6KaCX+6DXvz19Lf388oAACOuYXn6ACW8oIOSiKGCeGOU8884+++++++++++++++++888NIosPqsHPw0Wu6EsG2iL44cBsslS0UoAQWLbfzHfrCW8sf/AG14/py18/PPPLPvvvvvvvvvvvvvvnvvPOCg+gOBRP8A9QzjrDQA53//APiQ82V5288oAAmX/mmaabCW8o3L7z/D7b7728+88+++++++++++++++++888/bryCag9ZO6GeeeOqCGuyC8Y0y1e884AAC33frDrqCWooGjHbzHfnfn88888+++++++++++++++++888169599xhFyiSO6S2CGDDCiSmqWd+88oCCWfHn77/uCWgoCCCnzzvLCC+88042+6ymy2++++++++++844w++u+//APyxxzz3zjzzzzzjjjjnhnvPPAggt24935/LliBAhjiv78QghhDDDDAgggggggggghljvriDDDDLKmecdfffeccdcdffffffdeOAoBAENBjHCny361/PukMAEMMMMEAAAMMMMAMBDDDDDDDDDDDDCEIMMMMMANKEABnny0073zxw5uhjHAAAAAAAAIMMMIMMMMBDCOPLNMKDDADCNLCDNDCOBAAAPHAAACBCHIAABCDCAPPMMIChDjPvPuPOMABACEJAADAAGNNNMMNPPACBEP/aAAwDAQACAAMAAAAQkMc888ccccccYYYIIIIIIIYYwIw0www00000k000UkkkEkkEsskkkEEskg8s8s88ss8ccYccYcccIEMoA88ss888s8888kIAAAAAAEAEAAAAAAAAAAgQAAAAAAAAAAIAAAAAAAAIAggoosss8s888s888888A88gU8+88uCK8su8u+8888884088w8gw8ggwwww8wwwwwwww4wwwww8Q4ww04wwww0w8888888888++8o84AU88w88sAAAAAAAA04w84w8IAcEAEMAogEEAIYIAwwoMAAMISoAMAAQ088Q0MAQ0AAAAEAAEMEAAAA0AIU+AE88UgIAAgAIAEI44oYw4QEw0McEM8sIkYYsMwkUYQM8cI4Ms80Yc4oE4AoAgMAEEAAAEUgkEEcUAoU+UM4wQQkEM4AEAMwwgM0kU80sAMwEGEogEuwc4oA4k04IUMgK8IokIIo4cMoUU8oAQQAAQsEQAkAUIAU+o8o0cIAQwAAcQ8ggwc8QMss8004gMcooscMM8kkEssQg88c+cQA8840AwMIgcUogAIAAAQwgkAQEAI08IIAAAwwAAM80sEAQA8QAASy2wU4qQwkAC40ygAU4yAA2q66gAgy4wAQAwggAAAAgAAUIMIAEAICAIAU8EwoggQkMc888088+888+++88888840sO+884488uguse+OM08+++us88OIos8oEsK8e8cIcc8kEIoAU8QIAAAQAUwwgAU84AigE+ywAEU2wQAg44gAQA0kAAIQYggAA2SAAggwIAAAwwwAAAQgQAAA+oAAAgoEU8EMMMIAsMIAAM88oAA8MgIM88csEEMMMsAMsMs8McOM0AMc8sccMMsIIEMcsIMEMMMMIACCAAAAMUIUU8w40wwAAMYIAEcAcIIIMAEMMkAAMGEkgMAM4OoMss0AAkMc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"

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


GITHUB_PROJECT_URL = "https://github.com/LaianeNR/MBA_Eng_Dados_TurmaG_Energia_Solar/tree/desenv"


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
qr_html = (
    f'<div class="brand-qr">'
    f'<img src="{qr_uri}" alt="QR Code do projeto">'
    f'<div class="brand-qr-label">Acesse<br>o projeto</div>'
    f'</div>'
    if qr_uri else ""
)

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
        {qr_html}
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
tabs = st.tabs([
    "01  Contexto",
    "02  Metodologia",
    "03  Jornada completa",
    "04  Histórico",
    "05  Sinais",
    "06  Modelo",
    "07  Previsão",
])

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
with tabs[0]:
    st.markdown(
        """
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
# TAB 4 — HISTÓRICO
# ------------------------------------------------------------
with tabs[3]:
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
        st.caption("A cor acompanha o nível oficial observado em cada mês.")
        flag_history_chart(df_bandeiras)

        dist = hist.copy()
        dist["Bandeira"] = dist["NivelBandeira"].map(flag_name)
        dist = dist["Bandeira"].value_counts().rename_axis("Bandeira").reset_index(name="Meses")
        dist["Percentual"] = dist["Meses"] / dist["Meses"].sum() * 100
        st.markdown("### Frequência das bandeiras")
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
# TAB 5 — SINAIS
# ------------------------------------------------------------
with tabs[4]:
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
          <div class="impact-card"><div class="impact-icon">◆</div><h4>HIDROLOGIA</h4><p>EAR do subsistema Sudeste, usado como sinal do estado dos reservatórios.</p></div>
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
# TAB 6 — MODELO
# ------------------------------------------------------------
with tabs[5]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como os sinais viram uma probabilidade?</h2>
            <p>O modelo final transforma o histórico em uma classificação binária de risco.</p>
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
            <div class="eyebrow">Modelo final</div>
            <div class="question-text" style="font-size:16px;">
              Regressão logística + balanceamento de classes + ponderação de observações recentes
              → probabilidade de bandeira vermelha
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### O protocolo de modelagem")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>ALVO</h4><p>Bandeira vermelha = 1 (patamar 1 ou 2); demais situações = 0.</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>SEM VAZAMENTO</h4><p>Cada previsão usa apenas informações disponíveis até o mês de origem.</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>BACKTEST</h4><p>Janela expansiva, com mínimo de 36 meses de treino e avaliação temporal.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid" style="margin-top:14px;">
          <div class="impact-card"><div class="impact-icon">04</div><h4>NORMAL DE CHUVA</h4><p>A normal climatológica usada no percentual de chuva é construída de forma expansiva, somente com o passado.</p></div>
          <div class="impact-card"><div class="impact-icon">05</div><h4>CLASSES</h4><p>A classe vermelha é menos frequente; a ponderação evita que o modelo simplesmente favoreça a classe majoritária.</p></div>
          <div class="impact-card"><div class="impact-icon">06</div><h4>RECÊNCIA</h4><p>Observações a partir de abril de 2024 recebem peso 5x, aproximando o treino do regime mais recente.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_training.empty:
        st.markdown("### Base de treinamento")
        st.caption(f"{len(df_training):,} observações disponíveis na camada refined para apoio à leitura da modelagem.")
        st.dataframe(df_training.tail(12), use_container_width=True, hide_index=True)

    st.markdown("### O que a avaliação precisa responder")
    st.info("A qualidade do modelo deve ser lida principalmente pela capacidade de identificar meses vermelhos sem usar o futuro. Acurácia, precisão, recall e F1 ajudam a interpretar esse equilíbrio.")

# ------------------------------------------------------------
# TAB 7 — PREVISÃO
# ------------------------------------------------------------
with tabs[6]:
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
# TAB 2 — METODOLOGIA
# ------------------------------------------------------------
with tabs[1]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Metodologia e limites da leitura</h2>
            <p>O resultado é uma estimativa probabilística, não uma certeza sobre o futuro.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>DADOS</h4><p>ANEEL, ONS e INMET fornecem as séries utilizadas para construir o painel mensal.</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>MODELAGEM</h4><p>Regressão logística, engenharia de atributos e validação temporal preservam a ordem cronológica.</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>SAÍDA</h4><p>Probabilidade estimada de bandeira vermelha para M+1, M+2 e M+3.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Limitações")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">•</div><h4>BASE HISTÓRICA</h4><p>A série mensal é relativamente pequena e a classe vermelha é menos frequente, o que limita a complexidade do modelo.</p></div>
          <div class="impact-card"><div class="impact-icon">•</div><h4>CLIMA FUTURO</h4><p>A previsão utiliza dados observados do mês de referência; não incorpora uma previsão meteorológica futura.</p></div>
          <div class="impact-card"><div class="impact-icon">•</div><h4>PROBABILIDADE</h4><p>Uma probabilidade elevada representa sinal de risco, não garantia de ocorrência da bandeira.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(closing_section(), unsafe_allow_html=True)



def render_f1_notebook05_chart():
    """Gráfico de barras agrupadas: F1-macro de CV do notebook 05 (persistência vs. modelos aprendidos)."""
    horizons = ["t+1", "t+2", "t+3"]
    candidates = [
        ("Persistência", [0.7907, 0.7076, 0.6117], "#16a0ff"),
        ("v5: receita fixa por horizonte", [0.7818, 0.5359, 0.4881], "#3d5b74"),
        ("Regressão ordinal + CMO", [0.5758, 0.5178, 0.4151], "#3d5b74"),
        ("Controle nominal + CMO", [0.5605, 0.4854, 0.4151], "#3d5b74"),
        ("v3: clima + bandeira + ONI", [0.4611, 0.4567, 0.4063], "#3d5b74"),
        ("Classe majoritária", [0.2500, 0.2500, 0.2557], "#26384a"),
    ]
    fig = go.Figure()
    for name, values, color in candidates:
        fig.add_trace(go.Bar(
            name=name,
            x=horizons,
            y=values,
            marker_color=color,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(size=10, color="#8fa8bb"),
        ))
    fig.update_layout(
        barmode="group",
        height=370,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#07131f",
        plot_bgcolor="#07131f",
        font=dict(color="#cfe0eb", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9.5)),
        xaxis=dict(showgrid=False, color="#8fa8bb"),
        yaxis=dict(title="F1-macro (CV)", gridcolor="#183042", zeroline=False, color="#8fa8bb", range=[0, 0.95]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------
# TAB 3 — JORNADA DO PROJETO (making-of / storytelling)
# ------------------------------------------------------------
with tabs[2]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como chegamos até aqui</h2>
            <p>Bastidores do projeto: notebooks testados, decisões técnicas e caminhos abandonados.</p>
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Por que contar essa história</div>
            <div class="question-text" style="font-size:14px;">
              Um projeto de dados não nasce pronto do jeito que aparece nas abas anteriores. O grupo
              (Alberto, Fabio, Laiane, Sweeli e Tatiane) começou em outro tema, com outras ferramentas,
              e mudou de rumo mais de uma vez. Esta seção documenta o caminho completo — da ideia
              original de energia solar até a versão validada do modelo — o que funcionou, o que foi
              abandonado e por quê, para dar transparência ao número final apresentado.
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
            <h2>O diagrama original da arquitetura</h2>
            <p>A arquitetura "Previsão de Bandeira Vermelha" desenhada pelo grupo, ponto de partida de toda a implementação.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    architecture_bytes = base64.b64decode(ARCHITECTURE_IMAGE_B64)
    st.image(architecture_bytes, use_container_width=True)
    st.caption("Arquitetura da solução de previsão de bandeira vermelha.")

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como os dados chegam ao produto</h2>
            <p>Do dado bruto às tabelas utilizadas na modelagem e no consumo analítico.</p>
          </div>
        </div>

        <style>
          .pipeline-wrap { display:flex; align-items:stretch; gap:0; margin:18px 0 6px 0; flex-wrap:wrap; }
          .pipeline-col { flex:1 1 190px; min-width:170px; background:#0d1e2e; border:1px solid #1c3348;
                          border-radius:10px; padding:14px 12px; display:flex; flex-direction:column; gap:8px; }
          .pipeline-col-title { font-size:10.5px; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
                                 color:#8fa8bb; margin-bottom:2px; }
          .pipeline-box { background:#132638; border-radius:6px; padding:8px 10px; font-size:12px; line-height:1.4;
                           color:#cfe0eb; border-left:3px solid var(--blue); }
          .pipeline-box b { color:#fff; }
          .pipeline-box.stage { border-left-color:var(--orange); }
          .pipeline-box.raw { border-left-color:#8fa8bb; }
          .pipeline-box.trusted { border-left-color:#ffcf4d; }
          .pipeline-box.refined { border-left-color:var(--blue-2); }
          .pipeline-arrow { display:flex; align-items:center; justify-content:center; color:#4c6a80;
                             font-size:22px; flex:0 0 26px; }
          @media (max-width:900px) {
            .pipeline-wrap { flex-direction:column; }
            .pipeline-arrow { transform:rotate(90deg); padding:2px 0; }
          }
        </style>

        <div class="pipeline-wrap">
          <div class="pipeline-col">
            <div class="pipeline-col-title">Fontes de dados</div>
            <div class="pipeline-box"><b>APIs meteorológicas</b><br>INMET / CPTEC</div>
            <div class="pipeline-box"><b>Dados operacionais</b><br>ONS — ENA, EAR, CMO, carga</div>
            <div class="pipeline-box"><b>Histórico regulatório</b><br>ANEEL — bandeiras tarifárias</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-col">
            <div class="pipeline-col-title">Orquestração &amp; ingestão</div>
            <div class="pipeline-box"><b>Scripts de ingestão</b><br><code>scripts/01_ingestao_stage</code></div>
            <div class="pipeline-box"><b>Versionamento</b><br>Databricks Repos ↔ GitHub, branch <code>desenv</code></div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-col">
            <div class="pipeline-col-title">Data lake (Databricks)</div>
            <div class="pipeline-box stage"><b>Stage</b> — dados brutos ingeridos, limpeza básica</div>
            <div class="pipeline-box raw"><b>Raw</b> — tipagem e padronização</div>
            <div class="pipeline-box trusted"><b>Trusted</b> — dados validados e estruturados</div>
            <div class="pipeline-box refined"><b>Refined</b> — tabelas prontas para consumo</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-col">
            <div class="pipeline-col-title">Consumo de dados</div>
            <div class="pipeline-box"><b>Modelagem</b><br><code>scripts/05_modelagem</code>, notebooks 01–07</div>
            <div class="pipeline-box"><b>Streamlit</b><br>este painel e o portal operacional</div>
            <div class="pipeline-box"><b>Relatórios</b><br>previsão executiva por horizonte</div>
          </div>
        </div>
        <p style="font-size:11px; color:#63788a; margin-top:8px;">
          Versão adaptada à estrutura efetivamente implementada no repositório
          <a href="https://github.com/FabioFumioWada/MBA_Eng_Dados_TurmaG_Energia_Solar/tree/desenv" target="_blank">MBA_Eng_Dados_TurmaG_Energia_Solar</a>
          (branch <code>desenv</code>).
        </p>

        <div class="impact-grid" style="margin-top:14px;">
          <div class="impact-card">
            <div class="impact-icon">◇</div>
            <h4>INMET &amp; CPTEC</h4>
            <p>APIs públicas do Instituto Nacional de Meteorologia e do CPTEC/INPE — chuva, temperatura,
            umidade e o índice El Niño (ONI) usados como sinal climático mensal.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">◇</div>
            <h4>ONS</h4>
            <p>API/portal de dados abertos do Operador Nacional do Sistema Elétrico — ENA, EAR, CMO e
            carga, atualizados mensalmente por subsistema.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">◇</div>
            <h4>ANEEL</h4>
            <p>Histórico oficial das bandeiras tarifárias — a variável que o modelo tenta prever, também
            consumida via dados públicos do regulador.</p>
          </div>
        </div>
        <p style="font-size:11px; color:#63788a; margin-top:10px;">
          As três são APIs/portais públicos e gratuitos — nenhuma credencial paga ou acesso restrito é
          necessário, o que manteve o pipeline reprodutível para qualquer integrante do grupo rodar do zero.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Quatro fases até o modelo atual</h2>
            <p>O projeto trocou de tema, de escopo e de abordagem de modelagem antes de chegar à versão em produção.</p>
          </div>
        </div>

        <div class="impact-grid">
          <div class="impact-card abandoned">
            <div class="impact-icon">F1</div>
            <h4>FASE 1 · ESCOPO ORIGINAL</h4>
            <p>"Energia em Dados: Geração Solar, Tarifas e Reservatórios no Brasil". Geração distribuída
            solar, tarifas TE/TUSD, ENA por bacia e clusterização de UF, com dashboard em Power BI.
            Registrado no repositório <code>MACK_MBA_Eng_Dados_TurmaG_Energia_Solar</code>.</p>
          </div>
          <div class="impact-card abandoned">
            <div class="impact-icon">F2</div>
            <h4>FASE 2 · SPRINT 2</h4>
            <p>Escopo reduzido para o Brasil como um todo, com a solar ainda central: a janela de 30 a
            120 dias para instalar um sistema solar vira a métrica de decisão do consumidor. Camadas
            Raw/Trusted/Refined chegaram a ser criadas no Databricks, mas o push para o GitHub nunca
            foi concluído.</p>
          </div>
          <div class="impact-card abandoned">
            <div class="impact-icon">01-06</div>
            <h4>MODELAGEM MULTICLASSE</h4>
            <p>Seis notebooks depois, com clima e hidrologia (ENA/EAR/CMO/El Niño) no lugar da solar, o
            resultado foi honesto e desconfortável: o modelo de persistência (repetir a última bandeira)
            venceu todos os modelos treinados, em todos os horizontes, em validação cruzada. Conduzida
            em <code>scripts/05_modelagem</code> — a mesma pasta, no repositório sem o prefixo
            MACK, onde depois nasceria o notebook 07.</p>
          </div>
          <div class="impact-card done">
            <div class="impact-icon">07</div>
            <h4>MODELO EM PRODUÇÃO</h4>
            <p>Virada de chave: classificação binária (vermelha ou não) com régua de marcos regulatórios
            e peso 5x para dados recentes. É a única tentativa, entre todas, que superou a persistência
            — e é o que alimenta esta aplicação.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Linha do tempo do desenvolvimento</h2>
            <p>Da ideia original de energia solar ao modelo binário em produção.</p>
          </div>
        </div>

        <div class="timeline">

          <div class="tl-item">
            <div class="tl-dot abandoned"></div>
            <div class="tl-tag abandoned">Fase 1 · 15/08/2026</div>
            <div class="tl-title">Projeto começa com foco em energia solar</div>
            <div class="tl-text">
              Escopo original: relação entre a expansão da geração distribuída solar, as tarifas TE/TUSD
              das distribuidoras e o nível dos reservatórios (ENA), com clusterização de estados por
              perfil de penetração solar e um dashboard em Power BI / Looker Studio. Estrutura registrada
              no repositório
              <a href="https://github.com/FabioFumioWada/MACK_MBA_Eng_Dados_TurmaG_Energia_Solar" target="_blank">MACK_MBA_Eng_Dados_TurmaG_Energia_Solar</a>.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot warn"></div>
            <div class="tl-tag warn">Fase 2 · Sprint 2</div>
            <div class="tl-title">Escopo nacional e a janela de decisão do consumidor</div>
            <div class="tl-text">
              A pergunta de negócio foi reformulada: além de prever a bandeira, mostrar o quanto a
              energia solar mitigaria o impacto — aproveitando que instalar um sistema residencial leva
              de 30 a 120 dias, quase um trimestre de antecedência. Camadas Raw/Trusted/Refined chegaram
              a ser criadas no workspace do Databricks para o mesmo repositório, mas o commit e push para
              o GitHub nunca foram concluídos por integração externa indisponível na sessão.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot abandoned"></div>
            <div class="tl-tag abandoned">Notebooks 01–06</div>
            <div class="tl-title">Resultado honesto: a persistência venceu</div>
            <div class="tl-text">
              A energia solar saiu do modelo. O grupo conduziu seis notebooks testando classificação
              multiclasse (Verde / Amarela / Vermelha) com clima, ENA, EAR, CMO e índice El Niño (ONI),
              dentro de <code>scripts/05_modelagem</code> no repositório
              <a href="https://github.com/FabioFumioWada/MBA_Eng_Dados_TurmaG_Energia_Solar" target="_blank">MBA_Eng_Dados_TurmaG_Energia_Solar</a>
              (sem o prefixo MACK) — a mesma pasta onde, mais tarde, nasceria o notebook 07. Com validação
              temporal rigorosa (TimeSeriesSplit com gap, métricas OOF, F1-macro), em todos os horizontes
              o modelo de persistência — simplesmente repetir a última bandeira — superou todos os
              modelos treinados. O resultado negativo foi documentado e mantido, sem maquiagem, para a
              banca.
            </div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chart-card" style="margin:4px 0 18px 30px;">
          <div class="chart-title">Notebook 05 — a persistência venceu em todos os horizontes</div>
          <div class="chart-sub">F1-macro de validação cruzada (CV), previsões OOF, calendário comum de 45 meses
          por horizonte · Fonte: Relatório Consolidado das Modelagens, notebook 05
          (resultado_databricks.json, campo ResumoCV), corte em 08/09/2026.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_f1_notebook05_chart()

    st.markdown(
        """
        <div class="timeline" style="margin-top:0;">

          <div class="tl-item">
            <div class="tl-dot done"></div>
            <div class="tl-tag done">Virada</div>
            <div class="tl-title">Simplificar para binário e usar conhecimento de domínio</div>
            <div class="tl-text">
              Diante do resultado anterior, o grupo trocou o problema de multiclasse para binário
              (vermelha ou não) e passou a incorporar conhecimento de domínio: uma régua de marcos
              regulatórios e peso 5x para observações recentes. Essa combinação foi a única, entre todas
              as tentativas, a superar a persistência — e deu origem ao notebook 07.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot"></div>
            <div class="tl-tag">Ponto de partida</div>
            <div class="tl-title">Arquitetura em camadas já definida</div>
            <div class="tl-text">
              Stage → Raw → Trusted → Refined → Modelagem, com três variantes de notebook de
              regressão logística (07 principal, 07_P e 07_PP) alimentando aplicações diferentes do grupo.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot"></div>
            <div class="tl-tag">Diagnóstico</div>
            <div class="tl-title">"Por que o notebook 07 não avança para os meses mais recentes?"</div>
            <div class="tl-text">
              Investigação mostrou que não era um bug: o backtest só avalia meses cuja bandeira
              oficial já existe em <code>mba.trusted.f_bandeira</code>. É uma proteção contra
              vazamento de dados (anti-leakage), não uma falha de código.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot done"></div>
            <div class="tl-tag done">Correção de dados</div>
            <div class="tl-title">Defasagem de publicação da ANEEL</div>
            <div class="tl-text">
              A ANEEL antecipa a divulgação da bandeira do mês seguinte — setembro/2026 já havia
              sido publicada em 28/08 como amarela. A base foi reprocessada para ligar o clima de
              agosto à competência de setembro, deslocando a fronteira real de previsão para
              outubro em diante.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot warn"></div>
            <div class="tl-tag warn">Governança de dados</div>
            <div class="tl-title">Padronização do fluxo de versionamento entre integrantes</div>
            <div class="tl-text">
              O grupo formalizou boas práticas de versionamento — <code>git status</code> /
              <code>git add</code> explícitos antes de cada commit, com <code>pull</code> cuidadoso
              nos clones de cada integrante — para garantir consistência do histórico do repositório
              ao longo da colaboração.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot abandoned"></div>
            <div class="tl-tag abandoned">Testado e abandonado</div>
            <div class="tl-title">Dashboard antigo só de EDA</div>
            <div class="tl-text">
              O <code>dashboards/streamlit_app.py</code> criado na Fase 2 — dentro do repositório
              MACK_MBA — lia três parquets (<code>solar_mensal_nacional</code>, <code>ena_mensal</code>,
              <code>bandeira_mensal</code>) que eram apenas ponteiros de Git LFS dentro do Databricks
              Repos — as abas de Solar, ENA e Bandeiras ficavam em branco, e um <code>try/except</code>
              escondia o erro em vez de expô-lo. Não havia aba de previsão nem de acurácia, insuficiente
              para contar a história completa do projeto.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot done"></div>
            <div class="tl-tag done">Estado atual</div>
            <div class="tl-title">Notebook 07 principal validado e em produção</div>
            <div class="tl-text">
              Régua de 4 marcos regulatórios, peso 5x para observações recentes, backtest expansivo
              com intervalo de confiança de Wilson. Esta aplicação consulta os mesmos dados reais do
              Databricks usados pelo notebook para gerar o painel executivo e a previsão.
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
            <h2>Onde você vai ver o modelo funcionando</h2>
            <p>O mesmo notebook de regressão logística alimenta três aplicações — duas delas serão demonstradas ao vivo na apresentação.</p>
          </div>
        </div>

        <div class="impact-grid">
          <div class="impact-card done">
            <div class="impact-icon">07</div>
            <h4>PRINCIPAL · ESTE PAINEL</h4>
            <p>Versão de referência com a régua regulatória completa. É a que alimenta esta aplicação e a aba de Previsão mostrada nesta jornada.</p>
          </div>
          <div class="impact-card done">
            <div class="impact-icon">P</div>
            <h4>07_P · PORTAL OPERACIONAL</h4>
            <p>Envia ano/mês ao notebook no Databricks e devolve a previsão em JSON para os três horizontes (t+1, t+2, t+3).</p>
            <p style="margin-top:6px;"><a href="https://previsao-bandeira.streamlit.app/" target="_blank">Abrir portal ↗</a></p>
          </div>
          <div class="impact-card done">
            <div class="impact-icon">PP</div>
            <h4>07_PP · PORTAL DE PESQUISA</h4>
            <p>Recebe cinco parâmetros meteorológicos digitados pelo usuário e retorna a previsão parametrizada para o horizonte t+1.</p>
            <p style="margin-top:6px;"><a href="https://previsao-bandeira-pesquisa.streamlit.app/" target="_blank">Abrir portal ↗</a></p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Ferramentas e pilha técnica</h2>
            <p>O que sustenta o projeto do dado bruto até esta tela.</p>
          </div>
        </div>
        <div>
          <span class="tool-pill">● Databricks · Workspace &amp; Notebooks</span>
          <span class="tool-pill">● Databricks · Repos / Git Folder</span>
          <span class="tool-pill">● Unity Catalog</span>
          <span class="tool-pill">● SQL Warehouse</span>
          <span class="tool-pill">◆ GitHub · branch desenv</span>
          <span class="tool-pill">◆ Git LFS</span>
          <span class="tool-pill">● Python</span>
          <span class="tool-pill">● PySpark</span>
          <span class="tool-pill">● pandas / NumPy</span>
          <span class="tool-pill">● scikit-learn · Regressão logística</span>
          <span class="tool-pill">● Plotly</span>
          <span class="tool-pill">● Streamlit · dois apps distintos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Ferramentas testadas e substituídas</h2>
            <p>Pilha planejada nas fases 1 e 2, antes da migração para o fluxo atual em Databricks.</p>
          </div>
        </div>
        <div>
          <span class="tool-pill">◆ Power BI (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ Google Looker Studio (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ K-Means · clusterização de UF (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ DuckDB (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ Prefect / cron (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ Trello / Jira (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ ANEEL · Geração Distribuída Solar (Fase 1-2 · descontinuado)</span>
          <span class="tool-pill">◆ Índice ONI / El Niño (Notebooks 01–06 · descontinuado)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="closing" style="margin-top:26px;">
          <div class="quote">"</div>
          <div class="closing-text">
            O objetivo desta aba não é esconder as tentativas que não deram certo — da energia solar
            abandonada em MACK_MBA à persistência que venceu seis notebooks de modelagem.
            É mostrar o raciocínio — e as correções de rota — que sustentam o número final.
            <small>SOLAR · MULTICLASSE · VIRADA · VALIDAÇÃO</small>
          </div>
          <div class="closing-side">
            Solar<br>
            Multiclasse<br>
            Virada<br>
            Validação
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

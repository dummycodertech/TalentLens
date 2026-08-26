"""
app.py — AI-Powered Candidate Screening Platform
Streamlit single-app entry point. Five tabs, one pipeline state machine.

Pipeline:
  Upload candidates (CSV) → Provide JD → Parse resumes → AI-evaluate → GitHub analyze
  → Score & rank → Email test link → Upload test results → Re-rank → Schedule interviews

All secrets read from st.secrets (Streamlit Cloud) or .streamlit/secrets.toml (local).
"""

import json
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Candidate Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Module imports ────────────────────────────────────────────────────────────

import db
from modules import resume_parser, github_analyzer, ai_evaluator, scorer, emailer, calendar_scheduler
from modules import archetype_clustering


@st.cache_resource
def _get_clustering_models():
    return archetype_clustering.load_or_fit_clustering()

_archetype_scaler, _archetype_kmeans, _archetype_label_map = _get_clustering_models()

# ─── Init ─────────────────────────────────────────────────────────────────────

db.init_db()

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600;700&display=swap');

  /* ─── Base Reset ─── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, sans-serif;
    color: #e8e5de;
  }
  code, pre, .mono { font-family: 'IBM Plex Mono', monospace; }

  /* ─── Dark canvas ─── */
  .stApp, .main { background: #0c0c14 !important; }
  .main .block-container { padding-top: 1rem; max-width: 1480px; }

  /* ─── Custom scrollbar ─── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0c0c14; }
  ::-webkit-scrollbar-thumb { background: #2a2a3d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #3a3a55; }

  /* ─── Sidebar ─── */
  section[data-testid="stSidebar"] {
    background: #0a0a12 !important;
    border-right: 1px solid #1a1a28;
  }
  section[data-testid="stSidebar"] * {
    color: #8a8a9a !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #e8e5de !important;
    font-family: 'DM Serif Display', serif !important;
    letter-spacing: -0.02em;
  }
  section[data-testid="stSidebar"] .stSlider label {
    font-size: 13px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #6a6a7a !important;
  }
  section[data-testid="stSidebar"] hr {
    border-color: #1a1a28 !important;
  }
  /* API dot colors — must be more specific than the wildcard * rule above */
  section[data-testid="stSidebar"] .api-dot-green { color: #2dd4a8 !important; font-size: 9px !important; }
  section[data-testid="stSidebar"] .api-dot-red   { color: #f87171 !important; font-size: 9px !important; }

  /* ─── Tab bar ─── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #12121c;
    border-radius: 0;
    padding: 0;
    border-bottom: 1px solid #1a1a28;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 14px 24px;
    font-weight: 600;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    color: #5a5a6a;
    border: none;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    transition: color 0.2s, border-color 0.2s;
    border-bottom: 2px solid transparent;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #a0a0b0;
  }
  .stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #2dd4a8 !important;
    border-bottom: 2px solid #2dd4a8 !important;
  }

  /* ─── Headers ─── */
  h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8e5de !important;
    letter-spacing: -0.02em;
  }
  h3 { font-size: 22px !important; }

  /* ─── Text ─── */
  p, span, label, .stCaption, .stMarkdown {
    color: #b0b0be;
  }
  .stCaption p, small {
    color: #5a5a6a !important;
    font-size: 12px !important;
  }

  /* ─── Badges ─── */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-green  { background: rgba(45,212,168,0.12); color: #2dd4a8; }
  .badge-blue   { background: rgba(96,165,250,0.12); color: #60a5fa; }
  .badge-yellow { background: rgba(251,191,36,0.12); color: #fbbf24; }
  .badge-red    { background: rgba(248,113,113,0.12); color: #f87171; }
  .badge-gray   { background: rgba(160,160,176,0.08); color: #a0a0b0; }
  .badge-purple { background: rgba(168,130,255,0.12); color: #a882ff; }

  /* ─── Hero Banner ─── */
  .hero-banner {
    background: #12121c;
    border: 1px solid #1a1a28;
    border-radius: 2px;
    padding: 40px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, #2dd4a8 0%, transparent 60%);
  }
  .hero-banner h1 {
    margin: 0;
    font-size: 28px;
    font-family: 'DM Serif Display', serif;
    color: #e8e5de;
    letter-spacing: -0.02em;
  }
  .hero-banner p {
    margin: 10px 0 0;
    color: #6a6a7a;
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    max-width: 600px;
    line-height: 1.6;
  }

  /* ─── Score Card (dark) ─── */
  .score-card {
    background: #12121c;
    border: 1px solid #1a1a28;
    border-radius: 2px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .score-card:hover { border-color: #2a2a3d; }
  .score-card .value {
    font-size: 32px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #e8e5de;
    letter-spacing: -0.02em;
  }
  .score-card .label {
    font-size: 11px;
    color: #5a5a6a;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
  }

  /* ─── Stat Counter (JS animated) ─── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1px;
    background: #1a1a28;
    border: 1px solid #1a1a28;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 24px;
  }
  .stat-cell {
    background: #12121c;
    padding: 24px 20px;
    text-align: center;
  }
  .stat-val {
    font-size: 36px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #e8e5de;
    letter-spacing: -0.03em;
    line-height: 1;
  }
  .stat-val.accent { color: #2dd4a8; }
  .stat-val.warn { color: #fbbf24; }
  .stat-val.danger { color: #f87171; }
  .stat-label {
    font-size: 11px;
    color: #5a5a6a;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 8px;
  }

  /* ─── Candidate Card (Rankings) ─── */
  .cand-card {
    background: #12121c;
    border: 1px solid #1a1a28;
    border-radius: 2px;
    margin-bottom: 2px;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .cand-card:hover { border-color: #2a2a3d; }
  .cand-header {
    display: flex;
    align-items: center;
    padding: 18px 24px;
    gap: 16px;
    cursor: default;
  }
  .cand-rank {
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    color: #5a5a6a;
    min-width: 32px;
  }
  .cand-rank.top { color: #2dd4a8; }
  .cand-name {
    font-size: 16px;
    font-weight: 600;
    font-family: 'DM Serif Display', serif;
    color: #e8e5de;
    flex: 1;
    letter-spacing: -0.01em;
  }
  .cand-score-pill {
    font-size: 14px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #e8e5de;
    background: #1a1a28;
    padding: 4px 14px;
    border-radius: 2px;
  }
  .cand-sno {
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #3a3a4a;
  }

  /* ─── Score Bar ─── */
  .sbar-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
  }
  .sbar-label {
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #5a5a6a;
    min-width: 110px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: right;
  }
  .sbar-track {
    flex: 1;
    height: 4px;
    background: #1a1a28;
    border-radius: 0;
    overflow: hidden;
  }
  .sbar-fill {
    height: 100%;
    border-radius: 0;
    transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .sbar-val {
    font-size: 13px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #e8e5de;
    min-width: 40px;
    text-align: right;
  }

  /* ─── Section Divider ─── */
  .sec-divider {
    height: 1px;
    background: #1a1a28;
    margin: 28px 0;
  }
  .sec-title {
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #3a3a4a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
  }

  /* ─── Archetype Tags (dark) ─── */
  .archetype-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .archetype-tag-research-deep { background: rgba(251,191,36,0.12); color: #fbbf24; }
  .archetype-tag-builder        { background: rgba(45,212,168,0.12); color: #2dd4a8; }
  .archetype-tag-fast-learner   { background: rgba(96,165,250,0.12); color: #60a5fa; }
  .archetype-tag-all-rounder    { background: rgba(168,130,255,0.12); color: #a882ff; }

  .archetype-disclosure {
    font-size: 12px;
    color: #3a3a4a;
    font-family: 'DM Sans', sans-serif;
    font-style: italic;
    line-height: 1.6;
    margin: 4px 0 20px 0;
    max-width: 800px;
  }

  .archetype-breakdown {
    margin-top: 16px;
    padding: 16px 20px;
    background: #0a0a12;
    border: 1px solid #1a1a28;
    border-radius: 2px;
  }
  .archetype-breakdown-title {
    font-size: 11px;
    font-weight: 600;
    color: #5a5a6a;
    margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .archetype-dist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .archetype-dist-name {
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #5a5a6a;
    min-width: 120px;
    text-align: right;
  }
  .archetype-dist-bar-bg {
    flex: 1;
    height: 3px;
    background: #1a1a28;
    border-radius: 0;
    overflow: hidden;
  }
  .archetype-dist-bar-fill {
    height: 100%;
    border-radius: 0;
    transition: width 0.5s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .archetype-dist-bar-fill-research-deep { background: #fbbf24; }
  .archetype-dist-bar-fill-builder        { background: #2dd4a8; }
  .archetype-dist-bar-fill-fast-learner   { background: #60a5fa; }
  .archetype-dist-bar-fill-all-rounder    { background: #a882ff; }
  .archetype-distance-val {
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    color: #5a5a6a;
    min-width: 40px;
    text-align: right;
  }

  /* ─── Interview Card ─── */
  .interview-card {
    background: #12121c;
    border: 1px solid #1a1a28;
    border-radius: 2px;
    padding: 18px 20px;
    margin-bottom: 4px;
    transition: border-color 0.15s;
  }
  .interview-card:hover { border-color: #2a2a3d; }
  .interview-card .ic-name {
    font-weight: 600;
    font-size: 15px;
    font-family: 'DM Serif Display', serif;
    color: #e8e5de;
  }
  .interview-card .ic-sno {
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #3a3a4a;
    margin-left: 8px;
  }
  .interview-card .ic-time {
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    color: #5a5a6a;
    margin-top: 6px;
  }
  .interview-card a {
    color: #2dd4a8;
    font-size: 13px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    text-decoration: none;
  }
  .interview-card a:hover { text-decoration: underline; }

  /* ─── Buttons ─── */
  .stButton > button {
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
  }

  /* Primary (green) — ALL known Streamlit data-testid variants */
  [data-testid="baseButton-primary"],
  [data-testid="stBaseButton-primary"],
  .stButton > button[kind="primary"] {
    background: #2dd4a8 !important;
    border: 2px solid #2dd4a8 !important;
    font-weight: 800 !important;
  }
  /* Target the text node inside primary buttons — covers p, span, div children */
  [data-testid="baseButton-primary"] *,
  [data-testid="stBaseButton-primary"] *,
  .stButton > button[kind="primary"],
  .stButton > button[kind="primary"] * {
    color: #000000 !important;
    text-shadow: none !important;
  }
  [data-testid="baseButton-primary"]:hover,
  [data-testid="stBaseButton-primary"]:hover,
  .stButton > button[kind="primary"]:hover {
    background: #1fbf96 !important;
    border-color: #1fbf96 !important;
    transform: translateY(-1px);
  }

  /* Secondary (dark) */
  [data-testid="baseButton-secondary"],
  [data-testid="stBaseButton-secondary"],
  .stButton > button[kind="secondary"] {
    background: #1a1a2e !important;
    border: 1px solid #2a2a3d !important;
  }
  [data-testid="baseButton-secondary"] *,
  [data-testid="stBaseButton-secondary"] *,
  .stButton > button[kind="secondary"],
  .stButton > button[kind="secondary"] * {
    color: #d0d0e0 !important;
  }
  [data-testid="baseButton-secondary"]:hover,
  [data-testid="stBaseButton-secondary"]:hover,
  .stButton > button[kind="secondary"]:hover {
    background: #22223a !important;
    border-color: #3a3a55 !important;
    transform: translateY(-1px);
  }
  [data-testid="baseButton-secondary"]:hover *,
  [data-testid="stBaseButton-secondary"]:hover *,
  .stButton > button[kind="secondary"]:hover * {
    color: #ffffff !important;
  }

  .stButton > button:hover {
    transform: translateY(-1px);
  }

  /* ─── Expanders ─── */
  div[data-testid="stExpander"] {
    background: #12121c !important;
    border: 1px solid #1a1a28 !important;
    border-radius: 2px !important;
    margin-bottom: 2px !important;
  }
  div[data-testid="stExpander"]:hover {
    border-color: #2a2a3d !important;
  }
  div[data-testid="stExpander"] summary span {
    color: #e8e5de !important;
  }

  /* ─── Inputs ─── */
  .stTextInput input, .stTextArea textarea, .stSelectbox select,
  .stNumberInput input {
    background: #12121c !important;
    border: 1px solid #1a1a28 !important;
    border-radius: 2px !important;
    color: #e8e5de !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2dd4a8 !important;
    box-shadow: 0 0 0 1px rgba(45,212,168,0.2) !important;
  }

  /* ─── File uploader ─── */
  [data-testid="stFileUploader"] {
    background: #12121c;
    border: 1px dashed #2a2a3d;
    border-radius: 2px;
    padding: 20px;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #2dd4a8;
  }

  /* ─── Dataframe ─── */
  [data-testid="stDataFrame"] {
    border: 1px solid #1a1a28 !important;
    border-radius: 2px !important;
  }

  /* ─── Progress bar ─── */
  .stProgress > div > div {
    background: #1a1a28 !important;
  }
  .stProgress > div > div > div {
    background: #2dd4a8 !important;
  }

  /* ─── Metric cards ─── */
  [data-testid="stMetric"] {
    background: #12121c;
    border: 1px solid #1a1a28;
    border-radius: 2px;
    padding: 16px;
  }
  [data-testid="stMetricLabel"] {
    color: #5a5a6a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }
  [data-testid="stMetricValue"] {
    color: #e8e5de !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }

  /* ─── Dividers ─── */
  hr { border-color: #1a1a28 !important; }

  /* ─── Info/Warning/Error ─── */
  .stAlert { border-radius: 2px !important; }

  /* ─── Pipeline log ─── */
  .pipeline-step {
    background: #12121c;
    border-bottom: 1px solid #1a1a28;
    padding: 12px 16px;
    margin-bottom: 0;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    color: #6a6a7a;
  }

  /* ─── Hide default Streamlit chrome ─── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] {
    background: #0c0c14 !important;
    border-bottom: 1px solid #1a1a28;
  }

  /* ─── Hide keyboard_doc / toolbar buttons (all Streamlit versions) ─── */
  [data-testid="stToolbar"]               { display: none !important; visibility: hidden !important; }
  [data-testid="stDecoration"]            { display: none !important; }
  [data-testid="stToolbarActions"]        { display: none !important; }
  button[title="Keyboard shortcuts"]      { display: none !important; }
  button[aria-label="Keyboard shortcuts"] { display: none !important; }
  [data-testid="baseButton-headerNoPadding"] { display: none !important; }
  /* Catch-all: every button inside the header bar */
  header[data-testid="stHeader"] button  { display: none !important; }
  header[data-testid="stHeader"] a       { display: none !important; }

  /* Button contrast already handled above via data-testid selectors */
</style>
""", unsafe_allow_html=True)

# ─── JS: nuke the keyboard shortcut button from the DOM ───────────────────────
st.markdown("""
<script>
(function removeKeyboardBtn() {
  function nuke() {
    // Target by title, aria-label, and data-testid variants
    var selectors = [
      'button[title="Keyboard shortcuts"]',
      'button[aria-label="Keyboard shortcuts"]',
      '[data-testid="stToolbar"]',
      '[data-testid="stToolbarActions"]',
      '[data-testid="baseButton-headerNoPadding"]',
    ];
    selectors.forEach(function(sel) {
      document.querySelectorAll(sel).forEach(function(el) {
        el.style.display = 'none';
        el.style.visibility = 'hidden';
        if (el.parentNode && el.tagName !== 'HEADER') {
          try { el.parentNode.removeChild(el); } catch(e) {}
        }
      });
    });
    // Also hide any button in the header whose text/icon looks like keyboard_doc
    document.querySelectorAll('header button').forEach(function(btn) {
      if (!btn.closest('.stButton')) {
        btn.style.display = 'none';
      }
    });
  }
  // Run immediately and then poll for a few seconds (Streamlit injects lazily)
  nuke();
  var tries = 0;
  var interval = setInterval(function() {
    nuke();
    tries++;
    if (tries > 20) clearInterval(interval);
  }, 300);
})();
</script>
""", unsafe_allow_html=True)

# ─── Session state initialization ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "jd_text": "",
        "pipeline_running": False,
        "pipeline_log": [],
        "weights": dict(scorer.DEFAULT_WEIGHTS),
        "test_url": "",
        "shortlist_threshold": 60.0,
        "rerank_threshold": 65.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Secrets helper ────────────────────────────────────────────────────────────

def _secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return fallback

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AI Screener")
    st.markdown("<span style='font-size:12px;color:#5a5a6a;font-family:IBM Plex Mono,monospace'>Candidate Evaluation Platform</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Weights")
    st.caption("Auto-normalized to 1.0")

    w1 = st.slider("JD Match (w1)", 0.0, 1.0, st.session_state["weights"]["w1"], 0.05, key="sl_w1")
    w2 = st.slider("Project Quality (w2)", 0.0, 1.0, st.session_state["weights"]["w2"], 0.05, key="sl_w2")
    w3 = st.slider("GitHub Score (w3)", 0.0, 1.0, st.session_state["weights"]["w3"], 0.05, key="sl_w3")
    w4 = st.slider("Test Score (w4)", 0.0, 1.0, st.session_state["weights"]["w4"], 0.05, key="sl_w4")

    raw_sum = w1 + w2 + w3 + w4
    if raw_sum > 0:
        st.session_state["weights"] = {
            "w1": round(w1 / raw_sum, 4),
            "w2": round(w2 / raw_sum, 4),
            "w3": round(w3 / raw_sum, 4),
            "w4": round(w4 / raw_sum, 4),
        }
        st.caption(f"Sum: {sum(st.session_state['weights'].values()):.2f}")

    st.divider()
    st.markdown("### Integrations")
    groq_ok = bool(_secret("GROQ_API_KEY"))
    gh_ok = bool(_secret("GITHUB_PAT"))
    gmail_ok = bool(_secret("GMAIL_ADDRESS")) and bool(_secret("GMAIL_APP_PASSWORD"))
    cal_ok = bool(_secret("GOOGLE_OAUTH_CLIENT_JSON"))

    def _dot(ok):
        cls = "api-dot-green" if ok else "api-dot-red"
        return f'<span class="{cls}">&#x25CF;</span>'

    st.markdown(f"{_dot(groq_ok)} Groq LLM {'connected' if groq_ok else 'missing'}", unsafe_allow_html=True)
    st.markdown(f"{_dot(gh_ok)} GitHub PAT {'connected' if gh_ok else 'missing'}", unsafe_allow_html=True)
    st.markdown(f"{_dot(gmail_ok)} Gmail SMTP {'connected' if gmail_ok else 'missing'}", unsafe_allow_html=True)
    st.markdown(f"{_dot(cal_ok)} Calendar OAuth {'connected' if cal_ok else 'missing'}", unsafe_allow_html=True)

    st.divider()
    if st.button("Reset Database", type="secondary", use_container_width=True, key="btn_reset"):
        db.reset_db()
        st.session_state["pipeline_log"] = []
        st.success("Database reset.")
        st.rerun()

# ─── Status badge helper ───────────────────────────────────────────────────────

STATUS_BADGE = {
    "uploaded":             ("gray",   "📤 Uploaded"),
    "resume_parsed":        ("blue",   "📄 Resume Parsed"),
    "resume_failed":        ("red",    "❌ Resume Failed"),
    "resume_failed_scan":   ("yellow", "⚠️ Scan Warning"),
    "github_analyzed":      ("blue",   "🐙 GitHub Done"),
    "github_failed":        ("yellow", "⚠️ No GitHub"),
    "ai_scored":            ("purple", "🤖 AI Scored"),
    "ranked":               ("green",  "🏆 Ranked"),
    "test_sent":            ("blue",   "📨 Test Sent"),
    "test_scored":          ("blue",   "📊 Test Scored"),
    "shortlisted":          ("green",  "✅ Shortlisted"),
    "interview_scheduled":  ("purple", "📅 Scheduled"),
    "invited":              ("green",  "🎉 Invited"),
    "email_failed":         ("red",    "❌ Email Failed"),
    "scheduling_failed":    ("red",    "❌ Schedule Failed"),
}

def status_badge_html(status: str) -> str:
    color, label = STATUS_BADGE.get(status, ("gray", status))
    return f'<span class="badge badge-{color}">{label}</span>'


def score_bar(value, max_val=100, color="#2dd4a8", label=""):
    if value is None:
        return f'<div class="sbar-wrap"><span class="sbar-label">{label}</span><span class="sbar-val" style="color:#3a3a4a">--</span></div>'
    pct = min(100, (value / max_val) * 100)
    return f'''<div class="sbar-wrap">
      <span class="sbar-label">{label}</span>
      <div class="sbar-track">
        <div class="sbar-fill" style="width:{pct:.0f}%;background:{color}"></div>
      </div>
      <span class="sbar-val">{value:.1f}</span>
    </div>'''


# ─── Pipeline log helper ───────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["pipeline_log"].append(f"[{ts}] {msg}")


# ─── Archetype assignment helper ──────────────────────────────────────────────

def _assign_archetypes_to_all(status_callback=None):
    """
    Assign archetype to every candidate with status >= ai_scored.
    Uses the pre-fitted K-means model loaded at app start.
    Writes archetype, archetype_confidence, archetype_breakdown to scores table.
    """
    target_statuses = ["ai_scored", "ranked", "test_sent", "test_scored", "shortlisted"]
    candidates = []
    seen = set()
    for status in target_statuses:
        for row in db.get_by_status(status):
            if row["s_no"] not in seen:
                seen.add(row["s_no"])
                candidates.append(row)

    for row in candidates:
        sno = row["s_no"]
        if status_callback:
            status_callback(sno, "assigning archetype")

        score_row = db.get_score(sno)
        if not score_row:
            continue

        candidate_scores = {
            "jd_match": score_row["jd_match"],
            "project_quality": score_row["project_quality"],
            "github_score": score_row["github_score"],
            "cgpa": row["cgpa"],
        }

        result = archetype_clustering.assign_candidate_archetype(
            candidate_scores,
            _archetype_scaler,
            _archetype_kmeans,
            _archetype_label_map,
        )

        db.upsert_score(
            sno,
            archetype=result["archetype"],
            archetype_confidence=result["confidence"].get(result["archetype"], 0.0),
            archetype_breakdown=json.dumps({
                "confidence": result["confidence"],
                "distances_all": result["distances_all"],
            }),
        )

        if status_callback:
            status_callback(sno, f"archetype: {result['archetype']}")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload & Configure",
    "Run Pipeline",
    "Rankings",
    "Test Results",
    "Interviews",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Configure
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    # ── Hero ──
    st.markdown("""
    <div class="hero-banner">
      <h1>AI Candidate Screener</h1>
      <p>Upload the candidate CSV and job description to begin. The pipeline runs
         resume parsing, GitHub analysis, LLM evaluation, WSM scoring, and
         unsupervised archetype clustering in sequence.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_jd = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="sec-title">Candidate CSV</div>', unsafe_allow_html=True)
        st.caption("Required: `s_no, name, email, college, branch, cgpa, best_ai_project, research_work, github, resume`")
        uploaded_file = st.file_uploader(
            "Choose candidates CSV",
            type=["csv"],
            key="candidates_csv_upload",
            label_visibility="collapsed",
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(f'<div class="sec-title" style="margin-top:16px">{len(df)} rows detected</div>', unsafe_allow_html=True)
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)
                if st.button("Load Candidates", type="primary", use_container_width=True, key="btn_load_candidates"):
                    try:
                        inserted, skipped = db.insert_candidates(df)
                        if inserted > 0:
                            st.success(f"{inserted} candidates loaded ({skipped} duplicates skipped)")
                        else:
                            st.warning(f"All {skipped} rows already exist.")
                    except ValueError as e:
                        st.error(f"CSV format error: {e}")
                    except Exception as e:
                        st.error(f"Load failed: {e}")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    with col_jd:
        st.markdown('<div class="sec-title">Job Description</div>', unsafe_allow_html=True)
        st.caption("Full JD text — used for embedding similarity and LLM evaluation.")
        jd = st.text_area(
            "Job Description",
            value=st.session_state["jd_text"],
            height=300,
            placeholder="Paste the job description here…\n\nE.g.: We are looking for a GTM Engineering Intern with experience in LLMs, Python, and AI product development…",
            key="jd_input",
            label_visibility="collapsed",
        )
        st.session_state["jd_text"] = jd
        if jd:
            word_count = len(jd.split())
            color = "#2dd4a8" if word_count > 50 else "#fbbf24"
            st.markdown(f'<span style="font-size:12px;font-family:\'IBM Plex Mono\',monospace;color:{color}">{word_count} words — {"good length" if word_count > 50 else "too short for accurate matching"}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Database</div>', unsafe_allow_html=True)
    all_cands = db.get_all_candidates()
    if all_cands.empty:
        st.info("No candidates loaded yet. Upload a CSV above.")
    else:
        st.markdown(f'<span style="font-size:13px;font-family:\'IBM Plex Mono\',monospace;color:#5a5a6a">{len(all_cands)} candidates in database</span>', unsafe_allow_html=True)
        display_cols = ["s_no", "name", "college", "branch", "cgpa", "github_url", "status"]
        display_cols = [c for c in display_cols if c in all_cands.columns]
        st.dataframe(all_cands[display_cols], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Run Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    all_cands = db.get_all_candidates()
    total = len(all_cands)
    status_counts = all_cands["status"].value_counts().to_dict() if "status" in all_cands.columns else {}
    ranked_cnt = status_counts.get("ranked", 0) + sum(
        v for k, v in status_counts.items()
        if k in ("test_sent", "test_scored", "shortlisted", "interview_scheduled", "invited")
    )
    failed_cnt = sum(v for k, v in status_counts.items() if "failed" in k)
    avg_score_val = all_cands["final_score"].dropna().mean() if "final_score" in all_cands.columns and not all_cands.empty else None
    avg_str = f"{avg_score_val:.1f}" if avg_score_val and not pd.isna(avg_score_val) else "--"

    # Animated stat grid via JS counter
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-cell">
        <div class="stat-val" id="sc-total" data-target="{total}">0</div>
        <div class="stat-label">Candidates</div>
      </div>
      <div class="stat-cell">
        <div class="stat-val accent" id="sc-ranked" data-target="{ranked_cnt}">0</div>
        <div class="stat-label">Ranked</div>
      </div>
      <div class="stat-cell">
        <div class="stat-val {'danger' if failed_cnt > 0 else ''}" id="sc-fail" data-target="{failed_cnt}">0</div>
        <div class="stat-label">Errors</div>
      </div>
      <div class="stat-cell">
        <div class="stat-val" id="sc-avg">{avg_str}</div>
        <div class="stat-label">Avg Score</div>
      </div>
    </div>
    <script>
      (function(){{
        document.querySelectorAll('.stat-val[data-target]').forEach(function(el){{
          var target = parseInt(el.getAttribute('data-target'), 10);
          var cur = 0;
          var step = Math.ceil(target / 30) || 1;
          var timer = setInterval(function(){{
            cur = Math.min(cur + step, target);
            el.textContent = cur;
            if (cur >= target) clearInterval(timer);
          }}, 30);
        }});
      }})();
    </script>
    """, unsafe_allow_html=True)

    if total == 0:
        st.warning("No candidates loaded. Go to Upload & Configure first.")
    else:
        # Pipeline stage visualiser
        jd_ready = bool(st.session_state["jd_text"].strip())
        groq_ready = bool(_secret("GROQ_API_KEY"))
        can_run = jd_ready and groq_ready and total > 0

        stages = [
            ("Resume Parsing",   "Extracts text, detects scan-only PDFs"),
            ("GitHub Analysis",  "Stars, commits, original repos, recency"),
            ("LLM Evaluation",   "Groq LLM + MiniLM embedding similarity"),
            ("WSM Scoring",      "Weighted Sum Model, renormalises missing signals"),
            ("Archetypes",       "K-means clustering on 4-dimensional profile"),
        ]
        st.markdown("""
        <div style="margin-bottom:20px">
        """ + "".join(
            f'<div class="pipeline-step"><span style="color:#2dd4a8;font-family:\'IBM Plex Mono\',monospace;font-size:11px;margin-right:12px">0{i+1}</span>{name} <span style="color:#3a3a4a;font-size:12px;margin-left:8px">{desc}</span></div>'
            for i, (name, desc) in enumerate(stages)
        ) + """</div>""", unsafe_allow_html=True)

        if not jd_ready:
            st.warning("Job Description is empty — paste it in Upload & Configure first.")
        if not groq_ready:
            st.warning("GROQ_API_KEY missing — add it to .streamlit/secrets.toml")

        if st.button(
            "Run Evaluation Pipeline",
            type="primary",
            disabled=not can_run or st.session_state["pipeline_running"],
            use_container_width=True,
            key="btn_run_pipeline",
        ):
            st.session_state["pipeline_running"] = True
            st.session_state["pipeline_log"] = []

            progress = st.progress(0, text="Starting…")
            status_placeholder = st.empty()
            log_placeholder = st.empty()

            def update_status_table():
                fresh = db.get_all_candidates()
                if not fresh.empty:
                    cols = ["s_no", "name", "status", "error_notes"]
                    cols = [c for c in cols if c in fresh.columns]
                    status_placeholder.dataframe(fresh[cols], use_container_width=True, hide_index=True)

            def pipeline_cb(sno, msg):
                log(f"[s_no {sno}] {msg}")
                log_placeholder.markdown(
                    '<div style="background:#0a0a12;border:1px solid #1a1a28;border-radius:2px;padding:12px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#5a5a6a;max-height:180px;overflow-y:auto">' +
                    "<br>".join(st.session_state["pipeline_log"][-10:]) +
                    '</div>',
                    unsafe_allow_html=True
                )

            try:
                log("Stage 1/5 — Parsing resumes")
                progress.progress(5, "Parsing resumes…")
                resume_parser.parse_all(db, status_callback=pipeline_cb)
                progress.progress(25, "Resumes parsed")
                update_status_table()

                log("Stage 2/5 — Analyzing GitHub repos")
                progress.progress(30, "GitHub analysis…")
                github_analyzer.analyze_all(db, pat=_secret("GITHUB_PAT") or None, status_callback=pipeline_cb)
                progress.progress(50, "GitHub done")
                update_status_table()

                log("Stage 3/5 — LLM evaluation (Groq + MiniLM)")
                progress.progress(55, "LLM evaluation…")
                ai_evaluator.evaluate_all(
                    jd_text=st.session_state["jd_text"],
                    db=db,
                    groq_api_key=_secret("GROQ_API_KEY"),
                    status_callback=pipeline_cb,
                )
                progress.progress(75, "LLM done")
                update_status_table()

                log("Stage 4/5 — WSM scoring & ranking")
                progress.progress(80, "Scoring…")
                scorer.score_all(db, weights=st.session_state["weights"], status_callback=pipeline_cb)
                progress.progress(88, "Scores computed")
                update_status_table()

                log("Stage 5/5 — Archetype clustering")
                progress.progress(92, "Archetypes…")
                _assign_archetypes_to_all(pipeline_cb)
                progress.progress(100, "Done")
                update_status_table()

                log("Pipeline complete.")
                st.success("Pipeline complete — view results in Rankings.")

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                log(f"ERROR: {e}")
            finally:
                st.session_state["pipeline_running"] = False

        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Candidate Status</div>', unsafe_allow_html=True)
        all_cands = db.get_all_candidates()
        if not all_cands.empty:
            display = all_cands[["s_no", "name", "status", "error_notes"]].copy()
            display["status"] = display["status"].apply(
                lambda s: status_badge_html(s) if isinstance(s, str) else s
            )
            st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)

        if st.session_state["pipeline_log"]:
            with st.expander("Pipeline Log", expanded=False):
                st.code("\n".join(st.session_state["pipeline_log"]), language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Rankings Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    all_cands = db.get_all_candidates()
    ranked_mask = all_cands["status"].isin([
        "ranked", "test_sent", "test_scored", "shortlisted",
        "interview_scheduled", "invited"
    ]) if "status" in all_cands.columns else pd.Series([False] * len(all_cands))
    ranked_df = all_cands[ranked_mask].copy()

    if ranked_df.empty:
        st.markdown("""
        <div class="hero-banner" style="text-align:center;padding:60px 44px">
          <h1 style="font-size:20px">No ranked candidates</h1>
          <p>Run the pipeline in the Run Pipeline tab to see results here.</p>
        </div>""", unsafe_allow_html=True)
    else:
        if "rank" in ranked_df.columns:
            ranked_df = ranked_df.sort_values("rank", na_position="last")

        col_filter, col_spacer = st.columns([2, 3])
        with col_filter:
            min_score = st.slider("Min score", 0.0, 100.0, 0.0, 5.0, key="rank_min_score")

        filtered = ranked_df[ranked_df["final_score"].fillna(0) >= min_score]

        # Summary bar
        total_r = len(ranked_df)
        showing = len(filtered)
        has_arch = filtered["archetype"].notna().any() if "archetype" in filtered.columns else False
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;margin-bottom:20px">
          <span style="font-size:13px;font-family:'IBM Plex Mono',monospace;color:#5a5a6a">
            Showing <span style="color:#e8e5de">{showing}</span> of <span style="color:#e8e5de">{total_r}</span> ranked
          </span>
          {'<span style="font-size:12px;font-family:\'IBM Plex Mono\',monospace;color:#3a3a4a;font-style:italic">Archetypes are descriptive only — they do not affect rank.</span>' if has_arch else ''}
        </div>""", unsafe_allow_html=True)

        for _, row in filtered.iterrows():
            rank_num = int(row.get("rank") or 0)
            name = row.get("name", "Unknown")
            sno = int(row.get("s_no", 0))
            final = row.get("final_score")
            jd_m  = row.get("jd_match")
            pq    = row.get("project_quality")
            gh    = row.get("github_score")
            ts    = row.get("test_score")
            cgpa  = row.get("cgpa")
            status = row.get("status", "")
            reasoning = row.get("llm_reasoning", "")
            breakdown_json = row.get("github_breakdown", "{}")
            cand_arch = row.get("archetype", None)
            cand_arch_conf = row.get("archetype_confidence", None)
            cand_arch_bkdn = row.get("archetype_breakdown", None)
            college = row.get("college", "")
            branch  = row.get("branch", "")

            is_top = rank_num <= 3
            rank_class = "top" if is_top else ""
            score_str = f"{final:.1f}" if final is not None else "--"

            arch_tag = ""
            if cand_arch and isinstance(cand_arch, str):
                tag_class = "archetype-tag-" + cand_arch.lower().replace(" ", "-")
                conf_str = f" {cand_arch_conf*100:.0f}%" if cand_arch_conf else ""
                arch_tag = f'<span class="archetype-tag {tag_class}">{cand_arch}{conf_str}</span>'

            with st.expander(
                f"#{rank_num}  {name}  {score_str}",
                expanded=is_top,
            ):
                # Top strip: meta + archetype
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap">
                  <span style="font-size:28px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:{'#2dd4a8' if is_top else '#e8e5de'}">{score_str}</span>
                  {status_badge_html(status)}
                  {arch_tag}
                  <span style="font-size:12px;font-family:'IBM Plex Mono',monospace;color:#3a3a4a">{college} / {branch}</span>
                  {'<span style="font-size:12px;font-family:\'IBM Plex Mono\',monospace;color:#3a3a4a">CGPA ' + str(cgpa) + '</span>' if cgpa else ''}
                </div>""", unsafe_allow_html=True)

                # Score bars
                bars_html = '<div style="margin-bottom:16px">'
                bars_html += score_bar(jd_m,  label="JD Match",        color="#2dd4a8")
                bars_html += score_bar(pq,    label="Project Quality",  color="#60a5fa")
                bars_html += score_bar(gh,    label="GitHub Score",     color="#fbbf24")
                if ts is not None:
                    bars_html += score_bar(ts, label="Test Score",      color="#a882ff")
                bars_html += '</div>'
                st.markdown(bars_html, unsafe_allow_html=True)

                # Archetype distance breakdown
                if cand_arch_bkdn:
                    try:
                        bkdn = json.loads(cand_arch_bkdn) if isinstance(cand_arch_bkdn, str) else cand_arch_bkdn
                        conf_data = bkdn.get("confidence", {})
                        if conf_data:
                            bars = '<div class="archetype-breakdown">'
                            bars += '<div class="archetype-breakdown-title">Archetype Distance</div>'
                            for arch_name, conf_val in sorted(conf_data.items(), key=lambda x: x[1], reverse=True):
                                pct = conf_val * 100
                                bc = "archetype-dist-bar-fill-" + arch_name.lower().replace(" ", "-")
                                bars += (
                                    f'<div class="archetype-dist-row">'
                                    f'<span class="archetype-dist-name">{arch_name}</span>'
                                    f'<div class="archetype-dist-bar-bg"><div class="archetype-dist-bar-fill {bc}" style="width:{pct:.0f}%"></div></div>'
                                    f'<span class="archetype-distance-val">{pct:.0f}%</span>'
                                    f'</div>'
                                )
                            bars += '</div>'
                            st.markdown(bars, unsafe_allow_html=True)
                    except Exception:
                        pass

                if reasoning:
                    st.markdown('<div class="sec-title" style="margin-top:16px">AI Reasoning</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="background:#0a0a12;border:1px solid #1a1a28;border-radius:2px;padding:14px 16px;font-size:13px;color:#8a8a9a;line-height:1.7">{reasoning}</div>', unsafe_allow_html=True)

                if breakdown_json and breakdown_json not in ("{}", "null"):
                    try:
                        bd = json.loads(breakdown_json) if isinstance(breakdown_json, str) else breakdown_json
                        if bd and "repos" in bd:
                            st.markdown('<div class="sec-title" style="margin-top:16px">GitHub Repos</div>', unsafe_allow_html=True)
                            repos = bd.get("repos", [])
                            repo_rows = []
                            for r in repos[:10]:
                                if isinstance(r, dict) and "name" in r:
                                    repo_rows.append({
                                        "Repo": r["name"],
                                        "Language": r.get("language", "--"),
                                        "Stars": r.get("stars", 0),
                                        "Commits (6mo)": r.get("commit_count_6mo", 0),
                                        "README": "yes" if r.get("has_readme") else "no",
                                        "Fork": "yes" if r.get("is_fork") else "no",
                                        "Days since push": r.get("days_since_push", "--"),
                                    })
                            if repo_rows:
                                st.dataframe(pd.DataFrame(repo_rows), use_container_width=True, hide_index=True)
                            sub = bd.get("sub_scores", {})
                            if sub:
                                st.markdown(
                                    f'<div style="font-size:11px;font-family:\'IBM Plex Mono\',monospace;color:#3a3a4a;margin-top:8px">'
                                    f'activity={sub.get("activity_recency",0):.2f} &nbsp; commits={sub.get("commit_frequency",0):.2f} &nbsp; originals={sub.get("original_repo_ratio",0):.2f} &nbsp; docs={sub.get("documentation_quality",0):.2f}'
                                    f'</div>', unsafe_allow_html=True
                                )
                    except Exception:
                        pass

        # Send test links
        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Send Test Links</div>', unsafe_allow_html=True)
        col_turl, col_tthresh = st.columns([2, 1])
        with col_turl:
            test_url = st.text_input(
                "Assessment URL",
                value=st.session_state["test_url"],
                placeholder="https://forms.gle/... or https://hackerrank.com/...",
                key="test_url_input",
            )
            st.session_state["test_url"] = test_url
        with col_tthresh:
            threshold = st.number_input(
                "Min score to send test",
                min_value=0.0, max_value=100.0,
                value=st.session_state["shortlist_threshold"],
                step=5.0, key="send_threshold",
            )
            st.session_state["shortlist_threshold"] = threshold

        eligible = filtered[filtered["final_score"].fillna(0) >= threshold]
        st.caption(f"{len(eligible)} candidates above threshold {threshold:.0f}")

        can_send = bool(test_url.strip()) and len(eligible) > 0 and bool(_secret("GMAIL_ADDRESS")) and bool(_secret("GMAIL_APP_PASSWORD"))
        if not _secret("GMAIL_ADDRESS"):
            st.warning("Gmail credentials not configured in secrets.")

        if st.button(
            f"Send Test Links ({len(eligible)} candidates)",
            type="primary",
            disabled=not can_send,
            key="btn_send_test_links",
        ):
            eligible_snos = set(eligible["s_no"].tolist())
            to_send = [r for r in db.get_candidates_past_status("ranked") if r["s_no"] in eligible_snos]
            with st.spinner("Sending…"):
                results = emailer.send_test_links(
                    candidates=to_send,
                    test_url=test_url.strip(),
                    db=db,
                    gmail_address=_secret("GMAIL_ADDRESS"),
                    app_password=_secret("GMAIL_APP_PASSWORD"),
                )
            ok = sum(1 for r in results.values() if r.get("success"))
            fail = len(results) - ok
            if ok: st.success(f"{ok} emails sent.")
            if fail: st.error(f"{fail} failed: " + "; ".join(f's_no {k}: {v["error"]}' for k, v in results.items() if not v.get("success")))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Test Results Upload & Re-rank
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="sec-title">Test Results Upload & Re-rank</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([1, 1], gap="large")

    with col_up:
        st.caption("Upload CSV with: `s_no, test_la, test_code`")
        test_file = st.file_uploader(
            "Upload test results",
            type=["csv"],
            key="test_results_upload",
            label_visibility="collapsed",
        )
        if test_file:
            try:
                test_df = pd.read_csv(test_file)
                st.dataframe(test_df.head(10), use_container_width=True, hide_index=True)
                if st.button("Merge & Re-rank", type="primary", use_container_width=True, key="btn_merge_rerank"):
                    try:
                        merged, warnings = db.merge_test_results(test_df)
                        for w in warnings:
                            st.warning(w)
                        st.info(f"Merged test scores for {merged} candidates.")
                        with st.spinner("Re-ranking…"):
                            scored = scorer.score_all(db, weights=st.session_state["weights"])
                        st.success(f"{len(scored)} candidates re-ranked with test scores.")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"CSV format error: {e}")
                    except Exception as e:
                        st.error(f"Re-rank failed: {e}")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    with col_info:
        st.markdown('<div class="sec-title">Expected Format</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([
                {"s_no": 1, "test_la": 85.0, "test_code": 72.0},
                {"s_no": 3, "test_la": 91.0, "test_code": 88.0},
                {"s_no": 7, "test_la": 65.0, "test_code": 70.0},
            ]),
            use_container_width=True, hide_index=True,
        )
        st.markdown("""
        <div style="font-size:12px;font-family:'IBM Plex Mono',monospace;color:#5a5a6a;margin-top:12px;line-height:2">
          s_no &nbsp;&nbsp;&nbsp;→ must match candidate IDs in DB<br>
          test_la &nbsp;→ logical aptitude score (0–100)<br>
          test_code → coding score (0–100)<br>
          test_score = average of the two
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Updated Rankings</div>', unsafe_allow_html=True)
    all_cands = db.get_all_candidates()
    if not all_cands.empty and "final_score" in all_cands.columns:
        display_cols = ["rank", "s_no", "name", "final_score", "jd_match",
                        "project_quality", "github_score", "test_score", "status"]
        display_cols = [c for c in display_cols if c in all_cands.columns]
        sorted_df = all_cands[all_cands["final_score"].notna()].sort_values("rank", na_position="last")
        st.dataframe(sorted_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Mark Shortlist</div>', unsafe_allow_html=True)
        rerank_thresh = st.slider(
            "Shortlist threshold (final score ≥)",
            0.0, 100.0, st.session_state["rerank_threshold"], 5.0,
            key="rerank_threshold_slider",
        )
        st.session_state["rerank_threshold"] = rerank_thresh
        to_shortlist = sorted_df[sorted_df["final_score"].fillna(0) >= rerank_thresh]
        st.caption(f"{len(to_shortlist)} candidates at score ≥ {rerank_thresh:.0f}")
        if st.button(
            f"Mark {len(to_shortlist)} as Shortlisted",
            type="primary",
            disabled=len(to_shortlist) == 0,
            key="btn_shortlist",
        ):
            for _, row in to_shortlist.iterrows():
                db.update_status(int(row["s_no"]), "shortlisted")
            st.success(f"{len(to_shortlist)} candidates marked shortlisted.")
            st.rerun()
    else:
        st.info("No scored candidates yet. Run the pipeline first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Interview Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="sec-title">Interview Scheduling</div>', unsafe_allow_html=True)
    st.caption("Creates Google Calendar events with Meet links for all shortlisted candidates.")

    shortlisted = db.get_by_status("shortlisted")
    scheduled   = db.get_by_status("interview_scheduled")

    col_sched, col_status = st.columns([1, 1], gap="large")

    with col_sched:
        st.markdown('<div class="sec-title">Schedule Configuration</div>', unsafe_allow_html=True)

        # Shortlist summary card
        st.markdown(f"""
        <div style="background:#12121c;border:1px solid #1a1a28;border-radius:2px;padding:20px;margin-bottom:20px">
          <div style="font-size:36px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#2dd4a8;letter-spacing:-0.02em">{len(shortlisted)}</div>
          <div style="font-size:11px;font-family:'IBM Plex Mono',monospace;color:#5a5a6a;text-transform:uppercase;letter-spacing:0.08em;margin-top:6px">Shortlisted candidates</div>
        </div>""", unsafe_allow_html=True)

        if len(shortlisted) == 0 and len(scheduled) == 0:
            st.info("No shortlisted candidates. Complete test results and mark shortlist in Test Results tab.")
        else:
            IST = pytz.timezone("Asia/Kolkata")
            interview_date = st.date_input(
                "First interview date",
                value=datetime.now(IST).date() + timedelta(days=3),
                key="interview_date",
            )
            interview_time = st.time_input(
                "Start time (IST)",
                value=datetime.now(IST).replace(hour=10, minute=0, second=0, microsecond=0).time(),
                key="interview_time",
            )
            spacing = st.number_input(
                "Minutes between interviews",
                min_value=15, max_value=120, value=45, step=15,
                key="interview_spacing",
            )

            slot_start = IST.localize(datetime.combine(interview_date, interview_time))
            last_slot  = slot_start + timedelta(minutes=spacing * max(len(shortlisted)-1, 0))
            st.markdown(f"""
            <div style="font-size:12px;font-family:'IBM Plex Mono',monospace;color:#5a5a6a;margin-top:8px;line-height:2">
              First: <span style="color:#e8e5de">{slot_start.strftime('%a %d %b %Y, %H:%M IST')}</span><br>
              Last:  <span style="color:#e8e5de">{last_slot.strftime('%H:%M IST')}</span>
            </div>""", unsafe_allow_html=True)

            can_schedule = (
                len(shortlisted) > 0
                and bool(_secret("GOOGLE_OAUTH_CLIENT_JSON"))
                and bool(_secret("RECRUITER_EMAIL") or _secret("GMAIL_ADDRESS"))
            )
            if not _secret("GOOGLE_OAUTH_CLIENT_JSON"):
                st.warning("Google Calendar OAuth not configured.")

            if st.button(
                f"Schedule {len(shortlisted)} Interviews",
                type="primary",
                disabled=not can_schedule,
                key="btn_schedule_interviews",
            ):
                recruiter_email = _secret("RECRUITER_EMAIL") or _secret("GMAIL_ADDRESS")
                with st.spinner("Creating Calendar events…"):
                    results = calendar_scheduler.schedule_all(
                        db=db,
                        oauth_json_b64=_secret("GOOGLE_OAUTH_CLIENT_JSON"),
                        recruiter_email=recruiter_email,
                        slot_start=slot_start,
                        spacing_mins=int(spacing),
                    )
                if isinstance(results, dict) and "error" in results and len(results) == 1:
                    st.error(f"Calendar error: {results['error']}")
                else:
                    ok   = sum(1 for r in results.values() if isinstance(r, dict) and r.get("success"))
                    fail = sum(1 for r in results.values() if isinstance(r, dict) and not r.get("success"))
                    if ok:   st.success(f"{ok} interviews scheduled on Google Calendar.")
                    if fail: st.error(f"{fail} scheduling failures.")

                    # ── Also send meet-link email to each successfully scheduled candidate ──
                    gmail_addr = _secret("GMAIL_ADDRESS")
                    gmail_pass  = _secret("GMAIL_APP_PASSWORD")
                    if gmail_addr and gmail_pass and ok > 0:
                        # Build list of candidates who got a meet link
                        invite_list = []
                        for sno, res in results.items():
                            if isinstance(res, dict) and res.get("success") and res.get("meet_link"):
                                cand_row = db.get_candidate(sno)
                                if cand_row:
                                    invite_list.append({
                                        "s_no": sno,
                                        "name": cand_row["name"],
                                        "email": cand_row["email"],
                                        "meet_link": res["meet_link"],
                                        "scheduled_time": res.get("scheduled_time", ""),
                                    })
                        if invite_list:
                            with st.spinner(f"Sending Meet link emails to {len(invite_list)} candidates…"):
                                email_results = emailer.send_interview_links(
                                    candidates=invite_list,
                                    gmail_address=gmail_addr,
                                    app_password=gmail_pass,
                                )
                            e_ok   = sum(1 for r in email_results.values() if r.get("success"))
                            e_fail = len(email_results) - e_ok
                            if e_ok:   st.success(f"✉️ Meet link emails sent to {e_ok} candidates.")
                            if e_fail: st.warning(f"{e_fail} meet-link email(s) failed.")
                st.rerun()


    with col_status:
        st.markdown('<div class="sec-title">Scheduled Interviews</div>', unsafe_allow_html=True)
        sched_df = db.get_all_candidates()
        if "status" in sched_df.columns:
            sched_df = sched_df[sched_df["status"] == "interview_scheduled"]

        if sched_df.empty:
            st.markdown("""
            <div style="background:#12121c;border:1px dashed #1a1a28;border-radius:2px;padding:32px;text-align:center">
              <div style="font-size:13px;font-family:'IBM Plex Mono',monospace;color:#3a3a4a">No interviews scheduled yet</div>
            </div>""", unsafe_allow_html=True)
        else:
            for _, row in sched_df.iterrows():
                name = row.get("name", f"Candidate {row['s_no']}")
                sno  = int(row.get("s_no", 0))
                meet = row.get("meet_link", "")
                sched_time = row.get("scheduled_time", "")
                time_str = sched_time[:16].replace("T", " ") if sched_time else "--"
                meet_html = f'<a href="{meet}" target="_blank">Join Meet</a>' if meet else '<span style="color:#3a3a4a">Meet link pending</span>'
                st.markdown(f"""
                <div class="interview-card">
                  <div><span class="ic-name">{name}</span><span class="ic-sno">s_no {sno}</span></div>
                  <div class="ic-time">{time_str} IST</div>
                  <div style="margin-top:8px">{meet_html}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">All Interview Events</div>', unsafe_allow_html=True)
    all_cands_full = db.get_all_candidates()
    scheduled_full = all_cands_full[
        all_cands_full["status"].isin(["interview_scheduled", "invited"])
    ] if "status" in all_cands_full.columns else pd.DataFrame()

    if not scheduled_full.empty:
        display_cols = ["s_no", "name", "email", "scheduled_time", "meet_link", "invite_sent", "final_score"]
        display_cols = [c for c in display_cols if c in scheduled_full.columns]
        st.dataframe(scheduled_full[display_cols], use_container_width=True, hide_index=True)
    else:
        st.caption("No scheduled events to display.")


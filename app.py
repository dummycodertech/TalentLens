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
    page_title="AI Candidate Screener — myNachiketa",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Module imports ────────────────────────────────────────────────────────────

import db
from modules import resume_parser, github_analyzer, ai_evaluator, scorer, emailer, calendar_scheduler

# ─── Init ─────────────────────────────────────────────────────────────────────

db.init_db()

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
  }
  section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

  /* Main area */
  .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(103, 126, 234, 0.08);
    border-radius: 12px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 500;
    color: #6b7280;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
  }

  /* Status badges */
  .badge { display: inline-block; padding: 2px 10px; border-radius: 99px;
            font-size: 12px; font-weight: 600; }
  .badge-green  { background: #d1fae5; color: #065f46; }
  .badge-blue   { background: #dbeafe; color: #1e40af; }
  .badge-yellow { background: #fef3c7; color: #92400e; }
  .badge-red    { background: #fee2e2; color: #991b1b; }
  .badge-gray   { background: #f3f4f6; color: #374151; }
  .badge-purple { background: #ede9fe; color: #5b21b6; }

  /* Score cards */
  .score-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .score-card .value { font-size: 28px; font-weight: 700;
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .score-card .label { font-size: 12px; color: #6b7280; font-weight: 500; margin-top: 4px; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 24px;
  }
  .hero-banner h1 { margin: 0; font-size: 26px; font-weight: 700; }
  .hero-banner p  { margin: 8px 0 0; opacity: 0.85; font-size: 14px; }

  /* Pipeline step */
  .pipeline-step {
    background: rgba(103,126,234,0.06);
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 14px;
  }

  /* Metric row */
  .metric-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }

  /* Button overrides */
  .stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(103,126,234,0.3) !important; }

  div[data-testid="stExpander"] { border-radius: 10px !important; border: 1px solid #e5e7eb !important; }
</style>
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
    st.markdown("## 🎯 AI Screener")
    st.markdown("**myNachiketa** GTM Engineering")
    st.divider()

    st.markdown("### ⚖️ Scoring Weights")
    st.caption("Weights auto-normalize to sum to 1.0")

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
        st.caption(f"Normalized: {sum(st.session_state['weights'].values()):.2f}")

    st.divider()
    st.markdown("### ⚙️ API Status")
    groq_ok = bool(_secret("GROQ_API_KEY"))
    gh_ok = bool(_secret("GITHUB_PAT"))
    gmail_ok = bool(_secret("GMAIL_ADDRESS")) and bool(_secret("GMAIL_APP_PASSWORD"))
    cal_ok = bool(_secret("GOOGLE_OAUTH_CLIENT_JSON"))

    st.markdown(f"{'🟢' if groq_ok else '🔴'} Groq API {'connected' if groq_ok else '— key missing'}")
    st.markdown(f"{'🟢' if gh_ok else '🔴'} GitHub PAT {'connected' if gh_ok else '— token missing'}")
    st.markdown(f"{'🟢' if gmail_ok else '🔴'} Gmail SMTP {'connected' if gmail_ok else '— credentials missing'}")
    st.markdown(f"{'🟢' if cal_ok else '🔴'} Google Calendar {'connected' if cal_ok else '— OAuth missing'}")

    st.divider()
    if st.button("🗑️ Reset Database", type="secondary", use_container_width=True, key="btn_reset"):
        db.reset_db()
        st.session_state["pipeline_log"] = []
        st.success("Database reset. Reload to start fresh.")
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


def score_bar(value, max_val=100, color="#667eea"):
    if value is None:
        return "—"
    pct = min(100, (value / max_val) * 100)
    return f"""
    <div style="display:flex;align-items:center;gap:8px">
      <div style="flex:1;background:#e5e7eb;border-radius:4px;height:6px">
        <div style="width:{pct:.0f}%;background:{color};height:100%;border-radius:4px"></div>
      </div>
      <span style="font-size:13px;font-weight:600;color:#374151;min-width:36px">{value:.1f}</span>
    </div>"""


# ─── Pipeline log helper ───────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["pipeline_log"].append(f"[{ts}] {msg}")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Configure",
    "🚀 Run Pipeline",
    "🏆 Rankings",
    "📊 Test Results",
    "📅 Interviews",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Configure
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class="hero-banner">
      <h1>🎯 AI-Powered Candidate Screener</h1>
      <p>Upload your candidate CSV and job description to begin the automated pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_jd = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📋 Candidate CSV")
        st.caption("Required columns: `s_no, name, email, college, branch, cgpa, best_ai_project, research_work, github, resume`")
        uploaded_file = st.file_uploader(
            "Choose candidates CSV",
            type=["csv"],
            key="candidates_csv_upload",
            label_visibility="collapsed",
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(f"**Preview** — {len(df)} rows detected")
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)

                col_btn, col_info = st.columns([1, 1])
                with col_btn:
                    if st.button("⬆️ Load Candidates", type="primary", use_container_width=True, key="btn_load_candidates"):
                        try:
                            inserted, skipped = db.insert_candidates(df)
                            if inserted > 0:
                                st.success(f"✅ Loaded {inserted} candidates. ({skipped} duplicate s_no skipped)")
                            else:
                                st.warning(f"All {skipped} rows already exist in DB (duplicate s_no).")
                        except ValueError as e:
                            st.error(f"CSV format error: {e}")
                        except Exception as e:
                            st.error(f"Load failed: {e}")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    with col_jd:
        st.markdown("### 📝 Job Description")
        st.caption("Paste the full JD text. Used for embedding similarity and LLM evaluation.")
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
            st.caption(f"{word_count} words — {'✅ good length' if word_count > 50 else '⚠️ short — add more detail for better matching'}")

    st.divider()
    st.markdown("### 📊 Current Database")
    all_cands = db.get_all_candidates()
    if all_cands.empty:
        st.info("No candidates loaded yet. Upload a CSV above.")
    else:
        st.markdown(f"**{len(all_cands)} candidates** in database")
        display_cols = ["s_no", "name", "college", "branch", "cgpa", "github_url", "status"]
        display_cols = [c for c in display_cols if c in all_cands.columns]
        st.dataframe(all_cands[display_cols], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Run Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🚀 Evaluation Pipeline")
    st.caption("Runs resume parsing → GitHub analysis → AI evaluation → scoring in sequence. Each stage updates status live.")

    all_cands = db.get_all_candidates()
    total = len(all_cands)

    if total == 0:
        st.warning("No candidates loaded. Go to **Upload & Configure** first.")
    else:
        # Pipeline status overview
        status_counts = all_cands["status"].value_counts().to_dict() if "status" in all_cands.columns else {}

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Candidates", total)
        with col_b:
            ranked = status_counts.get("ranked", 0) + sum(
                v for k, v in status_counts.items()
                if k in ("test_sent", "test_scored", "shortlisted", "interview_scheduled", "invited")
            )
            st.metric("Ranked", ranked)
        with col_c:
            failed = sum(v for k, v in status_counts.items() if "failed" in k)
            st.metric("⚠️ Errors", failed)
        with col_d:
            if not all_cands.empty and "final_score" in all_cands.columns:
                avg_score = all_cands["final_score"].dropna().mean()
                st.metric("Avg Score", f"{avg_score:.1f}" if not pd.isna(avg_score) else "—")
            else:
                st.metric("Avg Score", "—")

        st.divider()

        # Pre-flight checks
        jd_ready = bool(st.session_state["jd_text"].strip())
        groq_ready = bool(_secret("GROQ_API_KEY"))
        gh_ready = bool(_secret("GITHUB_PAT"))

        if not jd_ready:
            st.warning("⚠️ Job Description is empty. Go to **Upload & Configure** and paste the JD first.")
        if not groq_ready:
            st.warning("⚠️ GROQ_API_KEY not set. Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")

        can_run = jd_ready and groq_ready and total > 0

        if st.button(
            "▶️ Run Evaluation Pipeline",
            type="primary",
            disabled=not can_run or st.session_state["pipeline_running"],
            use_container_width=True,
            key="btn_run_pipeline",
        ):
            st.session_state["pipeline_running"] = True
            st.session_state["pipeline_log"] = []

            progress = st.progress(0, text="Starting pipeline…")
            status_placeholder = st.empty()
            log_placeholder = st.empty()

            def update_status_table():
                fresh = db.get_all_candidates()
                if not fresh.empty:
                    cols = ["s_no", "name", "status", "error_notes"]
                    cols = [c for c in cols if c in fresh.columns]
                    status_placeholder.dataframe(fresh[cols], use_container_width=True, hide_index=True)

            def pipeline_cb(sno, msg):
                log(f"Candidate {sno}: {msg}")
                log_placeholder.markdown("\n".join(st.session_state["pipeline_log"][-8:]))

            try:
                # Stage 1: Resume parsing
                log("━━ Stage 1/4: Parsing resumes…")
                progress.progress(5, "Parsing resumes…")
                resume_parser.parse_all(db, status_callback=pipeline_cb)
                progress.progress(25, "Resumes parsed")
                update_status_table()

                # Stage 2: GitHub analysis
                log("━━ Stage 2/4: Analyzing GitHub repos…")
                progress.progress(30, "Analyzing GitHub…")
                github_analyzer.analyze_all(
                    db,
                    pat=_secret("GITHUB_PAT") or None,
                    status_callback=pipeline_cb,
                )
                progress.progress(50, "GitHub analyzed")
                update_status_table()

                # Stage 3: AI evaluation
                log("━━ Stage 3/4: Running AI evaluation (Groq LLM + MiniLM)…")
                progress.progress(55, "Running AI evaluation…")
                ai_evaluator.evaluate_all(
                    jd_text=st.session_state["jd_text"],
                    db=db,
                    groq_api_key=_secret("GROQ_API_KEY"),
                    status_callback=pipeline_cb,
                )
                progress.progress(80, "AI evaluation complete")
                update_status_table()

                # Stage 4: Scoring + ranking
                log("━━ Stage 4/4: Computing scores and ranking…")
                progress.progress(85, "Scoring and ranking…")
                scorer.score_all(db, weights=st.session_state["weights"], status_callback=pipeline_cb)
                progress.progress(100, "Pipeline complete! ✅")
                update_status_table()

                log("✅ Pipeline complete! Check the Rankings tab.")
                st.success("Pipeline finished. View results in the **Rankings** tab.")

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                log(f"❌ Pipeline error: {e}")
            finally:
                st.session_state["pipeline_running"] = False

        # Live status table (always shown)
        st.divider()
        st.markdown("#### Candidate Status")
        all_cands = db.get_all_candidates()
        if not all_cands.empty:
            display = all_cands[["s_no", "name", "status", "error_notes"]].copy()
            display["status"] = display["status"].apply(
                lambda s: status_badge_html(s) if isinstance(s, str) else s
            )
            st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Pipeline log
        if st.session_state["pipeline_log"]:
            with st.expander("📋 Pipeline Log", expanded=False):
                st.code("\n".join(st.session_state["pipeline_log"]), language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Rankings Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🏆 Candidate Rankings")

    all_cands = db.get_all_candidates()
    # Only show ranked candidates
    ranked_mask = all_cands["status"].isin([
        "ranked", "test_sent", "test_scored", "shortlisted",
        "interview_scheduled", "invited"
    ]) if "status" in all_cands.columns else pd.Series([False] * len(all_cands))
    ranked_df = all_cands[ranked_mask].copy()

    if ranked_df.empty:
        st.info("No ranked candidates yet. Run the pipeline in **Run Pipeline** tab.")
    else:
        # Sort by rank
        if "rank" in ranked_df.columns:
            ranked_df = ranked_df.sort_values("rank", na_position="last")

        col_filter, col_spacer = st.columns([2, 3])
        with col_filter:
            min_score = st.slider(
                "Minimum score filter", 0.0, 100.0, 0.0, 5.0, key="rank_min_score"
            )

        filtered = ranked_df[ranked_df["final_score"].fillna(0) >= min_score]
        st.caption(f"Showing {len(filtered)} of {len(ranked_df)} ranked candidates")

        # Ranking table
        for _, row in filtered.iterrows():
            rank_num = int(row.get("rank") or 0)
            name = row.get("name", "Unknown")
            sno = int(row.get("s_no", 0))
            final = row.get("final_score")
            jd_m = row.get("jd_match")
            pq = row.get("project_quality")
            gh = row.get("github_score")
            ts = row.get("test_score")
            status = row.get("status", "")
            reasoning = row.get("llm_reasoning", "")
            breakdown_json = row.get("github_breakdown", "{}")

            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank_num, f"#{rank_num}")
            score_str = f"{final:.1f}" if final is not None else "—"

            with st.expander(
                f"{medal} **{name}** (s_no {sno}) — Score: {score_str} \u00a0 {status_badge_html(status)}",
                expanded=rank_num <= 3,
            ):
                final_s = f"{final:.1f}" if final is not None else "—"
                jdm_s   = f"{jd_m:.1f}" if jd_m is not None else "—"
                gh_s    = f"{gh:.1f}" if gh is not None else "N/A"
                ts_s    = f"{ts:.1f}" if ts is not None else "—"

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="score-card"><div class="value">{final_s}</div><div class="label">Final Score</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="score-card"><div class="value">{jdm_s}</div><div class="label">JD Match</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="score-card"><div class="value">{gh_s}</div><div class="label">GitHub Score</div></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="score-card"><div class="value">{ts_s}</div><div class="label">Test Score</div></div>', unsafe_allow_html=True)

                st.markdown("**Score Breakdown**")
                st.markdown(score_bar(jd_m), unsafe_allow_html=True)
                st.caption("JD Match")
                st.markdown(score_bar(pq, color="#10b981"), unsafe_allow_html=True)
                st.caption("Project Quality")
                st.markdown(score_bar(gh, color="#f59e0b"), unsafe_allow_html=True)
                st.caption("GitHub Score")
                if ts is not None:
                    st.markdown(score_bar(ts, color="#8b5cf6"), unsafe_allow_html=True)
                    st.caption("Test Score")

                if reasoning:
                    st.markdown("**🤖 AI Reasoning**")
                    st.info(reasoning)

                if breakdown_json and breakdown_json not in ("{}", "null"):
                    try:
                        bd = json.loads(breakdown_json) if isinstance(breakdown_json, str) else breakdown_json
                        if bd and "repos" in bd:
                            st.markdown("**🐙 GitHub Breakdown**")
                            repos = bd.get("repos", [])
                            repo_rows = []
                            for r in repos[:10]:  # cap display at 10 repos
                                if isinstance(r, dict) and "name" in r:
                                    repo_rows.append({
                                        "Repo": r["name"],
                                        "Language": r.get("language", "—"),
                                        "Stars": r.get("stars", 0),
                                        "Commits (6mo)": r.get("commit_count_6mo", 0),
                                        "README": "✅" if r.get("has_readme") else "❌",
                                        "Fork": "Yes" if r.get("is_fork") else "No",
                                        "Days since push": r.get("days_since_push", "—"),
                                    })
                            if repo_rows:
                                st.dataframe(pd.DataFrame(repo_rows), use_container_width=True, hide_index=True)
                            sub = bd.get("sub_scores", {})
                            if sub:
                                st.caption(f"Sub-scores: Activity={sub.get('activity_recency',0):.2f}, "
                                           f"Commits={sub.get('commit_frequency',0):.2f}, "
                                           f"Originals={sub.get('original_repo_ratio',0):.2f}, "
                                           f"Docs={sub.get('documentation_quality',0):.2f}")
                    except Exception:
                        pass

        # Send test links section
        st.divider()
        st.markdown("### 📨 Send Test Links")
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

        can_send = (
            bool(test_url.strip())
            and len(eligible) > 0
            and bool(_secret("GMAIL_ADDRESS"))
            and bool(_secret("GMAIL_APP_PASSWORD"))
        )
        if not _secret("GMAIL_ADDRESS"):
            st.warning("Gmail credentials not configured in secrets.")
        if not test_url.strip():
            st.warning("Enter the assessment URL above before sending.")

        if st.button(
            f"📨 Send Test Links to {len(eligible)} candidates",
            type="primary",
            disabled=not can_send,
            key="btn_send_test_links",
        ):
            eligible_rows = db.get_candidates_past_status("ranked")
            # Filter to eligible s_nos
            eligible_snos = set(eligible["s_no"].tolist())
            to_send = [r for r in eligible_rows if r["s_no"] in eligible_snos]

            with st.spinner("Sending emails…"):
                results = emailer.send_test_links(
                    candidates=to_send,
                    test_url=test_url.strip(),
                    db=db,
                    gmail_address=_secret("GMAIL_ADDRESS"),
                    app_password=_secret("GMAIL_APP_PASSWORD"),
                )

            ok = sum(1 for r in results.values() if r.get("success"))
            fail = len(results) - ok
            if ok:
                st.success(f"✅ Sent {ok} emails successfully.")
            if fail:
                failed_msgs = [f"s_no {k}: {v['error']}" for k, v in results.items() if not v.get("success")]
                st.error(f"❌ {fail} failed:\n" + "\n".join(failed_msgs))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Test Results Upload & Re-rank
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 📊 Upload Test Results & Re-rank")
    st.caption("Upload the test results CSV (`s_no, test_la, test_code`). Scores are merged and all candidates are re-ranked using current sidebar weights.")

    col_up, col_info = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("#### 📁 Test Results CSV")
        st.caption("Required columns: `s_no, test_la, test_code`")
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

                if st.button("🔄 Merge & Re-rank", type="primary", use_container_width=True, key="btn_merge_rerank"):
                    try:
                        # Merge test scores into DB (joined on s_no — C9)
                        merged, warnings = db.merge_test_results(test_df)
                        if warnings:
                            for w in warnings:
                                st.warning(w)
                        st.info(f"Merged test scores for {merged} candidates.")

                        # Re-run scorer with current session_state weights (C4)
                        with st.spinner("Re-ranking…"):
                            scored = scorer.score_all(
                                db,
                                weights=st.session_state["weights"],
                            )

                        st.success(f"✅ Re-ranked {len(scored)} candidates with test scores included.")
                        st.rerun()

                    except ValueError as e:
                        st.error(f"CSV format error: {e}")
                    except Exception as e:
                        st.error(f"Re-rank failed: {e}")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    with col_info:
        st.markdown("#### 📋 Expected Format")
        st.dataframe(
            pd.DataFrame([
                {"s_no": 1, "test_la": 85.0, "test_code": 72.0},
                {"s_no": 3, "test_la": 91.0, "test_code": 88.0},
                {"s_no": 7, "test_la": 65.0, "test_code": 70.0},
            ]),
            use_container_width=True, hide_index=True,
        )
        st.caption("`s_no` must match the candidate IDs in the database.")
        st.caption("`test_la` = logical aptitude score (0–100)")
        st.caption("`test_code` = coding score (0–100)")
        st.caption("Final `test_score` = average of the two.")

    # Updated rankings table
    st.divider()
    st.markdown("#### Updated Rankings (with Test Scores)")
    all_cands = db.get_all_candidates()
    has_test = all_cands["test_score"].notna().any() if "test_score" in all_cands.columns else False

    if not all_cands.empty and "final_score" in all_cands.columns:
        display_cols = ["rank", "s_no", "name", "final_score", "jd_match",
                        "project_quality", "github_score", "test_score", "status"]
        display_cols = [c for c in display_cols if c in all_cands.columns]
        sorted_df = all_cands[all_cands["final_score"].notna()].sort_values("rank", na_position="last")
        st.dataframe(sorted_df[display_cols], use_container_width=True, hide_index=True)

        # Shortlist controls
        st.divider()
        st.markdown("#### ✅ Mark Shortlist")
        rerank_thresh = st.slider(
            "Shortlist threshold (final score ≥)",
            0.0, 100.0,
            st.session_state["rerank_threshold"],
            5.0,
            key="rerank_threshold_slider",
        )
        st.session_state["rerank_threshold"] = rerank_thresh

        to_shortlist = sorted_df[sorted_df["final_score"].fillna(0) >= rerank_thresh]
        st.caption(f"{len(to_shortlist)} candidates would be shortlisted at score ≥ {rerank_thresh:.0f}")

        if st.button(
            f"✅ Mark {len(to_shortlist)} as Shortlisted",
            type="primary",
            disabled=len(to_shortlist) == 0,
            key="btn_shortlist",
        ):
            for _, row in to_shortlist.iterrows():
                db.update_status(int(row["s_no"]), "shortlisted")
            st.success(f"Marked {len(to_shortlist)} candidates as shortlisted.")
            st.rerun()
    else:
        st.info("No scored candidates yet. Run the pipeline and/or upload test results.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Interview Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 📅 Interview Scheduling")
    st.caption("Schedules Google Calendar events with Meet links for all shortlisted candidates. Google sends its own invite email to attendees.")

    shortlisted = db.get_by_status("shortlisted")
    scheduled = db.get_by_status("interview_scheduled")

    col_sched, col_status = st.columns([1, 1], gap="large")

    with col_sched:
        st.markdown("#### ⏰ Schedule Configuration")
        st.metric("Shortlisted candidates", len(shortlisted))

        if len(shortlisted) == 0 and len(scheduled) == 0:
            st.info("No shortlisted candidates yet. Complete test results and mark shortlist in **Test Results** tab.")
        else:
            st.markdown("**Interview slot settings**")
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
            st.caption(f"First slot: **{slot_start.strftime('%A, %d %b %Y at %H:%M IST')}**")
            st.caption(f"Last slot: **{(slot_start + timedelta(minutes=spacing * max(len(shortlisted)-1, 0))).strftime('%H:%M IST')}**")

            can_schedule = (
                len(shortlisted) > 0
                and bool(_secret("GOOGLE_OAUTH_CLIENT_JSON"))
                and bool(_secret("RECRUITER_EMAIL") or _secret("GMAIL_ADDRESS"))
            )
            if not _secret("GOOGLE_OAUTH_CLIENT_JSON"):
                st.warning("Google Calendar OAuth token not configured. See README for setup instructions.")

            if st.button(
                f"📅 Schedule {len(shortlisted)} Interviews",
                type="primary",
                disabled=not can_schedule,
                key="btn_schedule_interviews",
            ):
                recruiter_email = _secret("RECRUITER_EMAIL") or _secret("GMAIL_ADDRESS")
                with st.spinner("Creating Google Calendar events…"):
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
                    ok = sum(1 for r in results.values() if isinstance(r, dict) and r.get("success"))
                    fail = sum(1 for r in results.values() if isinstance(r, dict) and not r.get("success"))
                    if ok:
                        st.success(f"✅ Scheduled {ok} interviews. Google sent invite emails to all attendees.")
                    if fail:
                        st.error(f"❌ {fail} scheduling failures.")
                st.rerun()

    with col_status:
        st.markdown("#### 📋 Scheduled Interviews")
        sched_df = db.get_all_candidates()
        if "status" in sched_df.columns:
            sched_df = sched_df[sched_df["status"] == "interview_scheduled"]

        if sched_df.empty:
            st.info("No interviews scheduled yet.")
        else:
            for _, row in sched_df.iterrows():
                name = row.get("name", f"Candidate {row['s_no']}")
                sno = int(row.get("s_no", 0))
                meet = row.get("meet_link", "")
                sched_time = row.get("scheduled_time", "")

                st.markdown(f"""
                <div style="border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; margin-bottom:10px; background:white;">
                  <div style="font-weight:600; font-size:15px;">🎯 {name} <span style="color:#9ca3af; font-size:12px;">(s_no {sno})</span></div>
                  <div style="color:#6b7280; font-size:13px; margin-top:4px;">📅 {sched_time[:16].replace('T', ' ') if sched_time else '—'}</div>
                  {'<a href="' + meet + '" target="_blank" style="color:#667eea; font-size:13px; font-weight:600;">🎥 Join Meet</a>' if meet else '<span style="color:#9ca3af; font-size:13px;">Meet link pending</span>'}
                </div>
                """, unsafe_allow_html=True)

    # Full scheduled table
    st.divider()
    st.markdown("#### 📊 All Interview Events")
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

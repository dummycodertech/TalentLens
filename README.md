# 🎯 AI-Powered Candidate Screening Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Compound-F55036)](https://groq.com)
[![scikit-learn](https://img.shields.io/badge/Clustering-scikit--learn%20KMeans-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![GitHub API](https://img.shields.io/badge/GitHub-REST%20API%20v3-181717?logo=github&logoColor=white)](https://docs.github.com/en/rest)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Cost: $0](https://img.shields.io/badge/Infrastructure%20Cost-%240%2Fmonth-brightgreen)](README.md)

> A fully automated, end-to-end AI recruiter pipeline. Upload a candidate CSV → paste a JD → click one button. The platform parses resumes from Google Drive, analyzes GitHub repositories via REST API, evaluates candidates with a Groq LLM + sentence-transformer embeddings, assigns unsupervised K-means archetypes, scores with a Weighted Sum Model, sends test-link emails, re-ranks after test results, and auto-schedules Google Calendar interviews with Google Meet links. **Total infrastructure cost: $0.**

---

## 🔗 Live Demo

> ⚠️ The live deployment requires valid API keys in Streamlit Cloud secrets. If the deployed version shows API connection errors, refer to the setup guide below to configure your own deployment.

---

## 📸 App Screenshots

<table>
  <tr>
    <td align="center"><b>Tab 1 — Upload & Configure</b></td>
    <td align="center"><b>Tab 2 — Evaluation Pipeline (5 Stages)</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/tab1_upload.jpg" alt="Upload & Configure Tab" width="100%"/></td>
    <td><img src="docs/screenshots/tab2_pipeline.jpg" alt="Run Pipeline Tab" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Tab 3 — AI Rankings + Archetype Tags</b></td>
    <td align="center"><b>Tab 5 — Google Calendar Interview Scheduling</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/tab3_rankings.jpg" alt="Rankings Tab" width="100%"/></td>
    <td><img src="docs/screenshots/tab5_interviews.jpg" alt="Interviews Tab" width="100%"/></td>
  </tr>
</table>

---

## 🚀 Full Pipeline (5 Stages)

```
Upload CSV → Parse Resumes (PDF) → GitHub Analysis → AI Evaluation
    → K-means Archetype Clustering → WSM Score & Rank
    → Send Test Links → Upload Results → Re-rank
    → Mark Shortlist → Auto-Schedule Google Calendar Interviews
```

Every stage is automatic once triggered.

---

## ✨ What's Actually Built

| Stage | Module | Technology | What Happens |
|---|---|---|---|
| **CSV Ingestion** | `db.py` | pandas + SQLite | Normalize column aliases (`github`→`github_url`), dedup on `s_no` (not email), persist |
| **Resume Parsing** | `resume_parser.py` | pdfplumber + PyMuPDF | Download PDF from Google Drive, pdfplumber primary / PyMuPDF fallback for scanned PDFs |
| **GitHub Analysis** | `github_analyzer.py` | GitHub REST API v3 | Per-repo WSM: activity recency, 6-month commit count, original-repo ratio, README quality |
| **AI Evaluation** | `ai_evaluator.py` | MiniLM-L6-v2 + Groq LLM | Cosine similarity (local) + structured JSON reasoning (API): `jd_match`, `project_quality`, `reasoning` |
| **Archetype Clustering** | `archetype_clustering.py` | scikit-learn KMeans | K-means on 4 features → assigns one of 4 named archetypes with distance-to-centroid confidence |
| **WSM Scoring** | `scorer.py` | MCDA / Weighted Sum | `final_score = Σ(wᵢ × scoreᵢ)` with auto-renormalization for missing signals |
| **Send Test Links** | `emailer.py` | Gmail SMTP | Branded HTML email with `name + s_no` in subject line for inbox disambiguation |
| **Test Results** | `db.py` | SQLite JOIN on `s_no` | Merge uploaded scores, re-run full WSM with all 4 signals active |
| **Interview Scheduling** | `calendar_scheduler.py` | Google Calendar API | Create event + Google Meet link, Google sends its own invite email to attendees |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────── Streamlit App (single-process) ─────────────────────────────┐
│                                                                                          │
│  ┌──── UI Layer (5 Tabs) ─────────────────────────────────────────────────────────────┐ │
│  │  Tab 1: Upload CSV + paste Job Description                                          │ │
│  │  Tab 2: Run Pipeline (live log, 5 stages including archetype clustering)            │ │
│  │  Tab 3: Rankings (score cards, archetype tags, AI reasoning, GitHub breakdown)      │ │
│  │  Tab 4: Upload test_results.csv → merge → re-rank → mark shortlist                 │ │
│  │  Tab 5: Schedule Google Calendar interviews with Meet links                         │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌──── modules/ ──────────────────────────────────────────────────────────────────────┐ │
│  │  resume_parser.py         Drive URL → bytes → pdfplumber → PyMuPDF fallback        │ │
│  │  github_analyzer.py       REST API → per-repo metrics → WSM sub-score              │ │
│  │  ai_evaluator.py          MiniLM (local cosine sim) + Groq (JSON reasoning)        │ │
│  │  archetype_clustering.py  StandardScaler + KMeans → 4 named archetypes             │ │
│  │  scorer.py                WSM final_score + weight renorm + RANK()                 │ │
│  │  emailer.py               smtplib → Gmail SMTP → branded HTML email                │ │
│  │  calendar_scheduler.py    Google Calendar API → event + Meet + invite               │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  db.py ──► SQLite (candidates + scores + interview_events tables)                        │
│  model_cache/ ──► joblib-persisted KMeans model, scaler, label_map                      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
       ↑ free              ↑ free/5000rph      ↑ local           ↑ free tier
 Streamlit Cloud       GitHub REST API     HF MiniLM-L6-v2   Groq Compound
```

### Data Flow

```
candidates.csv
    │
    ├─ normalize_csv_columns()     ← alias "github"→"github_url", "resume"→"resume_url"
    ├─ insert_candidates(df)       ← dedup on s_no (NOT email)
    │
    ▼  status: uploaded
Stage 1 — resume_parser.py
    ├─ drive_to_direct_url()       ← extract file ID from share URL
    ├─ download_pdf()              ← with Drive virus-scan confirm handling
    ├─ extract_text_pdfplumber()   ← primary
    └─ extract_text_pymupdf()      ← fallback for scanned/image PDFs
    ▼  status: resume_parsed | resume_failed

Stage 2 — github_analyzer.py
    ├─ GET /users/{username}/repos (paginated, all public repos)
    ├─ GET /repos/{owner}/{repo}/commits?since=6mo  (per-repo commit count)
    └─ WSM sub-score:
         activity_recency     (w=0.35) ← days since last push, decaying
         commit_frequency     (w=0.25) ← 6mo commits, capped at 100
         original_repo_ratio  (w=0.20) ← non-forks / total repos
         documentation_quality(w=0.20) ← README presence + size bonus
       blank github_url → {github_score: None} → weight renormalized in scorer
    ▼  status: github_analyzed | github_failed

Stage 3 — ai_evaluator.py
    ├─ MiniLM-L6-v2 (local, cached)   ← embed JD + candidate text → cosine similarity
    ├─ Groq compound (API)             ← system prompt + structured JSON output:
    │    {"jd_match": 0-100, "project_quality": 0-100, "reasoning": "..."}
    └─ Fallback: if Groq fails → embedding_sim used as jd_match proxy
    ▼  status: ai_scored

Stage 4 — archetype_clustering.py
    ├─ load_or_fit_clustering()
    │    ├─ If model_cache/ exists → load scaler + kmeans + label_map (joblib)
    │    └─ Else → generate 800 synthetic profiles (200/archetype) →
    │              StandardScaler.fit() → KMeans(n_clusters=4, n_init=10).fit()
    │              → majority-vote label_map → persist to model_cache/
    ├─ 4 Archetypes (fit on synthetic populations):
    │    🟡 Research-Deep   high cgpa, strong JD match, lower GitHub activity
    │    🟢 Builder         high GitHub score, strong projects, moderate JD match
    │    🔵 Fast-Learner    lower scores across all signals, high growth potential
    │    🟣 All-Rounder     balanced profile across all 4 dimensions
    ├─ assign_candidate_archetype()
    │    ├─ Impute missing github_score with synthetic population mean
    │    ├─ StandardScaler.transform()
    │    ├─ KMeans.predict() → cluster_id → archetype name
    │    └─ Distance to all 4 centroids → confidence score + breakdown bars
    └─ Stores: archetype, archetype_confidence, archetype_breakdown in scores table
    ▼  status: ai_scored (archetype data written to scores table)

Stage 5 — scorer.py
    ├─ get_active_weights(weights, has_test, has_github)
    │    ← zeros w3/w4 if signal missing, renormalizes remainder to 1.0
    ├─ final_score = w1×jd_match + w2×project_quality + w3×github_score + w4×test_score
    └─ RANK() by final_score DESC, written back to DB
    ▼  status: ranked

emailer.py → Gmail SMTP HTML email
    Subject: "Assessment Link — {name} (s_no {n})"
    ▼  status: test_sent

test_results.csv (uploaded, JOIN on s_no)
    └─ test_score = (test_la + test_code) / 2
    └─ scorer.score_all() re-runs with all 4 signals active
    ▼  status: shortlisted

calendar_scheduler.py
    └─ Calendar events.insert() + conferenceData requestId
    └─ Google auto-generates Meet link, sends own invite to attendee
    ▼  status: interview_scheduled
```

### Database Schema

```sql
-- Primary key: s_no (candidate number), NOT email
-- Multiple candidates can share the same email (recruiter forwarding inbox)

CREATE TABLE candidates (
    s_no            INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    email           TEXT,
    college         TEXT,
    branch          TEXT,
    cgpa            REAL,
    best_ai_project TEXT,
    research_work   TEXT,
    github_url      TEXT,        -- NULL handled: weight renormalized in scorer
    resume_url      TEXT,        -- Google Drive sharing link
    resume_text     TEXT,        -- extracted by pdfplumber/PyMuPDF
    status          TEXT DEFAULT 'uploaded',
    error_notes     TEXT,
    created_at      TEXT
);

CREATE TABLE scores (
    s_no                  INTEGER PRIMARY KEY REFERENCES candidates(s_no),
    embedding_sim         REAL,   -- MiniLM cosine similarity ×100 (0–100)
    jd_match              REAL,   -- Groq LLM score (0–100)
    project_quality       REAL,   -- Groq LLM score (0–100)
    github_score          REAL,   -- GitHub WSM sub-score (0–100), NULL if no URL
    test_la               REAL,   -- logical aptitude (uploaded)
    test_code             REAL,   -- coding score (uploaded)
    test_score            REAL,   -- (test_la + test_code) / 2
    final_score           REAL,   -- WSM weighted sum
    rank                  INTEGER,
    llm_reasoning         TEXT,   -- Groq explanation string
    github_breakdown      TEXT,   -- JSON: per-repo stats + WSM sub-scores
    archetype             TEXT,   -- one of: Research-Deep, Builder, Fast-Learner, All-Rounder
    archetype_confidence  REAL,   -- distance-based confidence (0–100)
    archetype_breakdown   TEXT    -- JSON: distance to each centroid
);

CREATE TABLE interview_events (
    s_no              INTEGER PRIMARY KEY REFERENCES candidates(s_no),
    calendar_event_id TEXT,
    meet_link         TEXT,
    scheduled_time    TEXT,
    invite_sent       INTEGER DEFAULT 0
);
```

---

## 🧠 Scoring & Clustering Methodology

### Why Weighted Sum Model (MCDA)?

1. **No historical outcome labels exist** — no "hired/performed well" column to train a classifier on. SHAP/LIME don't apply here; those explain a *trained model's learned weights*, not a formula with recruiter-defined weights.
2. **Direct brief alignment** — the 4 WSM terms map 1:1 to stated criteria: Resume quality, GitHub activity, JD relevance, Test performance.
3. **Maximum auditability** — every sub-score, weight, and LLM reasoning string is stored in SQLite and displayed in the dashboard.

### WSM Formula

```
final_score = w1 × jd_match + w2 × project_quality + w3 × github_score + w4 × test_score
```

| Weight | Default | Signal | Source |
|---|---|---|---|
| w1 | 0.30 | JD Match | MiniLM cosine similarity + Groq LLM |
| w2 | 0.25 | Project Quality | Groq structured JSON |
| w3 | 0.25 | GitHub Score | GitHub REST API → 4-signal WSM |
| w4 | 0.20 | Test Score | Uploaded CSV |

Weights are **recruiter-adjustable via sidebar sliders** and auto-normalize to 1.0.

### Missing-Signal Renormalization

```python
def get_active_weights(weights, has_test, has_github):
    active = dict(weights)
    if not has_test:   active['w4'] = 0.0
    if not has_github: active['w3'] = 0.0
    remaining = sum(active.values())
    return {k: v/remaining if remaining > 0 else 0.0 for k, v in active.items()}
```

A candidate without GitHub is still scored 0–100 on 3 signals. A candidate without test results is scored on 3 signals. Both missing → scored on 2 signals. The final score never gets an artificial ceiling.

### K-Means Archetype Clustering

Implemented in `modules/archetype_clustering.py` using **scikit-learn**.

The clustering is **descriptive, not prescriptive** — it answers "what kind of candidate is this?" as a complement to WSM's "how highly do they rank?" The two answer different questions and neither overrides the other.

**Features used:** `jd_match`, `project_quality`, `github_score`, `cgpa` (4-dimensional)

**Training data:** 800 synthetic candidate profiles (200 per archetype), sampled from Gaussian distributions with deliberately separated centroids so K-means finds real structure.

```python
# 4 Archetypes and their synthetic population parameters
ARCHETYPES = {
    "Research-Deep":  {"jd_match": (68,7), "project_quality": (72,6), "github_score": (40,10), "cgpa": (9.0,0.3)},
    "Builder":        {"jd_match": (55,8), "project_quality": (78,7), "github_score": (90, 5), "cgpa": (6.8,0.5)},
    "Fast-Learner":   {"jd_match": (48,8), "project_quality": (48,8), "github_score": (35,12), "cgpa": (7.2,0.6)},
    "All-Rounder":    {"jd_match": (75,6), "project_quality": (70,6), "github_score": (72, 7), "cgpa": (8.2,0.4)},
}
```

**Pipeline:**
1. `StandardScaler.fit()` on synthetic data
2. `KMeans(n_clusters=4, n_init=10, random_state=42).fit()` on scaled data
3. Majority-vote cluster → archetype label mapping
4. Model persisted to `model_cache/` (joblib) — re-loaded on every app start
5. Real candidates: impute missing `github_score` with synthetic population mean → scale → `predict()`
6. Distance to all 4 centroids stored as confidence breakdown bars in the UI

**Missing `github_score` handling:** Imputed with the synthetic population mean before scaling (explicitly documented as a limitation in the module docstring).

---

## 📁 Project Structure

```
candidate-screening-platform/
├── app.py                          # Main Streamlit app (~1000+ lines, 5 tabs)
├── db.py                           # SQLite schema + CRUD + CSV normalization
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py            # Drive PDF download + pdfplumber/PyMuPDF
│   ├── github_analyzer.py          # GitHub REST API + WSM sub-scorer
│   ├── ai_evaluator.py             # MiniLM-L6-v2 + Groq LLM evaluator
│   ├── archetype_clustering.py     # K-means archetype clustering (sklearn)
│   ├── scorer.py                   # WSM final scorer + renormalization
│   ├── emailer.py                  # Gmail SMTP HTML mailer
│   └── calendar_scheduler.py       # Google Calendar + Meet scheduler
├── tests/
│   ├── __init__.py
│   └── test_archetype_clustering.py  # Unit tests for clustering module
├── sample_data/
│   ├── candidates.csv              # 10-row demo CSV
│   ├── test_results.csv            # Demo test scores
│   └── synthetic_archetypes.csv    # 800-row synthetic training data (auto-generated)
├── model_cache/
│   ├── archetype_kmeans.joblib     # Persisted KMeans model
│   ├── archetype_scaler.joblib     # Persisted StandardScaler
│   └── archetype_label_map.joblib  # Cluster-ID → archetype-name map
├── docs/screenshots/               # UI screenshots embedded in this README
├── .streamlit/
│   ├── secrets.toml.example        # Template — copy and fill in your keys
│   └── secrets.toml                # ← git-ignored (NEVER committed)
├── .devcontainer/
│   └── devcontainer.json           # VS Code Dev Container config
├── requirements.txt
├── README.md
└── USER_GUIDE.md                   # Step-by-step recruiter operating guide
```

---

## ⚙️ Local Setup

### 1. Clone & Install

```bash
git clone https://github.com/dummycodertech/mynachiketa_assessment.git
cd mynachiketa_assessment
pip install -r requirements.txt
```

### 2. Configure Secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and fill in your credentials
```

```toml
GROQ_API_KEY             = "gsk_..."
GITHUB_PAT               = "ghp_..."
GMAIL_ADDRESS            = "you@gmail.com"
GMAIL_APP_PASSWORD       = "xxxx xxxx xxxx xxxx"
GOOGLE_OAUTH_CLIENT_JSON = ""    # generated separately — see below
RECRUITER_EMAIL          = "you@gmail.com"
```

### 3. Required API Keys

| Key | Where to Get It | Time |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys | 2 min |
| `GITHUB_PAT` | [github.com/settings/tokens](https://github.com/settings/tokens) → Classic → `public_repo` scope | 1 min |
| `GMAIL_APP_PASSWORD` | [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) (requires 2FA) | 3 min |
| `GOOGLE_OAUTH_CLIENT_JSON` | Google Cloud Console → Calendar API → OAuth Desktop → base64 token | 10 min |

### 4. Google Calendar OAuth (One-Time)

```bash
# After downloading your OAuth client_secret.json from Google Cloud Console:
python -c "
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
flow = InstalledAppFlow.from_client_secrets_file('client_secret.json',
    scopes=['https://www.googleapis.com/auth/calendar.events'])
creds = flow.run_local_server(port=0)
print(base64.b64encode(creds.to_json().encode()).decode())
"
# Paste the printed base64 string into GOOGLE_OAUTH_CLIENT_JSON in secrets.toml
```

Full instructions: **[USER_GUIDE.md](USER_GUIDE.md)**

### 5. Run

```bash
streamlit run app.py
```

The K-means model fits automatically on first run (~1 second) and caches to `model_cache/`. Subsequent starts load from cache.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo → `app.py`
3. **Advanced settings → Secrets**: paste full contents of your `secrets.toml`
4. Click **Deploy** → free public URL

> ⚠️ `secrets.toml` is git-ignored and never committed. Paste secrets directly into Streamlit Cloud dashboard.

---

## 📊 Input CSV Format

### `candidates.csv`

| Column | Required | Notes |
|---|---|---|
| `s_no` | ✅ | Unique integer — primary key, may be non-contiguous |
| `name` | ✅ | Full name |
| `email` | ✅ | Test link destination |
| `college` | ✅ | Institution |
| `branch` | ✅ | Department / field |
| `cgpa` | ✅ | GPA — used as 4th feature in K-means clustering |
| `best_ai_project` | ✅ | Key signal for LLM evaluation |
| `research_work` | ✅ | Research / publications |
| `github` / `github_url` | optional | Profile URL — blank = w3 renormalized, imputed mean for clustering |
| `resume` / `resume_url` | ✅ | Google Drive sharing link to PDF |

### `test_results.csv`

| Column | Notes |
|---|---|
| `s_no` | Joined on this — NOT email |
| `test_la` | Logical aptitude (0–100) |
| `test_code` | Coding score (0–100) |

---

## 🧪 Tests

```bash
pytest tests/
```

Includes unit tests for:
- Archetype clustering: synthetic data generation, KMeans fit, label mapping, candidate assignment
- Scorer: WSM formula correctness, renormalization edge cases (no test, no GitHub, both missing)

---

## 💰 Total Cost: $0/month

| Service | Usage | Cost |
|---|---|---|
| Streamlit Community Cloud | Hosting | **$0** |
| GitHub REST API | 5000 req/hr with PAT | **$0** |
| HuggingFace MiniLM-L6-v2 | Local inference | **$0** |
| Groq API (compound model) | Free tier | **$0** |
| Gmail SMTP | App password auth | **$0** |
| Google Calendar API | Personal OAuth | **$0** |
| scikit-learn KMeans | Local, in-process | **$0** |
| SQLite | Single file, stdlib | **$0** |

---

## 🔮 Production Upgrade Path

| Component | Current (Demo) | Production |
|---|---|---|
| Database | SQLite | Supabase Postgres |
| Processing | Synchronous Streamlit | Celery + Redis job queue |
| LLM | Groq free tier | GPT-4o / Claude 3.5 Sonnet |
| Clustering | Synthetic training data | Fit on historical hire outcomes when available |
| Auth | Single recruiter | Supabase Auth / Clerk (multi-recruiter) |
| Secrets | Streamlit secrets | AWS Secrets Manager |

---

## 🔒 Security

- `secrets.toml` is in `.gitignore` — never committed to this repo
- Google OAuth token stored as base64 in Streamlit Cloud secrets — not in code
- All candidate data stays in local SQLite — only resume text + JD sent to Groq API

---

## 📄 License

MIT — free to use, modify, and distribute.

# 🎯 AI-Powered Candidate Screening Platform
### myNachiketa GTM Engineering Intern Assignment

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Compound-F55036)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Cost](https://img.shields.io/badge/Infrastructure%20Cost-%240-brightgreen)](README.md)

> A fully automated, end-to-end recruiter pipeline that evaluates AI/ML candidates using resume parsing, GitHub repository analysis, LLM reasoning, and test performance — then schedules interviews via Google Calendar. **Total infrastructure cost: $0.**

---

## 📸 Screenshots

| Upload & Configure | Rankings Dashboard | Interview Scheduling |
|---|---|---|
| Upload candidate CSV + paste JD | AI-ranked candidates with scores & reasoning | Auto-schedule Google Meet interviews |

---

## 🚀 Live Pipeline

```
Upload CSV → Parse Resumes → Analyze GitHub → AI Evaluate → Score & Rank
     → Send Test Links → Upload Results → Re-rank → Schedule Interviews
```

Every stage is **automatic** once triggered. The recruiter clicks buttons — no manual work.

---

## ✨ Features

| Stage | Technology | What Happens |
|---|---|---|
| **CSV Upload** | pandas + SQLite | Candidate data normalized and stored |
| **Resume Parsing** | pdfplumber + PyMuPDF | Google Drive PDFs downloaded and text extracted |
| **GitHub Analysis** | GitHub REST API | Per-repo scoring: activity, commits, README quality, originality |
| **AI Evaluation** | MiniLM + Groq LLM | Semantic similarity + structured JSON reasoning from Llama |
| **WSM Scoring** | MCDA/WSM | Weighted Sum Model with automatic weight renormalization |
| **Email Test Links** | Gmail SMTP | Branded HTML email with name + s_no in subject |
| **Test Results** | SQLite join on s_no | Merge test scores, re-rank with all 4 signals |
| **Interview Scheduling** | Google Calendar API | Creates events with Meet links, sends invite emails automatically |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────── Streamlit App (single process) ───────────────────────────┐
│                                                                                        │
│  UI Layer (5 Tabs)                                                                     │
│  ├─ Tab 1: Upload candidates CSV + paste Job Description                               │
│  ├─ Tab 2: Run Pipeline (resume → GitHub → AI → score) with live status              │
│  ├─ Tab 3: Rankings dashboard (scores, AI reasoning, per-repo GitHub breakdown)        │
│  ├─ Tab 4: Upload test results → merge → re-rank → mark shortlist                     │
│  └─ Tab 5: Schedule Google Calendar interviews with Meet links                         │
│                                                                                        │
│  ┌──────────────────────────── modules/ ────────────────────────────────────────────┐ │
│  │ resume_parser.py      Drive URL → PDF bytes → pdfplumber/PyMuPDF → text          │ │
│  │ github_analyzer.py    GitHub REST API → per-repo metrics → WSM sub-score         │ │
│  │ ai_evaluator.py       MiniLM embeddings (cosine sim) + Groq LLM (JSON)           │ │
│  │ scorer.py             WSM final_score, weight renorm, rank assignment             │ │
│  │ emailer.py            Gmail SMTP → branded HTML email, name+s_no subject         │ │
│  │ calendar_scheduler.py Google Calendar API → event + Meet link + invite            │ │
│  └──────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                        │
│  db.py → SQLite (candidates, scores, interview_events tables)                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
          ↑ free            ↑ free/5000rph      ↑ free/local      ↑ free tier
    Streamlit Cloud     GitHub REST API      HF MiniLM-L6-v2   Groq (compound)
```

### Data Flow

```
candidates.csv (uploaded)
        │
        ▼
db.py: normalize_csv_columns()          ← alias shorthand headers (github→github_url)
        │ insert_candidates(df)          ← dedup on s_no (not email)
        ▼
resume_parser.py: parse_all()
        │ Drive share URL → direct download URL
        │ download_pdf() → pdfplumber (primary) / PyMuPDF (fallback)
        │ status: uploaded → resume_parsed / resume_failed
        ▼
github_analyzer.py: analyze_all()
        │ GitHub REST API: GET /users/{username}/repos
        │ Per repo: language, stars, last push, commit count (6mo), README size, fork
        │ WSM sub-score: activity_recency + commit_frequency + original_ratio + docs
        │ Blank github_url → {"github_score": None, "error": "no_github_url"}
        │ status: → github_analyzed / github_failed
        ▼
ai_evaluator.py: evaluate_all()
        │ MiniLM-L6-v2: embed JD + candidate text → cosine similarity → embedding_sim
        │ Groq compound: structured JSON → {"jd_match": X, "project_quality": Y, "reasoning": "..."}
        │ Rate limit: 1s delay between calls (free tier ~30 RPM)
        │ Fallback: if Groq fails → use embedding_sim as jd_match proxy
        │ status: → ai_scored
        ▼
scorer.py: score_all()
        │ get_active_weights(weights, has_test, has_github)
        │   → zeros out w3/w4 if signal missing, renormalizes remainder to 1.0
        │ final_score = w1*jd_match + w2*project_quality + w3*github_score + w4*test_score
        │ RANK() by final_score DESC
        │ status: → ranked
        ▼
[RECRUITER SENDS TEST LINKS via Gmail SMTP]
        │ emailer.py: send_test_links()
        │ Subject: "Assessment Link — {name} (s_no {n})"  ← disambiguated
        │ status: → test_sent
        ▼
test_results.csv (uploaded)
        │ db.merge_test_results() — JOIN on s_no (not email)
        │ test_score = (test_la + test_code) / 2
        │ scorer.score_all() re-runs with all 4 signals active
        │ Recruiter clicks "Mark as Shortlisted"
        │ status: → shortlisted
        ▼
calendar_scheduler.py: schedule_all()
        │ Google Calendar API: events.insert() with conferenceData
        │ → Creates Google Meet link automatically
        │ → Google sends its own invite email to all attendees
        │ status: → interview_scheduled
```

### Database Schema

```sql
-- candidates table
CREATE TABLE candidates (
    s_no          INTEGER PRIMARY KEY,  -- NOT email — multiple candidates share recruiter email
    name          TEXT,
    email         TEXT,
    college       TEXT,
    branch        TEXT,
    cgpa          REAL,
    best_ai_project TEXT,
    research_work TEXT,
    github_url    TEXT,
    resume_url    TEXT,
    resume_text   TEXT,                 -- extracted from PDF
    status        TEXT DEFAULT 'uploaded',
    error_notes   TEXT,
    created_at    TEXT
);

-- scores table (1:1 with candidates)
CREATE TABLE scores (
    s_no            INTEGER PRIMARY KEY REFERENCES candidates(s_no),
    embedding_sim   REAL,   -- MiniLM cosine similarity (0–100)
    jd_match        REAL,   -- Groq LLM score (0–100)
    project_quality REAL,   -- Groq LLM score (0–100)
    github_score    REAL,   -- GitHub WSM sub-score (0–100), NULL if no URL
    test_la         REAL,   -- logical aptitude (uploaded)
    test_code       REAL,   -- coding score (uploaded)
    test_score      REAL,   -- (test_la + test_code) / 2
    final_score     REAL,   -- WSM weighted sum
    rank            INTEGER,
    llm_reasoning   TEXT,   -- Groq reasoning string
    github_breakdown TEXT   -- JSON: per-repo data + sub-scores
);

-- interview_events table
CREATE TABLE interview_events (
    s_no              INTEGER PRIMARY KEY REFERENCES candidates(s_no),
    calendar_event_id TEXT,
    meet_link         TEXT,
    scheduled_time    TEXT,
    invite_sent       INTEGER DEFAULT 0
);
```

---

## 🧠 Scoring Methodology: Weighted Sum Model (MCDA)

### Why WSM?

The scoring approach is a **Weighted Sum Model (WSM)** — the simplest form of Multi-Criteria Decision Analysis. This was the methodologically correct choice:

**No historical outcome labels exist.** There is no "hired / performed well" column to train a classifier on. Training one would require inventing fake labels — indefensible. This also means SHAP/LIME explanations don't apply — those explain a *trained model's learned weights*. There is no trained model.

**The four terms map 1:1 to the brief's stated criteria.** The brief says evaluate using: Resume, GitHub, JD relevance, Test performance. That's exactly the four WSM terms.

**It is maximally auditable.** Every sub-score, weight, and reasoning string is stored in SQLite and displayed in the dashboard — satisfying the "Explainable AI" requirement at the formula level.

### Formula

```
final_score = w1 × jd_match + w2 × project_quality + w3 × github_score + w4 × test_score
```

**Default weights:**

| Weight | Default | Signal | Source |
|---|---|---|---|
| w1 | 0.30 | JD Match | MiniLM cosine sim + Groq LLM |
| w2 | 0.25 | Project Quality | Groq Llama structured JSON |
| w3 | 0.25 | GitHub Score | GitHub REST API per-repo WSM |
| w4 | 0.20 | Test Score | Uploaded CSV |

Weights are **recruiter-adjustable via sidebar sliders** and auto-renormalize to sum to 1.0.

### Graceful Degradation (Missing Signals)

```python
def get_active_weights(weights, has_test, has_github):
    active = dict(weights)
    if not has_test:   active['w4'] = 0
    if not has_github: active['w3'] = 0
    remaining = sum(active.values())
    return {k: v/remaining if remaining > 0 else 0 for k, v in active.items()}
```

A candidate without GitHub is scored on 3 signals, still normalized to 0–100. Their score is not unfairly capped at 75 just because one signal is missing. This prevents blank GitHub fields from silently destroying good candidates.

### GitHub Sub-Scoring

```
github_score = 0.35 × activity_recency
             + 0.25 × commit_frequency (6 months)
             + 0.20 × original_repo_ratio
             + 0.20 × documentation_quality (README presence + size)
```

Computed from real REST API calls — not profile stats. Every repo's data is stored as JSON and displayed in the rankings breakdown table.

---

## 📁 Project Structure

```
candidate-screening-platform/
├── app.py                          # Main Streamlit app — 5 tabs, all UI
├── db.py                           # SQLite schema, CRUD, CSV normalization
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py            # Drive PDF download + pdfplumber/PyMuPDF
│   ├── github_analyzer.py          # GitHub REST API + WSM sub-scorer
│   ├── ai_evaluator.py             # MiniLM + Groq LLM evaluator
│   ├── scorer.py                   # WSM final scorer + weight renormalization
│   ├── emailer.py                  # Gmail SMTP email sender
│   └── calendar_scheduler.py       # Google Calendar + Meet scheduler
├── sample_data/
│   ├── candidates.csv              # Demo candidate CSV (10 candidates)
│   └── test_results.csv            # Demo test results (s_no, test_la, test_code)
├── .streamlit/
│   ├── secrets.toml.example        # Template — copy and fill in your keys
│   └── secrets.toml                # ← NEVER commit this (in .gitignore)
├── requirements.txt                # All Python dependencies
├── README.md                       # This file
├── ARCHITECTURE.md                 # Deep-dive architecture doc
├── USER_GUIDE.md                   # Step-by-step recruiter guide
└── .gitignore
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
# Fill in your credentials — see below
```

### 3. Get Your API Keys

| Key | Source | Time |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys | 2 min |
| `GITHUB_PAT` | [github.com/settings/tokens](https://github.com/settings/tokens) → Classic token, `public_repo` scope | 1 min |
| `GMAIL_ADDRESS` | Your Gmail address | — |
| `GMAIL_APP_PASSWORD` | [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) (requires 2FA enabled) | 3 min |
| `GOOGLE_OAUTH_CLIENT_JSON` | See `USER_GUIDE.md` → Google Calendar Setup section | 10 min |
| `RECRUITER_EMAIL` | Same as your Gmail address | — |

### 4. Run

```bash
streamlit run app.py
```


## 📊 CSV Format

### `candidates.csv`

| Column | Required | Notes |
|---|---|---|
| `s_no` | ✅ | Primary key — must be unique, may be non-contiguous |
| `name` | ✅ | Candidate full name |
| `email` | ✅ | Used for sending test links |
| `college` | ✅ | Institution name |
| `branch` | ✅ | Department / field of study |
| `cgpa` | ✅ | GPA (used in AI context) |
| `best_ai_project` | ✅ | Project description — key AI signal |
| `research_work` | ✅ | Research/publication history |
| `github` / `github_url` | optional | GitHub profile URL — blank = score renormalized |
| `resume` / `resume_url` | ✅ | Google Drive sharing link to PDF resume |

> Column aliases `github` and `resume` are automatically mapped to `github_url` and `resume_url` — both shorthand and long-form headers work.

### `test_results.csv`

| Column | Notes |
|---|---|
| `s_no` | Must match existing candidate IDs — joined on this, NOT email |
| `test_la` | Logical aptitude score (0–100) |
| `test_code` | Coding score (0–100) |

---

## 🔒 Security Notes

- `secrets.toml` is in `.gitignore` — it is **never committed**.
- The Google OAuth token is stored base64-encoded in Streamlit secrets — never in code.
- The Gmail app password is separate from your Google account password.
- For production: migrate secrets to AWS Secrets Manager or GCP Secret Manager.

---

## 🔮 Production Upgrade Path

| Component | Demo (this repo) | Production |
|---|---|---|
| Database | SQLite | Supabase Postgres (free tier available) |
| Processing | Synchronous Streamlit | Celery + Redis job queue |
| LLM | Groq free tier | GPT-4o / Claude 3.5 Sonnet (paid) |
| Auth | Single recruiter | Supabase Auth / Clerk (multi-recruiter) |
| Secrets | Streamlit secrets | AWS Secrets Manager |
| Scale | ~10–50 candidates | Async processing, rate-limit pooling |

---

## 💰 Cost Summary

| Service | Usage | Cost |
|---|---|---|
| Streamlit Community Cloud | Hosting | **$0** |
| GitHub REST API | 5000 req/hr with PAT | **$0** |
| HuggingFace MiniLM-L6-v2 | Local inference | **$0** |
| Groq (compound model) | Free tier | **$0** |
| Gmail SMTP | App password | **$0** |
| Google Calendar API | Personal OAuth | **$0** |
| SQLite | Local file | **$0** |
| **Total** | | **$0** |

---

## 🛠️ Tech Stack

- **Frontend/App:** Streamlit
- **Database:** SQLite via `sqlite3` (stdlib)
- **Resume Parsing:** pdfplumber + PyMuPDF
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2, local)
- **LLM:** Groq API (`groq/compound`)
- **GitHub:** REST API v3 (`requests`)
- **Email:** `smtplib` + Gmail SMTP
- **Calendar:** `google-api-python-client` + `google-auth-oauthlib`
- **Data:** pandas + pytz



*Built for the myNachiketa GTM Engineering Intern Assignment*

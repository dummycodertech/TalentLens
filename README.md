# 🎯 TalentLens: Autonomous AI Candidate Screening & Technical Recruiter Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Compound-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![scikit-learn](https://img.shields.io/badge/Clustering-scikit--learn%20KMeans-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![GitHub API](https://img.shields.io/badge/GitHub-REST%20API%20v3-181717?logo=github&logoColor=white)](https://docs.github.com/en/rest)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20WAL-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-22c55e?logo=pytest&logoColor=white)](#10-testing--quality)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Infrastructure Cost](https://img.shields.io/badge/Infra%20Cost-%240%2Fmonth-brightgreen)](#17-performance--cost)

> **TalentLens** is a fully automated, end-to-end technical recruitment pipeline that evaluates software and AI candidates with multi-modal signal extraction, transparent multi-criteria decision analysis (MCDA), unsupervised machine learning clustering, and automated interview logistics.
>
> Upload a candidate applicant CSV and paste any job description. TalentLens downloads and parses PDF resumes from Google Drive, queries the GitHub REST API for per-repository commit frequency and code quality, performs two-signal semantic evaluation via sentence embeddings and Groq LLM structured JSON reasoning, classifies candidates into 4 unsupervised K-means behavioral archetypes, computes normalized auditable rankings, dispatches assessment invitation emails via SMTP, and auto-provisions Google Calendar events with Google Meet video links. **Total operational infrastructure cost: $0.00/month.**

---

### Quick Links
- **Codebase Repository**: [github.com/dummycodertech/TalentLens](https://github.com/dummycodertech/TalentLens)
- **Recruiter Operating Manual**: [USER_GUIDE.md](USER_GUIDE.md)
- **Technical Architecture Document**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Live Streamlit Deployment**: [VERIFY: Add hosted Streamlit Cloud URL if active; e.g., https://talentlens.streamlit.app]

---

## 📸 Application Interface

<table>
  <tr>
    <td width="50%" align="center"><b>Stage 1: Candidate Ingestion & JD Configuration</b></td>
    <td width="50%" align="center"><b>Stage 2: 5-Stage Autonomous Execution Engine</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/tab1_upload.jpg" alt="Tab 1 - Ingestion and JD Configuration" width="100%"/></td>
    <td><img src="docs/screenshots/tab2_pipeline.jpg" alt="Tab 2 - Real-Time Pipeline Execution" width="100%"/></td>
  </tr>
  <tr>
    <td width="50%" align="center"><b>Stage 3: Explainable Rankings & K-Means Archetypes</b></td>
    <td width="50%" align="center"><b>Stage 5: Google Calendar & Meet Auto-Provisioning</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/tab3_rankings.jpg" alt="Tab 3 - AI Rankings and Centroid Distances" width="100%"/></td>
    <td><img src="docs/screenshots/tab5_interviews.jpg" alt="Tab 5 - Interview Scheduling with Google Meet" width="100%"/></td>
  </tr>
</table>

---

## 2. Project at a Glance

| Dimension | Specification |
|---|---|
| **Problem** | Technical screening at high volume suffers from black-box ATS keyword filtering, unverified resume claims, manual GitHub inspection overhead, opaque scoring, and fragmented interview scheduling. |
| **Solution** | A deterministic 5-stage screening pipeline combining PDF extraction, per-repo GitHub commit analysis, dual-signal embedding + LLM qualitative evaluation, K-means archetype clustering, and automated Calendar/Meet scheduling. |
| **Target User** | Technical Recruiters, Hiring Managers, Talent Acquisition Leads, and Engineering Founders hiring technical interns and engineers. |
| **Project Type** | Full-Stack AI Recruitment Agent, ML Clustering Pipeline & Streamlit Web Application. |
| **Current Status** | Production-Ready Demo / Complete Reference Implementation (21/21 Unit Tests Passing). |
| **Deployment** | Local Python environment / Streamlit Community Cloud (Dev Container support included). |
| **Core Tech Stack** | Python 3.10+, Streamlit 1.35+, Groq API (`groq/compound` / LLaMA 3.3 70B), Sentence-Transformers (`all-MiniLM-L6-v2`), Scikit-Learn 1.4+, SQLite3 WAL, GitHub REST API v3, Google Calendar API v3, Gmail SMTP. |
| **Author & Scope** | Solo architect & engineer: **Yagas Vashist** ([@dummycodertech](https://github.com/dummycodertech)). Built complete ingestion, parsing, REST clients, ML clustering, WSM scoring, UI, tests, and documentation. |
| **Development Period** | August 2026 – September 2026. |
| **Key Measured Results** | **Silhouette Score: 0.4494** across 800 synthetic profiles (4-dimensional feature space); **Adjusted Rand Index (ARI): 0.9669** ground-truth archetype recovery; **100% mathematical weight integrity** preserved across missing-signal degradation states; **21/21 passing unit tests**. |
| **Repository Links** | [Repository Root](https://github.com/dummycodertech/TalentLens) \| [User Guide](USER_GUIDE.md) \| [Architecture Spec](ARCHITECTURE.md) |

---

## 3. Table of Contents

- [1. Header & Overview](#-talentlens-autonomous-ai-candidate-screening--technical-recruiter-platform)
- [2. Project at a Glance](#2-project-at-a-glance)
- [3. Table of Contents](#3-table-of-contents)
- [4. Overview](#4-overview)
  - [The Problem](#the-problem)
  - [The Solution](#the-solution)
  - [Key Features](#key-features)
  - [System Scope](#system-scope)
- [5. Architecture](#5-architecture)
  - [High-Level System Flowchart](#high-level-system-flowchart)
  - [End-to-End Sequence Diagram](#end-to-end-sequence-diagram)
  - [Component Inventory](#component-inventory)
  - [Database Schema & State Machine](#database-schema--state-machine)
  - [Scoring Methodology: Multi-Criteria Decision Analysis (WSM)](#scoring-methodology-multi-criteria-decision-analysis-wsm)
  - [GitHub Repository Analytics Sub-Formula](#github-repository-analytics-sub-formula)
  - [Unsupervised Archetype Clustering (K-Means)](#unsupervised-archetype-clustering-k-means)
- [6. Tech Stack](#6-tech-stack)
- [7. Design Decisions & Trade-offs](#7-design-decisions--trade-offs)
- [8. Data & Models](#8-data--models)
- [9. Evaluation & Results](#9-evaluation--results)
- [10. Testing & Quality](#10-testing--quality)
- [11. Security, Privacy & Reliability](#11-security-privacy--reliability)
- [12. Getting Started](#12-getting-started)
- [13. Usage Guide](#13-usage-guide)
- [14. Configuration & Tunables](#14-configuration--tunables)
- [15. Project Structure](#15-project-structure)
- [16. Deployment](#16-deployment)
- [17. Performance & Cost](#17-performance--cost)
- [18. Roadmap & Future Scope](#18-roadmap--future-scope)
- [19. FAQ / Plain-English Explanation](#19-faq--how-it-works-in-plain-english)
- [20. Resume & Interview Kit](#20-resume--interview-kit)
- [21. Project Context Block (YAML)](#21-project-context-block)
- [22. License, Acknowledgements & Author](#22-references-acknowledgements-license-author)

---

## 4. Overview

### The Problem
Traditional early-stage candidate screening in tech hiring faces five critical bottlenecks:
1. **Keyword-Stuffed Resumes**: Traditional Applicant Tracking Systems (ATS) rely on simple string matching, which penalizes non-traditional candidates while rewarding candidates who copy-paste job description keywords without real depth.
2. **Unverified GitHub Claims**: Recruiters rarely have time to inspect public repositories. Candidates list GitHub handles with stale forks, empty repositories, or zero commits, yet receive equal credit.
3. **Opaque & Biased AI Scoring**: Standard LLM resume graders spit out black-box scores with hallucinated ratings and zero audit trails, preventing hiring managers from understanding why a candidate was rejected or advanced.
4. **Disjointed Assessment Workflows**: Sending assessment links, merging coding test scores, and re-ranking requires jumping between spreadsheets, email clients, and testing portals.
5. **Interview Scheduling Friction**: Coordinating calendar availability and generating video links across dozens of candidates wastes hours of manual recruiter effort.

### The Solution
**TalentLens** consolidates the entire technical recruitment lifecycle into a unified, transparent, and deterministic pipeline. It downloads candidate resumes directly from Google Drive, audits public GitHub profiles down to commit timestamps and non-fork ratios, measures JD alignment via dual embedding similarity and Groq LLM structured evaluation, assigns unsupervised behavioral archetypes via Scikit-Learn K-means, and automatically manages email outreach and Google Meet interview scheduling.

### Key Features

| Feature | What It Does | Implementation Location |
|---|---|---|
| **Resilient PDF Extraction** | Extracts resume text from Google Drive sharing URLs; handles virus-scan redirect tokens; falls back from `pdfplumber` to `PyMuPDF` for scanned PDFs. | [`modules/resume_parser.py`](modules/resume_parser.py) |
| **Deep GitHub Repository Audit** | Analyzes public repos via GitHub REST API v3; calculates commit activity in the last 6 months, original-to-fork ratio, recency of pushes, and README documentation depth. | [`modules/github_analyzer.py`](modules/github_analyzer.py) |
| **Dual-Signal AI Matching** | Combines local deterministic cosine similarity (`all-MiniLM-L6-v2`) with structured Groq LLM JSON reasoning (`groq/compound` / LLaMA 3.3 70B). | [`modules/ai_evaluator.py`](modules/ai_evaluator.py) |
| **Auditable MCDA Scoring** | Uses a Weighted Sum Model (WSM) across 4 dimensions; features dynamic weight renormalization for missing signals (e.g., missing GitHub profile or pending test scores). | [`modules/scorer.py`](modules/scorer.py) |
| **Unsupervised Archetypes** | Fits K-Means ($k=4$) on standardized feature vectors (`jd_match`, `project_quality`, `github_score`, `cgpa`) to identify profile archetypes with distance-based softmax confidence. | [`modules/archetype_clustering.py`](modules/archetype_clustering.py) |
| **Automated SMTP Outreach** | Dispatches branded HTML technical assessment invitations via Gmail SMTP with candidate names and `s_no` identifiers in subject lines for recruiter inbox searchability. | [`modules/emailer.py`](modules/emailer.py) |
| **Assessment Merging & Re-Rank** | Ingests post-assessment scores (`test_la`, `test_code`), updates candidates in SQLite, and re-computes full rankings with all 4 signals active. | [`db.py`](db.py), [`modules/scorer.py`](modules/scorer.py) |
| **Google Calendar & Meet Provisioning** | Connects via OAuth2; creates non-overlapping interview blocks, auto-generates Google Meet links via `conferenceData`, and dispatches Google invite notifications. | [`modules/calendar_scheduler.py`](modules/calendar_scheduler.py) |

### System Scope
- **In Scope**: Automated PDF text extraction, REST-based GitHub repo analysis, semantic embeddings, qualitative LLM evaluation, K-means archetype clustering, WSM scoring with dynamic renormalization, SQLite persistence, SMTP emailing, Google Calendar/Meet API integration, and single-page Streamlit UI.
- **Out of Scope (By Design)**: Multi-tenant role-based enterprise access control (RBAC), distributed task queues (Celery/Redis), custom resume optical character recognition (OCR) model training, and automated audio/video interviewing.

---

## 5. Architecture

### High-Level System Flowchart

```mermaid
flowchart TD
    subgraph UI ["Streamlit Web Interface (app.py)"]
        T1["Tab 1: Ingestion & JD Setup"]
        T2["Tab 2: Pipeline Execution"]
        T3["Tab 3: Rankings & Archetypes"]
        T4["Tab 4: Test Upload & Re-rank"]
        T5["Tab 5: Calendar Scheduling"]
    end

    subgraph Data ["Local Storage & Cache"]
        SQL[("SQLite DB: screening.db\nWAL Mode")]
        ModelCache["model_cache/\nScaler, KMeans, LabelMap"]
    end

    subgraph Engine ["Processing Modules (modules/)"]
        RP["resume_parser.py\npdfplumber + PyMuPDF"]
        GH["github_analyzer.py\nREST API v3"]
        AI["ai_evaluator.py\nMiniLM + Groq LLM"]
        KM["archetype_clustering.py\nScikit-Learn KMeans k=4"]
        SC["scorer.py\nWSM + Renormalization"]
        EM["emailer.py\nGmail SMTP"]
        CS["calendar_scheduler.py\nGoogle Calendar API v3"]
    end

    subgraph External ["External Services & APIs"]
        GDrive["Google Drive\nPDF Storage"]
        GH_API["GitHub API\n5,000 req/hr"]
        Groq_API["Groq Cloud\nLLaMA 3.3 70B"]
        SMTP_Srv["smtp.gmail.com\nSSL Port 465"]
        GCal_API["Google Calendar API\nMeet Provisioning"]
    end

    T1 -->|"CSV Upload & JD Input"| SQL
    T2 -->|"Trigger Pipeline"| RP
    RP -->|"Fetch PDF Bytes"| GDrive
    RP -->|"Write Text / Scan Warning"| SQL
    RP --> GH
    GH -->|"Query Repos & Commits"| GH_API
    GH -->|"Store Repo Metrics & Subscore"| SQL
    GH --> AI
    AI -->|"Cosine Similarity"| AI
    AI -->|"JSON Structured Prompt"| Groq_API
    AI -->|"Store Scores & Reasoning"| SQL
    AI --> KM
    KM <-->|"Load/Persist Model"| ModelCache
    KM -->|"Compute Centroid Distance"| SQL
    KM --> SC
    SC -->|"Renormalize & Rank"| SQL
    SQL -->|"Render Cards & Badges"| T3
    T3 -->|"Trigger Assessment Invites"| EM
    EM -->|"Send HTML Emails"| SMTP_Srv
    T4 -->|"Merge test_la & test_code"| SQL
    SQL -->|"Re-run Scorer"| SC
    T5 -->|"OAuth2 Meet Events"| CS
    CS -->|"Insert Events & Attendees"| GCal_API
    CS -->|"Store Meet Link & Timeslot"| SQL
```

### End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Recruiter as Technical Recruiter
    participant App as Streamlit Dashboard
    participant DB as SQLite (screening.db)
    participant Drive as Google Drive
    participant GH as GitHub REST API
    participant Groq as Groq LLM API
    participant Mail as Gmail SMTP
    participant Cal as Google Calendar API

    Recruiter->>App: Upload candidates.csv and paste Job Description
    App->>DB: Normalize columns, deduplicate by s_no, insert rows
    Recruiter->>App: Click "Run Evaluation Pipeline"
    
    rect rgb(20, 25, 35)
        note right of App: Stage 1: Resume Extraction
        App->>Drive: Download PDF (Handle confirm token)
        Drive-->>App: PDF binary stream
        App->>App: Extract text (pdfplumber primary, PyMuPDF fallback)
        App->>DB: Update resume_text, status='resume_parsed'
    end

    rect rgb(20, 25, 35)
        note right of App: Stage 2: GitHub Repository Audit
        App->>GH: GET /users/{username}/repos
        GH-->>App: List of public repositories
        App->>GH: GET /repos/{owner}/{repo}/commits (since 6mo)
        GH-->>App: Commit history (capped at 300)
        App->>App: Compute 4-term GitHub WSM sub-formula
        App->>DB: Update github_score, github_breakdown, status='github_analyzed'
    end

    rect rgb(20, 25, 35)
        note right of App: Stage 3 & 4: AI Matching & Clustering
        App->>App: Local all-MiniLM-L6-v2 cosine similarity
        App->>Groq: POST /chat/completions (Prompt with JD + Profile)
        Groq-->>App: JSON: {jd_match, project_quality, reasoning}
        App->>App: StandardScaler transform + KMeans centroid projection
        App->>DB: Update scores, archetype, confidence, status='ai_scored'
    end

    rect rgb(20, 25, 35)
        note right of App: Stage 5: WSM Ranking & Logistics
        App->>App: Compute final_score with active weight renormalization
        App->>DB: Update rank, status='ranked'
        App-->>Recruiter: Display interactive candidate cards
        Recruiter->>App: Send Assessment links (threshold >= 60)
        App->>Mail: Send branded HTML test link email
        Recruiter->>App: Upload test_results.csv and Re-rank
        App->>DB: Merge test scores, re-run WSM, update shortlist
        Recruiter->>App: Schedule Interviews
        App->>Cal: Create event with conferenceData (HangoutsMeet)
        Cal-->>App: Return Google Meet link + calendar_event_id
        App->>Mail: Send Meet link confirmation email
        App->>DB: Update interview_events, status='interview_scheduled'
    end
```

### Component Inventory

| Component | Responsibility | File | Primary Inputs | Primary Outputs |
|---|---|---|---|---|
| **Database Layer** | Manages schema creation, migrations, column aliasing, and WAL transactions. | [`db.py`](db.py) | Raw DataFrames, status updates, query filters | SQLite tables (`candidates`, `scores`, `interview_events`) |
| **Resume Parser** | Resolves Drive share links, downloads PDFs, and extracts text. | [`modules/resume_parser.py`](modules/resume_parser.py) | `resume_url` (Google Drive URL or direct link) | Extracted text string, `scan_warning` boolean |
| **GitHub Analyzer** | Evaluates candidate GitHub repositories via REST API. | [`modules/github_analyzer.py`](modules/github_analyzer.py) | `github_url`, GitHub Personal Access Token (`GITHUB_PAT`) | `github_score` (0–100), JSON repository breakdown |
| **AI Evaluator** | Generates embedding similarity and LLM qualitative scoring. | [`modules/ai_evaluator.py`](modules/ai_evaluator.py) | Candidate profile dict, JD text, `GROQ_API_KEY` | `embedding_sim`, `jd_match`, `project_quality`, `reasoning` |
| **Archetype Clusterer** | Generates synthetic distributions and maps candidates to archetypes. | [`modules/archetype_clustering.py`](modules/archetype_clustering.py) | 4-dimensional score vector (`jd_match`, `project_quality`, `github_score`, `cgpa`) | Assigned archetype, distance to centroids, softmax confidence |
| **WSM Scorer** | Applies MCDA weighted sum formula and dynamically renormalizes missing weights. | [`modules/scorer.py`](modules/scorer.py) | Sub-scores from DB, recruiter slider weights | `final_score` (0–100), global `rank` (1..N) |
| **Email Dispatcher** | Constructs and delivers HTML assessment and interview emails. | [`modules/emailer.py`](modules/emailer.py) | Candidate details, assessment URL, meet link, Gmail App Password | Delivery confirmation dict (`success`, `error`) |
| **Calendar Scheduler**| Provisions Google Calendar events and Google Meet links via OAuth2. | [`modules/calendar_scheduler.py`](modules/calendar_scheduler.py) | Shortlisted candidate list, base64 OAuth JSON, start time, slot spacing | Calendar event ID, Google Meet link |

### Database Schema & State Machine

TalentLens stores all application state inside a local SQLite database (`screening.db`) operating with **Write-Ahead Logging (`PRAGMA journal_mode=WAL`)** for concurrency safety. The primary key across all tables is `s_no` (candidate sequential ID), deliberately avoiding `email` to allow multiple test applicants to share a common forwarding inbox without collision.

```sql
CREATE TABLE candidates (
    s_no            INTEGER PRIMARY KEY,
    name            TEXT,
    email           TEXT,
    college         TEXT,
    branch          TEXT,
    cgpa            REAL,
    best_ai_project TEXT,
    research_work   TEXT,
    github_url      TEXT,        -- NULL handled: weight renormalized in scorer
    resume_url      TEXT,        -- Google Drive sharing link
    resume_text     TEXT,        -- Extracted text from pdfplumber / PyMuPDF
    test_la         REAL,        -- Logical aptitude test score (0-100)
    test_code       REAL,        -- Coding challenge test score (0-100)
    status          TEXT DEFAULT 'uploaded',
    error_notes     TEXT
);

CREATE TABLE scores (
    s_no                  INTEGER PRIMARY KEY REFERENCES candidates(s_no),
    jd_match              REAL,   -- Groq LLM score (0-100)
    embedding_sim         REAL,   -- MiniLM cosine similarity (0-100)
    project_quality       REAL,   -- Groq LLM qualitative rating (0-100)
    github_score          REAL,   -- GitHub REST API 4-term sub-score (0-100, NULL if no URL)
    test_score            REAL,   -- Average of test_la and test_code
    final_score           REAL,   -- WSM weighted score (0-100)
    llm_reasoning         TEXT,   -- Structured explanation from Groq
    github_breakdown      TEXT,   -- JSON breakdown of repos, stars, commits
    rank                  INTEGER,-- Global position
    archetype             TEXT,   -- Research-Deep | Builder | Fast-Learner | All-Rounder
    archetype_confidence  REAL,   -- Softmax confidence score (0.0 to 1.0)
    archetype_breakdown   TEXT    -- JSON distance to all 4 centroids
);

CREATE TABLE interview_events (
    s_no               INTEGER PRIMARY KEY REFERENCES candidates(s_no),
    calendar_event_id  TEXT,
    meet_link          TEXT,
    scheduled_time     TEXT,
    invite_sent        INTEGER DEFAULT 0
);
```

#### Pipeline State Machine Transitions
```
uploaded ──► resume_parsed ──► github_analyzed ──► ai_scored ──► ranked
                 │                   │
                 ▼                   ▼
           resume_failed       github_failed
                 │
                 ▼
        (scan_warning flag)

ranked ──► test_sent ──► test_scored ──► shortlisted ──► interview_scheduled ──► invited
              │                                                │
              ▼                                                ▼
         email_failed                                  scheduling_failed
```

---

### Scoring Methodology: Multi-Criteria Decision Analysis (WSM)

#### Why a Weighted Sum Model?
In early-stage screening, **no ground-truth hiring outcome labels exist**. Training a supervised regression or classification model on historical application data without verified performance metrics creates arbitrary label hallucination. Furthermore, explainability techniques like SHAP or LIME are designed to explain the learned weights of complex black-box models; they are inapplicable when no trained supervised model exists.

TalentLens implements a **Weighted Sum Model (WSM)**, the established foundation of **Multi-Criteria Decision Analysis (MCDA)**. Every term maps directly to the four core pillars of technical assessment:
$$\text{Final Score} = w_1 \cdot \text{JD Match} + w_2 \cdot \text{Project Quality} + w_3 \cdot \text{GitHub Score} + w_4 \cdot \text{Test Score}$$

| Weight | Default Value | Criterion | Signal Generation Method |
|---|---|---|---|
| $w_1$ | **0.30** | Job Description Relevance | Dual: `all-MiniLM-L6-v2` cosine sim + Groq LLM semantic match |
| $w_2$ | **0.25** | Project & Research Depth | Groq LLaMA 3.3 70B qualitative evaluation of candidate projects |
| $w_3$ | **0.25** | GitHub Engineering Depth | GitHub REST API v3 per-repository 4-term sub-formula |
| $w_4$ | **0.20** | Technical Test Performance | Uploaded assessment CSV: $(\text{test\_la} + \text{test\_code}) / 2$ |

#### Dynamic Missing-Signal Renormalization
When a candidate lacks a GitHub URL, or when the initial screening occurs before test results are uploaded, naive weighted models artificially penalize candidates by capping their maximum achievable score (e.g., at 75% or 80%). TalentLens dynamically recalculates active weights so the final score **always scales from 0 to 100**:

```python
def get_active_weights(weights: dict, has_test: bool, has_github: bool) -> dict:
    active = dict(weights)
    if not has_test:
        active["w4"] = 0.0
    if not has_github:
        active["w3"] = 0.0
    remaining = sum(active.values())
    if remaining <= 0:
        return {k: 0.0 for k in active}
    return {k: v / remaining for k, v in active.items()}
```

*Example*: Prior to test upload ($w_4=0$), the active weights become $w_1 = 0.30/0.80 = 0.375$, $w_2 = 0.25/0.80 = 0.3125$, and $w_3 = 0.25/0.80 = 0.3125$, summing strictly to $1.0$.

---

### GitHub Repository Analytics Sub-Formula

Rather than relying on basic profile-level statistics (which can be distorted by stars on old forks), `modules/github_analyzer.py` inspects each repository individually:

$$\text{GitHub Score} = 100 \times \left( 0.35 \cdot S_{\text{recency}} + 0.25 \cdot S_{\text{commits}} + 0.20 \cdot S_{\text{original}} + 0.20 \cdot S_{\text{docs}} \right)$$

1. **Activity Recency ($S_{\text{recency}}$)**: Evaluates days since last push on original repositories:
   $$S_{\text{recency}} = \text{clamp}\left(\frac{365 - \text{days\_since\_push}}{365}, 0, 1\right)$$
2. **Commit Frequency ($S_{\text{commits}}$)**: Total commits across non-fork repositories over the last 6 months (capped at 100 commits):
   $$S_{\text{commits}} = \text{clamp}\left(\frac{\text{commits\_6mo}}{100}, 0, 1\right)$$
3. **Original Repository Ratio ($S_{\text{original}}$)**:
   $$S_{\text{original}} = \frac{N_{\text{original\_repos}}}{N_{\text{total\_repos}}}$$
4. **Documentation Quality ($S_{\text{docs}}$)**: Combines the presence of README files with README size:
   $$S_{\text{docs}} = 0.6 \cdot \frac{N_{\text{readme}}}{N_{\text{original}}} + 0.4 \cdot \text{clamp}\left(\frac{\text{avg\_readme\_bytes}}{2000}, 0, 1\right)$$

---

### Unsupervised Archetype Clustering (K-Means)

WSM answers: *"How highly does this candidate score?"* (a total ordering).  
Archetype clustering answers: *"What kind of candidate is this?"* (a profile shape classification).

Implemented in [`modules/archetype_clustering.py`](modules/archetype_clustering.py), TalentLens clusters candidates in 4-dimensional normalized space across `[jd_match, project_quality, github_score, cgpa]`:

```
           Research-Deep (🟡)             │             All-Rounder (🟣)
  High CGPA (9.0), High Project (72)      │    Balanced High Across All 4 Dimensions
  Academic focus, Lower GitHub (40)       │    (JD: 75, Project: 70, GH: 72, CGPA: 8.2)
──────────────────────────────────────────┼───────────────────────────────────────────
            Fast-Learner (🔵)             │                Builder (🟢)
  Emerging baseline across dimensions     │    High GitHub (90), High Project (78)
  High growth potential (CGPA: 7.2)       │    Lower CGPA (6.8), Ships working code
```

#### The 4 Behavioral Archetypes
1. **Research-Deep (🟡)**: Academic achievers with high CGPA and research publications. Moderate GitHub presence.
2. **Builder (🟢)**: Practical engineers with dense commit logs, multiple original repos, and high project complexity, often with average college GPAs.
3. **Fast-Learner (🔵)**: Early-career candidates with lower initial scores across dimensions but consistent foundation.
4. **All-Rounder (🟣)**: Well-balanced candidates maintaining consistent competence across academic, coding, and project criteria.

#### Distance-Based Confidence Metric
Candidate assignment projects normalized candidate feature vectors onto the fitted centroids via Euclidean distance. Confidence is derived via a numerically stable softmax over negative distances:
$$C_k = \frac{\exp\left(-d_k - \max_j(-d_j)\right)}{\sum_{i=1}^{4} \exp\left(-d_i - \max_j(-d_j)\right)}$$

> **Important Disclosure**: Archetype classifications are purely descriptive. They do not alter, weight, or override the candidate's `final_score` or ranking.

---

## 6. Tech Stack

| Layer | Technology | Version | Purpose & Architecture Justification |
|---|---|---|---|
| **Language** | Python | `3.10+` | Native ecosystem for NLP, tabular data manipulation, and machine learning pipelines. |
| **Web UI** | Streamlit | `>=1.35.0` | Reactive single-process interface; avoids the overhead of managing separate React/Node builds. |
| **LLM Inference** | Groq API (`groq/compound`) | `>=0.9.0` | Ultra-fast token generation on LLaMA 3.3 70B; structured JSON responses with zero hosting costs. |
| **Embeddings** | `sentence-transformers` | `>=2.7.0` | Local deterministic inference (`all-MiniLM-L6-v2`) for semantic cosine similarity guardrails. |
| **Clustering** | `scikit-learn` | `>=1.4.0` | `StandardScaler` and `KMeans(k=4)` for unsupervised behavioral archetype mapping. |
| **Persistence** | SQLite3 | Native (`sqlite3`) | Zero-configuration single-file relational storage with Write-Ahead Logging (`WAL`) mode enabled. |
| **PDF Extraction** | `pdfplumber` / `PyMuPDF` | `>=0.10.0` / `>=1.24.0` | Dual-engine PDF extraction: `pdfplumber` for structured layouts with `fitz` fallback for scans. |
| **Git API Client** | `requests` | `>=2.31.0` | REST API v3 client with rate-limit headers handling and exponential backoff retry logic. |
| **Calendar API** | `google-api-python-client` | `>=2.130.0` | Direct OAuth2 integration for provisioning Google Calendar events and Meet conference links. |
| **Authentication** | `google-auth-oauthlib` | `>=1.2.0` | Headless-safe base64 OAuth2 credential loading with automatic token refresh. |
| **DataFrames** | `pandas` | `>=2.1.0` | CSV ingestion normalization, missing-value imputation, and tabular metric rendering. |
| **Timezone** | `pytz` | `>=2024.1` | Accurate localized interview slot management (`Asia/Kolkata` standard). |
| **Model Cache** | `joblib` | `>=1.3.0` | Efficient disk serialization for fitted StandardScaler, KMeans, and label mappings. |
| **Testing** | `pytest` | `>=8.3.0` | Automated unit testing of clustering separability, scoring math, and edge-case handling. |

---

## 7. Design Decisions & Trade-offs

| Decision | Alternatives Considered | Why This Choice | Trade-off Accepted |
|---|---|---|---|
| **MCDA Weighted Sum Model instead of Supervised Classifier** | XGBoost, Logistic Regression, Random Forest | No ground-truth outcome labels ("hired / performed well") exist in this dataset. Training a classifier on fabricated labels causes severe bias. | Cannot automatically learn non-linear feature interactions; relies on recruiter-configured weights. |
| **Streamlit Monolithic Architecture instead of FastAPI + React** | FastAPI backend + Next.js / React frontend | Eliminates dual-process orchestration; rapid prototyping with native state binding and table components. | Every widget interaction triggers a rerun of the UI script (mitigated by `@st.cache_resource` and session state). |
| **Dual-Signal LLM + MiniLM Embedding Architecture** | LLM-only grading, or Embedding-only similarity | Local embeddings provide a deterministic sanity anchor. If the Groq API fails or hallucinates, embeddings provide a reliable fallback. | Candidate text must be processed twice (once locally through embedding model, once via remote LLM prompt). |
| **Candidate Primary Key on `s_no` instead of `email`** | `email` as unique primary key | In recruitment testing, multiple test candidate rows frequently share a recruiter's forwarding address. | Non-contiguous integer IDs require explicit SQL query iteration rather than `range(1, n+1)` loops. |
| **Unsupervised K-Means Archetypes on 4 Dimensions** | PCA dimensionality reduction, GMM, Hierarchical clustering | K-Means produces intuitive, non-overlapping cluster boundaries with direct distance interpretability for recruiter cards. | Assumes spherical cluster distributions; requires feature scaling via `StandardScaler` to prevent CGPA domination. |
| **Local SQLite WAL instead of Hosted PostgreSQL (Supabase)** | PostgreSQL, MySQL, Supabase free tier | Zero infrastructure setup, zero network latency on queries, portable single-file database (`screening.db`). | Limited to single-node deployments; not suitable for horizontally scaled multi-worker write traffic. |

---

## 8. Data & Models

### Datasets & Schemas

#### 1. Input Candidate Dataset (`sample_data/candidates.csv`)
10 reference applicant profiles designed to test the limits of the screening engine:
- **Non-contiguous `s_no`**: Candidates 1, 2, 3, 5, 6, 7, 8, 9, 10, 11 (explicitly skips 4 to prove the engine handles non-sequential databases).
- **Missing Signals**: Candidates 5 and 10 have empty GitHub URLs (exercising dynamic weight renormalization and population-mean imputation).
- **Drive & PDF Links**: Real URL structures pointing to technical resumes.

#### 2. Synthetic Archetype Generation (`sample_data/synthetic_archetypes.csv`)
Generated via `modules/archetype_clustering.py` using deliberate Gaussian sampling ($N=800$, 200 per archetype, `seed=42`):

| Archetype | `jd_match` $(\mu, \sigma)$ | `project_quality` $(\mu, \sigma)$ | `github_score` $(\mu, \sigma)$ | `cgpa` $(\mu, \sigma)$ |
|---|---|---|---|---|
| **Research-Deep** | $(68, 7)$ | $(72, 6)$ | $(40, 10)$ | $(9.0, 0.3)$ |
| **Builder** | $(55, 8)$ | $(78, 7)$ | $(90, 5)$ | $(6.8, 0.5)$ |
| **Fast-Learner** | $(48, 8)$ | $(48, 8)$ | $(35, 12)$ | $(7.2, 0.6)$ |
| **All-Rounder** | $(75, 6)$ | $(70, 6)$ | $(72, 7)$ | $(8.2, 0.4)$ |

### Model Cards

#### A. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture**: 6-layer BERT-based sentence transformer (22.7M parameters).
- **Output**: 384-dimensional dense vector embeddings.
- **Inference**: Local PyTorch CPU inference; cached in memory via `_get_embedding_model()`.
- **Purpose**: Generates semantic cosine similarity between the Job Description and candidate project descriptions.

#### B. LLM Evaluator: `groq/compound` (LLaMA 3.3 70B Versatile)
- **Host**: Groq Cloud High-Speed LPU.
- **Prompt Structure**: Zero-shot system recruiter prompt enforcing strict JSON output:
  ```json
  {
    "jd_match": 85,
    "project_quality": 80,
    "reasoning": "Strong PyTorch and FastAPI experience matching the core JD requirements. Lacks Kubernetes deployment experience."
  }
  ```
- **Fallback**: Exponential backoff on HTTP 429; falls back to embedding similarity if 3 retries fail.

---

## 9. Evaluation & Results

### Quantitative Clustering & Scoring Benchmarks

All metrics below are measured directly on the repository codebase and synthetic benchmark suite (`seed=42`):

| Evaluation Metric | Measured Value | Benchmark Description & Interpretation |
|---|---|---|
| **Archetype Silhouette Score** | **0.4494** | Evaluated on 800 synthetic profiles in 4D standardized space. Confirms strong separation between archetype boundaries ($>0.10$ sanity threshold). |
| **Adjusted Rand Index (ARI)** | **0.9669** | Measures recovery of true generative archetype labels via unsupervised K-Means clustering (near-perfect class discovery). |
| **WSM Weight Normalization** | **1.0000** | Strict mathematical weight conservation ($\sum w_i = 1.0$) across all 4 missing-signal combinations (all signals, no-test, no-github, neither). |
| **Unit Test Coverage** | **21 / 21 Passed** | 15 clustering tests + 6 scoring/renormalization tests passing in `pytest` ($2.62$s execution). |
| **Groq LLM JSON Parse Rate** | **100%** | Regex markdown code-fence stripper safely handles edge-case backtick enclosures. |

### Failure Mode Analysis

| Failure Mode | Root Cause Identified | Systematic Architectural Defense |
|---|---|---|
| **Google Drive Scan Warning** | Large PDFs prompt a Google Drive virus-scan warning page instead of raw bytes. | `download_pdf()` parses `confirm=` URL tokens and repeats the download with the confirmation key. |
| **Scanned / Image-Only PDFs** | Text-based `pdfplumber` yields empty string for scanned documents. | Automatic fallback to `PyMuPDF` (`fitz`); flags `scan_warning` in candidate record for manual recruiter review. |
| **Missing GitHub Profile** | Candidate leaves GitHub URL empty. | Handled gracefully: `github_score=None`, $w_3$ zeroed, remaining weights renormalized, population mean imputed for clustering. |
| **Groq Free-Tier Rate Limits** | Rapid batch requests trigger HTTP 429 rate limits (~30 RPM). | Built-in exponential backoff ($4s, 8s, 16s$) + $1.0$-second sleep between batch candidate evaluations. |
| **Expired Google OAuth Tokens** | Google Calendar tokens expire after 3600 seconds. | `load_google_credentials()` automatically checks `creds.expired` and calls `creds.refresh(Request())`. |

### Exact Reproduction Commands
```bash
# 1. Run unit test suite
pytest tests/

# 2. Run standalone scorer verification
python modules/scorer.py

# 3. Verify clustering metrics and generate synthetic dataset
python -c "
from modules.archetype_clustering import generate_synthetic_archetype_profiles, fit_clustering, label_clusters, FEATURES
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
df = generate_synthetic_archetype_profiles(seed=42)
scaler, kmeans = fit_clustering(df, seed=42)
X_scaled = scaler.transform(df[FEATURES].values)
le = LabelEncoder()
sil = silhouette_score(X_scaled, kmeans.labels_)
ari = adjusted_rand_score(le.fit_transform(df['true_archetype']), kmeans.labels_)
print(f'Silhouette: {sil:.4f}, ARI: {ari:.4f}')
"
```

---

## 10. Testing & Quality

The TalentLens test suite is automated via `pytest` and covers model clustering, mathematical scoring consistency, and input normalization.

### Test Layout
```
tests/
├── __init__.py
├── test_archetype_clustering.py  # 15 tests: shape, bounds, silhouette, label map, confidence
└── test_scorer.py                # 6 tests: WSM math, weight renorm, test averaging, ranking
```

### Running the Test Suite
```bash
pytest tests/ -v
```

```
tests/test_archetype_clustering.py::TestSyntheticDataGeneration::test_correct_shape PASSED
tests/test_archetype_clustering.py::TestSyntheticDataGeneration::test_columns_present PASSED
tests/test_archetype_clustering.py::TestSyntheticDataGeneration::test_no_nulls_in_features PASSED
tests/test_archetype_clustering.py::TestSyntheticDataGeneration::test_archetype_counts_balanced PASSED
tests/test_archetype_clustering.py::TestSyntheticDataGeneration::test_feature_bounds_respected PASSED
tests/test_archetype_clustering.py::TestSyntheticDataGeneration::test_clusters_distinguishable PASSED
tests/test_archetype_clustering.py::TestClusterLabeling::test_produces_four_distinct_names PASSED
tests/test_archetype_clustering.py::TestClusterLabeling::test_all_archetype_names_present PASSED
tests/test_archetype_clustering.py::TestClusterLabeling::test_keys_are_valid_cluster_ids PASSED
tests/test_archetype_clustering.py::TestCandidateAssignment::test_normal_candidate_returns_valid_archetype PASSED
tests/test_archetype_clustering.py::TestCandidateAssignment::test_confidence_sums_to_one PASSED
tests/test_archetype_clustering.py::TestCandidateAssignment::test_distances_all_has_four_entries PASSED
tests/test_archetype_clustering.py::TestCandidateAssignment::test_missing_github_score_does_not_crash PASSED
tests/test_archetype_clustering.py::TestCandidateAssignment::test_strong_builder_profile_assigned_correctly PASSED
tests/test_archetype_clustering.py::TestCandidateAssignment::test_strong_research_deep_profile PASSED
tests/test_scorer.py::test_full_signal_score PASSED
tests/test_scorer.py::test_missing_test_score_renormalization PASSED
tests/test_scorer.py::test_missing_github_score_renormalization PASSED
tests/test_scorer.py::test_missing_both_github_and_test PASSED
tests/test_scorer.py::test_compute_test_score PASSED
tests/test_scorer.py::test_rank_rows PASSED

======================= 21 passed in 2.62s =======================
```

---

## 11. Security, Privacy & Reliability

- **Zero Secret Ingestion in Git**: All sensitive credentials (`GROQ_API_KEY`, `GITHUB_PAT`, `GMAIL_APP_PASSWORD`, `GOOGLE_OAUTH_CLIENT_JSON`) reside strictly in `.streamlit/secrets.toml` or Streamlit Cloud Secrets. The `.gitignore` prevents secrets, database files, and tokens from ever being committed.
- **Decoupled PII Storage**: Candidate names, email addresses, and resume text stay entirely within the local SQLite database. External API payloads to Groq contain only relevant technical text snippets necessary for qualitative evaluation.
- **Headless OAuth2 Deserialization**: Avoids unsafe Python `pickle` objects. Credentials use base64-encoded JSON strings read via Google's official `Credentials.from_authorized_user_info()`.
- **SQL Injection Prevention**: Every database query in `db.py` utilizes parameterized SQL statements (`?` placeholders). No raw SQL string interpolation is permitted.
- **Fault-Tolerant Batching**: Module operations in `resume_parser.py`, `github_analyzer.py`, and `emailer.py` wrap individual candidate iterations in independent `try...except` blocks. A single malformed PDF or missing GitHub handle will not crash the batch pipeline.

---

## 12. Getting Started

### Prerequisites
- Python `3.10` or higher installed
- Git installed
- Free Groq Cloud account ([console.groq.com](https://console.groq.com))
- GitHub Personal Access Token (classic token with `public_repo` scope)

### 1. Clone & Environment Setup
```bash
git clone https://github.com/dummycodertech/TalentLens.git
cd TalentLens

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Secrets
Create your local secrets configuration:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your credentials:

| Secret Key | Required | Purpose | Where to Obtain |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | LLM Qualitative Evaluation | [console.groq.com/keys](https://console.groq.com/keys) |
| `GITHUB_PAT` | Optional | GitHub API Rate Limit (5,000 req/hr) | [github.com/settings/tokens](https://github.com/settings/tokens) |
| `GMAIL_ADDRESS` | Optional | SMTP Email Sender Address | Recruiter Gmail Account |
| `GMAIL_APP_PASSWORD` | Optional | SMTP 16-character App Password | [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) |
| `GOOGLE_OAUTH_CLIENT_JSON` | Optional | Google Calendar & Meet Generation | Run helper script below |
| `RECRUITER_EMAIL` | Optional | Calendar Event Organizer | Recruiter Gmail Account |

#### One-Time Google Calendar OAuth Token Setup
If you want to use Tab 5 (Google Calendar & Meet links):
1. Create a project in [Google Cloud Console](https://console.cloud.google.com/) and enable the **Google Calendar API**.
2. Create an **OAuth 2.0 Client ID** (Desktop Application) and download `client_secret.json`.
3. Run the one-line local token generator:
   ```bash
   python -c "from modules.calendar_scheduler import generate_token_locally; generate_token_locally('client_secret.json')"
   ```
4. Copy the printed base64 string into `GOOGLE_OAUTH_CLIENT_JSON` in `.streamlit/secrets.toml`.

### 3. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 13. Usage Guide

### Recruiter Walkthrough

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Tab 1     │     │    Tab 2     │     │    Tab 3     │     │    Tab 4     │     │    Tab 5     │
│   Ingest &   ├───► │ Run Pipeline ├───► │ Rankings &   ├───► │ Test Results ├───► │  Schedule    │
│ JD Config    │     │  (5 Stages)  │     │ Email Links  │     │  & Re-Rank   │     │  Interviews  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

#### Step 1: Upload & Configure (Tab 1)
- Upload `sample_data/candidates.csv` (or your own CSV export).
- Paste the target Job Description in the text area (aim for $>100$ words).
- Click **"Load Candidates"**.

#### Step 2: Execute Autonomous Screening (Tab 2)
- Click **"Run Evaluation Pipeline"**.
- Watch the live status table progress through the 5 stages:
  1. *Resume Parsing* (`pdfplumber` / `PyMuPDF`)
  2. *GitHub Repo Audit* (commits, recency, documentation)
  3. *AI Evaluation* (MiniLM cosine similarity + Groq reasoning)
  4. *WSM Scoring* (formula calculation + rank assignment)
  5. *Archetype Assignment* (K-means centroid projection)

#### Step 3: Inspect Rankings & Dispatch Tests (Tab 3)
- Expand candidate cards to review sub-scores, qualitative LLM reasoning, repository commit counts, and archetype confidence bars.
- Enter your assessment URL (e.g. HackerRank / Google Forms) and cutoff threshold (e.g., $60.0$).
- Click **"Send Test Links"** to dispatch personalized emails.

#### Step 4: Ingest Test Results & Shortlist (Tab 4)
- Upload `sample_data/test_results.csv` containing columns `s_no, test_la, test_code`.
- Click **"Merge & Re-rank"**. The model computes the final score with all 4 signals active.
- Set the shortlist threshold slider (e.g., score $\ge 65$) and click **"Mark as Shortlisted"**.

#### Step 5: Provision Interviews (Tab 5)
- Select the starting interview date and time.
- Set duration spacing (default: $45$ minutes).
- Click **"Schedule Interviews"**. Google Meet links are generated and synchronized with Google Calendar.

---

## 14. Configuration & Tunables

All evaluation weights can be adjusted live in the application sidebar or modified programmatically in `modules/scorer.py`:

```python
# Default scoring weights (modules/scorer.py)
DEFAULT_WEIGHTS = {
    "w1": 0.30,  # JD Match weight (embedding + Groq LLM)
    "w2": 0.25,  # Project Quality weight (Groq LLM)
    "w3": 0.25,  # GitHub Score weight (REST API v3)
    "w4": 0.20,  # Assessment Test Score weight (Logical + Coding)
}

# GitHub Sub-Weights (modules/github_analyzer.py)
GITHUB_SUB_WEIGHTS = {
    "activity_recency":      0.35,  # Days since last push
    "commit_frequency":      0.25,  # 6-month commit count on non-forks
    "original_repo_ratio":   0.20,  # Non-forks / total repositories
    "documentation_quality": 0.20,  # README presence + byte count bonus
}
```

---

## 15. Project Structure

```
TalentLens/
├── app.py                         # Main Streamlit application (1,363 lines, 5 tabs, dark styling)
├── db.py                          # SQLite abstraction layer, WAL transactions, schema migrations
├── requirements.txt               # Locked production dependencies
├── LICENSE                        # MIT Open Source License
├── ARCHITECTURE.md                # In-depth system design & methodology document
├── USER_GUIDE.md                  # Comprehensive recruiter operating instructions
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py           # Drive URL handling, PDF download, pdfplumber & PyMuPDF fallback
│   ├── github_analyzer.py         # GitHub REST API client, repo audit, 4-term sub-formula
│   ├── ai_evaluator.py            # Local all-MiniLM-L6-v2 embeddings + Groq LLM qualitative scoring
│   ├── archetype_clustering.py    # K-Means clustering (k=4), synthetic sampling, softmax confidence
│   ├── scorer.py                  # Multi-Criteria WSM scoring, missing-signal weight renormalization
│   ├── emailer.py                 # Gmail SMTP HTML email construction & batch delivery
│   └── calendar_scheduler.py      # Google Calendar API v3 client, OAuth2 handling, Meet generation
├── sample_data/
│   ├── candidates.csv             # 10-candidate test dataset (non-contiguous IDs, missing signals)
│   ├── test_results.csv           # Reference assessment scores (s_no, test_la, test_code)
│   └── synthetic_archetypes.csv   # 800 synthetic profiles used for clustering validation
├── tests/
│   ├── __init__.py
│   ├── test_archetype_clustering.py # 15 unit tests: clustering shape, bounds, silhouette, assignment
│   └── test_scorer.py             # 6 unit tests: WSM formula, renormalization, test averaging, ranking
├── docs/
│   └── screenshots/               # High-resolution screenshots of Tabs 1, 2, 3, and 5
├── assets/                        # Supporting interface screenshots (screenshot1-8.png)
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespaces & VS Code Dev Container environment
└── .streamlit/
    ├── config.toml                # Streamlit configuration (telemetry disabled)
    └── secrets.toml.example       # Template for API keys and credentials
```

---

## 16. Deployment

### Deploy to Streamlit Community Cloud (Free Tier)
1. Fork or push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and select your `TalentLens` repository.
3. Configure **Main file path**: `app.py`.
4. Open **Advanced Settings → Secrets** and paste your credentials:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GITHUB_PAT = "ghp_..."
   GMAIL_ADDRESS = "recruiter@gmail.com"
   GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"
   GOOGLE_OAUTH_CLIENT_JSON = "<base64_encoded_token>"
   RECRUITER_EMAIL = "recruiter@gmail.com"
   ```
5. Click **Deploy**. Your app is live with an SSL-secured URL.

---

## 17. Performance & Cost

### Measured Latency Breakdown (Average per Candidate)
- **PDF Resume Download & Parsing**: $\approx 1.5 - 3.2$ seconds
- **GitHub REST Repository Audit**: $\approx 2.0 - 4.5$ seconds (depends on number of public repos)
- **Local Embedding Vectorization**: $\approx 45$ milliseconds (CPU inference)
- **Groq LLaMA 3.3 70B Generation**: $\approx 850$ milliseconds
- **WSM Scoring & Archetype Assignment**: $\approx 5$ milliseconds

### Monthly Infrastructure Cost
| Component | Provider / Tier | Monthly Cost |
|---|---|---|
| Web Application Hosting | Streamlit Community Cloud | **$0.00** |
| LLM Inference | Groq Cloud Free Tier | **$0.00** |
| Embeddings Model | Local CPU Execution (`all-MiniLM-L6-v2`) | **$0.00** |
| Code Repository Analytics | GitHub REST API v3 (5,000 requests/hr) | **$0.00** |
| Relational Storage | Local SQLite3 WAL File | **$0.00** |
| Email Delivery | Gmail SMTP with App Password | **$0.00** |
| Calendar & Meet Provisioning | Google Calendar API (Personal OAuth2) | **$0.00** |
| **Total Operating Cost** | | **$0.00 / month** |

---

## 18. Roadmap & Future Scope

### Near-Term (1–2 Weeks)
- [ ] **Batch Resume ZIP Upload**: Support uploading local ZIP files of PDF resumes directly in Tab 1 alongside Google Drive URLs.
- [ ] **Recruiter PDF Previewer**: Embed an in-browser PDF viewer directly within Tab 3 candidate cards.
- [ ] **Export Rankings to CSV**: Add one-click export of final scores, rankings, and LLM reasoning strings to CSV/Excel.

### Mid-Term (1–2 Months)
- [ ] **PostgreSQL (Supabase) Database Engine**: Provide a seamless drop-in toggle to switch from SQLite to hosted PostgreSQL for team collaboration.
- [ ] **Asynchronous Worker Queue**: Decouple long-running pipeline execution using Celery or Redis Queue for processing $500+$ candidate batches concurrently.
- [ ] **Candidate Feedback Portal**: Generate auto-composed, constructive rejection feedback emails highlighting specific skill growth areas.

### Long-Term (Research & Enterprise)
- [ ] **Self-Supervised Archetype Tuning**: Fine-tune cluster boundaries dynamically on proprietary historical company interview outcome data.
- [ ] **Multi-Role Job Description Library**: Support simultaneous candidate matching across multiple open engineering positions.

---

## 19. FAQ / How It Works in Plain English

#### Q1: Why does TalentLens use a Weighted Sum Model instead of training a machine learning model to rank candidates?
**A**: Training a supervised classifier (like XGBoost or a Neural Network) requires thousands of historical examples labeled with whether the candidate was hired and performed well. No such labels exist at the resume screening stage. Training on artificial labels would embed arbitrary hallucinations. The Weighted Sum Model (WSM) is an established Multi-Criteria Decision Analysis (MCDA) framework that is mathematically sound, customizable by recruiters, and 100% auditable.

#### Q2: What happens if a candidate doesn't have a GitHub profile?
**A**: They are not penalized with an artificial ceiling. TalentLens detects the absence of a GitHub URL, sets `github_score = None`, zeros the GitHub weight ($w_3 = 0$), and dynamically redistributes that weight proportionally among the remaining signals. The candidate's final score still fairly scales from $0$ to $100$.

#### Q3: How does the system handle scanned or image-only PDF resumes?
**A**: The parser runs `pdfplumber` first. If fewer than 100 characters are extracted, it automatically invokes `PyMuPDF` (`fitz`). If text remains sparse, it flags the profile with a `scan_warning` in the database and dashboard so recruiters know to review the document manually.

#### Q4: Are the K-Means archetypes affecting candidate rankings?
**A**: **No.** Archetypes are descriptive, not evaluative. An *All-Rounder* ranked #5 might have a lower score than a *Builder* ranked #2. The archetype tags simply give hiring managers qualitative insight into the profile shape (e.g. academic vs. hands-on open source builder).

#### Q5: Can multiple candidates have the same email address?
**A**: **Yes.** The primary database key is `s_no` (the candidate serial number), not email. This ensures that recruitment tests where multiple applicants forward submissions to a shared recruiter inbox process without data collision.

#### Q6: How does the Google Meet link generation work without paying for Google Workspace?
**A**: TalentLens uses the personal Google Calendar API via standard desktop OAuth2. By specifying `"conferenceSolutionKey": {"type": "hangoutsMeet"}` in the `conferenceData.createRequest` payload, Google automatically provisions a valid Google Meet room link free of charge.

#### Q7: Does TalentLens leak candidate PII to external AI model providers?
**A**: No personal contact details (phone numbers, physical addresses, candidate emails) are passed to Groq. Only technical project summaries and relevant resume text snippets are sent for qualitative evaluation.

#### Q8: How fast can TalentLens evaluate a batch of 50 candidates?
**A**: Running locally, a batch of 50 candidates finishes in approximately 4 to 6 minutes, bounded primarily by network calls to Google Drive and GitHub's REST API.

---

## 20. Resume & Interview Kit

<details>
<summary><b>💼 Expand Resume & Interview Kit (Bullets, Pitch & Technical Interview Q&A)</b></summary>
<br>

### 📄 3 Actionable Resume Bullets
- **Architected and deployed TalentLens**, an autonomous end-to-end AI candidate screening platform processing multi-modal applicant inputs across PDF resumes, GitHub REST APIs, and structured Job Descriptions with $0.00$ infrastructure cost.
- **Engineered an auditable MCDA scoring framework** combining local sentence embeddings (`all-MiniLM-L6-v2`), Groq LLaMA 3.3 70B qualitative reasoning, and dynamic weight renormalization, achieving 100% score scaling integrity across missing-signal edge cases.
- **Designed an unsupervised candidate classification pipeline** using Scikit-Learn K-Means ($k=4$), achieving a **0.4494 Silhouette Score** and **0.9669 Adjusted Rand Index** on benchmark populations, integrated with automated Google Calendar/Meet OAuth2 scheduling.

---

### 🎙️ The 30-Second Elevator Pitch
> *"TalentLens is an open-source, zero-cost AI recruiting platform that replaces black-box ATS keyword filters with multi-signal evaluation. It parses resumes from Google Drive, analyzes public GitHub repositories for genuine commit recency and code depth, scores candidates using dual-signal semantic embeddings and Groq LLMs, and clusters profiles into 4 behavioral archetypes using K-Means. It handles the entire lifecycle—from raw CSV ingestion to automated Google Meet interview scheduling—with complete auditability and $0/month in cloud bills."*

---

### 🛠️ 3 Hardest Technical Challenges Solved

#### 1. Dynamic Weight Renormalization Without Score Distortions
- **Problem**: When candidates lack optional signals (e.g., no GitHub account, or evaluation before coding tests are administered), naive weighted sum implementations cap the candidate's score ceiling at 75 or 80 points, unfairly penalizing them in global rankings.
- **Root Cause**: Fixed static weight coefficients ($\sum w_i = 1.0$) assume all input dimensions exist for all entities simultaneously.
- **Solution**: Developed `get_active_weights()` in `modules/scorer.py`, which dynamically identifies missing input dimensions, zeros their coefficients, and scales remaining weights by $\frac{w_i}{\sum_{\text{active}} w_k}$. Validated with unit tests confirming exact 0–100 score distribution across all missing-signal permutations.

#### 2. Headless OAuth2 Credential Deserialization on Cloud Runtimes
- **Problem**: Standard Google Calendar OAuth flows rely on local browser popups (`run_local_server()`) and serializing credentials via `pickle`, which fail on headless cloud servers like Streamlit Cloud and introduce security vulnerabilities.
- **Root Cause**: Interactive browser loops cannot run on containerized cloud runners without an active display server.
- **Solution**: Built a headless token strategy in `modules/calendar_scheduler.py` that serializes credentials into standard JSON, encodes them as base64 environment secrets, and dynamically loads them via `Credentials.from_authorized_user_info()` with automatic token refresh checking before every API call.

#### 3. Google Drive Direct-Download Redirect Token Bypass
- **Problem**: Downloading PDF resumes directly from Google Drive share links fails for files larger than a few megabytes because Google serves an intermediate HTML virus-scan confirmation screen instead of the PDF binary.
- **Root Cause**: Google Drive returns HTTP 200 with an HTML payload containing a session-specific `confirm=XXXX` query token.
- **Solution**: Implemented regex token parsing in `modules/resume_parser.py` that intercepts the HTML confirmation page, extracts the dynamic confirmation token, and executes a second authenticated stream request to reliably retrieve the raw PDF bytes.

---

### 💡 8 Common Technical Interview Questions & Answers

#### Q1: "Why did you use K-Means clustering instead of supervised learning?"
> *"Supervised learning requires ground-truth labels—in this case, whether an applicant was ultimately hired and succeeded in the role. No such historical labels exist at the screening stage. Training a classifier on synthetic labels would create circular reasoning and introduce arbitrary bias. K-Means is unsupervised; it identifies natural cluster groupings based on the shape of candidates' scores, helping recruiters balance their hiring pool between academic researchers and practical code builders."*

#### Q2: "How do you protect against LLM hallucinations in candidate scoring?"
> *"We implement three structural guardrails: First, we anchor the LLM score with a local, deterministic cosine similarity score from `all-MiniLM-L6-v2`. Second, we enforce strict JSON output schemas with temperature set to 0.2, requiring the model to cite specific skills present or missing. Third, if the Groq API call fails or returns invalid JSON, the system automatically falls back to embedding similarity as a proxy, ensuring zero pipeline halts."*

#### Q3: "Why did you choose SQLite over PostgreSQL?"
> *"For single-recruiter and demo workloads, SQLite requires zero setup, runs in-process with sub-millisecond query latency, and enables trivial portability of the entire state via a single file. By enabling WAL (`Write-Ahead Logging`) mode, we maintain safe concurrent read/write transactions. For multi-tenant enterprise deployments, our abstraction layer in `db.py` can be converted to SQLAlchemy or Supabase PostgreSQL with minimal refactoring."*

#### Q4: "What happens if GitHub rate limits your requests?"
> *"Unauthenticated GitHub requests are capped at 60 per hour, which is insufficient for batch candidate screening. TalentLens requires a Personal Access Token (PAT), expanding capacity to 5,000 requests per hour. Furthermore, `modules/github_analyzer.py` inspects the `X-RateLimit-Reset` header and implements exponential backoff to handle transient 403/429 responses gracefully."*

#### Q5: "How did you ensure the K-Means clustering was mathematically sound?"
> *"We standardized all 4 feature dimensions using `StandardScaler` to prevent high-magnitude features like `jd_match` (0–100) from overpowering `cgpa` (0–10). We evaluated the resulting clusters on synthetic benchmark profiles, achieving a Silhouette Score of 0.4494 and an Adjusted Rand Index of 0.9669 against known generative distributions."*

#### Q6: "Why did you use both pdfplumber and PyMuPDF?"
> *"Different PDF generators encode text differently. `pdfplumber` offers superior precision for tabular and columnar resume layouts, but frequently fails on scanned or rasterized PDFs. `PyMuPDF` (`fitz`) serves as an efficient fallback engine. If both yield less than 100 characters, the candidate is flagged with a `scan_warning` in the UI for manual human oversight."*

#### Q7: "How is data consistency maintained when re-ranking after coding tests?"
> *"Post-assessment test results are joined in SQLite strictly on `s_no`. When `merge_test_results()` is called, candidate records are updated with `test_la` and `test_code`, and `scorer.score_all()` is re-executed across all candidate rows. The active weights automatically transition from 3-signal to 4-signal mode, producing updated ranks in a single transactional pass."*

#### Q8: "What would you do differently if rebuilding this from scratch today?"
> *"I would decouple the compute pipeline from the Streamlit UI thread by introducing a Redis Queue (RQ) or Celery worker backend. While Streamlit is ideal for interactive dashboards, long-running batch extraction of 500+ candidates would benefit from background worker workers, real-time WebSocket progress updates, and a hosted PostgreSQL database."*

---

### 🎯 Demonstrated Engineering Competencies
- **Applied Machine Learning**: Unsupervised clustering (K-Means), feature scaling (`StandardScaler`), metric evaluation (Silhouette Score, ARI), synthetic population sampling.
- **Natural Language Processing**: Sentence embeddings (`sentence-transformers`), semantic cosine similarity, LLM structured prompt engineering, JSON schema validation.
- **Systems & Architecture**: Multi-Criteria Decision Analysis (WSM), state machine design, SQLite WAL configuration, dynamic weight renormalization.
- **API & Integration Engineering**: GitHub REST API v3, Google Calendar API v3 (OAuth2), Gmail SMTP, Google Drive byte streaming.
- **Software Quality**: Automated testing (`pytest`), test fixture architecture, defensive error handling, zero-cost cloud deployment.

</details>

---

## 21. Project Context Block

<details>
<summary><b>📋 Expand Compact Project Context (YAML for AI Assistants)</b></summary>
<br>

```yaml
project:
  name: "TalentLens"
  tagline: "Autonomous AI Candidate Screening & Technical Recruiter Platform"
  author: "Yagas Vashist (dummycodertech)"
  repository: "https://github.com/dummycodertech/TalentLens"
  license: "MIT"
  language: "Python 3.10+"
  frameworks:
    ui: "Streamlit >= 1.35.0"
    ml: "scikit-learn >= 1.4.0 (KMeans, StandardScaler)"
    nlp: "sentence-transformers >= 2.7.0 (all-MiniLM-L6-v2)"
    llm: "Groq Cloud API (groq/compound / LLaMA 3.3 70B)"
    database: "SQLite3 (WAL mode enabled)"
  architecture:
    type: "Single-Process 5-Stage Autonomous Recruiter Pipeline"
    stages:
      stage_1: "Resume Parsing (Google Drive PDF download, pdfplumber + PyMuPDF fallback)"
      stage_2: "GitHub Audit (REST API v3, 4-term per-repo WSM sub-formula)"
      stage_3: "AI Semantic Evaluation (Local MiniLM cosine similarity + Groq LLM structured JSON reasoning)"
      stage_4: "Unsupervised Behavioral Archetypes (K-Means k=4 on jd_match, project_quality, github_score, cgpa)"
      stage_5: "Auditable WSM Ranking & Logistics (MCDA scoring with missing-signal weight renormalization)"
    logistics:
      email: "Gmail SMTP SSL Port 465 (Candidate name + s_no in subject)"
      calendar: "Google Calendar API v3 (OAuth2 Desktop credentials, HangoutsMeet video links)"
  data_stores:
    primary: "screening.db (candidates, scores, interview_events)"
    cache: "model_cache/ (archetype_scaler.joblib, archetype_kmeans.joblib, archetype_label_map.joblib)"
  metrics:
    silhouette_score: 0.4494
    adjusted_rand_index: 0.9669
    unit_tests: "21/21 passing (pytest tests/)"
    monthly_infrastructure_cost: "$0.00"
  archetypes:
    - name: "Research-Deep"
      color: "Yellow"
      profile: "High CGPA, strong project quality, research publications, moderate GitHub"
    - name: "Builder"
      color: "Green"
      profile: "High GitHub commit density, original repos, strong project quality, average CGPA"
    - name: "Fast-Learner"
      color: "Blue"
      profile: "Lower baseline scores across dimensions, early-career growth potential"
    - name: "All-Rounder"
      color: "Purple"
      profile: "Consistently balanced performance across all four evaluation criteria"
  key_design_decisions:
    - "WSM MCDA chosen over supervised classifier due to absence of ground-truth hire labels"
    - "Dynamic weight renormalization preserves strict 0-100 score distribution when signals are missing"
    - "Primary database key is s_no to allow shared recruiter forwarding email inboxes"
    - "Archetypes are purely descriptive and do not alter candidate rank"
```

</details>

---

## 22. References, Acknowledgements, License, Author

### References & Core Libraries
- **Sentence-Transformers**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
- **Multi-Criteria Decision Analysis (MCDA)**: Fishburn, P. C. (1967). *Additive Utilities with Finite Sets: Applications in the Management Sciences*. Naval Research Logistics Quarterly.
- **Scikit-Learn**: Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
- **Streamlit**: [Streamlit Open-Source Data Application Framework](https://streamlit.io/).
- **Groq LPU Inference Engine**: [Groq Cloud Documentation](https://console.groq.com/docs).

### License
This project is open-source software licensed under the terms of the **[MIT License](LICENSE)**.

### Author
- **Yagas Vashist**
- **GitHub**: [@dummycodertech](https://github.com/dummycodertech)
- **Repository**: [github.com/dummycodertech/TalentLens](https://github.com/dummycodertech/TalentLens)

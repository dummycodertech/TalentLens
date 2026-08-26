# Architecture — AI-Powered Candidate Screening Platform

**myNachiketa GTM Engineering Intern Assignment**
**Cost: $0 — every component runs on a free tier or locally**

---

## System Overview

```
┌─────────────────────────── Streamlit App (single process) ───────────────────────────┐
│                                                                                       │
│  UI Layer (st.tabs)                                                                   │
│  ├─ Tab 1: Upload candidates + JD                                                     │
│  ├─ Tab 2: Pipeline run + live status table                                           │
│  ├─ Tab 3: Ranking dashboard (scores, reasoning, per-repo breakdown)                  │
│  ├─ Tab 4: Test result upload + re-rank + shortlist marking                           │
│  └─ Tab 5: Interview scheduling log + Google Meet links                               │
│                                                                                       │
│  ┌───────────────────────────── modules/ ───────────────────────────────────────────┐ │
│  │ resume_parser.py     → Drive link → PDF bytes → pdfplumber → clean text          │ │
│  │ github_analyzer.py   → GitHub REST API → per-repo metrics → WSM github_score     │ │
│  │ ai_evaluator.py      → MiniLM embeddings (sim) + Groq LLM (reasoning JSON)       │ │
│  │ scorer.py            → WSM final_score + weight renormalization + rank            │ │
│  │ emailer.py           → Gmail SMTP → test-link email with name+s_no subject       │ │
│  │ calendar_scheduler.py → Google Calendar API → event + Meet link + invite         │ │
│  └──────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                       │
│  db.py → SQLite (candidates, scores, interview_events)                                │
└───────────────────────────────────────────────────────────────────────────────────────┘
         ↑ free            ↑ free              ↑ free              ↑ free tier
   Streamlit Cloud    GitHub REST API    HF MiniLM (local)    Groq (Llama 3.3 70B)
```

---

## Pipeline State Machine

```
uploaded → resume_parsed → github_analyzed → ai_scored → ranked
         → test_sent → test_scored → shortlisted → interview_scheduled → invited
```

Each module writes `status` to SQLite. The dashboard is a live `SELECT *` — no separate progress-tracking logic.

Error states: `resume_failed`, `github_failed`, `email_failed`, `scheduling_failed`

---

## Scoring Methodology: Weighted Sum Model (MCDA)

### Why WSM?

The scoring approach is a **Weighted Sum Model (WSM)**, the simplest form of **Multi-Criteria Decision Analysis (MCDA)**. This was the correct methodological choice, not just the convenient one:

**No outcome labels exist.** The dataset has candidate features and test scores, but no "hired / performed well" outcome column. A trained classifier or regressor would require historical outcomes to learn from — none exist here. Training one on this data would require inventing fake labels, which is indefensible. This also means SHAP/LIME don't apply: those tools explain a trained model's learned weights. There is no trained model.

**The four formula terms map 1:1 onto the assignment's own stated criteria.** Section 4.5 of the brief says: "evaluate candidates using: Resume information, GitHub activity, Job description relevance, Test performance." That's exactly four dimensions. The WSM has exactly four terms:

| Score term | Criterion | Method |
|---|---|---|
| `jd_match` | JD relevance | MiniLM embedding sim + Groq LLM |
| `project_quality` | Resume/project quality | Groq Llama 3.3 70B JSON reasoning |
| `github_score` | GitHub activity | REST API per-repo WSM sub-formula |
| `test_score` | Test performance | Uploaded CSV, `(test_la + test_code)/2` |

**It is maximally auditable.** Every sub-score, weight, and reasoning string is stored in SQLite and visible in the dashboard. The explainability requirement ("Explainable AI scoring") is satisfied at the formula level — no post-hoc explanation method needed.

---

## Candidate Archetypes (Unsupervised Clustering)

Unlike supervised approaches (which would require outcome labels that don't exist in this dataset), the archetype feature uses **K-means clustering**, which requires no labels at all — so it carries none of the label-circularity limitations that would apply to a supervised model trained on synthetic hire/no-hire labels.

### How it works

1. **Synthetic profile generation**: 800 candidate profiles (200 per archetype) are sampled from 4 deliberate Gaussian distributions, each representing a known candidate pattern:
   - **Research-Deep**: High CGPA, strong project quality, moderate GitHub, focused academic profile
   - **Builder**: High GitHub activity, strong projects, lower CGPA — ships code over papers
   - **Fast-Learner**: Lower scores across the board but balanced — early-career, high potential
   - **All-Rounder**: Above-average on everything, no single standout dimension

2. **K-means fitting**: `StandardScaler` + `KMeans(k=4)` on `[jd_match, project_quality, github_score, cgpa]`. Cluster IDs are mapped to archetype names by majority vote of `true_archetype` within each cluster.

3. **Real candidate assignment**: Each scored candidate is projected onto the fitted centroids by Euclidean distance in scaled feature space. Assignment is the nearest centroid; confidence is a softmax over negative distances across all 4 centroids.

4. **Missing signal handling**: If `github_score` is `None`, the synthetic population mean is imputed before scaling. This is documented as a stated limitation, consistent with the WSM missing-signal pattern.

### What it answers that WSM doesn't

WSM answers "how highly does this candidate rank?" (a total ordering by score). Archetypes answer "what *kind* of candidate is this?" (a descriptive grouping by profile shape). The two are complementary:

- A Research-Deep candidate ranked #2 and a Builder ranked #3 might have similar scores but very different strengths — the archetype surfaces this for a recruiter building a diverse team.
- Archetypes **do not affect** `final_score` or `rank`. They are a purely descriptive overlay, displayed separately, with a permanent disclosure in the UI.

### Formula

```
final_score = w1 * jd_match + w2 * project_quality + w3 * github_score + w4 * test_score
```

Default weights derived from equal-priority reading of the brief's four criteria, with a slight upweight for JD match (most direct relevance signal) and slight downweight for test (uploaded last):

| Weight | Default | Signal |
|---|---|---|
| w1 | 0.30 | JD Match |
| w2 | 0.25 | Project Quality |
| w3 | 0.25 | GitHub Score |
| w4 | 0.20 | Test Score |

Weights are recruiter-configurable via sidebar sliders. They auto-renormalize to sum to 1.0.

### Graceful degradation for missing signals

```python
def get_active_weights(weights, has_test, has_github):
    active = dict(weights)
    if not has_test:   active['w4'] = 0
    if not has_github: active['w3'] = 0
    remaining = sum(active.values())
    return {k: v/remaining if remaining > 0 else 0 for k, v in active.items()}
```

- Candidates without a GitHub URL: `github_score=None`, w3 zeroed and redistributed. Score still scales 0–100.
- Candidates without test results yet: w4 zeroed and redistributed. Pre-test maximum is still 100, not capped at 80.

### Where the AI is — and why the combination is simple

The sophistication is in **signal generation**:
- `jd_match`: MiniLM-L6-v2 cosine similarity (semantic relevance) + Groq LLaMA 3.3 70B qualitative reasoning
- `project_quality`: LLM evaluation of project complexity, domain relevance, and depth
- `github_score`: Per-repo REST API analysis with a WSM sub-formula (activity recency, commit frequency, original repo ratio, documentation quality)

The **combination** is deliberately transparent. A hand-tunable weighted sum is more auditable than a black-box ensemble with SHAP bolted on after the fact. Transparency is explicitly rewarded ("Explainable AI scoring" bonus), so this is a feature, not a limitation.

---

## GitHub Analysis: Per-Repo WSM Sub-formula

Per the brief's "repository-level evaluation" requirement, each candidate's GitHub score is computed from individual repo data, not profile statistics.

```
github_score = 0.35 * activity_recency
             + 0.25 * commit_frequency
             + 0.20 * original_repo_ratio
             + 0.20 * documentation_quality
```

**Per-repo signals collected:**
- Primary language, size, stars, forks
- Last push date (recency signal — penalizes stale repos)
- Commit count in last 6 months (activity signal)
- README presence + size (documentation signal)
- Fork vs. original (forks excluded/downweighted)

The per-repo breakdown is stored as JSON and displayed in the dashboard expandable row — this is what makes the GitHub evaluation defensible: "this candidate scored X because their last active original repo was pushed 12 days ago, has 47 commits in 6 months, and has a 2KB README."

---

## Module Design

### `db.py`
- `normalize_csv_columns(df)` — applies column aliases (`github→github_url`, `resume→resume_url`) at the CSV ingestion boundary before any schema validation
- `insert_candidates(df)` — deduplicates on `s_no` (not `email`, since real CSVs may share a recruiter forwarding address)
- All iteration over candidates uses actual DB query results — never `range(1, n+1)` — because `s_no` may be non-contiguous

### `resume_parser.py`
- Converts Drive sharing URLs to direct-download URLs
- Primary: pdfplumber; fallback: PyMuPDF for image-based PDFs
- Flags scan-warning candidates for manual review rather than silently failing

### `github_analyzer.py`
- Returns `{"github_score": None, "error": "no_github_url"}` for blank GitHub URLs
- scorer.py treats this as a missing signal and renormalizes weights accordingly
- 5000 req/hr GitHub PAT rate limit — ~5–10 calls per candidate, safe for 50+ candidates

### `ai_evaluator.py`
- Two signals stored independently: `embedding_sim` (MiniLM) and `jd_match` (LLM)
- If they diverge significantly, the dashboard shows both — the recruiter sees the discrepancy rather than a silently averaged number
- LLM fallback: if Groq fails after 3 retries, `embedding_sim` is used as `jd_match` proxy with a note in `llm_reasoning`
- Rate limit: 1-second delay between Groq calls to stay within free-tier limits (~30 RPM)

### `emailer.py`
- Gmail SMTP + app password (no paid transactional email API)
- Subject format: `"Assessment Link — {name} (s_no {sno})"` to disambiguate in a shared recruiter inbox
- Per-candidate try/except: one failure doesn't crash the batch

### `calendar_scheduler.py`
- Google Calendar API, OAuth2, personal Gmail (no Google Workspace billing)
- `conferenceData.createRequest` auto-generates a Meet link
- Adding candidate as attendee triggers Google's own invite email — one API call covers Calendar + Meet + invite
- Event title: `"Interview — {name} (s_no {sno})"` for inbox disambiguation
- Token handling: base64-decode → `Credentials.from_authorized_user_info()` → refresh before each call

---

## Tech Choices — Justification

| Choice | Why | Rejected alternative |
|---|---|---|
| **Streamlit, single app** | Native tables/uploaders/buttons = zero frontend build time; one deploy target | FastAPI + React: doubles surface for no additional requirements coverage |
| **SQLite** | Zero-setup, fine for demo scale; `screening.db` is a single file | Postgres: correct for production — Supabase free tier is the stated upgrade path |
| **MiniLM-L6-v2 (local)** | Free, fast, deterministic; acts as a cheap guardrail on LLM scores | OpenAI embeddings: paid, network dependency for something a local model handles |
| **Groq / Llama 3.3 70B** | Free tier, fast enough for 10 candidates, structured JSON output | GPT-4o/Claude: better reasoning but not free; easy swap-in for production |
| **GitHub REST API, repo-level** | Free (5000 req/hr with PAT), satisfies "repository-level evaluation" explicitly | GraphQL API: more efficient at scale, simpler to debug with REST for a demo |
| **Gmail SMTP + app password** | Free, satisfies "candidates use their own email service" literally | SendGrid/Postmark: paid past free tier, violates the stated constraint |
| **Google Calendar API, personal OAuth2** | Free, no Workspace needed, `conferenceData` gives real Meet link + auto-invite in one call | Calendly: adds a third-party dependency the brief doesn't ask for |
| **Streamlit Community Cloud** | Free, public URL, connects directly to GitHub, purpose-built for exactly this app type | Render/Railway: also viable, but adds configuration overhead |

---

## What Would Change for Production

| Component | Demo | Production |
|---|---|---|
| Database | SQLite | Supabase Postgres |
| Processing | Synchronous | Celery + Redis job queue |
| LLM | Groq free tier | Paid model with SLA (GPT-4o, Claude) |
| Scale | ~10–50 candidates | Async processing, rate-limit pooling |
| Auth | Single recruiter | Supabase Auth or Clerk (multi-recruiter) |
| Secrets | Streamlit secrets | AWS Secrets Manager / GCP Secret Manager |

---

## Cost Summary

| Service | Usage | Cost |
|---|---|---|
| Streamlit Community Cloud | Hosting | $0 |
| GitHub REST API | 5000 req/hr free with PAT | $0 |
| HuggingFace MiniLM | Local inference | $0 |
| Groq (Llama 3.3 70B) | Free tier | $0 |
| Gmail SMTP | App password | $0 |
| Google Calendar API | Personal OAuth | $0 |
| SQLite | Local file | $0 |
| **Total** | | **$0** |

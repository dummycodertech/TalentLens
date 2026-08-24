# 📖 User Guide — AI Candidate Screening Platform
### myNachiketa GTM Engineering Intern Assignment

**For:** Recruiters using the hosted app
**Reading time:** ~10 minutes

---

## Overview

This platform automates the entire candidate evaluation pipeline — from receiving applications to scheduling final interviews — using AI. You interact with a 5-tab web app. Each tab represents one stage of the pipeline.

```
Tab 1: Upload & Configure → Tab 2: Run Pipeline → Tab 3: Rankings
→ Tab 4: Test Results → Tab 5: Schedule Interviews
```

---

## Before You Begin

You need to have the following ready before starting:

| Item | What it is | Where to get it |
|---|---|---|
| `candidates.csv` | CSV file with all candidate data | Export from your application form |
| Job Description | Full text of the JD you're hiring for | Your internal JD document |
| API keys | Set up once, stored in secrets.toml | See README → Local Setup |
| `test_results.csv` | Test scores after candidates complete assessment | From your test platform |

---

## Step-by-Step Guide

---

### Tab 1 — Upload & Configure

**Purpose:** Load your candidates and job description into the system.

**Step 1: Prepare your CSV**

Your candidates CSV must have these columns (order doesn't matter):

```
s_no, name, email, college, branch, cgpa, best_ai_project, research_work, github, resume
```

- `s_no` — a unique number per candidate (1, 2, 3... or any non-contiguous numbers)
- `github` — candidate's GitHub profile URL (leave blank if they don't have one)
- `resume` — the **Google Drive sharing link** to their PDF resume
  - Must look like: `https://drive.google.com/file/d/XXXXXXXXX/view?usp=sharing`
  - The file must be set to "Anyone with the link can view"

> ⚠️ If multiple candidates share the same email address (e.g. all applications went to a recruiter inbox), that is fine — the system deduplicates by `s_no`, not email.

**Step 2: Upload the CSV**

1. Click **"Choose candidates CSV"** on the left side of Tab 1
2. Select your `candidates.csv` file
3. A preview of the first 5 rows will appear — verify it looks correct
4. Click **⬆️ Load Candidates**
5. You'll see a success message: *"Loaded X candidates"*

**Step 3: Paste the Job Description**

1. On the right side of Tab 1, paste the full job description text into the text area
2. The more detailed the JD, the better the AI matching will be
3. Aim for at least 100 words — short JDs give less accurate results

> 💡 The JD is used for two things: semantic similarity matching (comparing candidate text to JD text) and LLM evaluation (the AI reads both and scores the fit).

---

### Tab 2 — Run Pipeline

**Purpose:** Automatically evaluate all candidates in sequence.

**Before you run, check the API Status panel in the left sidebar:**
- 🟢 Groq API connected — required for AI scoring
- 🟢 GitHub PAT connected — required for GitHub analysis
- 🟢 Gmail SMTP connected — required for sending test emails
- 🟢 Google Calendar connected — required for scheduling interviews

**To run the pipeline:**

1. Click **▶️ Run Evaluation Pipeline**
2. A progress bar appears at the top
3. A live status table shows each candidate's progress in real time
4. The pipeline runs 4 stages automatically:

| Stage | What happens | Time estimate |
|---|---|---|
| **Stage 1: Resume Parsing** | Downloads each PDF from Google Drive, extracts text | ~2–5 sec per candidate |
| **Stage 2: GitHub Analysis** | Fetches all public repos, scores activity and quality | ~3–8 sec per candidate |
| **Stage 3: AI Evaluation** | Runs MiniLM embedding + Groq LLM for each candidate | ~5–15 sec per candidate |
| **Stage 4: Scoring** | Computes final weighted scores and assigns ranks | <1 sec total |

5. When done, you'll see: *"Pipeline finished. View results in the Rankings tab."*

**Common status messages and what they mean:**

| Status | Meaning | Action needed? |
|---|---|---|
| `resume_failed: download_failed` | Could not download the PDF — check Drive sharing permissions | Fix the resume URL / sharing settings |
| `scan_warning: low text yield` | PDF might be image-based (scanned) — text extraction was poor | Review manually, update resume if possible |
| `github_failed: no_github_url` | Candidate left GitHub field blank | Normal — score is renormalized automatically |
| `llm_failed: embedding used as fallback` | Groq API was temporarily unavailable, used embedding score instead | Safe to proceed — slight reduction in accuracy |
| `ranked` | Successfully evaluated and scored | ✅ All good |

---

### Tab 3 — Rankings

**Purpose:** View all ranked candidates with their AI scores, reasoning, and GitHub breakdown.

**Reading the rankings:**

Each candidate appears as a card (top 3 auto-expanded). Click any card to expand it. Inside you'll see:

- **Final Score** — the overall weighted score (0–100)
- **JD Match** — how well the candidate's text matches the job description
- **GitHub Score** — based on actual repo analysis (activity, commits, README quality, originality)
- **Test Score** — only populated after test results are uploaded in Tab 4
- **Score Breakdown bars** — visual breakdown of each signal
- **🤖 AI Reasoning** — the Groq LLM's explanation of why the candidate scored that way
- **🐙 GitHub Breakdown** — a table of their actual repos with commit counts, languages, README status, and days since last push

**Adjusting weights (left sidebar):**

The sliders on the left control how much each signal contributes to the final score:

- **JD Match (w1)** — weight for job description relevance
- **Project Quality (w2)** — weight for AI project and research quality
- **GitHub Score (w3)** — weight for GitHub activity and code quality
- **Test Score (w4)** — weight for assessment performance

Weights automatically normalize to sum to 100%. You can change them at any time — click **Re-rank** (Tab 4) to apply new weights to all candidates.

**Sending Test Links:**

1. Scroll down to **"📨 Send Test Links"** at the bottom of Tab 3
2. Paste your assessment URL (Google Form, HackerRank, etc.) in the **Assessment URL** field
3. Set the **minimum score threshold** — only candidates above this score will receive the test link
4. Click **📨 Send Test Links to X candidates**
5. Each candidate receives a branded email with the link — the subject line includes their name and candidate number for easy inbox tracking

> 📧 Test emails are sent to whatever email is in your candidates CSV. For demo purposes, all test emails go to the same address you configured.

---

### Tab 4 — Test Results

**Purpose:** Upload assessment scores, re-rank candidates, and mark the shortlist.

**Step 1: Prepare your test results CSV**

The file needs exactly 3 columns:

```
s_no, test_la, test_code
```

- `s_no` — must match the candidate numbers from your original CSV
- `test_la` — logical aptitude score (0–100)
- `test_code` — coding challenge score (0–100)

> You don't need a row for every candidate — only those who completed the test.

**Step 2: Upload and re-rank**

1. Click **Browse files** and upload your `test_results.csv`
2. Preview the data to verify it loaded correctly
3. Click **🔄 Merge & Re-rank**
4. The system merges scores into the database (joined by `s_no`) and re-runs the full scoring formula with test scores now included
5. The updated rankings table appears below

**Step 3: Mark your shortlist**

1. Use the **"Shortlist threshold"** slider to set your cut-off score
2. The app shows how many candidates fall above the threshold
3. Click **✅ Mark X as Shortlisted**
4. Those candidates' status changes to `shortlisted` and they appear in Tab 5

---

### Tab 5 — Interviews

**Purpose:** Automatically create Google Calendar events with Meet links and send invites.

> ⚠️ Requires Google Calendar OAuth to be configured. See README → Local Setup → GOOGLE_OAUTH_CLIENT_JSON.

**Step 1: Configure interview slots**

1. Set the **First interview date** (the date of your first interview)
2. Set the **Start time (IST)** (time of the first slot)
3. Set **Minutes between interviews** (how long each slot is — default 45 min)

The app shows you the first and last slot times before you confirm.

**Step 2: Schedule**

1. Click **📅 Schedule X Interviews**
2. The app calls the Google Calendar API for each shortlisted candidate:
   - Creates a calendar event on your recruiter calendar
   - Auto-generates a Google Meet link (no third-party tools needed)
   - Adds the candidate as an attendee
   - Google sends its own professional invite email to the candidate
3. Each candidate appears in the **Scheduled Interviews** panel on the right with their slot time and a "Join Meet" link

> ℹ️ Google sends its own invite email to the candidate directly. You do not need to send a separate email.

**Step 3: Verify**

- Check your Google Calendar — all events should appear with Meet links
- The candidate will receive a calendar invite to their email from Google
- The status of each candidate updates to `interview_scheduled`

---

## Frequently Asked Questions

**Q: Can I run the pipeline again if I add more candidates?**
A: Yes. New candidates added to the CSV get `uploaded` status and will be processed on the next pipeline run. Already-ranked candidates are skipped.

**Q: What if a candidate doesn't have a GitHub profile?**
A: Leave the `github` column blank. The system detects this and automatically redistributes that weight to the other signals — the candidate's final score still scales from 0–100 fairly.

**Q: Can I change the scoring weights after the pipeline runs?**
A: Yes. Adjust the sliders in the sidebar, then go to Tab 4 and click **Merge & Re-rank** — even without new test results, this re-runs the scorer with the new weights.

**Q: The resume shows `download_failed` — what do I do?**
A: The Google Drive sharing setting on that PDF is probably restricted. Ask the candidate to re-share with "Anyone with the link can view," then reset the database and re-run.

**Q: What is a "scan warning"?**
A: This means the PDF appears to be a scanned image rather than a text-based PDF. Text extraction was limited. The candidate's resume text is probably minimal — you should review their profile manually and consider asking for a text-based PDF.

**Q: Can two candidates share the same email?**
A: Yes. The system uses `s_no` (candidate number) as the primary key, not email. Multiple candidates can have the same email address (e.g., if all applications went to a recruiter inbox).

**Q: How do I reset and start over?**
A: Click **🗑️ Reset Database** at the bottom of the left sidebar. This clears all candidate data from the database. You'll need to re-upload the CSV.

---

## Sidebar Controls Reference

| Control | Description |
|---|---|
| **JD Match / Project Quality / GitHub / Test Score sliders** | Adjust WSM weights — renormalize automatically |
| **API Status indicators** | Shows which integrations are live — must be 🟢 green before pipeline |
| **Reset Database** | Clears all data — use when starting a new hiring round |

---

## Support & Troubleshooting

| Problem | Solution |
|---|---|
| Pipeline stuck / no progress | Check the Pipeline Log (expandable at bottom of Tab 2) for specific errors |
| Groq API errors | Free tier has rate limits (~30 RPM) — wait 1 minute and retry |
| GitHub API returns 401 | GitHub PAT expired — generate a new one at github.com/settings/tokens |
| Gmail send failed | App password may be wrong — regenerate at myaccount.google.com/apppasswords |
| Calendar scheduling fails | OAuth token expired — re-run the `generate_token_locally()` script |
| All scores are very similar | The dummy PDF used in demo has no text — use real resumes for meaningful scores |

---

*For technical questions about the architecture, refer to `ARCHITECTURE.md`.*
*For setup and deployment, refer to `README.md`.*

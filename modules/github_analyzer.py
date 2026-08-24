"""
modules/github_analyzer.py — Per-repo GitHub analysis via REST API.

C7: Blank/null github_url returns {"github_score": None, "error": "no_github_url"}.
    scorer.py calls get_active_weights() which zeroes and redistributes w3.
C9: analyze_all() iterates over actual DB rows, never range().
"""

import re
import json
import time
from datetime import datetime, timezone, timedelta

import requests

# ─── Constants ─────────────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"
HEADERS_BASE = {"Accept": "application/vnd.github+json"}
SIX_MONTHS_AGO = (datetime.now(timezone.utc) - timedelta(days=182)).isoformat()

# Sub-formula weights (transparent, stored in JSON breakdown per PRD)
GITHUB_SUB_WEIGHTS = {
    "activity_recency":     0.35,
    "commit_frequency":     0.25,
    "original_repo_ratio":  0.20,
    "documentation_quality": 0.20,
}


# ─── GitHub username extraction ────────────────────────────────────────────────

def extract_github_username(github_url: str) -> str | None:
    """Extract username from various GitHub URL formats."""
    if not github_url or not isinstance(github_url, str):
        return None
    github_url = github_url.strip().rstrip("/")
    # Handle: https://github.com/username or github.com/username
    m = re.match(r"(?:https?://)?github\.com/([a-zA-Z0-9_-]+)/?$", github_url)
    if m:
        return m.group(1)
    # Handle bare usernames (no URL)
    if re.match(r"^[a-zA-Z0-9_-]+$", github_url):
        return github_url
    return None


# ─── API helpers ───────────────────────────────────────────────────────────────

def _headers(pat: str | None) -> dict:
    h = dict(HEADERS_BASE)
    if pat:
        h["Authorization"] = f"Bearer {pat}"
    return h


def _get(url: str, pat: str | None, params: dict | None = None, retries: int = 3) -> dict | list | None:
    """GET with retry on rate-limit (403/429)."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(pat), params=params, timeout=15)
            if resp.status_code in (403, 429):
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 1)
                time.sleep(min(wait, 30))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


# ─── Per-repo analysis ─────────────────────────────────────────────────────────

def _get_commit_count_6mo(owner: str, repo: str, pat: str | None) -> int:
    """Count commits in the last 6 months. Returns 0 on failure."""
    # GitHub returns max 100 per page; we cap at 3 pages = 300 commits max for score
    total = 0
    for page in range(1, 4):
        data = _get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            pat,
            params={"since": SIX_MONTHS_AGO, "per_page": 100, "page": page},
        )
        if not data or not isinstance(data, list):
            break
        total += len(data)
        if len(data) < 100:
            break
    return total


def _has_readme(owner: str, repo: str, pat: str | None) -> tuple[bool, int]:
    """Check if README exists and return its approximate length."""
    data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/readme", pat)
    if not data or not isinstance(data, dict):
        return False, 0
    size = data.get("size", 0)
    return True, size


def analyze_repo(repo: dict, owner: str, pat: str | None) -> dict:
    """Analyze a single repo. Returns a structured dict."""
    name = repo.get("name", "")
    is_fork = repo.get("fork", False)
    last_push = repo.get("pushed_at") or repo.get("updated_at") or ""
    language = repo.get("language") or "Unknown"
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    size_kb = repo.get("size", 0)

    # Recency: days since last push (0 = today, 365+ = dead)
    days_since_push = 999
    if last_push:
        try:
            pushed_dt = datetime.fromisoformat(last_push.replace("Z", "+00:00"))
            days_since_push = (datetime.now(timezone.utc) - pushed_dt).days
        except Exception:
            pass

    commit_count_6mo = 0
    if not is_fork and name:
        commit_count_6mo = _get_commit_count_6mo(owner, name, pat)

    has_readme, readme_size = _has_readme(owner, name, pat)

    return {
        "name": name,
        "language": language,
        "stars": stars,
        "forks": forks,
        "size_kb": size_kb,
        "is_fork": is_fork,
        "days_since_push": days_since_push,
        "commit_count_6mo": commit_count_6mo,
        "has_readme": has_readme,
        "readme_size_bytes": readme_size,
    }


# ─── Score computation ─────────────────────────────────────────────────────────

def _normalize(value: float, min_v: float, max_v: float) -> float:
    """Clamp and normalize to [0, 1]."""
    if max_v <= min_v:
        return 0.0
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


def compute_github_score(repos_data: list[dict]) -> tuple[float, dict]:
    """
    Weighted Sum Model over four sub-signals. Returns (score_0_100, breakdown_dict).

    Sub-formula:
        github_score = 0.35 * activity_recency
                     + 0.25 * commit_frequency
                     + 0.20 * original_repo_ratio
                     + 0.20 * documentation_quality
    """
    if not repos_data:
        return 0.0, {"error": "no_repos"}

    originals = [r for r in repos_data if not r["is_fork"]]
    total = len(repos_data)
    orig_count = len(originals)

    # 1. activity_recency: based on best (most recent) non-fork repo push
    best_days = min((r["days_since_push"] for r in originals), default=999)
    # 0 days → 1.0, 365 days → 0.0, 730+ → 0.0
    activity_recency = _normalize(365 - best_days, 0, 365)

    # 2. commit_frequency: total 6-month commits across original repos (cap at 100)
    total_commits_6mo = sum(r["commit_count_6mo"] for r in originals)
    commit_frequency = _normalize(total_commits_6mo, 0, 100)

    # 3. original_repo_ratio
    original_repo_ratio = orig_count / total if total > 0 else 0.0

    # 4. documentation_quality: proportion of original repos with READMEs
    with_readme = sum(1 for r in originals if r["has_readme"])
    doc_ratio = with_readme / orig_count if orig_count > 0 else 0.0
    # Bonus: average README length signal (normalized to 2000 bytes)
    avg_readme_size = (
        sum(r["readme_size_bytes"] for r in originals if r["has_readme"]) / max(with_readme, 1)
    )
    readme_quality = _normalize(avg_readme_size, 0, 2000)
    documentation_quality = 0.6 * doc_ratio + 0.4 * readme_quality

    # Weighted sum
    sub_scores = {
        "activity_recency": round(activity_recency, 4),
        "commit_frequency": round(commit_frequency, 4),
        "original_repo_ratio": round(original_repo_ratio, 4),
        "documentation_quality": round(documentation_quality, 4),
    }
    github_score = sum(
        GITHUB_SUB_WEIGHTS[k] * v for k, v in sub_scores.items()
    ) * 100  # scale to 0–100

    breakdown = {
        "sub_scores": sub_scores,
        "sub_weights": GITHUB_SUB_WEIGHTS,
        "total_repos": total,
        "original_repos": orig_count,
        "total_commits_6mo": total_commits_6mo,
        "repos": repos_data,
    }

    return round(github_score, 2), breakdown


# ─── Main analyze entry point ──────────────────────────────────────────────────

def analyze_github(github_url: str | None, pat: str | None) -> dict:
    """
    Full pipeline for one candidate.

    C7: Returns {"github_score": None, "error": "no_github_url"} for blank/null URLs.
         scorer.get_active_weights() will zero + redistribute w3 for these candidates.

    Returns:
        {
            "github_score": float | None,
            "breakdown": dict | None,
            "error": str | None
        }
    """
    # C7: Explicit early return for missing GitHub URL
    if not github_url or not isinstance(github_url, str) or not github_url.strip():
        return {"github_score": None, "breakdown": None, "error": "no_github_url"}

    username = extract_github_username(github_url.strip())
    if not username:
        return {"github_score": None, "breakdown": None, "error": "invalid_github_url"}

    repos_raw = _get(
        f"{GITHUB_API}/users/{username}/repos",
        pat,
        params={"per_page": 100, "sort": "pushed"},
    )
    if repos_raw is None:
        return {"github_score": None, "breakdown": None, "error": "github_api_error"}
    if not isinstance(repos_raw, list) or len(repos_raw) == 0:
        return {"github_score": 0.0, "breakdown": {"error": "no_public_repos"}, "error": None}

    repos_data = []
    for repo in repos_raw:
        try:
            repos_data.append(analyze_repo(repo, username, pat))
        except Exception as e:
            repos_data.append({"name": repo.get("name", "?"), "error": str(e)})

    score, breakdown = compute_github_score(repos_data)
    return {"github_score": score, "breakdown": breakdown, "error": None}


def analyze_all(db, pat: str | None, status_callback=None) -> dict[int, dict]:
    """
    Analyze GitHub for all candidates with status='resume_parsed'.
    C9: Iterates over actual DB rows.

    Args:
        db: the db module
        pat: GitHub Personal Access Token
        status_callback: optional callable(s_no, status_str)

    Returns: dict of s_no -> analyze result
    """
    candidates = db.get_by_status("resume_parsed")
    results = {}

    for row in candidates:
        sno = row["s_no"]
        github_url = row["github_url"]

        if status_callback:
            status_callback(sno, "analyzing GitHub…")

        result = analyze_github(github_url, pat)
        results[sno] = result

        breakdown_json = json.dumps(result.get("breakdown") or {})

        if result["error"] == "no_github_url":
            # C7: Not a failure — just missing signal. Store None score, advance normally.
            db.upsert_score(
                sno,
                github_score=None,
                github_breakdown=json.dumps({"note": "no_github_url"}),
            )
            db.update_status(sno, "github_analyzed", error="no_github_url (score renormalized)")
        elif result["error"]:
            db.upsert_score(sno, github_score=None, github_breakdown=breakdown_json)
            db.update_status(sno, "github_failed", error=result["error"])
        else:
            db.upsert_score(
                sno,
                github_score=result["github_score"],
                github_breakdown=breakdown_json,
            )
            db.update_status(sno, "github_analyzed")

        if status_callback:
            final_status = "github_analyzed" if not result["error"] or result["error"] == "no_github_url" else "github_failed"
            status_callback(sno, final_status)

    return results

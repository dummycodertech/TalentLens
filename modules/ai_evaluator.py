"""
modules/ai_evaluator.py — MiniLM embedding similarity + Groq LLM qualitative reasoning.

Two deliberately separate signals:
  1. Embedding similarity (local, free, deterministic) — fast sanity check
  2. LLM structured JSON (Groq / Llama 3.3 70B, free tier) — qualitative reasoning

Both are stored independently so the dashboard can show divergence as a signal,
not suppress it.
"""

import json
import time
import re

# ─── Embedding signal ──────────────────────────────────────────────────────────

_model = None  # Loaded once, cached at module level


def _get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def compute_embedding_similarity(jd_text: str, candidate_text: str) -> float:
    """
    Cosine similarity between JD and candidate text, scaled to 0–100.
    Local, free, deterministic. No API cost.
    """
    if not jd_text or not candidate_text:
        return 0.0
    model = _get_embedding_model()
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    jd_emb = model.encode([jd_text])
    cand_emb = model.encode([candidate_text])
    sim = cosine_similarity(jd_emb, cand_emb)[0][0]
    # Cosine sim is -1 to 1; clamp to 0–1 then scale
    return round(float(max(0.0, sim)) * 100, 2)


# ─── LLM signal (Groq) ────────────────────────────────────────────────────────

GROQ_MODEL = "groq/compound"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

EVAL_PROMPT_TEMPLATE = """You are an expert technical recruiter evaluating a candidate for the following role.

## Job Description
{jd_text}

## Candidate Profile
**Name:** {name}
**College:** {college} | **Branch:** {branch} | **CGPA:** {cgpa}
**Best AI Project:** {best_ai_project}
**Research Work:** {research_work}

## Resume Text (extracted)
{resume_text}

## Task
Evaluate this candidate against the job description. Return ONLY a valid JSON object with these exact keys:
{{
  "jd_match": <integer 0-100>,
  "project_quality": <integer 0-100>,
  "reasoning": "<2-3 sentences explaining the scores, referencing specific skills or gaps>"
}}

Rules:
- jd_match: How well the candidate's skills and experience match the JD requirements
- project_quality: Quality, complexity, and relevance of AI/ML projects described
- reasoning: Be specific. Mention skills mentioned or missing. Do not hallucinate details not in the profile.
- Return ONLY the JSON object, no markdown, no explanation outside it."""


def build_prompt(jd_text: str, candidate: dict) -> str:
    resume_text = (candidate.get("resume_text") or "")[:3000]  # truncate to avoid token limits
    return EVAL_PROMPT_TEMPLATE.format(
        jd_text=jd_text[:2000],
        name=candidate.get("name", "Unknown"),
        college=candidate.get("college", ""),
        branch=candidate.get("branch", ""),
        cgpa=candidate.get("cgpa", ""),
        best_ai_project=candidate.get("best_ai_project", ""),
        research_work=candidate.get("research_work", ""),
        resume_text=resume_text or "(no resume text extracted)",
    )


def _parse_llm_json(raw: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    # Find the JSON object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        # Validate required keys
        if not all(k in data for k in ("jd_match", "project_quality", "reasoning")):
            return None
        # Coerce to expected types
        data["jd_match"] = max(0, min(100, int(data["jd_match"])))
        data["project_quality"] = max(0, min(100, int(data["project_quality"])))
        data["reasoning"] = str(data["reasoning"])
        return data
    except Exception:
        return None


def call_groq(prompt: str, groq_api_key: str, retries: int = 3) -> dict | None:
    """
    Call Groq API with exponential backoff on rate limits.
    Returns parsed JSON dict or None on failure.
    """
    import requests as req

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise technical recruiter. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    for attempt in range(retries):
        try:
            resp = req.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)  # 4s, 8s, 16s
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _parse_llm_json(content)
            if parsed:
                return parsed
            # If parsing failed, retry once
            if attempt < retries - 1:
                time.sleep(1)
                continue
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ─── Combined evaluation ───────────────────────────────────────────────────────

def evaluate_candidate(jd_text: str, candidate: dict, groq_api_key: str) -> dict:
    """
    Run both evaluation signals for one candidate.

    Returns:
        {
            "embedding_sim": float (0-100),   — embedding cosine similarity
            "jd_match": float (0-100),         — LLM score
            "project_quality": float (0-100),  — LLM score
            "reasoning": str,                  — LLM explanation
            "error": str | None
        }
    """
    # Build candidate text for embedding
    candidate_text = " ".join(filter(None, [
        candidate.get("resume_text", ""),
        candidate.get("best_ai_project", ""),
        candidate.get("research_work", ""),
        candidate.get("branch", ""),
    ]))

    # Signal 1: Embedding similarity (always runs, even if text is short)
    embedding_sim = compute_embedding_similarity(jd_text, candidate_text)

    # Signal 2: LLM structured reasoning
    prompt = build_prompt(jd_text, candidate)
    llm_result = call_groq(prompt, groq_api_key)

    if llm_result is None:
        return {
            "embedding_sim": embedding_sim,
            "jd_match": embedding_sim,  # fall back to embedding if LLM fails
            "project_quality": 50.0,    # neutral fallback
            "reasoning": f"LLM evaluation failed. Embedding similarity used as JD match proxy: {embedding_sim:.1f}",
            "error": "llm_failed",
        }

    return {
        "embedding_sim": embedding_sim,
        "jd_match": float(llm_result["jd_match"]),
        "project_quality": float(llm_result["project_quality"]),
        "reasoning": llm_result["reasoning"],
        "error": None,
    }


def evaluate_all(jd_text: str, db, groq_api_key: str, status_callback=None) -> dict[int, dict]:
    """
    Evaluate all candidates with status='github_analyzed' (or 'github_failed').
    C9: Iterates over actual DB rows.

    Args:
        jd_text: full job description text
        db: the db module
        groq_api_key: Groq API key from secrets
        status_callback: optional callable(s_no, status_str)

    Returns: dict of s_no -> evaluation result
    """
    # Include github_failed candidates — they still get AI-scored
    ready_statuses = ["github_analyzed", "github_failed"]
    candidates = []
    for status in ready_statuses:
        candidates.extend(db.get_by_status(status))
        candidates.extend(db.get_by_status(status.replace("failed", "analyzed")))

    # De-duplicate by s_no
    seen = set()
    unique_candidates = []
    for row in candidates:
        if row["s_no"] not in seen:
            seen.add(row["s_no"])
            unique_candidates.append(row)

    # Actually: just get all past github step
    unique_candidates = []
    seen = set()
    for status in ["github_analyzed", "github_failed"]:
        for row in db.get_by_status(status):
            if row["s_no"] not in seen:
                seen.add(row["s_no"])
                unique_candidates.append(row)

    results = {}

    for row in unique_candidates:
        sno = row["s_no"]
        if status_callback:
            status_callback(sno, "running AI evaluation…")

        candidate_dict = dict(row)
        result = evaluate_candidate(jd_text, candidate_dict, groq_api_key)
        results[sno] = result

        db.upsert_score(
            sno,
            jd_match=result["jd_match"],
            embedding_sim=result["embedding_sim"],
            project_quality=result["project_quality"],
            llm_reasoning=result["reasoning"],
        )

        if result["error"] == "llm_failed":
            db.update_status(sno, "ai_scored", error="llm_failed: embedding used as fallback")
        else:
            db.update_status(sno, "ai_scored")

        if status_callback:
            status_callback(sno, "ai_scored")

        # Small delay to respect Groq free-tier rate limits (~30 RPM on free)
        time.sleep(1.0)

    return results

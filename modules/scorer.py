"""
modules/scorer.py — Weighted Sum Model (WSM) scoring and ranking.

Scoring methodology: WSM is the simplest form of Multi-Criteria Decision Analysis (MCDA).
It's the correct approach here because:
  - No historical outcome labels exist — trained models and SHAP/LIME are inapplicable.
  - The four terms map 1:1 to the assignment's own stated criteria (resume, GitHub, JD, test).
  - It's maximally auditable — every input, weight, and output is stored and visible.

The AI sophistication is in signal generation (embeddings, LLM, per-repo GitHub analysis).
The combination is deliberately transparent.

C3 + C7: get_active_weights() renormalizes for ANY missing signal (test or GitHub),
          so final_score always scales 0–100.
C4: score_all() reads weights from session_state on every call — never hardcoded.
C9: Iterates over actual DB rows, never range().
"""

import json

# ─── Default weights (configurable via sidebar sliders) ────────────────────────
# Derived directly from the assignment's four stated evaluation dimensions.

DEFAULT_WEIGHTS = {
    "w1": 0.30,  # jd_match        — JD relevance (embedding + LLM)
    "w2": 0.25,  # project_quality — AI/ML project quality (LLM)
    "w3": 0.25,  # github_score    — GitHub activity & depth (REST API)
    "w4": 0.20,  # test_score      — Assessment test performance
}

WEIGHT_LABELS = {
    "w1": "JD Match",
    "w2": "Project Quality",
    "w3": "GitHub Score",
    "w4": "Test Score",
}


# ─── Weight renormalization (C3 + C7) ─────────────────────────────────────────

def get_active_weights(weights: dict, has_test: bool, has_github: bool) -> dict:
    """
    Renormalize weights when signals are missing, so final_score always
    scales 0–100 regardless of which signals are available.

    C3: If no test score yet, w4 → 0, others renormalized.
    C7: If no GitHub URL, w3 → 0, others renormalized.
    Both can be missing simultaneously.
    """
    active = dict(weights)
    if not has_test:
        active["w4"] = 0.0
    if not has_github:
        active["w3"] = 0.0
    remaining = sum(active.values())
    if remaining <= 0:
        return {k: 0.0 for k in active}
    return {k: v / remaining for k, v in active.items()}


# ─── Score computation ─────────────────────────────────────────────────────────

def compute_test_score(test_la: float | None, test_code: float | None) -> float | None:
    """
    Combine test_la and test_code into a single test_score (0–100).
    Equal weighting. Returns None if both are missing.
    """
    if test_la is None and test_code is None:
        return None
    values = [v for v in [test_la, test_code] if v is not None]
    return round(sum(values) / len(values), 2)


def compute_final_score(
    jd_match: float | None,
    project_quality: float | None,
    github_score: float | None,
    test_score: float | None,
    weights: dict,
) -> tuple[float, dict]:
    """
    Weighted Sum Model: final_score = Σ(wᵢ × scoreᵢ) over active signals.
    Returns (final_score_0_100, active_weights_used).
    """
    has_test = test_score is not None
    has_github = github_score is not None

    active_w = get_active_weights(weights, has_test=has_test, has_github=has_github)

    score = (
        active_w["w1"] * (jd_match or 0.0)
        + active_w["w2"] * (project_quality or 0.0)
        + active_w["w3"] * (github_score or 0.0)
        + active_w["w4"] * (test_score or 0.0)
    )
    return round(score, 2), active_w


# ─── Ranking ───────────────────────────────────────────────────────────────────

def rank_rows(rows: list[dict]) -> list[dict]:
    """
    Sort by final_score descending, assign integer rank.
    C9: Operates on a list of dicts from DB query results, not positional index.
    """
    sorted_rows = sorted(rows, key=lambda r: r.get("final_score") or 0.0, reverse=True)
    for i, row in enumerate(sorted_rows):
        row["rank"] = i + 1
    return sorted_rows


# ─── Main score_all entry point ────────────────────────────────────────────────

def score_all(db, weights: dict | None = None, status_callback=None) -> list[dict]:
    """
    Compute final_score for all candidates with status='ai_scored'.
    Also re-scores anyone already ranked (for re-rank after test upload).

    C4: Reads weights from caller (session_state['weights']) on every call.
        Never hardcodes w4=0.2 or any specific value.
    C9: Iterates over actual DB query rows.

    Args:
        db: the db module
        weights: dict with w1..w4. Defaults to DEFAULT_WEIGHTS if None.
        status_callback: optional callable(s_no, status_str)

    Returns: list of scored candidate dicts (for immediate UI render)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Score candidates at ai_scored AND re-score already ranked ones (for re-rank flow)
    target_statuses = ["ai_scored", "ranked", "test_sent", "test_scored", "shortlisted"]
    candidates = []
    seen = set()
    for status in target_statuses:
        for row in db.get_by_status(status):
            if row["s_no"] not in seen:
                seen.add(row["s_no"])
                candidates.append(row)

    scored = []

    for row in candidates:
        sno = row["s_no"]
        if status_callback:
            status_callback(sno, "computing score…")

        # Get current score record
        score_row = db.get_score(sno)

        jd_match = score_row["jd_match"] if score_row else None
        project_quality = score_row["project_quality"] if score_row else None
        github_score = score_row["github_score"] if score_row else None

        # Test score: average of test_la and test_code
        test_la = row["test_la"]
        test_code = row["test_code"]
        test_score = compute_test_score(test_la, test_code)

        final_score, active_w = compute_final_score(
            jd_match=jd_match,
            project_quality=project_quality,
            github_score=github_score,
            test_score=test_score,
            weights=weights,
        )

        db.upsert_score(
            sno,
            test_score=test_score,
            final_score=final_score,
        )

        # Keep previous rank temporarily; will be updated after global sort
        scored.append({
            "s_no": sno,
            "name": dict(row).get("name"),
            "jd_match": jd_match,
            "project_quality": project_quality,
            "github_score": github_score,
            "test_score": test_score,
            "final_score": final_score,
            "active_weights": active_w,
            "status": row["status"],
        })

    # Global rank across all scored candidates
    scored = rank_rows(scored)

    # Write ranks back to DB and advance status
    for item in scored:
        sno = item["s_no"]
        db.upsert_score(sno, rank=item["rank"])
        # Only advance status if currently at ai_scored
        if dict(db.get_candidate(sno)).get("status") == "ai_scored":
            db.update_status(sno, "ranked")
        if status_callback:
            status_callback(sno, "ranked")

    return scored


if __name__ == "__main__":
    # Quick unit test for the scoring formula
    w = DEFAULT_WEIGHTS

    # Test: all signals present
    score, aw = compute_final_score(80, 70, 90, 75, w)
    expected = 0.30 * 80 + 0.25 * 70 + 0.25 * 90 + 0.20 * 75
    assert abs(score - expected) < 0.01, f"Expected {expected}, got {score}"
    print(f"Full signal score: {score:.2f} ✓")

    # Test: no test score — weights renormalize
    score2, aw2 = compute_final_score(80, 70, 90, None, w)
    renorm_sum = w["w1"] + w["w2"] + w["w3"]
    expected2 = (w["w1"] / renorm_sum * 80) + (w["w2"] / renorm_sum * 70) + (w["w3"] / renorm_sum * 90)
    assert abs(score2 - expected2) < 0.01, f"Expected {expected2}, got {score2}"
    assert abs(sum(aw2.values()) - 1.0) < 0.001, "Weights don't sum to 1"
    print(f"No-test score: {score2:.2f} (max remains 100) ✓")

    # Test: no github + no test
    score3, aw3 = compute_final_score(80, 70, None, None, w)
    renorm_sum2 = w["w1"] + w["w2"]
    expected3 = (w["w1"] / renorm_sum2 * 80) + (w["w2"] / renorm_sum2 * 70)
    assert abs(score3 - expected3) < 0.01, f"Expected {expected3}, got {score3}"
    print(f"No-github, no-test score: {score3:.2f} ✓")

    print("All scorer unit tests passed.")

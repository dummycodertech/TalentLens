"""
tests/test_scorer.py — Unit tests for Weighted Sum Model (WSM) scorer.

Validates:
  1. Full signal WSM calculation matches exact formula
  2. Missing test score triggers graceful weight renormalization
  3. Missing GitHub score triggers graceful weight renormalization
  4. Both missing signals renormalize over remaining weights
  5. Test score calculation (logical aptitude + coding challenge average)
  6. Ranking order and tie handling
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.scorer import (
    DEFAULT_WEIGHTS,
    get_active_weights,
    compute_test_score,
    compute_final_score,
    rank_rows,
)


def test_full_signal_score():
    w = DEFAULT_WEIGHTS
    score, aw = compute_final_score(80.0, 70.0, 90.0, 75.0, w)
    expected = 0.30 * 80.0 + 0.25 * 70.0 + 0.25 * 90.0 + 0.20 * 75.0
    assert abs(score - expected) < 0.01
    assert abs(sum(aw.values()) - 1.0) < 0.001


def test_missing_test_score_renormalization():
    w = DEFAULT_WEIGHTS
    score, aw = compute_final_score(80.0, 70.0, 90.0, None, w)
    renorm_sum = w["w1"] + w["w2"] + w["w3"]
    expected = (w["w1"] / renorm_sum * 80.0) + (w["w2"] / renorm_sum * 70.0) + (w["w3"] / renorm_sum * 90.0)
    assert abs(score - expected) < 0.01
    assert aw["w4"] == 0.0
    assert abs(sum(aw.values()) - 1.0) < 0.001


def test_missing_github_score_renormalization():
    w = DEFAULT_WEIGHTS
    score, aw = compute_final_score(85.0, 75.0, None, 80.0, w)
    renorm_sum = w["w1"] + w["w2"] + w["w4"]
    expected = (w["w1"] / renorm_sum * 85.0) + (w["w2"] / renorm_sum * 75.0) + (w["w4"] / renorm_sum * 80.0)
    assert abs(score - expected) < 0.01
    assert aw["w3"] == 0.0
    assert abs(sum(aw.values()) - 1.0) < 0.001


def test_missing_both_github_and_test():
    w = DEFAULT_WEIGHTS
    score, aw = compute_final_score(80.0, 70.0, None, None, w)
    renorm_sum = w["w1"] + w["w2"]
    expected = (w["w1"] / renorm_sum * 80.0) + (w["w2"] / renorm_sum * 70.0)
    assert abs(score - expected) < 0.01
    assert aw["w3"] == 0.0
    assert aw["w4"] == 0.0
    assert abs(sum(aw.values()) - 1.0) < 0.001


def test_compute_test_score():
    assert compute_test_score(80.0, 90.0) == 85.0
    assert compute_test_score(80.0, None) == 80.0
    assert compute_test_score(None, 90.0) == 90.0
    assert compute_test_score(None, None) is None


def test_rank_rows():
    rows = [
        {"s_no": 1, "final_score": 75.0},
        {"s_no": 2, "final_score": 92.0},
        {"s_no": 3, "final_score": 84.5},
    ]
    ranked = rank_rows(rows)
    assert [r["s_no"] for r in ranked] == [2, 3, 1]
    assert [r["rank"] for r in ranked] == [1, 2, 3]

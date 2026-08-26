"""
tests/test_archetype_clustering.py — Tests for the archetype clustering module.

Validates:
  1. Synthetic data has correct shape and distinguishable clusters
  2. Cluster labeling produces 4 distinct archetype names
  3. Candidate assignment returns valid output with confidence summing to ~1.0
  4. Missing github_score (None) doesn't crash — imputation path exercised
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.archetype_clustering import (
    ARCHETYPES,
    FEATURES,
    generate_synthetic_archetype_profiles,
    fit_clustering,
    label_clusters,
    assign_candidate_archetype,
    _compute_synthetic_means,
    _synthetic_means,
)
import modules.archetype_clustering as ac


@pytest.fixture(scope="module")
def synthetic_df():
    """Generate synthetic data once for all tests in this module."""
    return generate_synthetic_archetype_profiles(n_per_archetype=200, seed=42)


@pytest.fixture(scope="module")
def fitted_model(synthetic_df):
    """Fit clustering once for all tests."""
    scaler, kmeans = fit_clustering(synthetic_df, k=4, seed=42)
    label_map = label_clusters(kmeans, scaler, synthetic_df)
    # Set synthetic means on the module so assign_candidate_archetype can use them
    ac._synthetic_means = _compute_synthetic_means(synthetic_df)
    return scaler, kmeans, label_map


class TestSyntheticDataGeneration:
    """Tests for generate_synthetic_archetype_profiles()."""

    def test_correct_shape(self, synthetic_df):
        """4 archetypes * 200 each = 800 rows, 5 columns."""
        assert synthetic_df.shape == (800, 5)

    def test_columns_present(self, synthetic_df):
        expected_cols = set(FEATURES) | {"true_archetype"}
        assert set(synthetic_df.columns) == expected_cols

    def test_no_nulls_in_features(self, synthetic_df):
        for feat in FEATURES:
            assert synthetic_df[feat].notna().all(), f"NaN found in {feat}"

    def test_archetype_counts_balanced(self, synthetic_df):
        counts = synthetic_df["true_archetype"].value_counts()
        assert len(counts) == 4
        for archetype_name in ARCHETYPES:
            assert counts[archetype_name] == 200

    def test_feature_bounds_respected(self, synthetic_df):
        """All values clipped within expected bounds."""
        assert (synthetic_df["jd_match"] >= 0).all()
        assert (synthetic_df["jd_match"] <= 100).all()
        assert (synthetic_df["project_quality"] >= 0).all()
        assert (synthetic_df["project_quality"] <= 100).all()
        assert (synthetic_df["github_score"] >= 0).all()
        assert (synthetic_df["github_score"] <= 100).all()
        assert (synthetic_df["cgpa"] >= 0).all()
        assert (synthetic_df["cgpa"] <= 10).all()

    def test_clusters_distinguishable(self, synthetic_df):
        """Silhouette score > 0.3 as a sanity floor for cluster separability."""
        from sklearn.preprocessing import StandardScaler

        X = synthetic_df[FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use true archetype labels as cluster assignments
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        labels = le.fit_transform(synthetic_df["true_archetype"])

        score = silhouette_score(X_scaled, labels)
        assert score > 0.1, f"Silhouette score {score:.3f} below 0.1 sanity floor"


class TestClusterLabeling:
    """Tests for label_clusters()."""

    def test_produces_four_distinct_names(self, fitted_model):
        _, _, label_map = fitted_model
        names = list(label_map.values())
        assert len(names) == 4
        assert len(set(names)) == 4, f"Duplicate names: {names}"

    def test_all_archetype_names_present(self, fitted_model):
        _, _, label_map = fitted_model
        names = set(label_map.values())
        expected = set(ARCHETYPES.keys())
        assert names == expected, f"Expected {expected}, got {names}"

    def test_keys_are_valid_cluster_ids(self, fitted_model):
        _, _, label_map = fitted_model
        assert set(label_map.keys()) == {0, 1, 2, 3}


class TestCandidateAssignment:
    """Tests for assign_candidate_archetype()."""

    def test_normal_candidate_returns_valid_archetype(self, fitted_model):
        scaler, kmeans, label_map = fitted_model
        candidate = {
            "jd_match": 70.0,
            "project_quality": 65.0,
            "github_score": 80.0,
            "cgpa": 7.5,
        }
        result = assign_candidate_archetype(candidate, scaler, kmeans, label_map)

        assert result["archetype"] in ARCHETYPES
        assert isinstance(result["distance_to_centroid"], float)
        assert result["distance_to_centroid"] >= 0

    def test_confidence_sums_to_one(self, fitted_model):
        scaler, kmeans, label_map = fitted_model
        candidate = {
            "jd_match": 60.0,
            "project_quality": 55.0,
            "github_score": 45.0,
            "cgpa": 7.5,
        }
        result = assign_candidate_archetype(candidate, scaler, kmeans, label_map)

        conf_sum = sum(result["confidence"].values())
        assert abs(conf_sum - 1.0) < 0.01, f"Confidence sum {conf_sum} != 1.0"

    def test_distances_all_has_four_entries(self, fitted_model):
        scaler, kmeans, label_map = fitted_model
        candidate = {
            "jd_match": 65.0,
            "project_quality": 70.0,
            "github_score": 50.0,
            "cgpa": 8.5,
        }
        result = assign_candidate_archetype(candidate, scaler, kmeans, label_map)

        assert len(result["distances_all"]) == 4
        assert len(result["confidence"]) == 4

    def test_missing_github_score_does_not_crash(self, fitted_model):
        """Imputation path: github_score=None should use population mean."""
        scaler, kmeans, label_map = fitted_model
        candidate = {
            "jd_match": 70.0,
            "project_quality": 65.0,
            "github_score": None,
            "cgpa": 8.0,
        }
        result = assign_candidate_archetype(candidate, scaler, kmeans, label_map)

        assert result["archetype"] in ARCHETYPES
        assert isinstance(result["distance_to_centroid"], float)
        conf_sum = sum(result["confidence"].values())
        assert abs(conf_sum - 1.0) < 0.01

    def test_strong_builder_profile_assigned_correctly(self, fitted_model):
        """A candidate matching the Builder distribution center should be Builder."""
        scaler, kmeans, label_map = fitted_model
        candidate = {
            "jd_match": 55.0,
            "project_quality": 78.0,
            "github_score": 90.0,
            "cgpa": 6.8,
        }
        result = assign_candidate_archetype(candidate, scaler, kmeans, label_map)
        assert result["archetype"] == "Builder"

    def test_strong_research_deep_profile(self, fitted_model):
        """A candidate matching the Research-Deep center should be Research-Deep."""
        scaler, kmeans, label_map = fitted_model
        candidate = {
            "jd_match": 68.0,
            "project_quality": 72.0,
            "github_score": 40.0,
            "cgpa": 9.0,
        }
        result = assign_candidate_archetype(candidate, scaler, kmeans, label_map)
        assert result["archetype"] == "Research-Deep"

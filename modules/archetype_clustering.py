"""
modules/archetype_clustering.py — Unsupervised candidate archetype clustering.

Uses K-means on 4 features (jd_match, project_quality, github_score, cgpa) to
assign each candidate to one of 4 named archetypes. Cluster centroids are fit
on synthetic profiles with known archetype structure; real candidates are
projected onto those centroids by feature distance.

This is purely descriptive — it surfaces "what kind of candidate is this"
as a complement to WSM's "how highly do they rank." The two answer different
questions and neither overrides the other.

Missing github_score is imputed with the synthetic population mean before
scaling — a stated limitation, same pattern as WSM missing-signal handling.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ─── Archetype definitions ────────────────────────────────────────────────────
# Each entry: feature_name -> (mean, std) for Gaussian sampling.
# Deliberately differentiated so K-means has real structure to find.

ARCHETYPES = {
    "Research-Deep": {
        "jd_match": (68, 7),
        "project_quality": (72, 6),
        "github_score": (40, 10),
        "cgpa": (9.0, 0.3),
    },
    "Builder": {
        "jd_match": (55, 8),
        "project_quality": (78, 7),
        "github_score": (90, 5),
        "cgpa": (6.8, 0.5),
    },
    "Fast-Learner": {
        "jd_match": (48, 8),
        "project_quality": (48, 8),
        "github_score": (35, 12),
        "cgpa": (7.2, 0.6),
    },
    "All-Rounder": {
        "jd_match": (75, 6),
        "project_quality": (70, 6),
        "github_score": (72, 7),
        "cgpa": (8.2, 0.4),
    },
}

FEATURES = ["jd_match", "project_quality", "github_score", "cgpa"]
FEATURE_BOUNDS = {
    "jd_match": (0, 100),
    "project_quality": (0, 100),
    "github_score": (0, 100),
    "cgpa": (0, 10),
}

CACHE_DIR = Path("model_cache")
SCALER_PATH = CACHE_DIR / "archetype_scaler.joblib"
KMEANS_PATH = CACHE_DIR / "archetype_kmeans.joblib"
LABEL_MAP_PATH = CACHE_DIR / "archetype_label_map.joblib"
SYNTHETIC_CSV_PATH = Path("sample_data") / "synthetic_archetypes.csv"


# ─── Synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic_archetype_profiles(
    n_per_archetype: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample from the archetype distributions. Each row is tagged with its
    true_archetype for cluster-label mapping after K-means.

    Returns a DataFrame with columns: jd_match, project_quality, github_score,
    cgpa, true_archetype.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for archetype_name, feature_dists in ARCHETYPES.items():
        for _ in range(n_per_archetype):
            row = {"true_archetype": archetype_name}
            for feat, (mean, std) in feature_dists.items():
                lo, hi = FEATURE_BOUNDS[feat]
                value = rng.normal(mean, std)
                value = float(np.clip(value, lo, hi))
                row[feat] = round(value, 2)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save to disk for reproducibility and inspection
    SYNTHETIC_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYNTHETIC_CSV_PATH, index=False)

    return df


# ─── Clustering ────────────────────────────────────────────────────────────────

def fit_clustering(
    df: pd.DataFrame,
    k: int = 4,
    seed: int = 42,
) -> tuple[StandardScaler, KMeans]:
    """
    Fit StandardScaler + KMeans on the synthetic data.
    Returns (scaler, kmeans_model).
    """
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(X_scaled)

    return scaler, kmeans


def label_clusters(
    kmeans_model: KMeans,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> dict[int, str]:
    """
    Map each cluster ID to the archetype name that dominates it, by majority
    vote of true_archetype within each cluster.

    Returns dict: cluster_id -> archetype_name.
    If two clusters would get the same name (shouldn't happen with well-separated
    distributions), the second one gets the name of the next-most-common
    archetype in that cluster.
    """
    X_scaled = scaler.transform(df[FEATURES].values)
    labels = kmeans_model.predict(X_scaled)
    df = df.copy()
    df["cluster"] = labels

    cluster_label_map = {}
    used_names = set()

    # Sort clusters by size (largest first) so the biggest cluster gets
    # first pick of its majority archetype name.
    cluster_sizes = df["cluster"].value_counts().sort_values(ascending=False)

    for cluster_id in cluster_sizes.index:
        cluster_rows = df[df["cluster"] == cluster_id]
        vote_counts = cluster_rows["true_archetype"].value_counts()

        for name in vote_counts.index:
            if name not in used_names:
                cluster_label_map[int(cluster_id)] = name
                used_names.add(name)
                break
        else:
            # Fallback: all names taken (shouldn't happen with k=4 and 4 archetypes)
            cluster_label_map[int(cluster_id)] = f"Cluster-{cluster_id}"

    return cluster_label_map


# ─── Per-candidate assignment ──────────────────────────────────────────────────

def _compute_synthetic_means(df: pd.DataFrame) -> dict[str, float]:
    """Population means from synthetic data, used for missing-value imputation."""
    return {feat: float(df[feat].mean()) for feat in FEATURES}


# Module-level cache for synthetic means (computed once at load_or_fit time)
_synthetic_means: dict[str, float] = {}


def assign_candidate_archetype(
    candidate_scores: dict,
    scaler: StandardScaler,
    kmeans_model: KMeans,
    cluster_label_map: dict[int, str],
) -> dict:
    """
    Assign a single real candidate to an archetype.

    Args:
        candidate_scores: dict with keys from FEATURES. github_score may be None.
        scaler: fitted StandardScaler
        kmeans_model: fitted KMeans
        cluster_label_map: cluster_id -> archetype_name mapping

    Returns:
        {
            "archetype": str,           — assigned archetype name
            "distance_to_centroid": float,
            "distances_all": {name: float, ...},  — distance to each centroid
            "confidence": {name: float, ...},     — softmax-normalized inverse distances
        }
    """
    # Build feature vector, imputing missing github_score
    feature_vec = []
    for feat in FEATURES:
        val = candidate_scores.get(feat)
        if val is None:
            val = _synthetic_means.get(feat, 50.0)  # population mean fallback
        feature_vec.append(float(val))

    X = np.array([feature_vec])
    X_scaled = scaler.transform(X)

    # Distances to all centroids
    centroids = kmeans_model.cluster_centers_
    distances_raw = {}
    for cluster_id, centroid in enumerate(centroids):
        name = cluster_label_map.get(cluster_id, f"Cluster-{cluster_id}")
        dist = float(np.linalg.norm(X_scaled[0] - centroid))
        distances_raw[name] = round(dist, 4)

    # Assigned cluster
    predicted_cluster = int(kmeans_model.predict(X_scaled)[0])
    assigned_name = cluster_label_map.get(predicted_cluster, f"Cluster-{predicted_cluster}")
    assigned_distance = distances_raw[assigned_name]

    # Confidence via softmax on negative distances (closer = higher confidence)
    # Using temperature=1.0 on negative distances
    neg_dists = np.array([-distances_raw[name] for name in distances_raw])
    # Shift for numerical stability
    neg_dists_shifted = neg_dists - np.max(neg_dists)
    exp_vals = np.exp(neg_dists_shifted)
    softmax_vals = exp_vals / exp_vals.sum()

    confidence = {}
    for i, name in enumerate(distances_raw.keys()):
        confidence[name] = round(float(softmax_vals[i]), 4)

    return {
        "archetype": assigned_name,
        "distance_to_centroid": assigned_distance,
        "distances_all": distances_raw,
        "confidence": confidence,
    }


# ─── Cached loader ─────────────────────────────────────────────────────────────

def load_or_fit_clustering() -> tuple[StandardScaler, KMeans, dict[int, str]]:
    """
    Load cached scaler/kmeans/label_map from disk, or fit fresh from
    synthetic data and cache. Thread-safe for Streamlit's @st.cache_resource.
    """
    global _synthetic_means

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if SCALER_PATH.exists() and KMEANS_PATH.exists() and LABEL_MAP_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        kmeans = joblib.load(KMEANS_PATH)
        label_map = joblib.load(LABEL_MAP_PATH)

        # Recompute synthetic means (cheap, needed for imputation)
        if SYNTHETIC_CSV_PATH.exists():
            df = pd.read_csv(SYNTHETIC_CSV_PATH)
            _synthetic_means = _compute_synthetic_means(df)
        else:
            # Regenerate if CSV is missing
            df = generate_synthetic_archetype_profiles()
            _synthetic_means = _compute_synthetic_means(df)

        return scaler, kmeans, label_map

    # Fit fresh
    df = generate_synthetic_archetype_profiles()
    _synthetic_means = _compute_synthetic_means(df)

    scaler, kmeans = fit_clustering(df)
    label_map = label_clusters(kmeans, scaler, df)

    # Persist
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)

    return scaler, kmeans, label_map

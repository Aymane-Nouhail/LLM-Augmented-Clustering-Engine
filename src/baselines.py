import numpy as np
from sklearn.cluster import KMeans


def run_naive_kmeans(
    features: np.ndarray, n_clusters: int, random_state: int = 0
) -> np.ndarray:
    """K-Means with L2-normalized features (cosine-equivalent distance)."""
    print("\n--- Running Naive KMeans Baseline (Cosine Similarity) ---")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_normalized = features / np.where(norms > 0, norms, 1.0)
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_assignments = kmeans.fit_predict(features_normalized)
        print("Naive KMeans completed.")
        return cluster_assignments
    except Exception as e:
        print(f"Error during Naive KMeans clustering: {e}")
        return np.array([-1] * features.shape[0])


def run_spherical_kmeans(
    features: np.ndarray, n_clusters: int, random_state: int = 0
) -> np.ndarray:
    """Spherical K-Means (L2-normalize then K-Means). Used for JoSE embeddings."""
    print("\n--- Running Spherical K-Means ---")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_normalized = features / np.where(norms > 0, norms, 1.0)
    try:
        return KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        ).fit_predict(features_normalized)
    except Exception as e:
        print(f"Error during Spherical K-Means: {e}")
        return np.array([-1] * features.shape[0])

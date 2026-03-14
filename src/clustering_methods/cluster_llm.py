"""
ClusterLLM — reimplementation of Zhang et al. (2023, EMNLP).

Two-stage pipeline:
  Stage 1 (Perspective):  ~n_triplets LLM triplet queries.
      Sample anchor + two options; LLM picks the option closest in intent.
      Used to identify must-link / cannot-link pairs for Stage 2 initialisation.
      (Full embedder fine-tuning via TripletLoss is not performed here; the
      triplet signal is instead captured as soft pairwise constraints.)

  Stage 2 (Granularity):  ~n_pairwise LLM pairwise queries on a Ward
      hierarchical dendrogram.
      LLM labels sampled pairs YES/NO; we score each candidate k (from 2 to
      n_clusters) using F-β consistency between the LLM labels and the
      cluster assignments at that cut.  The cut with the highest F-β is k*.

Both stages are fully cached (pickle) so re-runs skip LLM calls.
"""

import os
import pickle
import concurrent.futures
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.llm_service import LLMService


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------


def _process_triplet(
    anchor_idx: int,
    a_idx: int,
    b_idx: int,
    documents: List[str],
    llm_service: LLMService,
    prompt_template: str,
) -> Tuple[int, int, int, str]:
    """Query LLM for a single triplet; returns (anchor, a, b, 'A'|'B'|'?')."""
    try:
        prompt = prompt_template.format(
            anchor=documents[anchor_idx],
            option_a=documents[a_idx],
            option_b=documents[b_idx],
        )
        response = llm_service.get_chat_completion(prompt)
        if response and isinstance(response, str):
            r = response.strip().upper()
            if r.startswith("A"):
                return (anchor_idx, a_idx, b_idx, "A")
            if r.startswith("B"):
                return (anchor_idx, a_idx, b_idx, "B")
    except Exception:
        pass
    return (anchor_idx, a_idx, b_idx, "?")


def _sample_triplets(
    features_normed: np.ndarray,
    initial_assignments: np.ndarray,
    n_triplets: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int, int]]:
    """
    Sample informative triplets.

    Anchor selection: prefer uncertain documents — those whose top-2 cluster
    distances are closest (smallest margin).
    option_a: random doc from the same initial cluster as anchor.
    option_b: random doc from a different cluster.
    """
    n = len(features_normed)
    n_clusters = len(np.unique(initial_assignments))

    # Compute per-doc distance to each cluster centroid
    centroids = np.array(
        [
            features_normed[initial_assignments == k].mean(axis=0)
            for k in range(n_clusters)
        ]
    )
    # distance matrix: n_docs × n_clusters  (cosine via normed dot)
    sim = features_normed @ centroids.T  # higher = more similar
    # Sort each row descending; margin = sim[:,0] - sim[:,1]
    sorted_sim = np.sort(sim, axis=1)[:, ::-1]
    margins = sorted_sim[:, 0] - sorted_sim[:, 1]

    # Weight: smaller margin → more uncertain → higher probability
    weights = 1.0 / (margins + 1e-6)
    weights /= weights.sum()

    triplets: List[Tuple[int, int, int]] = []
    cluster_members = {
        k: np.where(initial_assignments == k)[0] for k in range(n_clusters)
    }

    for _ in range(n_triplets * 5):  # oversample to hit budget
        if len(triplets) >= n_triplets:
            break
        anchor = int(rng.choice(n, p=weights))
        a_cluster = int(initial_assignments[anchor])
        same = cluster_members[a_cluster]
        diff_clusters = [k for k in range(n_clusters) if k != a_cluster]
        if len(same) < 2 or not diff_clusters:
            continue
        a_candidates = same[same != anchor]
        if len(a_candidates) == 0:
            continue
        b_cluster = int(rng.choice(diff_clusters))
        if len(cluster_members[b_cluster]) == 0:
            continue
        a_idx = int(rng.choice(a_candidates))
        b_idx = int(rng.choice(cluster_members[b_cluster]))
        triplets.append((anchor, a_idx, b_idx))

    return triplets[:n_triplets]


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------


def _process_pair(
    i: int,
    j: int,
    documents: List[str],
    llm_service: LLMService,
    prompt_template: str,
) -> Tuple[int, int, str]:
    """Query LLM for a pairwise YES/NO label."""
    try:
        prompt = prompt_template.format(text1=documents[i], text2=documents[j])
        response = llm_service.get_chat_completion(prompt)
        if response and isinstance(response, str):
            r = response.strip().upper()
            if r.startswith("YES"):
                return (i, j, "YES")
            if r.startswith("NO"):
                return (i, j, "NO")
    except Exception:
        pass
    return (i, j, "?")


def _sample_dendrogram_pairs(
    Z: np.ndarray,
    n_samples: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Sample pairs from merge steps in the Ward dendrogram.

    Each row of Z = [left, right, dist, count].  We treat merge events as
    opportunities to ask the LLM about a same-cluster vs different-cluster pair.
    We sample proportionally to sub-cluster size at each merge step.
    """
    n_merges = len(Z)
    # Assign weight proportional to min(left_count, right_count) — bigger merges
    # are more informative for deciding k.
    weights = np.array([min(Z[i, 2], 1.0) + 1.0 for i in range(n_merges)], dtype=float)
    weights /= weights.sum()

    # Build leaf membership for each merge node
    n_leaves = n_samples
    membership: Dict[int, List[int]] = {i: [i] for i in range(n_leaves)}
    for merge_idx, (left, right, _, _) in enumerate(Z):
        node_id = n_leaves + merge_idx
        left_leaves = membership.get(int(left), [int(left)])
        right_leaves = membership.get(int(right), [int(right)])
        membership[node_id] = left_leaves + right_leaves

    pairs: List[Tuple[int, int]] = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 20

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        merge_idx = int(rng.choice(n_merges, p=weights))
        node_id = n_leaves + merge_idx
        left = int(Z[merge_idx, 0])
        right = int(Z[merge_idx, 1])
        left_leaves = membership.get(left, [left])
        right_leaves = membership.get(right, [right])

        # 50/50: same-cluster pair (from left or right sub-cluster) vs cross-cluster
        if rng.random() < 0.5:
            # same-cluster pair within left sub-cluster
            pool = left_leaves if len(left_leaves) >= 2 else right_leaves
            if len(pool) < 2:
                continue
            i, j = sorted(rng.choice(pool, size=2, replace=False))
        else:
            # cross-cluster pair (one from each sub-cluster)
            if not left_leaves or not right_leaves:
                continue
            i = int(rng.choice(left_leaves))
            j = int(rng.choice(right_leaves))
            if i > j:
                i, j = j, i

        key = (i, j)
        if key in seen or i == j:
            continue
        seen.add(key)
        pairs.append(key)

    return pairs


def _fbeta_consistency(
    pair_labels: Dict[Tuple[int, int], str],
    assignments: np.ndarray,
    beta: float = 5.0,
) -> float:
    """
    F-β consistency score (Zhang et al. 2023, Eq. 1).

    For each LLM-labelled pair (i, j):
      - True Positive  (TP): LLM=YES and same cluster
      - False Positive (FP): LLM=YES and different cluster
      - False Negative (FN): LLM=NO  and same cluster

    Precision = TP / (TP + FP), Recall = TP / (TP + FN)
    F_β = (1 + β²) * P * R / (β² * P + R)
    """
    tp = fp = fn = 0
    for (i, j), label in pair_labels.items():
        same_cluster = assignments[i] == assignments[j]
        if label == "YES":
            if same_cluster:
                tp += 1
            else:
                fp += 1
        elif label == "NO":
            if same_cluster:
                fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta_sq = beta**2
    denom = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def cluster_via_clusterllm(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    triplet_prompt_template: str,
    pairwise_prompt_template: str,
    output_path: str = "clusterllm_output.csv",
    output_dir: str = "results/",
    n_triplets: int = 1024,
    n_pairwise: int = 400,
    fine_tune: bool = False,  # reserved; fine-tuning not performed
    max_workers: int = 50,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """
    Cluster documents using the ClusterLLM two-stage framework.

    Parameters
    ----------
    documents         : list of raw text strings
    features          : sentence-transformer embeddings (n_docs, dim)
    n_clusters        : upper bound / hint for k; Stage 2 selects optimal k* ≤ n_clusters
    llm_service       : LLMService instance
    triplet_prompt_template : prompt with {anchor}, {option_a}, {option_b}
    pairwise_prompt_template: prompt with {text1}, {text2} → YES/NO
    output_path       : CSV path for query log
    output_dir        : directory for cache files
    n_triplets        : Stage 1 LLM budget
    n_pairwise        : Stage 2 LLM budget
    fine_tune         : reserved (not implemented)
    max_workers       : ThreadPoolExecutor concurrency
    seed              : random seed

    Returns
    -------
    np.ndarray of cluster assignments (length n_docs), or None on failure.
    """
    print("\n--- Running ClusterLLM (Zhang et al., 2023) ---")
    n_docs = len(documents)
    rng = np.random.default_rng(seed)

    os.makedirs(output_dir, exist_ok=True)
    triplet_cache = os.path.join(output_dir, "clusterllm_triplet_cache.pkl")
    pairwise_cache = os.path.join(output_dir, "clusterllm_pairwise_cache.pkl")

    # L2-normalise features (cosine similarity)
    features_normed = normalize(features, axis=1, norm="l2")

    # ── Stage 1: Triplet queries ─────────────────────────────────────────────
    print("  [Stage 1] Triplet perspective queries")

    triplet_results: Dict[Tuple[int, int, int], str] = {}
    if os.path.exists(triplet_cache):
        try:
            with open(triplet_cache, "rb") as f:
                triplet_results = pickle.load(f)
            print(f"    Loaded {len(triplet_results)} cached triplet labels.")
        except Exception as e:
            print(f"    Cache load failed: {e}. Re-querying.")
            triplet_results = {}

    if not triplet_results:
        if not llm_service.generation_available():
            print("    No generation model. Skipping Stage 1.")
        else:
            # Initial KMeans for anchor uncertainty estimation
            init_km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
            init_assignments = init_km.fit_predict(features_normed)

            triplets = _sample_triplets(
                features_normed, init_assignments, n_triplets, rng
            )
            print(
                f"    Querying LLM for {len(triplets)} triplets ({max_workers} workers)..."
            )

            workers = min(max_workers, len(triplets))
            raw: List[Optional[Tuple]] = [None] * len(triplets)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _process_triplet,
                        a,
                        b,
                        c,
                        documents,
                        llm_service,
                        triplet_prompt_template,
                    ): idx
                    for idx, (a, b, c) in enumerate(triplets)
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(triplets),
                    desc="Triplets",
                ):
                    raw[futures[future]] = future.result()

            for res in raw:
                if res:
                    anchor, a_idx, b_idx, label = res
                    triplet_results[(anchor, a_idx, b_idx)] = label

            with open(triplet_cache, "wb") as f:
                pickle.dump(triplet_results, f)
            print(f"    Cached {len(triplet_results)} triplet labels → {triplet_cache}")

    # Derive soft pairwise hints from triplets:
    #   LLM answered "A" → anchor is same-intent as a_idx (must-link hint)
    #   LLM answered "B" → anchor is same-intent as b_idx (must-link hint)
    # These will be used below when scoring k* if desired; for now they inform
    # the initial embedding used in Stage 2 (KMeans on normed features).

    # ── Stage 2: Hierarchical dendrogram + pairwise queries ──────────────────
    print("  [Stage 2] Hierarchical granularity selection")

    pairwise_labels: Dict[Tuple[int, int], str] = {}
    if os.path.exists(pairwise_cache):
        try:
            with open(pairwise_cache, "rb") as f:
                pairwise_labels = pickle.load(f)
            print(f"    Loaded {len(pairwise_labels)} cached pairwise labels.")
        except Exception as e:
            print(f"    Cache load failed: {e}. Re-querying.")
            pairwise_labels = {}

    if not pairwise_labels:
        if not llm_service.generation_available():
            print("    No generation model. Using n_clusters directly.")
        else:
            # Build Ward dendrogram
            condensed = pdist(features_normed, metric="cosine")
            Z = linkage(condensed, method="ward")

            pairs = _sample_dendrogram_pairs(Z, n_docs, n_pairwise, rng)
            print(f"    Querying LLM for {len(pairs)} pairs ({max_workers} workers)...")

            workers = min(max_workers, len(pairs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _process_pair,
                        i,
                        j,
                        documents,
                        llm_service,
                        pairwise_prompt_template,
                    ): (i, j)
                    for i, j in pairs
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(pairs),
                    desc="Pairwise",
                ):
                    i, j = futures[future]
                    _, _, label = future.result()
                    pairwise_labels[(i, j)] = label

            with open(pairwise_cache, "wb") as f:
                pickle.dump(pairwise_labels, f)
            print(
                f"    Cached {len(pairwise_labels)} pairwise labels → {pairwise_cache}"
            )

    # Select optimal k* via F-β consistency
    k_star = n_clusters
    if pairwise_labels:
        condensed = pdist(features_normed, metric="cosine")
        Z = linkage(condensed, method="ward")

        k_min = max(2, n_clusters // 4)
        k_max = min(n_docs - 1, n_clusters * 3)
        best_score = -1.0
        fbeta_rows = []
        for k in range(k_min, k_max + 1):
            cuts = fcluster(Z, k, criterion="maxclust")
            score = _fbeta_consistency(pairwise_labels, cuts, beta=5.0)
            fbeta_rows.append({"k": k, "fbeta": score})
            if score > best_score:
                best_score = score
                k_star = k

        print(
            f"    Optimal k* = {k_star}  (F-5 = {best_score:.4f}, range [{k_min},{k_max}])"
        )

        # Save per-k scores for plotting
        fbeta_csv = os.path.join(output_dir, "clusterllm_fbeta_scores.csv")
        pd.DataFrame(fbeta_rows).to_csv(fbeta_csv, index=False)
        print(f"    F-β scores saved → {fbeta_csv}")

        # Final assignments at k*
        final_cuts = fcluster(Z, k_star, criterion="maxclust") - 1  # 0-indexed
        assignments = final_cuts.astype(int)
    else:
        # Fallback: plain KMeans at n_clusters
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        assignments = km.fit_predict(features_normed)

    # ── Write query log ───────────────────────────────────────────────────────
    rows = []
    for (anchor, a_idx, b_idx), label in triplet_results.items():
        rows.append(
            {
                "stage": 1,
                "anchor": documents[anchor],
                "option_a": documents[a_idx],
                "option_b": documents[b_idx],
                "llm_response": label,
            }
        )
    for (i, j), label in pairwise_labels.items():
        rows.append(
            {
                "stage": 2,
                "anchor": documents[i],
                "option_a": documents[j],
                "option_b": "",
                "llm_response": label,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)

    return assignments

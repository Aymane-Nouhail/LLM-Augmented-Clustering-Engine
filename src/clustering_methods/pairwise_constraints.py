import numpy as np
import random
import pickle
import pandas as pd
from typing import Any, List, Dict, Tuple, Set
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
import os


def _repair_constraints(
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return a consistent (must_links, cannot_links) pair.

    PCKMeans raises an error when a cannot-link connects two nodes that are
    transitively joined by must-links (e.g. A-ML-B, B-ML-C, A-CL-C).
    This function:
      1. Removes direct contradictions (same pair in both lists).
      2. Builds must-link connected components via union-find.
      3. Drops any cannot-link whose endpoints share a component.
    """
    must_set = set(map(tuple, must_links))
    cannot_set = set(map(tuple, cannot_links))

    # 1. Remove direct contradictions
    direct = must_set & cannot_set
    if direct:
        print(f"  Dropping {len(direct)} directly contradictory pair(s).")
        must_set -= direct
        cannot_set -= direct

    # 2. Union-find over must-link graph
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for a, b in must_set:
        union(a, b)

    # 3. Drop cannot-links that connect same must-link component
    invalid = {p for p in cannot_set if find(p[0]) == find(p[1])}
    if invalid:
        print(f"  Dropping {len(invalid)} transitively inconsistent cannot-link(s).")
        cannot_set -= invalid

    return list(must_set), list(cannot_set)


def process_pair(
    pair_data: Tuple[int, int, str, str, ChatPromptTemplate, LLMService],
) -> Dict[str, Any]:
    idx1, idx2, doc1, doc2, template, llm = pair_data
    prompt = template.format(text1=doc1, text2=doc2)
    response = ""
    constraint = None
    try:
        response = llm.get_chat_completion(prompt).strip().upper()
        if response == "YES":
            constraint = ("MUST", (idx1, idx2))
            const_type = "Must-Link"
        elif response == "NO":
            constraint = ("CANNOT", (idx1, idx2))
            const_type = "Cannot-Link"
        else:
            const_type = "Ambiguous Response"
    except Exception as e:
        const_type = f"Error: {str(e)[:50]}..."

    return {
        "pair_indices": f"({idx1}, {idx2})",
        "doc1_full_text": doc1,
        "doc2_full_text": doc2,
        "full_prompt": prompt,
        "raw_llm_response": response,
        "constraint_type": const_type,
        "constraint": constraint,
    }


def cluster_via_pairwise_constraints(
    dataset_name: str,
    documents: List[str],
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    prompt_template: str,
    num_pairs: int,
    strategy: str = "random",
    output_path: str = "pairwise_queries_output.csv",
    output_dir: str = "results/",
    max_workers: int = 50,
) -> np.ndarray:
    """Semi-supervised clustering via LLM-generated pairwise must-link / cannot-link constraints."""
    print("\n--- Running Pairwise Constraint Clustering ---")

    n_samples = len(documents)
    num_pairs = min(num_pairs, n_samples * (n_samples - 1) // 2)

    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "pairwise_cache.pkl")

    # {(i, j): "YES" | "NO"} — keyed by sorted pair index
    pair_labels: Dict[Tuple[int, int], str] = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                pair_labels = pickle.load(f)
            print(f"  Loaded {len(pair_labels)} cached pair labels from {cache_file}")
        except Exception as e:
            print(f"  Cache load failed: {e}. Re-querying LLM.")
            pair_labels = {}

    if not pair_labels:
        if strategy == "explore_consolidate":
            # Explore-Consolidate from Viswanathan et al. (2023), adapted from
            # Basu et al. (2004).
            # Explore: greedy farthest-point sampling builds a diverse set of anchor
            #   points spanning the entire embedding space.
            # Consolidate: each anchor's nearest neighbours are must-link candidates;
            #   mid-range neighbours are cannot-link candidates (boundary region).
            features_norm = normalize(features)
            half = num_pairs // 2

            # Number of anchors: enough that each contributes ~5 pairs on average.
            n_anchors = max(10, min(half // 5, n_samples // 2))
            seed_idx = random.randrange(n_samples)
            anchors: List[int] = [seed_idx]
            min_dists = 1.0 - (features_norm @ features_norm[seed_idx])

            while len(anchors) < n_anchors:
                next_anchor = int(np.argmax(min_dists))
                anchors.append(next_anchor)
                d = 1.0 - (features_norm @ features_norm[next_anchor])
                min_dists = np.minimum(min_dists, d)
                min_dists[next_anchor] = -np.inf  # prevent re-selection

            k_nn = min(40, n_samples - 1)
            nn = NearestNeighbors(n_neighbors=k_nn, metric="cosine", algorithm="brute")
            nn.fit(features_norm)
            _, neighbor_indices = nn.kneighbors(features_norm[anchors])

            must_link_candidates: List[Tuple[int, int]] = []
            cannot_link_candidates: List[Tuple[int, int]] = []
            seen: Set[Tuple[int, int]] = set()

            for anchor_pos, anchor_idx in enumerate(anchors):
                near = [j for j in neighbor_indices[anchor_pos] if j != anchor_idx]
                for j in near[: k_nn // 4]:   # nearest → must-link
                    key = (min(anchor_idx, j), max(anchor_idx, j))
                    if key not in seen:
                        seen.add(key)
                        must_link_candidates.append(key)
                for j in near[k_nn // 2 :]:   # mid-range → cannot-link
                    key = (min(anchor_idx, j), max(anchor_idx, j))
                    if key not in seen:
                        seen.add(key)
                        cannot_link_candidates.append(key)

            # Pad with random pairs if anchor coverage didn't fill the budget.
            attempts = 0
            while len(must_link_candidates) < half and attempts < num_pairs * 10:
                a, b = sorted(random.sample(range(n_samples), 2))
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    must_link_candidates.append(key)
                attempts += 1

            attempts = 0
            while len(cannot_link_candidates) < half and attempts < num_pairs * 10:
                a, b = sorted(random.sample(range(n_samples), 2))
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    cannot_link_candidates.append(key)
                attempts += 1

            selected = must_link_candidates[:half] + cannot_link_candidates[:half]

        elif strategy == "similarity":
            # Most-similar pairs → must-link candidates (k-NN, avoids O(n²) enumeration).
            # Random pairs → cannot-link candidates (naturally dissimilar).
            features_norm = normalize(features)
            k = min(20, n_samples - 1)
            nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            nn.fit(features_norm)
            _, neighbor_indices = nn.kneighbors(features_norm)

            most_similar: List[Tuple[int, int]] = []
            seen: Set[Tuple[int, int]] = set()
            for i, neighbors in enumerate(neighbor_indices):
                for j in neighbors:
                    if i == j:
                        continue
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        most_similar.append(key)
                    if len(most_similar) >= num_pairs // 2:
                        break
                if len(most_similar) >= num_pairs // 2:
                    break

            least_similar: Set[Tuple[int, int]] = set()
            attempts = 0
            while len(least_similar) < num_pairs // 2 and attempts < num_pairs * 10:
                a, b = sorted(random.sample(range(n_samples), 2))
                key = (a, b)
                if key not in seen and key not in least_similar:
                    least_similar.add(key)
                attempts += 1

            selected = most_similar[: num_pairs // 2] + list(least_similar)
        else:
            all_pairs = [
                (i, j) for i in range(n_samples) for j in range(i + 1, n_samples)
            ]
            selected = random.sample(all_pairs, num_pairs)

        query_data = []
        template = ChatPromptTemplate.from_template(prompt_template)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_pair,
                    (i, j, documents[i], documents[j], template, llm_service),
                ): (i, j)
                for i, j in selected
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Querying LLM"
            ):
                result = future.result()
                query_data.append(result)
                if result["constraint"]:
                    const_type, (i, j) = result["constraint"]
                    label = "YES" if const_type == "MUST" else "NO"
                    pair_labels[(i, j)] = label

        with open(cache_file, "wb") as f:
            pickle.dump(pair_labels, f)
        print(f"  Cached {len(pair_labels)} pair labels → {cache_file}")

        if query_data:
            pd.DataFrame(query_data).to_csv(output_path, index=False)

    must_links: List[Tuple[int, int]] = []
    cannot_links: List[Tuple[int, int]] = []
    for (i, j), label in pair_labels.items():
        if label == "YES":
            must_links.append((i, j))
        elif label == "NO":
            cannot_links.append((i, j))

    must_links, cannot_links = _repair_constraints(must_links, cannot_links)

    assignments = None
    if must_links or cannot_links:
        features_norm = normalize(features)
        try:
            pckmeans = PCKMeans(n_clusters=n_clusters)
            pckmeans.fit(features_norm, ml=must_links, cl=cannot_links)
            assignments = pckmeans.labels_
        except Exception as e:
            print(f"Clustering error: {e}")

    return assignments

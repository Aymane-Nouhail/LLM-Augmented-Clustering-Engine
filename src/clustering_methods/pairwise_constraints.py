import numpy as np
import random
import pandas as pd
from typing import Any, List, Dict, Tuple, Set
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService
from src.metrics import calculate_clustering_metrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
import os

METRICS_CSV_PATH = "clustering_metrics_results.csv"


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
    must_set   = set(map(tuple, must_links))
    cannot_set = set(map(tuple, cannot_links))

    # 1. Remove direct contradictions
    direct = must_set & cannot_set
    if direct:
        print(f"  Dropping {len(direct)} directly contradictory pair(s).")
        must_set   -= direct
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


def process_pair(pair_data: Tuple[int, int, str, str, ChatPromptTemplate, LLMService]) -> Dict[str, Any]:
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
        "pair_indices":    f"({idx1}, {idx2})",
        "doc1_full_text":  doc1,
        "doc2_full_text":  doc2,
        "full_prompt":     prompt,
        "raw_llm_response": response,
        "constraint_type": const_type,
        "constraint":      constraint,
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
    strategy: str = 'random',
    output_path: str = "pairwise_queries_output.csv",
    max_workers: int = 50,
) -> np.ndarray:
    """Semi-supervised clustering via LLM-generated pairwise must-link / cannot-link constraints."""
    print("\n--- Running Pairwise Constraint Clustering ---")

    n_samples = len(documents)
    num_pairs = min(num_pairs, n_samples * (n_samples - 1) // 2)

    if strategy == 'similarity':
        # Most-similar pairs → must-link candidates (k-NN, avoids O(n²) enumeration).
        # Random pairs → cannot-link candidates (naturally dissimilar).
        features_norm = normalize(features)
        k = min(20, n_samples - 1)
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
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

        selected = most_similar[:num_pairs // 2] + list(least_similar)
    else:
        all_pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
        selected = random.sample(all_pairs, num_pairs)

    must_links: List[Tuple[int, int]] = []
    cannot_links: List[Tuple[int, int]] = []
    query_data = []
    template = ChatPromptTemplate.from_template(prompt_template)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_pair, (i, j, documents[i], documents[j], template, llm_service)): (i, j)
            for i, j in selected
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Querying LLM"):
            result = future.result()
            query_data.append(result)
            if result["constraint"]:
                const_type, (i, j) = result["constraint"]
                if const_type == "MUST":
                    must_links.append((i, j))
                else:
                    cannot_links.append((i, j))

    if query_data:
        pd.DataFrame(query_data).to_csv(output_path, index=False)

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

    metrics = calculate_clustering_metrics(labels, assignments, n_clusters) if assignments is not None else {}
    pd.DataFrame([{
        "Dataset": dataset_name,
        "Method": "Pairwise Constraints",
        "Status": "Success" if assignments is not None else "Failed",
        **metrics,
    }]).to_csv(METRICS_CSV_PATH, mode='a', header=not os.path.exists(METRICS_CSV_PATH), index=False)

    return assignments

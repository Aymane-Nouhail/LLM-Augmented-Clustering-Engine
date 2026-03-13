import numpy as np
import pandas as pd
import concurrent.futures
import os
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from src.llm_service import LLMService
from src.metrics import calculate_clustering_metrics

METRICS_CSV_PATH = "clustering_metrics_results.csv"


def find_cluster_info(
    features: np.ndarray, assignments: np.ndarray, documents: List[str], n_clusters: int
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Return cluster centroids and the index of each cluster's representative (closest to centroid)."""
    centroids = np.zeros((n_clusters, features.shape[1]))
    representatives = {}
    for cluster_id in range(n_clusters):
        idx = np.where(assignments == cluster_id)[0]
        if idx.size > 0:
            centroids[cluster_id] = np.mean(features[idx], axis=0)
            dists = euclidean_distances([centroids[cluster_id]], features[idx])
            representatives[cluster_id] = idx[np.argmin(dists)]
    return centroids, representatives


def identify_low_confidence_points(
    features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray, n_clusters: int, k: int
) -> List[int]:
    """Return indices of the k documents least confident about their current cluster assignment."""
    distances = euclidean_distances(features, centroids)
    margins = np.array([
        np.sort(row)[1] - np.sort(row)[0] if len(np.unique(row)) > 1 else -np.inf
        for row in distances
    ])
    valid = np.isfinite(margins)
    return np.where(valid)[0][np.argsort(margins[valid])[:k]].tolist()


def process_low_confidence_point(
    doc_index: int,
    document: str,
    current_assignment: int,
    features: np.ndarray,
    centroids: np.ndarray,
    representatives: Dict[int, int],
    n_clusters: int,
    llm_service: LLMService,
    prompt_template: ChatPromptTemplate,
    num_candidates: int,
    documents: List[str],
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Ask the LLM whether a low-confidence document belongs to its current cluster or a better one."""
    queries = []
    new_assignment = current_assignment

    current_rep_idx = representatives.get(current_assignment)
    if current_rep_idx is None:
        return (doc_index, current_assignment, current_assignment, queries)

    prompt = prompt_template.format(
        document_text=document,
        rep_doc_text=documents[current_rep_idx],
        query_type="is_linked_to_assigned",
    )
    response = llm_service.get_chat_completion(prompt).strip().upper()
    queries.append({
        "document_index": doc_index,
        "full_prompt": prompt,
        "llm_answer": response,
        "resulting_action": "Evaluated Current Assignment",
    })

    if response != "YES":
        queries[-1]["resulting_action"] = "LLM Said NO (Checking Candidates)"
        distances = euclidean_distances([features[doc_index]], centroids)[0]
        candidates = [
            c for c in np.argsort(distances)
            if c != current_assignment and c in representatives
        ][:num_candidates]

        for candidate in candidates:
            prompt = prompt_template.format(
                document_text=document,
                rep_doc_text=documents[representatives[candidate]],
                query_type=f"is_linked_to_candidate_{candidate}",
            )
            response = llm_service.get_chat_completion(prompt).strip().upper()
            queries.append({
                "document_index": doc_index,
                "full_prompt": prompt,
                "llm_answer": response,
                "resulting_action": "Evaluated Candidate",
            })
            if response == "YES":
                new_assignment = candidate
                queries[-1]["resulting_action"] = f"LLM Said YES (Reassigned to {candidate})"
                break
            else:
                queries[-1]["resulting_action"] = "LLM Said NO (Remains in Original)"
    else:
        queries[-1]["resulting_action"] = "LLM Said YES (Remains in Original)"

    return (doc_index, current_assignment, new_assignment, queries)


def cluster_via_correction(
    dataset_name: str,
    documents: List[str],
    features: np.ndarray,
    initial_assignments: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    correction_prompt: str,
    k_low_confidence: int,
    num_candidates: int,
    queries_output_path: str = "correction_queries_output.csv",
) -> np.ndarray:
    """Correct low-confidence cluster assignments by querying the LLM."""
    if (not llm_service or not llm_service.is_available()
            or len(documents) != features.shape[0]
            or len(documents) != len(labels)
            or k_low_confidence <= 0
            or num_candidates <= 0):
        return initial_assignments

    corrected = np.copy(initial_assignments)
    centroids, reps = find_cluster_info(features, corrected, documents, n_clusters)
    low_conf = identify_low_confidence_points(features, corrected, centroids, n_clusters, k_low_confidence)

    if not low_conf:
        return corrected

    template = ChatPromptTemplate.from_template(correction_prompt)
    all_queries = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(
                process_low_confidence_point,
                i, documents[i], corrected[i], features, centroids, reps,
                n_clusters, llm_service, template, num_candidates, documents,
            ): i
            for i in low_conf if 0 <= corrected[i] < n_clusters
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Correcting"):
            doc_idx, old, new, queries = future.result()
            if new != old:
                corrected[doc_idx] = new
            all_queries.extend(queries)

    if all_queries:
        pd.DataFrame(all_queries).to_csv(queries_output_path, index=False)

    metrics = calculate_clustering_metrics(labels, corrected, n_clusters)
    pd.DataFrame([{"Dataset": dataset_name, "Method": "LLM Correction", **metrics}]).to_csv(
        METRICS_CSV_PATH, mode='a', header=not os.path.exists(METRICS_CSV_PATH), index=False
    )

    return corrected

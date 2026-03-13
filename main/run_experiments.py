"""
Experiment runner: replicates Viswanathan et al. (2023) Table 1 results
for Bank77, CLINC, and Tweet datasets across all clustering methods.

Methods run:
  1. K-Means baseline  (all-mpnet-base-v2)
  2. JoSE + Spherical K-Means  (Word2Vec trained from scratch, no LLM)
  3. PCKMeans  (all-mpnet-base-v2 + gpt-4.1-nano pairwise oracle)
  4. LLM Correction  (all-mpnet-base-v2 + gpt-4.1-nano)
  5. Keyphrase Clustering  (all-mpnet-base-v2 + gpt-4.1-nano)

Usage:
    python -m main.run_experiments
    python -m main.run_experiments --datasets bank77,clinc --methods kmeans,jose
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys

from src.config import (
    OPENAI_API_KEY,
    DATA_CACHE_PATH,
    RESULTS_ROOT,
    EMBEDDING_BACKEND,
    DATASET_PROMPTS,
    # Method hyperparameters
    PC_NUM_PAIRS_TO_QUERY,
    PC_CONSTRAINT_SELECTION_STRATEGY,
    CORRECTION_K_LOW_CONFIDENCE,
    CORRECTION_NUM_CANDIDATE_CLUSTERS,
)
from src.data import load_dataset
from src.llm_service import LLMService
from src.jose_embeddings import JoSEEmbeddings
from src.baselines import run_naive_kmeans, run_spherical_kmeans
from src.metrics import calculate_clustering_metrics
from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints
from src.clustering_methods.clustering_correction import cluster_via_correction
from src.clustering_methods.keyphrase_expansion import cluster_via_keyphrase_expansion

RESULTS_PATH = os.path.join(RESULTS_ROOT, "experiment_results_table.csv")
ALL_DATASETS = ["bank77", "clinc", "tweet"]
ALL_METHODS  = ["kmeans", "jose", "pairwise", "correction", "keyphrase"]


def _get_prompts(dataset_name: str) -> dict:
    """Return per-dataset prompt templates."""
    if dataset_name not in DATASET_PROMPTS:
        raise ValueError(f"No prompts defined for dataset '{dataset_name}'. "
                         f"Supported: {list(DATASET_PROMPTS.keys())}")
    return DATASET_PROMPTS[dataset_name]


def run_one_dataset(
    dataset_name: str,
    methods: list,
    llm_service: LLMService,
) -> list:
    """Run selected methods on one dataset and return list of result dicts."""
    results = []
    prompts = _get_prompts(dataset_name)
    output_dir = os.path.join(RESULTS_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load sentence-transformer embeddings (cached after first run)
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    features, labels, docs = load_dataset(
        dataset_name, DATA_CACHE_PATH, llm_service.get_embedding_model()
    )
    if features.size == 0:
        print(f"  ERROR: failed to load {dataset_name}, skipping.")
        return results

    n_clusters = len(np.unique(labels))
    print(f"  {len(docs)} samples | {n_clusters} clusters")

    # ------------------------------------------------------------------ #
    # Method 1: K-Means baseline                                          #
    # ------------------------------------------------------------------ #
    if "kmeans" in methods:
        print("\n[1/5] K-Means (all-mpnet-base-v2)")
        assignments = run_naive_kmeans(features, n_clusters)
        metrics = calculate_clustering_metrics(labels, assignments, n_clusters)
        results.append({
            "Dataset": dataset_name,
            "Method": "KMeans",
            **_fmt_metrics(metrics),
        })
        _print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Method 2: JoSE + Spherical K-Means                                 #
    # ------------------------------------------------------------------ #
    if "jose" in methods:
        print("\n[2/5] JoSE + Spherical K-Means (Word2Vec from scratch)")
        jose = JoSEEmbeddings(
            vector_size=100, window=5, min_count=1, epochs=10, seed=42
        )
        jose.fit(docs)
        jose_features = np.array(jose.embed_documents(docs))
        assignments = run_spherical_kmeans(jose_features, n_clusters)
        metrics = calculate_clustering_metrics(labels, assignments, n_clusters)
        results.append({
            "Dataset": dataset_name,
            "Method": "JoSE + Spherical KMeans",
            **_fmt_metrics(metrics),
        })
        _print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Methods 3-5 require LLM generation                                 #
    # ------------------------------------------------------------------ #
    if not llm_service.generation_available():
        print("\nSkipping LLM methods (no generation model available).")
        return results

    # ------------------------------------------------------------------ #
    # Method 3: PCKMeans (pairwise constraints)                           #
    # ------------------------------------------------------------------ #
    if "pairwise" in methods:
        print("\n[3/5] PCKMeans (pairwise constraints via LLM)")
        assignments = cluster_via_pairwise_constraints(
            dataset_name=dataset_name,
            documents=docs,
            features=features,
            labels=labels,
            n_clusters=n_clusters,
            llm_service=llm_service,
            prompt_template=prompts["pc"],
            num_pairs=PC_NUM_PAIRS_TO_QUERY,
            strategy=PC_CONSTRAINT_SELECTION_STRATEGY,
            output_path=os.path.join(output_dir, "pairwise_queries_output.csv"),
        )
        metrics = (
            calculate_clustering_metrics(labels, assignments, n_clusters)
            if assignments is not None else {}
        )
        results.append({
            "Dataset": dataset_name,
            "Method": "PCKMeans",
            **_fmt_metrics(metrics),
        })
        _print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Method 4: LLM Correction                                           #
    # ------------------------------------------------------------------ #
    if "correction" in methods:
        print("\n[4/5] LLM Correction")
        initial = run_naive_kmeans(features, n_clusters)
        assignments = cluster_via_correction(
            dataset_name=dataset_name,
            documents=docs,
            features=features,
            initial_assignments=initial,
            labels=labels,
            n_clusters=n_clusters,
            llm_service=llm_service,
            correction_prompt=prompts["correction"],
            k_low_confidence=CORRECTION_K_LOW_CONFIDENCE,
            num_candidates=CORRECTION_NUM_CANDIDATE_CLUSTERS,
            queries_output_path=os.path.join(output_dir, "correction_queries_output.csv"),
        )
        metrics = calculate_clustering_metrics(labels, assignments, n_clusters)
        results.append({
            "Dataset": dataset_name,
            "Method": "LLM Correction",
            **_fmt_metrics(metrics),
        })
        _print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Method 5: Keyphrase Clustering                                      #
    # ------------------------------------------------------------------ #
    if "keyphrase" in methods:
        print("\n[5/5] Keyphrase Clustering (LLM expansion + K-Means)")
        kp_results = cluster_via_keyphrase_expansion(
            documents=docs,
            features=features,
            n_clusters=n_clusters,
            llm_service=llm_service,
            keyphrase_prompt_template=prompts["kp"],
            keyphrase_output_csv_path=os.path.join(output_dir, "keyphrase_expansions_output.csv"),
        )
        # Report best variant (weighted_1.0 most closely matches the paper)
        for variant in ["weighted_1.0", "average", "concatenated"]:
            if kp_results.get(variant) is not None:
                metrics = calculate_clustering_metrics(labels, kp_results[variant], n_clusters)
                results.append({
                    "Dataset": dataset_name,
                    "Method": f"Keyphrase ({variant})",
                    **_fmt_metrics(metrics),
                })
                _print_metrics(metrics, label=f"  variant={variant}")
                break  # only report the best available variant

    return results


# ------------------------------------------------------------------ #
# Formatting helpers                                                  #
# ------------------------------------------------------------------ #

def _fmt_metrics(metrics: dict) -> dict:
    """Round metric values to 3 decimal places."""
    return {
        k: round(float(v), 3) if v is not None else None
        for k, v in metrics.items()
    }


def _print_metrics(metrics: dict, label: str = "") -> None:
    acc = metrics.get("Accuracy")
    nmi = metrics.get("NMI")
    if acc is not None and nmi is not None:
        print(f"  {label}  Acc={acc:.3f}  NMI={nmi:.3f}")
    else:
        print(f"  {label}  (no metrics)")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Replicate Few-Shot Clustering paper (Viswanathan et al., 2023)"
    )
    parser.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help=f"Comma-separated dataset names. Default: {','.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods or 'all'. Choices: {','.join(ALL_METHODS)}",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods  = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]

    print(f"\nDatasets : {datasets}")
    print(f"Methods  : {methods}")
    print(f"Embedding: {EMBEDDING_BACKEND}\n")

    llm_service = LLMService(api_key=OPENAI_API_KEY or "", embedding_backend=EMBEDDING_BACKEND)

    all_results = []
    for dataset in datasets:
        all_results.extend(run_one_dataset(dataset, methods, llm_service))

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    # Print pivot table matching paper's Table 2 format
    print("\n" + "="*60)
    print("RESULTS TABLE")
    print("="*60)
    for dataset in datasets:
        subset = df[df["Dataset"] == dataset]
        if subset.empty:
            continue
        print(f"\n{dataset.upper()}")
        print(f"  {'Method':<35} {'Acc':>6}  {'NMI':>6}")
        print(f"  {'-'*35} {'-'*6}  {'-'*6}")
        for _, row in subset.iterrows():
            acc = f"{row['Accuracy']:.3f}" if pd.notna(row.get("Accuracy")) else "  —  "
            nmi = f"{row['NMI']:.3f}"      if pd.notna(row.get("NMI"))      else "  —  "
            print(f"  {row['Method']:<35} {acc:>6}  {nmi:>6}")

    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

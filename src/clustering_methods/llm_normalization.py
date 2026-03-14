import pickle
import os
import numpy as np
import pandas as pd
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm
from src.llm_service import LLMService


def _process_doc(
    doc_index: int,
    document: str,
    llm_service: LLMService,
    prompt_template: str,
) -> Tuple[int, str, str]:
    """Query LLM to rewrite a document in canonical form."""
    try:
        prompt = prompt_template.format(text=document)
        response = llm_service.get_chat_completion(prompt)
        normalized = (
            response.strip() if response and response not in ("ERROR", "") else document
        )
        return (doc_index, document, normalized)
    except Exception:
        return (doc_index, document, document)


def _embed_normalized(
    doc_index: int,
    document: str,
    normalized_text: str,
    features: np.ndarray,
    llm_service: LLMService,
    embedding_dim: int,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Embed the normalized text and original; return L2-normalized pair."""
    try:
        orig_feature = features[doc_index].reshape(1, -1)
        orig_norm = (
            normalize(orig_feature, axis=1, norm="l2").flatten()
            if np.linalg.norm(orig_feature) > 0
            else np.zeros_like(orig_feature).flatten()
        )

        norm_embedding = np.array(llm_service.get_embedding(normalized_text))
        if len(norm_embedding) != embedding_dim:
            return (doc_index, orig_norm, None)

        exp_2d = norm_embedding.reshape(1, -1)
        exp_norm = (
            normalize(exp_2d, axis=1, norm="l2").flatten()
            if np.linalg.norm(exp_2d) > 0
            else np.zeros_like(exp_2d).flatten()
        )
        return (doc_index, orig_norm, exp_norm)
    except Exception:
        return (doc_index, None, None)


def cluster_via_llm_normalization(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    prompt_template: str,
    output_path: str = "normalization_output.csv",
    output_dir: str = "results/",
    max_workers: int = 50,
) -> Optional[np.ndarray]:
    """Cluster using LLM-rewritten canonical forms.

    Phase 1: LLM rewrites each doc to canonical form (cached in normalization_cache.pkl).
    Phase 2: Embed canonical form; concatenate with original embedding → KMeans.
    Falls back to original text on LLM error (graceful degradation).
    """
    print("\n--- Running LLM Normalization Clustering ---")
    n_samples = len(documents)
    embedding_dim = features.shape[1]

    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "normalization_cache.pkl")
    normalized_map: Dict[int, str] = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                normalized_map = pickle.load(f)
            print(
                f"  Loaded normalizations from cache: {cache_file} ({len(normalized_map)} docs)"
            )
        except Exception as e:
            print(f"  Normalization cache load failed: {e}. Re-querying LLM.")
            normalized_map = {}

    if not normalized_map:
        if not llm_service.generation_available():
            print("  No normalization cache and no generation model. Cannot run.")
            return None

        print(
            f"  Querying LLM for canonical forms ({n_samples} docs, up to {max_workers} workers)..."
        )
        workers = min(max_workers, n_samples)
        raw_results: List[Optional[Tuple]] = [None] * n_samples

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_doc, i, documents[i], llm_service, prompt_template
                ): i
                for i in range(n_samples)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=n_samples,
                desc="Normalizing",
            ):
                raw_results[futures[future]] = future.result()

        for result in raw_results:
            if result:
                doc_index, _, normalized = result
                normalized_map[doc_index] = normalized

        with open(cache_file, "wb") as f:
            pickle.dump(normalized_map, f)
        print(f"  Normalizations cached to: {cache_file}")

        pd.DataFrame(
            [
                {
                    "document_index": i,
                    "original": documents[i],
                    "normalized": normalized_map.get(i, documents[i]),
                }
                for i in range(n_samples)
            ]
        ).to_csv(output_path, index=False)

    print(f"  Embedding canonical forms ({n_samples} docs)...")
    workers = min(max_workers, n_samples)
    embed_results: List[Optional[Tuple]] = [None] * n_samples

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _embed_normalized,
                i,
                documents[i],
                normalized_map.get(i, documents[i]),
                features,
                llm_service,
                embedding_dim,
            ): i
            for i in range(n_samples)
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=n_samples, desc="Embedding"
        ):
            embed_results[futures[future]] = future.result()

    original_features: List[np.ndarray] = []
    expanded_features: List[np.ndarray] = []
    successful_indices: List[int] = []

    for result in embed_results:
        if result:
            doc_index, orig, exp = result
            if orig is not None and exp is not None:
                original_features.append(orig)
                expanded_features.append(exp)
                successful_indices.append(doc_index)

    if not original_features:
        print("  No valid embeddings — cannot cluster.")
        return None

    concat_features = np.hstack(
        [np.array(original_features), np.array(expanded_features)]
    )

    try:
        clusters = KMeans(
            n_clusters=n_clusters, random_state=0, n_init="auto"
        ).fit_predict(concat_features)
        assignments = np.full(n_samples, -1, dtype=int)
        for i, idx in enumerate(successful_indices):
            assignments[idx] = clusters[i]
        return assignments
    except Exception as e:
        print(f"  Clustering error: {e}")
        return None

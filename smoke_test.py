"""
Smoke test — verifies all 5 clustering methods run end-to-end without errors.

Real sentence-transformer embeddings, fake LLM generation (no API key needed).
Run from the repo root:
    python smoke_test.py
"""

import os
import sys
import tempfile
import traceback
import numpy as np

# ── Suppress noisy HuggingFace / tokenizers warnings ──────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================
# Synthetic dataset: 3 well-separated themes × 10 docs each (30 total)
# =============================================================================
_BANKING = [
    "how do I activate my new debit card",
    "my card is not working at the ATM",
    "I need to enable my credit card for online purchases",
    "card activation not working on the mobile app",
    "how to activate contactless payments on my card",
    "transfer money to my savings account",
    "what are the fees for international wire transfers",
    "how do I send money abroad to a European bank",
    "overseas transfer fee and exchange rate",
    "how to set up a direct debit for my rent",
]
_WEATHER = [
    "what is the weather like today in London",
    "will it rain tomorrow morning",
    "weekend weather forecast for New York",
    "is it going to snow tonight",
    "temperature and humidity outside right now",
    "sunny or cloudy tomorrow",
    "severe weather alert for my area",
    "check the hourly weather forecast",
    "wind speed and direction this afternoon",
    "UV index and sunscreen recommendation today",
]
_SPORTS = [
    "who won the NBA game last night",
    "football Premier League match results today",
    "Champions League standings this season",
    "top scorer in the World Cup so far",
    "tennis Grand Slam bracket and schedule",
    "baseball game scores from yesterday",
    "NHL playoff schedule and results",
    "golf tournament leaderboard after round two",
    "cricket test match result and highlights",
    "Olympics medal count by country",
]

DOCS   = _BANKING + _WEATHER + _SPORTS
LABELS = np.array([0] * 10 + [1] * 10 + [2] * 10)
N_CLUSTERS = 3


# =============================================================================
# Mock LLM service — real embedder, fake generation
# =============================================================================
class MockLLMService:
    """Wraps a real SentenceTransformerEmbeddings with stub LLM generation.

    Pairwise / correction calls answer based on ground-truth labels — if both
    texts belong to the same cluster, returns YES, otherwise NO.  This avoids
    contradictory constraints (must-link + cannot-link on the same pair) that
    would cause PCKMeans to crash, while still exercising the full pipeline.
    """

    def __init__(self, embedder, docs: list, labels: np.ndarray):
        from src.llm_service import KeyphraseList
        self._KeyphraseList = KeyphraseList
        self.embedding_model = embedder
        # Non-None sentinel so is_available() is True
        self.generation_model = object()
        self._embedding_dim = None
        # Ground-truth lookup: exact doc text → label
        self._doc_label = {doc: int(label) for doc, label in zip(docs, labels)}

    # ── Availability ──────────────────────────────────────────────────────────
    def is_available(self) -> bool:
        return True

    def generation_available(self) -> bool:
        return True

    # ── Embedding ─────────────────────────────────────────────────────────────
    def get_embedding_model(self):
        return self.embedding_model

    def get_embedding(self, text: str):
        return self.embedding_model.embed_query(text)

    def get_embedding_dimension(self) -> int:
        if self._embedding_dim is None:
            self._embedding_dim = len(self.get_embedding("test"))
        return self._embedding_dim

    # ── Generation (stubbed) ──────────────────────────────────────────────────
    def get_chat_completion(self, prompt, output_structure=None):
        if output_structure is self._KeyphraseList:
            return self._KeyphraseList(keyphrases=[
                "mock keyphrase alpha",
                "mock keyphrase beta",
                "mock keyphrase gamma",
            ])
        # For pairwise and correction: look up the last two labelled texts in the
        # prompt (last occurrence avoids matching the few-shot example lines).
        import re
        # Pairwise patterns: "Query 1:", "Utterance 1:", "Tweet 1:"
        m1_all = re.findall(r'(?:Query|Utterance|Tweet) 1:\s*(.+)', prompt)
        m2_all = re.findall(r'(?:Query|Utterance|Tweet) 2:\s*(.+)', prompt)
        # Correction patterns: "User Query:", "User Utterance:", "Tweet:"
        if not m1_all:
            m1_all = re.findall(r'(?:User Query|User Utterance|Tweet):\s*(.+)', prompt)
            m2_all = re.findall(r'Representative (?:Query|Utterance|Tweet):\s*(.+)', prompt)

        if m1_all and m2_all:
            t1 = m1_all[-1].strip()
            t2 = m2_all[-1].strip()
            l1 = self._doc_label.get(t1)
            l2 = self._doc_label.get(t2)
            if l1 is not None and l2 is not None:
                return "YES" if l1 == l2 else "NO"

        # Fallback for unrecognised prompt shape
        return "YES"


# =============================================================================
# Helpers
# =============================================================================
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def _check(assignments, name):
    """Return True if assignments look valid."""
    if assignments is None:
        print(f"  [{FAIL}] {name}: returned None")
        return False
    arr = np.asarray(assignments)
    if arr.shape != (len(DOCS),):
        print(f"  [{FAIL}] {name}: shape {arr.shape}, expected ({len(DOCS)},)")
        return False
    unique = np.unique(arr[arr >= 0])
    if len(unique) < 2:
        print(f"  [{FAIL}] {name}: only {len(unique)} unique cluster(s)")
        return False
    print(f"  [{PASS}] {name}: {len(unique)} clusters, shape {arr.shape}")
    return True


# =============================================================================
# Main smoke test
# =============================================================================
def main():
    print("=" * 60)
    print("  Smoke test — LLM-Augmented Clustering Engine")
    print("=" * 60)

    # ── 1. Load real embedder ─────────────────────────────────────────────────
    print("\n[Setup] Loading sentence-transformer embedder...")
    from src.llm_service import SentenceTransformerEmbeddings
    embedder = SentenceTransformerEmbeddings("all-mpnet-base-v2")
    features = np.array(embedder.embed_documents(DOCS))
    print(f"  Embedded {len(DOCS)} docs → shape {features.shape}")

    llm = MockLLMService(embedder, DOCS, LABELS)

    results = {}

    # ── 2. K-Means baseline ───────────────────────────────────────────────────
    print("\n[1/5] K-Means baseline")
    try:
        from src.baselines import run_naive_kmeans
        assignments = run_naive_kmeans(features, N_CLUSTERS)
        results["kmeans"] = _check(assignments, "K-Means")
    except Exception:
        print(f"  [{FAIL}] K-Means raised an exception:")
        traceback.print_exc()
        results["kmeans"] = False

    # ── 3. JoSE + Spherical K-Means ──────────────────────────────────────────
    print("\n[2/5] JoSE + Spherical K-Means")
    try:
        from src.jose_embeddings import JoSEEmbeddings
        from src.baselines import run_spherical_kmeans
        jose = JoSEEmbeddings(vector_size=50, window=3, min_count=1, epochs=5, seed=42)
        jose.fit(DOCS)
        jose_features = np.array(jose.embed_documents(DOCS))
        assignments = run_spherical_kmeans(jose_features, N_CLUSTERS)
        results["jose"] = _check(assignments, "JoSE + Spherical K-Means")
    except Exception:
        print(f"  [{FAIL}] JoSE raised an exception:")
        traceback.print_exc()
        results["jose"] = False

    # ── 4. PCKMeans (pairwise constraints) ───────────────────────────────────
    print("\n[3/5] PCKMeans (pairwise constraints, mock LLM)")
    try:
        from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints
        from src.config import BANK77_PC_PROMPT_TEMPLATE
        with tempfile.TemporaryDirectory() as tmpdir:
            assignments = cluster_via_pairwise_constraints(
                dataset_name="smoke",
                documents=DOCS,
                features=features,
                labels=LABELS,
                n_clusters=N_CLUSTERS,
                llm_service=llm,
                prompt_template=BANK77_PC_PROMPT_TEMPLATE,
                num_pairs=30,           # tiny budget for speed
                strategy='similarity',
                output_path=os.path.join(tmpdir, "pairwise_out.csv"),
            )
        results["pairwise"] = _check(assignments, "PCKMeans")
    except Exception:
        print(f"  [{FAIL}] PCKMeans raised an exception:")
        traceback.print_exc()
        results["pairwise"] = False

    # ── 5. LLM Correction ─────────────────────────────────────────────────────
    print("\n[4/5] LLM Correction (mock LLM)")
    try:
        from src.baselines import run_naive_kmeans
        from src.clustering_methods.clustering_correction import cluster_via_correction
        from src.config import BANK77_CORRECTION_PROMPT_TEMPLATE
        initial = run_naive_kmeans(features, N_CLUSTERS)
        with tempfile.TemporaryDirectory() as tmpdir:
            assignments = cluster_via_correction(
                dataset_name="smoke",
                documents=DOCS,
                features=features,
                initial_assignments=initial,
                labels=LABELS,
                n_clusters=N_CLUSTERS,
                llm_service=llm,
                correction_prompt=BANK77_CORRECTION_PROMPT_TEMPLATE,
                k_low_confidence=5,     # only correct 5 points
                num_candidates=2,
                queries_output_path=os.path.join(tmpdir, "correction_out.csv"),
            )
        results["correction"] = _check(assignments, "LLM Correction")
    except Exception:
        print(f"  [{FAIL}] LLM Correction raised an exception:")
        traceback.print_exc()
        results["correction"] = False

    # ── 6. Keyphrase Expansion ────────────────────────────────────────────────
    print("\n[5/5] Keyphrase Expansion (mock LLM)")
    try:
        from src.clustering_methods.keyphrase_expansion import cluster_via_keyphrase_expansion
        from src.config import BANK77_KP_PROMPT_TEMPLATE
        with tempfile.TemporaryDirectory() as tmpdir:
            kp_results = cluster_via_keyphrase_expansion(
                documents=DOCS,
                features=features,
                n_clusters=N_CLUSTERS,
                llm_service=llm,
                keyphrase_prompt_template=BANK77_KP_PROMPT_TEMPLATE,
                keyphrase_output_csv_path=os.path.join(tmpdir, "kp_out.csv"),
            )
        ok = False
        for variant in ["weighted_1.0", "average", "concatenated"]:
            if kp_results.get(variant) is not None:
                ok = _check(kp_results[variant], f"Keyphrase ({variant})")
                break
        if not ok and not any(v is not None for v in kp_results.values()):
            print(f"  [{FAIL}] Keyphrase: all variants returned None")
        results["keyphrase"] = ok
    except Exception:
        print(f"  [{FAIL}] Keyphrase raised an exception:")
        traceback.print_exc()
        results["keyphrase"] = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    names = {
        "kmeans":    "K-Means baseline",
        "jose":      "JoSE + Spherical K-Means",
        "pairwise":  "PCKMeans",
        "correction":"LLM Correction",
        "keyphrase": "Keyphrase Expansion",
    }
    all_pass = True
    for key, label in names.items():
        status = PASS if results.get(key) else FAIL
        print(f"  [{status}] {label}")
        if not results.get(key):
            all_pass = False

    print()
    if all_pass:
        print("All methods passed.")
        sys.exit(0)
    else:
        print("Some methods FAILED — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

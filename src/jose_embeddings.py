"""
Joint Spherical Embedding (JoSE) — Meng et al., NeurIPS 2019.

Pipeline:
  1. Tokenize corpus (lowercase + strip punctuation).
  2. Train Word2Vec (CBOW) on tokenized corpus.
  3. L2-normalize all word vectors (spherical constraint).
  4. Compute document embedding = mean of normalized word vectors.
  5. L2-normalize document embeddings.

Implements langchain_core.embeddings.Embeddings for drop-in compatibility
with load_dataset() in data.py.

Note: JoSE must be fit() on the full corpus before use. The embedding
quality depends on corpus size; on short utterance datasets (Bank77, CLINC,
Tweet with ~3k–4.5k samples), co-occurrence signal is limited and JoSE
typically underperforms pre-trained encoders.
"""

import re
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings


class JoSEEmbeddings(Embeddings):
    """
    Joint Spherical Embedding following Meng et al. (NeurIPS 2019).

    Usage:
        jose = JoSEEmbeddings(vector_size=100, epochs=10)
        jose.fit(corpus_documents)   # must call before embed_*
        embeddings = jose.embed_documents(corpus_documents)
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 10,
        workers: int = 4,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.seed = seed
        self._model = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str]) -> "JoSEEmbeddings":
        """
        Train Word2Vec on corpus and apply spherical normalization to word vectors.

        Args:
            corpus: List of raw text documents (the full dataset).
        Returns:
            self (for method chaining)
        """
        from gensim.models import Word2Vec

        tokenized = [self._tokenize(doc) for doc in corpus]

        self._model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=0,  # CBOW, as used in JoSE paper
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
        )

        # Spherical constraint: L2-normalize all word vectors in-place
        norms = np.linalg.norm(self._model.wv.vectors, axis=1, keepdims=True)
        self._model.wv.vectors /= np.where(norms > 0, norms, 1.0)

        vocab_size = len(self._model.wv)
        print(
            f"JoSE: trained on {len(tokenized)} docs | vocab={vocab_size} | dim={self.vector_size}"
        )
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def _embed_one(self, text: str) -> List[float]:
        """
        Embed a single document as the mean of its normalized word vectors,
        then L2-normalize the result.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "JoSEEmbeddings.fit() must be called before embed_documents()."
            )

        tokens = self._tokenize(text)
        known = [t for t in tokens if t in self._model.wv]

        if not known:
            return [0.0] * self.vector_size

        # Word vectors are already L2-normalized after fit()
        vecs = np.array([self._model.wv[t] for t in known])
        doc_vec = vecs.mean(axis=0)

        # L2-normalize document vector (spherical output)
        norm = np.linalg.norm(doc_vec)
        if norm > 1e-8:
            doc_vec = doc_vec / norm

        return doc_vec.tolist()

    @property
    def embedding_dimension(self) -> int:
        return self.vector_size

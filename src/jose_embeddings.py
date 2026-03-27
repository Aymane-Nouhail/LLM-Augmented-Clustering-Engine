"""
Joint Spherical Embedding (JoSE) — Meng et al., NeurIPS 2019.

Implements the two-step spherical generative model:
  1. Center word u generated from document d:  p(u|d) ∝ exp(cos(u, d))
  2. Context word v generated from center u:   p(v|u) ∝ exp(cos(v, u))

Training uses the max-margin loss (Eq. 3 of the paper):
  L(u,v,d) = max(0, m - cos(v,u) - cos(u,d) + cos(v,u') + cos(u',d))

Optimization uses Riemannian SGD on the unit sphere with exponential mapping
(Section 4 of the paper), keeping all embeddings on S^{p-1} throughout training.

Three embedding matrices are jointly learned:
  - Target (center) word embeddings  {u_i}
  - Context word embeddings          {v_i}
  - Document embeddings              {d_j}

Implements langchain_core.embeddings.Embeddings for drop-in compatibility.
"""

import re
import numpy as np
from typing import List, Optional, Tuple
from langchain_core.embeddings import Embeddings


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    """L2-normalize each row to the unit sphere."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(norms > 1e-10, norms, 1.0)


def _exp_map(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Exponential mapping on the unit sphere (Eq. 4).

    Maps a tangent vector z ∈ T_x S^{p-1} to a point on the sphere.
    exp_x(z) = cos(||z||) x + sin(||z||) z / ||z||
    """
    norm_z = np.linalg.norm(z)
    if norm_z < 1e-10:
        return x
    return np.cos(norm_z) * x + np.sin(norm_z) * (z / norm_z)


def _riemannian_grad(x: np.ndarray, euc_grad: np.ndarray) -> np.ndarray:
    """
    Project Euclidean gradient onto the tangent space at x on the sphere (Eq. 5).

    grad f(x) = (I - x x^T) ∇f(x) = ∇f(x) - (x^T ∇f(x)) x
    """
    return euc_grad - np.dot(x, euc_grad) * x


class JoSEEmbeddings(Embeddings):
    """
    Joint Spherical Embedding following Meng et al. (NeurIPS 2019).

    Usage:
        jose = JoSEEmbeddings(vector_size=100, epochs=10)
        jose.fit(corpus_documents)
        embeddings = jose.embed_documents(corpus_documents)
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 10,
        learning_rate: float = 0.05,
        margin: float = 1.0,
        n_negatives: int = 5,
        seed: int = 42,
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.margin = margin
        self.n_negatives = n_negatives
        self.seed = seed
        self.workers = workers

        self._word2idx: dict = {}
        self._idx2word: list = []
        self._W_target: Optional[np.ndarray] = None  # center word embeddings {u_i}
        self._W_context: Optional[np.ndarray] = None  # context word embeddings {v_i}
        self._D: Optional[np.ndarray] = None  # document embeddings {d_j}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    # ------------------------------------------------------------------
    # Vocabulary building
    # ------------------------------------------------------------------

    def _build_vocab(self, tokenized_corpus: List[List[str]]) -> None:
        """Build vocabulary from tokenized corpus, respecting min_count."""
        word_counts: dict = {}
        for tokens in tokenized_corpus:
            for t in tokens:
                word_counts[t] = word_counts.get(t, 0) + 1

        self._idx2word = [
            w for w, c in sorted(word_counts.items()) if c >= self.min_count
        ]
        self._word2idx = {w: i for i, w in enumerate(self._idx2word)}

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------

    def _generate_training_pairs(
        self, tokenized_corpus: List[List[str]]
    ) -> List[Tuple[int, int, int]]:
        """
        Generate (center_word_idx, context_word_idx, doc_idx) tuples
        by sliding a symmetric window over each document.
        """
        pairs = []
        for doc_idx, tokens in enumerate(tokenized_corpus):
            indices = [self._word2idx[t] for t in tokens if t in self._word2idx]
            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window)
                end = min(len(indices), i + self.window + 1)
                for j in range(start, end):
                    if j != i:
                        pairs.append((center_idx, indices[j], doc_idx))
        return pairs

    # ------------------------------------------------------------------
    # Negative sampling distribution
    # ------------------------------------------------------------------

    def _build_negative_table(
        self, tokenized_corpus: List[List[str]], table_size: int = 100_000
    ) -> np.ndarray:
        """
        Build unigram^(3/4) negative sampling table (same as Word2Vec).
        """
        counts = np.zeros(len(self._idx2word), dtype=np.float64)
        for tokens in tokenized_corpus:
            for t in tokens:
                if t in self._word2idx:
                    counts[self._word2idx[t]] += 1

        powered = np.power(counts, 0.75)
        powered /= powered.sum()

        table = np.zeros(table_size, dtype=np.int32)
        cumulative = np.cumsum(powered)
        idx = 0
        for i in range(table_size):
            while idx < len(cumulative) - 1 and i / table_size > cumulative[idx]:
                idx += 1
            table[i] = idx
        return table

    # ------------------------------------------------------------------
    # Riemannian SGD update
    # ------------------------------------------------------------------

    def _rsgd_update(
        self, embedding: np.ndarray, idx: int, euc_grad: np.ndarray, lr: float
    ) -> None:
        """
        Riemannian SGD step on the unit sphere for a single vector.

        1. Project Euclidean gradient to tangent space (Eq. 5)
        2. Update via exponential mapping (Eq. 4)
        """
        x = embedding[idx]
        rgrad = _riemannian_grad(x, euc_grad)
        embedding[idx] = _exp_map(x, -lr * rgrad)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str]) -> "JoSEEmbeddings":
        """
        Train JoSE on corpus using the max-margin loss with Riemannian SGD.
        """
        rng = np.random.RandomState(self.seed)

        # Tokenize and build vocab
        tokenized = [self._tokenize(doc) for doc in corpus]
        self._build_vocab(tokenized)
        vocab_size = len(self._idx2word)
        n_docs = len(corpus)

        if vocab_size == 0:
            raise ValueError("Empty vocabulary after tokenization.")

        # Initialize embeddings uniformly on the sphere
        self._W_target = _normalize_rows(
            rng.randn(vocab_size, self.vector_size).astype(np.float32)
        )
        self._W_context = _normalize_rows(
            rng.randn(vocab_size, self.vector_size).astype(np.float32)
        )
        self._D = _normalize_rows(
            rng.randn(n_docs, self.vector_size).astype(np.float32)
        )

        # Build negative sampling table and training pairs
        neg_table = self._build_negative_table(tokenized)
        pairs = self._generate_training_pairs(tokenized)
        n_pairs = len(pairs)

        print(
            f"JoSE: {n_docs} docs | vocab={vocab_size} | "
            f"dim={self.vector_size} | {n_pairs} training pairs"
        )

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle training pairs each epoch
            order = rng.permutation(n_pairs)
            total_loss = 0.0
            n_updates = 0

            # Linear learning rate decay
            lr = self.learning_rate * (1.0 - epoch / self.epochs)
            lr = max(lr, self.learning_rate * 0.01)

            for idx in order:
                u_idx, v_idx, d_idx = pairs[idx]

                u = self._W_target[u_idx]   # center word
                v = self._W_context[v_idx]  # context word
                d = self._D[d_idx]          # document

                # Negative samples: sample negative center words
                for _ in range(self.n_negatives):
                    u_neg_idx = neg_table[rng.randint(len(neg_table))]
                    if u_neg_idx == u_idx:
                        continue

                    u_neg = self._W_target[u_neg_idx]

                    # Max-margin loss (Eq. 3):
                    # L = max(0, m - cos(v,u) - cos(u,d) + cos(v,u') + cos(u',d))
                    # Since all vectors are unit-norm: cos(a,b) = a^T b
                    pos_score = np.dot(v, u) + np.dot(u, d)
                    neg_score = np.dot(v, u_neg) + np.dot(u_neg, d)
                    loss = self.margin - pos_score + neg_score

                    if loss <= 0:
                        continue

                    total_loss += loss
                    n_updates += 1

                    # Euclidean gradients of L w.r.t. each embedding:
                    # ∂L/∂u = -v - d
                    # ∂L/∂v = -u + u'
                    # ∂L/∂d = -u + u'
                    # ∂L/∂u' = v + d
                    grad_u = -v - d
                    grad_v = -u + u_neg
                    grad_d = -u + u_neg
                    grad_u_neg = v + d

                    # Riemannian SGD updates
                    self._rsgd_update(self._W_target, u_idx, grad_u, lr)
                    self._rsgd_update(self._W_context, v_idx, grad_v, lr)
                    self._rsgd_update(self._D, d_idx, grad_d, lr)
                    self._rsgd_update(self._W_target, u_neg_idx, grad_u_neg, lr)

            avg_loss = total_loss / max(n_updates, 1)
            print(
                f"  epoch {epoch + 1}/{self.epochs} — "
                f"avg_loss={avg_loss:.4f} | updates={n_updates}"
            )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Return document embeddings. For documents seen during fit(), return
        the jointly learned document embedding. For unseen documents, compute
        as mean of target word vectors (then L2-normalize).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "JoSEEmbeddings.fit() must be called before embed_documents()."
            )
        return [self._embed_one(t, idx) for idx, t in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text, doc_idx=None)

    def _embed_one(self, text: str, doc_idx: Optional[int] = None) -> List[float]:
        """
        Return the document embedding for a single text.

        If doc_idx is provided and within range of trained documents, return
        the jointly learned document vector. Otherwise, fall back to the mean
        of target word vectors (L2-normalized).
        """
        if doc_idx is not None and self._D is not None and doc_idx < len(self._D):
            return self._D[doc_idx].tolist()

        # Fallback for unseen documents
        tokens = self._tokenize(text)
        known = [t for t in tokens if t in self._word2idx]
        if not known:
            return [0.0] * self.vector_size
        vecs = np.array([self._W_target[self._word2idx[t]] for t in known])
        doc_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(doc_vec)
        if norm > 1e-8:
            doc_vec = doc_vec / norm
        return doc_vec.tolist()

    @property
    def embedding_dimension(self) -> int:
        return self.vector_size

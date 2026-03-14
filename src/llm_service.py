import time
import os
import threading
import csv
from typing import List, Any, Type

# Suppress HuggingFace tokenizers parallelism warning when forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.config import (
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    MOCKING_MODE,
    EMBEDDING_BACKEND,
    SENTENCE_TRANSFORMER_MODEL,
)
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# GPT-4.1-nano pricing (USD per token, as of 2025-04)
# ---------------------------------------------------------------------------
_PRICE_INPUT = 0.100 / 1_000_000  # $0.100 / 1M input tokens
_PRICE_CACHED = 0.025 / 1_000_000  # $0.025 / 1M cached input tokens
_PRICE_OUTPUT = 0.400 / 1_000_000  # $0.400 / 1M output tokens


class CostTracker:
    """Thread-safe accumulator for LLM token usage and estimated USD cost."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._current_method: str = "unknown"
        self._current_dataset: str = "unknown"

    def set_context(self, method: str, dataset: str) -> None:
        with self._lock:
            self._current_method = method
            self._current_dataset = dataset

    def record(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> None:
        non_cached_input = max(0, input_tokens - cached_tokens)
        cost = (
            non_cached_input * _PRICE_INPUT
            + cached_tokens * _PRICE_CACHED
            + output_tokens * _PRICE_OUTPUT
        )
        with self._lock:
            self._rows.append(
                {
                    "dataset": self._current_dataset,
                    "method": self._current_method,
                    "input_tokens": input_tokens,
                    "cached_tokens": cached_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost, 8),
                }
            )

    def summary(self) -> list[dict]:
        """Aggregate rows by (dataset, method) and return summary list."""
        from collections import defaultdict

        totals: dict[tuple, dict] = defaultdict(
            lambda: {
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "cost_usd": 0.0,
            }
        )
        with self._lock:
            for row in self._rows:
                key = (row["dataset"], row["method"])
                t = totals[key]
                t["input_tokens"] += row["input_tokens"]
                t["cached_tokens"] += row["cached_tokens"]
                t["output_tokens"] += row["output_tokens"]
                t["calls"] += 1
                t["cost_usd"] += row["cost_usd"]
        result = []
        for (dataset, method), t in sorted(totals.items()):
            result.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "calls": t["calls"],
                    "input_tokens": t["input_tokens"],
                    "cached_tokens": t["cached_tokens"],
                    "output_tokens": t["output_tokens"],
                    "cost_usd": round(t["cost_usd"], 6),
                }
            )
        return result

    def save(self, path: str) -> None:
        """Write per-call detail CSV and aggregated summary CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Per-call detail
        with self._lock:
            rows = list(self._rows)
        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        # Summary alongside
        summary_path = path.replace(".csv", "_summary.csv")
        summary = self.summary()
        if summary:
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary[0].keys())
                writer.writeheader()
                writer.writerows(summary)
            print(f"  Cost summary saved to {summary_path}")
            total_cost = sum(r["cost_usd"] for r in summary)
            print(f"  Total estimated cost: ${total_cost:.4f}")


class KeyphraseList(BaseModel):
    keyphrases: List[str] = Field(description="A list of keyphrases.")


class ParaphraseList(BaseModel):
    paraphrases: List[str] = Field(description="A list of paraphrases.")


class SentenceTransformerEmbeddings(Embeddings):
    """
    Langchain-compatible wrapper for sentence-transformers models.

    Optimizations:
    - Batched encoding (batch_size=128) to maximise GPU/CPU throughput.
    - Progress bar for corpora > 100 documents.
    - Automatic device selection (MPS on Apple Silicon, CUDA if available, else CPU).
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 128):
        from sentence_transformers import SentenceTransformer
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        print(f"SentenceTransformer '{model_name}' loaded on {device}.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        show_progress = len(texts) > 100
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode([text], convert_to_numpy=True)
        return vector[0].tolist()


class LLMService:
    def __init__(self, api_key: str, embedding_backend: str = EMBEDDING_BACKEND):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.embedding_model = None
        self.generation_model = None
        self._embedding_dim = 0
        self._api_delay_seconds = 0
        self.cost_tracker = CostTracker()

        print("Initializing LLM models...")

        # --- Embedding model ---
        if embedding_backend == "sentence_transformers":
            try:
                self.embedding_model = SentenceTransformerEmbeddings(
                    SENTENCE_TRANSFORMER_MODEL
                )
                test_embedding = self.embedding_model.embed_query("test")
                self._embedding_dim = len(test_embedding)
                print(f"Embedding dimension: {self._embedding_dim}")
            except Exception as e:
                print(f"Error initializing SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            try:
                self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
                test_embedding = self.embedding_model.embed_query("test")
                self._embedding_dim = len(test_embedding)
                print(
                    f"Embedding model '{EMBEDDING_MODEL_NAME}' initialized. Dimension: {self._embedding_dim}"
                )
            except Exception as e:
                print(f"Error testing embedding model '{EMBEDDING_MODEL_NAME}': {e}")
                self.embedding_model = None

        # --- Generation model (always OpenAI) ---
        if api_key:
            try:
                self.generation_model = ChatOpenAI(
                    model=GENERATION_MODEL_NAME, temperature=0
                )
                self.generation_model.invoke("test")
                print(f"Generation model '{GENERATION_MODEL_NAME}' initialized.")
            except Exception as e:
                print(f"Error testing generation model '{GENERATION_MODEL_NAME}': {e}")
                self.generation_model = None
        else:
            print("No API key provided — generation model unavailable.")
            self.generation_model = None

    def is_available(self) -> bool:
        return self.embedding_model is not None and self.generation_model is not None

    def generation_available(self) -> bool:
        return self.generation_model is not None

    def get_embedding_model(self) -> Embeddings:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not available.")
        return self.embedding_model

    def get_generation_model(self) -> ChatOpenAI:
        if not self.is_available() or self.generation_model is None:
            raise RuntimeError("Generation model is not available.")
        return self.generation_model

    def get_embedding_dimension(self) -> int:
        if self._embedding_dim == 0 and self.is_available() and self.embedding_model:
            try:
                test_embedding = self.get_embedding("test")
                if test_embedding:
                    self._embedding_dim = len(test_embedding)
            except Exception:
                pass

        if self._embedding_dim == 0:
            default_dim = 768 if EMBEDDING_BACKEND == "sentence_transformers" else 1536
            print(f"Warning: Embedding dimension unknown, assuming {default_dim}.")
            return default_dim
        return self._embedding_dim

    def get_embedding(self, text: str) -> List[float]:
        if self.embedding_model is None:
            dim = self.get_embedding_dimension()
            if dim > 0:
                return [0.0] * dim
            else:
                print(
                    "LLMService not available and embedding dimension unknown, returning empty list."
                )
                return []

        time.sleep(self._api_delay_seconds)
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"Error during embedding call for text '{text[:50]}...': {e}")
            dim = self.get_embedding_dimension()
            if dim > 0:
                print(
                    f"  Returning zero vector of dimension {dim} due to embedding error."
                )
                return [0.0] * dim
            else:
                print("  Embedding error and dimension unknown, returning empty list.")
                return []

    def get_chat_completion(
        self, prompt: Any, output_structure: Type[BaseModel] | None = None
    ) -> str | BaseModel | None:
        if MOCKING_MODE:
            return "YES"
        if self.generation_model is None:
            print("LLMService generation model not available.")
            return "ERROR" if output_structure is None else None

        time.sleep(self._api_delay_seconds)
        try:
            if output_structure:
                # include_raw=True gives us {"raw": AIMessage, "parsed": Pydantic}
                chain = self.generation_model.with_structured_output(
                    output_structure, include_raw=True
                )
                result = chain.invoke(prompt)
                raw_msg = result.get("raw")
                parsed = result.get("parsed")
                self._record_usage(raw_msg)
                return parsed
            else:
                response = self.generation_model.invoke(prompt)
                self._record_usage(response)
                return response.content

        except Exception as e:
            print(
                f"Error during generation call (structured: {output_structure is not None}): {e}"
            )
            return "ERROR" if output_structure is None else None

    def _record_usage(self, msg) -> None:
        """Extract usage_metadata from an AIMessage and log to cost_tracker."""
        if msg is None:
            return
        meta = getattr(msg, "usage_metadata", None)
        if meta is None:
            return
        # meta is a dict: {input_tokens, output_tokens, total_tokens, input_token_details}
        input_tokens = meta.get("input_tokens", 0)
        output_tokens = meta.get("output_tokens", 0)
        details = meta.get("input_token_details") or {}
        cached_tokens = (
            details.get("cache_read", 0)
            if isinstance(details, dict)
            else getattr(details, "cache_read", 0) or 0
        )
        self.cost_tracker.record(input_tokens, cached_tokens, output_tokens)

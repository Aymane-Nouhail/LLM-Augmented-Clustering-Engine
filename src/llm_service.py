import time
import os
from typing import List, Any, Type

# Suppress HuggingFace tokenizers parallelism warning when forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.config import EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, MOCKING_MODE, EMBEDDING_BACKEND, SENTENCE_TRANSFORMER_MODEL
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

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

        print("Initializing LLM models...")

        # --- Embedding model ---
        if embedding_backend == "sentence_transformers":
            try:
                self.embedding_model = SentenceTransformerEmbeddings(SENTENCE_TRANSFORMER_MODEL)
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
                print(f"Embedding model '{EMBEDDING_MODEL_NAME}' initialized. Dimension: {self._embedding_dim}")
            except Exception as e:
                print(f"Error testing embedding model '{EMBEDDING_MODEL_NAME}': {e}")
                self.embedding_model = None

        # --- Generation model (always OpenAI) ---
        if api_key:
            try:
                self.generation_model = ChatOpenAI(model=GENERATION_MODEL_NAME, temperature=0)
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
                print("LLMService not available and embedding dimension unknown, returning empty list.")
                return []

        time.sleep(self._api_delay_seconds)
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"Error during embedding call for text '{text[:50]}...': {e}")
            dim = self.get_embedding_dimension()
            if dim > 0:
                print(f"  Returning zero vector of dimension {dim} due to embedding error.")
                return [0.0] * dim
            else:
                print("  Embedding error and dimension unknown, returning empty list.")
                return []

    def get_chat_completion(self, prompt: Any, output_structure: Type[BaseModel] | None = None) -> str | BaseModel | None:
        if MOCKING_MODE:
            return "YES"
        if self.generation_model is None:
            print("LLMService generation model not available.")
            return "ERROR" if output_structure is None else None

        time.sleep(self._api_delay_seconds)
        try:
            chain = self.generation_model
            if output_structure:
                chain = chain.with_structured_output(output_structure)

            response = chain.invoke(prompt)
            return response if output_structure else response.content

        except Exception as e:
            print(f"Error during generation call (structured: {output_structure is not None}): {e}")
            return "ERROR" if output_structure is None else None
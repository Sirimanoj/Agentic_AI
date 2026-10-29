"""FAISS-based vector retrieval engine for legal document chunks.

Uses normalized embeddings with FAISS IndexFlatIP so inner-product scores
are exact cosine similarities in [-1.0, 1.0]. Preserves full DocumentChunk
provenance metadata for all retrieved passages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import faiss
import numpy as np

from legal_uq.ingestion.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved evidence passage with cosine similarity score and rank.

    Attributes:
        chunk: The full source DocumentChunk object with intact provenance.
        score: Cosine similarity score in [-1.0, 1.0].
               Note: This is a geometric relevance signal, NOT a probability of correctness.
        rank: 1-based retrieval ranking (1 = most relevant).
    """
    chunk: DocumentChunk
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_name(self) -> str:
        return self.chunk.doc_name

    @property
    def page_number(self) -> Optional[int]:
        return self.chunk.page_number

    @property
    def section_title(self) -> Optional[str]:
        return self.chunk.section_title

    @property
    def source_locator(self) -> str:
        return self.chunk.source_locator

    @property
    def text(self) -> str:
        return self.chunk.text


class Embedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Compute 2D numpy array of shape (N, dimension)."""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Compute 1D numpy array of shape (dimension,)."""
        res = self.embed_texts([query])
        return res[0]


class DeterministicFakeEmbedder(Embedder):
    """Fast, deterministic pseudo-semantic embedder for unit testing and CI/CD.

    Creates reproducible, normalized dense vectors using word-hash projections
    so identical or overlapping words yield higher cosine similarity without
    requiring PyTorch, heavy model weights, or network calls.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_word(self, word: str) -> np.ndarray:
        h = int(hashlib.md5(word.lower().encode("utf-8")).hexdigest(), 16)
        # Seed deterministic pseudo-random projection
        rng = np.random.RandomState(h % (2**31 - 1))
        return rng.randn(self._dim).astype(np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = []
        for text in texts:
            words = [w.strip() for w in text.split() if w.strip()]
            if not words:
                vec = np.zeros(self._dim, dtype=np.float32)
            else:
                word_vecs = [self._hash_word(w) for w in words]
                vec = np.sum(word_vecs, axis=0)
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.zeros(self._dim, dtype=np.float32)
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class CachedSentenceTransformerEmbedder(Embedder):
    """Local-only Sentence Transformers embedder wrapping all-MiniLM-L6-v2.

    Explicitly loads from the local HuggingFace cache without network queries.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def _find_cached_snapshot_path(self) -> Optional[str]:
        """Locate the exact on-disk snapshot folder in the user's HuggingFace cache."""
        home = Path.home()
        cache_hub = home / ".cache" / "huggingface" / "hub"
        folder_name = "models--" + self.model_name.replace("/", "--")
        snapshots_dir = cache_hub / folder_name / "snapshots"
        if snapshots_dir.exists():
            snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            if snapshots:
                # Return the latest modified snapshot directory
                snapshots.sort(key=lambda s: s.stat().st_mtime, reverse=True)
                return str(snapshots[0])
        return None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        snapshot_path = self._find_cached_snapshot_path()
        last_error = None

        if snapshot_path:
            try:
                self._model = SentenceTransformer(snapshot_path, local_files_only=True)
                return self._model
            except Exception as exc:
                last_error = exc

        # Try local cache lookup first
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            return self._model
        except Exception as exc:
            last_error = exc

        # Fallback to online download if cache is empty (e.g. fresh Streamlit Community Cloud container)
        try:
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:
            last_error = exc

        raise RuntimeError(
            f"Failed to load embedding model '{self.model_name}'. "
            f"Ensure sentence-transformers is installed and the model is reachable or cached. Details: {last_error}"
        )

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class VectorStore:
    """FAISS-based vector index for legal document chunks.

    Maintains a decoupled mapping between FAISS row IDs and rich DocumentChunk metadata.
    Uses FAISS IndexFlatIP with L2-normalized embeddings so inner products equal cosine similarity.

    Important Methodological Note:
        Cosine similarity scores are relative geometric relevance indicators in [-1.0, 1.0].
        They indicate how closely query terms and passage context align in vector space.
        A high similarity score does NOT guarantee that the retrieved legal rule is correct,
        applicable, or legally binding.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        relevance_heuristic_threshold: float = 0.50,
    ) -> None:
        """Initialize VectorStore.

        Args:
            embedder: Custom embedder instance. Defaults to CachedSentenceTransformerEmbedder.
            relevance_heuristic_threshold: Initial heuristic threshold for inspection.
                                          Configurable by the user; not a universal truth.
        """
        self.embedder: Embedder = embedder if embedder is not None else CachedSentenceTransformerEmbedder()
        self.relevance_heuristic_threshold = float(relevance_heuristic_threshold)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[DocumentChunk] = []

    @property
    def total_chunks(self) -> int:
        """Count of indexed chunks currently in the store."""
        return len(self._chunks)

    def build_index(self, chunks: Sequence[DocumentChunk]) -> int:
        """Construct or replace the FAISS index from the provided document chunks.

        Args:
            chunks: Sequence of DocumentChunk objects to index.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            self._index = None
            self._chunks = []
            return 0

        clean_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not clean_chunks:
            self._index = None
            self._chunks = []
            return 0

        texts = [c.text for c in clean_chunks]
        embeddings = self.embedder.embed_texts(texts)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(clean_chunks):
            raise ValueError(
                f"Embedding shape mismatch: expected ({len(clean_chunks)}, {self.embedder.dimension}), "
                f"got {embeddings.shape}"
            )

        # L2-normalize vectors so inner product equals cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        embeddings_f32 = embeddings.astype(np.float32)

        index = faiss.IndexFlatIP(self.embedder.dimension)
        index.add(embeddings_f32)

        self._index = index
        self._chunks = list(clean_chunks)
        return len(self._chunks)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """Retrieve top-k most relevant evidence passages for a legal query.

        Args:
            query: The legal question or query string.
            k: Maximum number of results to return (k >= 1).

        Returns:
            List of RetrievalResult objects ordered descending by cosine similarity.
        """
        if not query or not query.strip():
            return []

        if k <= 0:
            raise ValueError(f"k must be positive (k >= 1), got {k}")

        if self._index is None or len(self._chunks) == 0:
            return []

        effective_k = min(k, len(self._chunks))
        query_vec = self.embedder.embed_query(query.strip())

        # Normalize query vector
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-12:
            query_vec = query_vec / q_norm
        query_f32 = query_vec.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query_f32, effective_k)

        results: List[RetrievalResult] = []
        for r_idx in range(effective_k):
            chunk_row = int(indices[0][r_idx])
            if chunk_row < 0 or chunk_row >= len(self._chunks):
                continue
            sim_score = float(scores[0][r_idx])
            # Clamp precision bounds to valid cosine range [-1.0, 1.0]
            sim_score = max(-1.0, min(1.0, sim_score))
            chunk = self._chunks[chunk_row]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(sim_score, 4),
                    rank=r_idx + 1,
                )
            )

        # Stable sort: primary descending by score, tiebreak by chunk_id
        results.sort(key=lambda r: (-r.score, r.chunk_id))
        return results

    def clear(self) -> None:
        """Reset the index and delete all indexed chunk references."""
        self._index = None
        self._chunks = []


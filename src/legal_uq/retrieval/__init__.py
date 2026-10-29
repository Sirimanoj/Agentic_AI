"""Vector Retrieval module using FAISS and Sentence Transformers."""

from legal_uq.retrieval.vector_store import (
    Embedder,
    CachedSentenceTransformerEmbedder,
    DeterministicFakeEmbedder,
    RetrievalResult,
    VectorStore,
)

__all__ = [
    "Embedder",
    "CachedSentenceTransformerEmbedder",
    "DeterministicFakeEmbedder",
    "RetrievalResult",
    "VectorStore",
]


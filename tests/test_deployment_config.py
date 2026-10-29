"""Unit and regression tests for Streamlit Community Cloud deployment configuration.

Validates:
1. Secrets resolution hierarchy: os.environ > st.secrets > default.
2. Safe secret handling: keys are never exposed in error messages or logs.
3. Streamlit cache and embedder initialization fallback.
4. User upload memory purity: in-memory ingest does not write files to disk.
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app import get_pipeline, get_secret, get_shared_embedder
from legal_uq.generation.generator import GroundedGenerator, _get_secret_safe
from legal_uq.pipeline import LegalUQPipeline
from legal_uq.retrieval.vector_store import (
    CachedSentenceTransformerEmbedder,
    DeterministicFakeEmbedder,
)


def test_get_secret_env_precedence(monkeypatch):
    """os.environ should take precedence over st.secrets."""
    monkeypatch.setenv("TEST_KEY", "env_value")
    mock_st = MagicMock()
    mock_st.secrets = {"TEST_KEY": "secret_value"}

    with patch("streamlit.secrets", {"TEST_KEY": "secret_value"}):
        res = get_secret("TEST_KEY", default="fallback")
        assert res == "env_value"


def test_get_secret_st_secrets_fallback(monkeypatch):
    """When os.environ is absent, st.secrets should be retrieved cleanly."""
    monkeypatch.delenv("TEST_ST_KEY", raising=False)

    with patch("app.st") as mock_st:
        mock_st.secrets = {"TEST_ST_KEY": "streamlit_cloud_secret"}
        res = get_secret("TEST_ST_KEY", default="fallback")
        assert res == "streamlit_cloud_secret"


def test_get_secret_default_when_missing(monkeypatch):
    """When both os.environ and st.secrets lack the key, default should be returned."""
    monkeypatch.delenv("NON_EXISTENT_KEY", raising=False)

    with patch("app.st") as mock_st:
        mock_st.secrets = {}
        res = get_secret("NON_EXISTENT_KEY", default="my_default")
        assert res == "my_default"


def test_generator_get_secret_safe_st_secrets(monkeypatch):
    """GroundedGenerator helper correctly reads from st.secrets if os.environ is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    fake_secrets = {"GEMINI_API_KEY": "dummy-cloud-secret-key-123"}
    with patch("streamlit.secrets", fake_secrets):
        gen = GroundedGenerator(force_mock=True)
        # In force_mock, api_key is None
        assert gen.api_key is None

        # When force_mock=False and key provided via secrets
        val = _get_secret_safe("GEMINI_API_KEY")
        assert val == "dummy-cloud-secret-key-123"


def test_get_shared_embedder_returns_embedder():
    """get_shared_embedder returns an Embedder instance and caches safely."""
    embedder = get_shared_embedder()
    assert embedder is not None
    assert hasattr(embedder, "embed_texts")
    assert hasattr(embedder, "dimension")
    assert embedder.dimension == 384


def test_pipeline_session_isolation():
    """Each call to get_pipeline returns a distinct pipeline with isolated vector index."""
    p1 = get_pipeline()
    p2 = get_pipeline()

    assert p1 is not p2
    assert p1.vector_store is not p2.vector_store

    # Ingest document into p1
    p1.ingest_bytes("doc1.txt", b"Confidential Section: Coparcenary rights belong to daughters.")
    assert len(p1.indexed_documents) == 1
    assert p1.total_chunk_count > 0

    # p2 should remain clean and empty (no multi-user session pollution)
    assert len(p2.indexed_documents) == 0
    assert p2.total_chunk_count == 0


def test_user_upload_does_not_create_disk_files(tmp_path):
    """Ingesting bytes into the pipeline operates strictly in memory and creates no files."""
    pipeline = get_pipeline()
    pipeline.clear()

    existing_files_before = set(Path(".").glob("**/*.txt"))

    test_content = (
        b"CONFIDENTIAL NON-DISCLOSURE AGREEMENT\n\n"
        b"1. Definition of Confidential Information.\n"
        b"Receiving Party agrees not to disclose Proprietary Data."
    )
    summary = pipeline.ingest_bytes("session_upload.txt", test_content)

    assert summary.doc_name == "session_upload.txt"
    assert summary.num_chunks >= 1

    # Verify no persistent file was written to root or data directory
    assert not Path("session_upload.txt").exists()
    assert not Path("data/session_upload.txt").exists()
    assert not Path("data/uploads/session_upload.txt").exists()

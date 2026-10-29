"""Deterministic unit tests for grounded candidate response generation."""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

from legal_uq.generation.generator import (
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS,
    CandidateResponse,
    GenerationRun,
    GroundedGenerator,
    LEGAL_DISCLAIMER,
    MOCK_AGREEMENT_WARNING,
)
from legal_uq.ingestion.models import DocumentChunk
from legal_uq.retrieval.vector_store import RetrievalResult


def make_evidence(locator: str, text: str, page_number: int | None = 1) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chk_{abs(hash(locator)) % 1000}",
        doc_name="contract.pdf",
        doc_type="pdf",
        page_number=page_number,
        section_title="TERMS",
        source_locator=locator,
        text=text,
        character_count=len(text),
        token_count=len(text.split()),
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=0.88, rank=1)


def test_offline_mock_mode_produces_three_labeled_candidates() -> None:
    # Ensure no API key is active
    generator = GroundedGenerator(api_key=None, client=None)
    assert generator.is_live is False

    ev1 = make_evidence("contract.pdf:page_1 [SECTION 1]", "The agreement commences on January 1, 2026.")
    ev2 = make_evidence("contract.pdf:page_2 [SECTION 2]", "Payment is due net 30 days after invoice issuance.")

    run = generator.generate_candidates(
        query="When does the agreement commence?",
        evidence=[ev1, ev2],
        num_candidates=3,
    )

    assert run.mode == "MOCK"
    assert run.model_name == "offline-deterministic-mock"
    assert len(run.candidates) == 3

    for i, c in enumerate(run.candidates):
        assert c.candidate_index == i
        assert c.generation_mode == "MOCK"
        assert len(c.answer_text) > 0
        # Sources cited must be real source locators
        assert "contract.pdf:page_1 [SECTION 1]" in c.cited_sources

    assert LEGAL_DISCLAIMER in run.disclaimer
    assert "MOCK/DEMO MODE NOTICE" in run.disclaimer


def test_offline_mock_mode_with_empty_evidence_abstains() -> None:
    generator = GroundedGenerator(api_key=None, client=None)
    run = generator.generate_candidates(
        query="What are the liquidated damages?",
        evidence=[],
        num_candidates=3,
    )

    assert run.mode == "MOCK"
    assert len(run.candidates) == 3

    for c in run.candidates:
        assert c.generation_mode == "MOCK"
        assert c.cited_sources == []
        # Must clearly state that evidence is missing and abstain
        assert "no retrieved evidence addressing" in c.answer_text.lower() or "not contain sufficient" in c.answer_text.lower()


def test_live_generation_with_mocked_genai_client() -> None:
    # Create a mock Google GenAI client adhering to modern client.models.generate_content interface
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Under [S1], the agreement commences on January 1, 2026 with 30 days termination notice."
    mock_client.models.generate_content.return_value = mock_resp

    generator = GroundedGenerator(
        api_key="synthetic-test-key",
        model_name="gemini-3.8-flash",
        client=mock_client,
    )
    assert generator.is_live is True

    ev = make_evidence("contract.pdf:page_1", "The agreement commences on January 1, 2026.")
    run = generator.generate_candidates(
        query="When does the agreement commence?",
        evidence=[ev],
        num_candidates=3,
        temperature=0.7,
    )

    assert run.mode == "LIVE"
    assert run.model_name == "gemini-3.8-flash"
    assert len(run.candidates) == 3
    assert mock_client.models.generate_content.call_count == 3

    for i, c in enumerate(run.candidates):
        assert c.candidate_index == i
        assert c.generation_mode == "LIVE"
        assert "January 1, 2026" in c.answer_text
        # [S1] mapping
        assert "contract.pdf:page_1" in c.cited_sources


def test_live_generation_with_interactions_api_interface() -> None:
    # Create a mock client adhering to Google Interactions API (client.interactions.create)
    mock_client = MagicMock()
    del mock_client.models  # Ensure models interface is not present
    mock_interaction = MagicMock()
    mock_interaction.output_text = "According to [S1], liability is capped at $50,000."
    mock_client.interactions.create.return_value = mock_interaction

    generator = GroundedGenerator(
        api_key="synthetic-test-key",
        model_name="gemini-3.8-flash",
        client=mock_client,
    )
    assert generator.is_live is True

    ev = make_evidence("contract.pdf:page_2", "Liability is capped at $50,000.")
    run = generator.generate_candidates(
        query="What is the liability cap?",
        evidence=[ev],
        num_candidates=3,
        temperature=0.5,
    )

    assert run.mode == "LIVE"
    assert run.model_name == "gemini-3.8-flash"
    assert len(run.candidates) == 3
    assert mock_client.interactions.create.call_count == 3
    assert "contract.pdf:page_2" in run.candidates[0].cited_sources
    assert "$50,000" in run.candidates[0].answer_text


def test_live_generation_api_failure_raises_error_not_silent_mock() -> None:
    # Failing client
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("ResourceExhausted: Quota exceeded")

    generator = GroundedGenerator(
        api_key="synthetic-test-key",
        client=mock_client,
        max_retries=1,
    )
    assert generator.is_live is True

    ev = make_evidence("contract.pdf:page_1", "Sample clause.")

    # Must raise RuntimeError and NOT silently pretend to succeed with mock
    with pytest.raises(RuntimeError) as exc_info:
        generator.generate_candidates("query", [ev], num_candidates=3)

    assert "Live Gemini generation failed" in str(exc_info.value)
    assert "Quota exceeded" in str(exc_info.value)


def test_safe_configuration_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test default model when environment variable is unset
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    gen_default = GroundedGenerator()
    assert gen_default.model_name == DEFAULT_GEMINI_MODEL
    assert gen_default.model_name == "gemini-3.8-flash"
    assert gen_default.is_live is False

    # Test environment variable override
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    gen_override = GroundedGenerator()
    assert gen_override.model_name == "gemini-3.6-flash"
    assert gen_override.is_live is False

    # Never expose API key in string representations
    gen_with_key = GroundedGenerator(api_key="secret-key-12345")
    assert "secret-key-12345" not in repr(gen_with_key.__dict__.get("model_name"))


def test_force_mock_overrides_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure force_mock=True strictly guarantees mock execution even if env key exists."""
    monkeypatch.setenv("GEMINI_API_KEY", "real-looking-env-key-abcdef12345")
    gen = GroundedGenerator(force_mock=True)
    assert gen.is_live is False
    assert gen.api_key is None

    ev = make_evidence("contract.pdf:page_1", "Sample text.")
    run = gen.generate_candidates("query", [ev], num_candidates=3)
    assert run.mode == "MOCK"
    assert run.candidates[0].generation_mode == "MOCK"


def test_missing_key_live_client_initialization_raises_error() -> None:
    """Directly invoking client creation without an API key must raise an explicit RuntimeError."""
    gen = GroundedGenerator(api_key=None, client=None, force_mock=False)
    with pytest.raises(RuntimeError) as exc_info:
        gen._get_or_create_client()
    assert "No API key provided" in str(exc_info.value)


def test_error_sanitization_redacts_api_key_and_tokens() -> None:
    """Ensure API keys and Bearer tokens are scrubbed from error strings."""
    from legal_uq.generation.generator import _sanitize_error_message

    raw_err = (
        "HTTP 403 Forbidden: Request failed for key secret-api-key-998877 with "
        "AIzaSyD_TestGoogleKeyPattern1234567 and Bearer ya29.a0AfH6SMTestToken12345"
    )
    sanitized = _sanitize_error_message(raw_err, api_key="secret-api-key-998877")

    assert "secret-api-key-998877" not in sanitized
    assert "AIzaSyD_TestGoogleKeyPattern1234567" not in sanitized
    assert "ya29.a0AfH6SMTestToken12345" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_legacy_genai_interface_fallback() -> None:
    """Ensure client with client.generate_content (legacy SDK interface) works properly."""
    mock_legacy_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "According to [S1], the term is 2 years."
    mock_legacy_client.generate_content.return_value = mock_resp
    # Ensure it doesn't have client.models or client.interactions
    del mock_legacy_client.models
    del mock_legacy_client.interactions

    generator = GroundedGenerator(
        api_key="synthetic-test-key",
        client=mock_legacy_client,
    )
    assert generator.is_live is True

    ev = make_evidence("contract.pdf:page_1", "The term is 2 years.")
    run = generator.generate_candidates("What is the term?", [ev], num_candidates=2)

    assert run.mode == "LIVE"
    assert len(run.candidates) == 2
    assert mock_legacy_client.generate_content.call_count == 2
    assert "contract.pdf:page_1" in run.candidates[0].cited_sources


def test_pipeline_missing_key_live_mode_refuses_silent_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure pipeline in live mode without a key fails explicitly rather than falling back to mock."""
    from legal_uq.pipeline import LegalUQPipeline
    from legal_uq.retrieval.vector_store import DeterministicFakeEmbedder

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    pipeline = LegalUQPipeline(embedder=DeterministicFakeEmbedder(dimension=384))
    # User selects Live mode with no key
    pipeline.set_api_key(None, model_name="gemini-3.8-flash", force_mock=False)

    pipeline.ingest_bytes("terms.txt", b"Payment is due within 30 days of invoice.")

    # Must raise RuntimeError about missing API key, not silently generate mock answers
    with pytest.raises(RuntimeError) as exc_info:
        pipeline.query("What is the payment window?")

    assert "No API key provided" in str(exc_info.value)


# ============================================================================
# Concurrency, Ordering, Latency, and Partial Failure Tests
# ============================================================================


def test_concurrent_candidate_generation_ordering_and_latency() -> None:
    """Verify that concurrent execution preserves candidate index order and tracks per-candidate latency."""
    import time

    mock_client = MagicMock()

    def mock_gen_content(model: str, contents: str, config: Any = None) -> Any:
        # Simulate small variable execution latency
        time.sleep(0.02)
        resp = MagicMock()
        resp.text = f"Candidate response under [S1] from model {model}."
        return resp

    mock_client.models.generate_content.side_effect = mock_gen_content

    gen = GroundedGenerator(
        api_key="synthetic-key",
        model_name="gemini-3.8-flash",
        client=mock_client,
        max_concurrency=3,
        request_timeout_seconds=10.0,
    )

    ev = make_evidence("contract.pdf:page_1", "Grounded text.")
    run = gen.generate_candidates("What are the terms?", [ev], num_candidates=3)

    assert run.mode == "LIVE"
    assert len(run.candidates) == 3
    assert run.metadata.get("concurrency") == 3

    for i, c in enumerate(run.candidates):
        # Strict index order check
        assert c.candidate_index == i
        assert c.generation_mode == "LIVE"
        assert "contract.pdf:page_1" in c.cited_sources
        # Safe latency check
        assert "latency_seconds" in c.metadata
        assert isinstance(c.metadata["latency_seconds"], float)
        assert c.metadata["latency_seconds"] >= 0.01


def test_concurrent_candidate_partial_failure_raises_sanitized_error() -> None:
    """Verify that a failure in one of the concurrent candidate tasks cleanly aborts without mock fallback."""
    mock_client = MagicMock()
    call_count = 0

    def mock_flaky_gen(model: str, contents: str, config: Any = None) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("API quota exhausted for key secret-live-token-12345")
        resp = MagicMock()
        resp.text = "Valid answer under [S1]."
        return resp

    mock_client.models.generate_content.side_effect = mock_flaky_gen

    gen = GroundedGenerator(
        api_key="secret-live-token-12345",
        model_name="gemini-3.8-flash",
        client=mock_client,
        max_concurrency=3,
        max_retries=0,  # Fail fast
    )

    ev = make_evidence("contract.pdf:page_1", "Grounded text.")

    with pytest.raises(RuntimeError) as exc_info:
        gen.generate_candidates("Inquiry", [ev], num_candidates=3)

    err_str = str(exc_info.value)
    assert "Live Gemini generation failed" in err_str
    # Secret must be strictly scrubbed
    assert "secret-live-token-12345" not in err_str
    assert "[REDACTED_API_KEY]" in err_str or "quota exhausted" in err_str


def test_sequential_candidate_generation_when_concurrency_is_one() -> None:
    """Verify that setting max_concurrency=1 safely executes sequentially with latency tracking."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Sequential answer text under [S1]."
    mock_client.models.generate_content.return_value = mock_resp

    gen = GroundedGenerator(
        api_key="synthetic-key",
        model_name="gemini-3.6-flash",
        client=mock_client,
        max_concurrency=1,
    )

    ev = make_evidence("contract.pdf:page_1", "Grounded text.")
    run = gen.generate_candidates("Inquiry", [ev], num_candidates=2)

    assert len(run.candidates) == 2
    assert run.candidates[0].candidate_index == 0
    assert run.candidates[1].candidate_index == 1
    assert "latency_seconds" in run.candidates[0].metadata
    assert "latency_seconds" in run.candidates[1].metadata





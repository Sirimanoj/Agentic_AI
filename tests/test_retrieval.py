"""Deterministic unit tests for FAISS vector retrieval engine."""

import numpy as np
import pytest

from legal_uq.ingestion.models import DocumentChunk
from legal_uq.retrieval.vector_store import (
    DeterministicFakeEmbedder,
    RetrievalResult,
    VectorStore,
)


def make_chunk(
    chunk_id: str,
    doc_name: str,
    text: str,
    page_number: int | None = None,
    section_title: str | None = None,
    doc_type: str = "pdf",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_name=doc_name,
        doc_type=doc_type,
        page_number=page_number,
        section_title=section_title,
        source_locator=f"{doc_name}:p{page_number or 'na'} [{section_title or 'General'}]",
        text=text,
        character_count=len(text),
        token_count=len(text.split()),
        chunk_index=0,
    )


@pytest.fixture
def fake_store() -> VectorStore:
    embedder = DeterministicFakeEmbedder(dimension=128)
    return VectorStore(embedder=embedder, relevance_heuristic_threshold=0.50)


def test_retrieval_ranking_relevant_ahead_of_unrelated(fake_store: VectorStore) -> None:
    c_relevant = make_chunk(
        chunk_id="c_indemnity",
        doc_name="contract.pdf",
        page_number=3,
        section_title="INDEMNIFICATION",
        text="The Contractor shall indemnify and hold harmless the Client against intellectual property infringement claims.",
    )
    c_unrelated = make_chunk(
        chunk_id="c_boilerplate",
        doc_name="contract.pdf",
        page_number=7,
        section_title="MISCELLANEOUS",
        text="All notices under this agreement shall be delivered via certified postal mail or registered courier.",
    )

    indexed = fake_store.build_index([c_unrelated, c_relevant])
    assert indexed == 2

    # Query matching indemnification vocabulary
    results = fake_store.retrieve("intellectual property infringement indemnity contractor", k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c_indemnity"
    assert results[0].rank == 1
    assert results[0].score > results[1].score
    assert -1.0 <= results[0].score <= 1.0


def test_returned_passages_retain_all_source_metadata(fake_store: VectorStore) -> None:
    c = make_chunk(
        chunk_id="docA::p4::c1",
        doc_name="lease_agreement.pdf",
        page_number=4,
        section_title="SECURITY DEPOSIT",
        text="Tenant shall deposit the sum of twenty thousand rupees upon signing.",
    )
    fake_store.build_index([c])

    results = fake_store.retrieve("security deposit amount", k=1)
    assert len(results) == 1
    res = results[0]

    assert res.chunk_id == "docA::p4::c1"
    assert res.doc_name == "lease_agreement.pdf"
    assert res.page_number == 4
    assert res.section_title == "SECURITY DEPOSIT"
    assert "lease_agreement.pdf:p4 [SECURITY DEPOSIT]" in res.source_locator
    assert "twenty thousand rupees" in res.text
    assert res.rank == 1


def test_empty_index_and_invalid_inputs(fake_store: VectorStore) -> None:
    # Query empty store
    assert fake_store.retrieve("any query", k=3) == []

    # Blank query
    c = make_chunk("c1", "doc.pdf", "Some text")
    fake_store.build_index([c])
    assert fake_store.retrieve("", k=3) == []
    assert fake_store.retrieve("   \n ", k=3) == []

    # Invalid k
    with pytest.raises(ValueError):
        fake_store.retrieve("valid query", k=0)

    # k larger than total chunks
    results = fake_store.retrieve("Some text", k=10)
    assert len(results) == 1  # Capped at available count


def test_index_replacement_and_clear(fake_store: VectorStore) -> None:
    c1 = make_chunk("c1", "doc1.pdf", "First text")
    c2 = make_chunk("c2", "doc1.pdf", "Second text")
    fake_store.build_index([c1, c2])
    assert fake_store.total_chunks == 2

    # Replace index with single new chunk
    c3 = make_chunk("c3", "doc2.pdf", "Brand new replacement text")
    fake_store.build_index([c3])
    assert fake_store.total_chunks == 1

    res = fake_store.retrieve("Brand new", k=5)
    assert len(res) == 1
    assert res[0].chunk_id == "c3"

    # Clear
    fake_store.clear()
    assert fake_store.total_chunks == 0
    assert fake_store.retrieve("Brand new", k=1) == []


def test_multiple_documents_retrieval(fake_store: VectorStore) -> None:
    c_pdf = make_chunk("c_pdf", "statute.pdf", "Section 10 of Indian Contract Act requires lawful consideration.", page_number=2)
    c_docx = make_chunk("c_docx", "mou.docx", "Payment milestones are outlined in Exhibit B.", page_number=None, doc_type="docx")
    c_txt = make_chunk("c_txt", "notice.txt", "Failure to cure default within fifteen days will result in termination.", page_number=None, doc_type="txt")

    fake_store.build_index([c_pdf, c_docx, c_txt])
    assert fake_store.total_chunks == 3

    # Query targeting mou.docx
    res = fake_store.retrieve("payment milestones exhibit", k=1)
    assert len(res) == 1
    assert res[0].doc_name == "mou.docx"
    assert res[0].page_number is None  # DOCX page remains None

    # Query targeting statute.pdf
    res_pdf = fake_store.retrieve("lawful consideration contract", k=1)
    assert len(res_pdf) == 1
    assert res_pdf[0].doc_name == "statute.pdf"
    assert res_pdf[0].page_number == 2


def test_embedding_normalization_exact_match(fake_store: VectorStore) -> None:
    text = "Confidential Information shall not be disclosed to any competitor."
    c = make_chunk("c_exact", "nda.pdf", text)
    fake_store.build_index([c])

    # Exact query
    results = fake_store.retrieve(text, k=1)
    assert len(results) == 1
    # Normalized self-similarity should be ~1.0
    assert pytest.approx(results[0].score, abs=1e-3) == 1.0


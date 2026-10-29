"""Automated deterministic unit tests for legal document ingestion and chunking."""

import io
import os
import tempfile
from pathlib import Path

import docx
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from legal_uq.ingestion.chunker import LegalChunker
from legal_uq.ingestion.models import (
    CorruptDocumentError,
    DocumentChunk,
    DocumentType,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    UnsupportedFormatError,
)
from legal_uq.ingestion.parser import (
    parse_document_bytes,
    parse_document_file,
    parse_docx_bytes,
    parse_pdf_bytes,
    parse_txt_bytes,
)


# ============================================================================
# Fixture Helpers
# ============================================================================


def create_synthetic_pdf_bytes(pages_text: list[str]) -> bytes:
    """Generate an authentic in-memory multi-page PDF using reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages_text:
        t = c.beginText(50, 750)
        t.setFont("Helvetica", 11)
        for line in text.split("\n"):
            t.textLine(line)
        c.drawText(t)
        c.showPage()
    c.save()
    return buf.getvalue()


def create_synthetic_docx_bytes() -> bytes:
    """Generate an authentic in-memory DOCX file with headings and tables."""
    doc = docx.Document()
    doc.add_heading("NON-DISCLOSURE AGREEMENT", level=0)

    doc.add_heading("1. CONFIDENTIAL INFORMATION", level=1)
    doc.add_paragraph(
        "The Recipient shall hold and maintain the Confidential Information in strictest confidence "
        "for the sole and exclusive benefit of the Disclosing Party. (a) Confidential Information includes "
        "technical and business information; (b) Trade secrets shall be maintained indefinitely."
    )

    doc.add_heading("2. TERM AND TERMINATION", level=1)
    doc.add_paragraph(
        "This Agreement shall remain in effect for a term of three (3) years from the Effective Date."
    )

    # Add a structured table
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Party"
    table.rows[0].cells[1].text = "Notice Address"
    table.rows[1].cells[0].text = "Disclosing Party"
    table.rows[1].cells[1].text = "legal@discloser.example"

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ============================================================================
# TXT Parser Tests
# ============================================================================


def test_txt_parsing_utf8_and_headings() -> None:
    text = (
        "ARTICLE I: DEFINITIONS\n\n"
        "1.1 'Agreement' means this synthetic master service agreement.\n"
        "1.2 'Services' means the software testing and evaluation services.\n\n"
        "SECTION 2: OBLIGATIONS\n\n"
        "The Service Provider will perform all duties with professional diligence."
    )
    doc = parse_txt_bytes("agreement.txt", text.encode("utf-8"))

    assert doc.doc_name == "agreement.txt"
    assert doc.doc_type == DocumentType.TXT.value
    assert len(doc.blocks) == 4
    # All non-paginated formats must have page_number=None (never fabricated)
    assert all(b.page_number is None for b in doc.blocks)
    assert doc.blocks[0].section_title == "ARTICLE I: DEFINITIONS"
    assert doc.blocks[3].section_title == "SECTION 2: OBLIGATIONS"
    assert "agreement.txt:block_0" in doc.blocks[0].source_locator


def test_txt_parsing_cp1252_fallback() -> None:
    # Text containing Windows-1252 specific characters (em-dash, smart quotes, pound sign)
    raw_cp1252 = (
        "SECTION 3: COMPENSATION\n\n"
        "The fee shall be \u00a35,000 \u2014 payable upon delivery of the 'Milestone 1' report."
    ).encode("cp1252")

    doc = parse_txt_bytes("notice.txt", raw_cp1252)
    assert doc.total_characters > 0
    assert "5,000" in doc.full_text
    assert doc.blocks[0].section_title == "SECTION 3: COMPENSATION"


# ============================================================================
# DOCX Parser Tests
# ============================================================================


def test_docx_parsing_paragraphs_and_headings() -> None:
    docx_bytes = create_synthetic_docx_bytes()
    doc = parse_docx_bytes("sample_nda.docx", docx_bytes)

    assert doc.doc_name == "sample_nda.docx"
    assert doc.doc_type == DocumentType.DOCX.value
    assert len(doc.blocks) >= 4

    # Verify no page numbers were invented for DOCX
    for b in doc.blocks:
        assert b.page_number is None

    # Verify headings are captured
    headings = [b.text for b in doc.blocks if b.metadata.get("is_heading")]
    assert "NON-DISCLOSURE AGREEMENT" in headings
    assert "1. CONFIDENTIAL INFORMATION" in headings
    assert "2. TERM AND TERMINATION" in headings

    # Verify table row extraction
    table_blocks = [b for b in doc.blocks if b.metadata.get("is_table")]
    assert len(table_blocks) == 2
    assert "Disclosing Party | legal@discloser.example" in table_blocks[1].text


# ============================================================================
# PDF Parser Tests
# ============================================================================


def test_pdf_parsing_page_extraction_and_1based_indexing() -> None:
    pages = [
        "SECTION 1: APPOINTMENT\nThe Company hereby appoints the Consultant as advisor.",
        "SECTION 2: COMPENSATION\nThe Consultant shall receive a monthly retainer.",
        "SECTION 3: GOVERNING LAW\nThis Agreement is governed by the laws of India.",
    ]
    pdf_bytes = create_synthetic_pdf_bytes(pages)
    doc = parse_pdf_bytes("contract.pdf", pdf_bytes)

    assert doc.doc_name == "contract.pdf"
    assert doc.doc_type == DocumentType.PDF.value
    assert len(doc.blocks) == 3

    # Check 1-based page numbers
    assert doc.blocks[0].page_number == 1
    assert doc.blocks[1].page_number == 2
    assert doc.blocks[2].page_number == 3

    assert "contract.pdf:page_1" in doc.blocks[0].source_locator
    assert "contract.pdf:page_2" in doc.blocks[1].source_locator
    assert "contract.pdf:page_3" in doc.blocks[2].source_locator

    assert doc.metadata["page_count"] == 3
    assert doc.metadata["extracted_pages"] == 3


# ============================================================================
# Error Handling & Edge Cases
# ============================================================================


def test_unsupported_extensions() -> None:
    with pytest.raises(UnsupportedFormatError) as exc_info:
        parse_document_bytes("spreadsheet.xlsx", b"fake binary data")
    assert "Unsupported document format '.xlsx'" in str(exc_info.value)

    with pytest.raises(UnsupportedFormatError):
        parse_document_bytes("archive.zip", b"fake zip data")


def test_empty_documents() -> None:
    # Zero bytes
    with pytest.raises(EmptyDocumentError):
        parse_document_bytes("empty.pdf", b"")

    with pytest.raises(EmptyDocumentError):
        parse_document_bytes("empty.docx", b"")

    with pytest.raises(EmptyDocumentError):
        parse_document_bytes("empty.txt", b"")

    # Whitespace only
    with pytest.raises(EmptyDocumentError):
        parse_document_bytes("spaces.txt", b"   \n\t   \n  ")


def test_corrupt_documents() -> None:
    # Corrupt PDF header
    with pytest.raises(CorruptDocumentError):
        parse_document_bytes("broken.pdf", b"%PDF-1.4 garbage binary that cannot be parsed")

    # Corrupt DOCX (invalid zip)
    with pytest.raises(CorruptDocumentError):
        parse_document_bytes("broken.docx", b"PK\x03\x04corrupted-docx-header")


# ============================================================================
# Legal Chunker Tests
# ============================================================================


def test_chunker_size_and_overlap_behavior() -> None:
    # Synthetic contract provision
    text = (
        "1.1 Obligation of Confidentiality. The Receiving Party agrees that it shall take all "
        "reasonable measures to protect the secrecy of and avoid disclosure and unauthorized use "
        "of the Confidential Information. "
        "1.2 Non-Disclosure to Third Parties. Neither party will disclose to any third party "
        "the existence or terms of this Agreement without prior written consent. "
        "1.3 Permitted Disclosures. A party may disclose Confidential Information to the extent "
        "required by applicable law, regulation, or court order, provided prompt advance written "
        "notice is given to the other party to allow seeking a protective order."
    )
    block = ParsedBlock(
        text=text,
        doc_name="terms.txt",
        doc_type="txt",
        page_number=None,
        paragraph_index=0,
        section_title="1. CONFIDENTIALITY",
    )

    chunker = LegalChunker(target_chunk_size=250, chunk_overlap=60)
    chunks = chunker.chunk_block(block)

    assert len(chunks) > 1
    # Stable IDs and metadata
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"terms.txt::pna::c{i}"
        assert c.doc_name == "terms.txt"
        assert c.page_number is None
        assert c.section_title == "1. CONFIDENTIALITY"
        assert c.character_count == len(c.text)
        assert c.token_count > 0

    # Overlap check: text from chunk 0 should partially appear in chunk 1
    assert any(word in chunks[1].text for word in chunks[0].text.split()[-3:])


def test_chunker_pdf_page_citation_traceability() -> None:
    """Test that chunks never combine text from different PDF pages."""
    p1_text = (
        "ARTICLE 1. RECITALS.\n"
        "WHEREAS, Company is engaged in technology research and development;\n"
        "WHEREAS, Consultant possesses expertise in artificial intelligence;\n"
        "NOW, THEREFORE, the parties agree as follows."
    )
    p2_text = (
        "ARTICLE 2. DELIVERABLES.\n"
        "The Consultant shall deliver quarterly technical reports and source code.\n"
        "All work product shall be submitted in electronic format."
    )

    doc = ParsedDocument(
        doc_name="contract.pdf",
        doc_type="pdf",
        blocks=[
            ParsedBlock(text=p1_text, doc_name="contract.pdf", doc_type="pdf", page_number=1),
            ParsedBlock(text=p2_text, doc_name="contract.pdf", doc_type="pdf", page_number=2),
        ],
        total_characters=len(p1_text) + len(p2_text),
    )

    chunker = LegalChunker(target_chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2

    # Check page 1 chunks have page_number=1, page 2 chunks have page_number=2
    p1_chunks = [c for c in chunks if c.page_number == 1]
    p2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(p1_chunks) > 0
    assert len(p2_chunks) > 0
    assert len(p1_chunks) + len(p2_chunks) == len(chunks)

    # Assert no page 1 chunk contains text from page 2
    for c in p1_chunks:
        assert "DELIVERABLES" not in c.text
        assert "ARTICLE 2" not in c.text

    # Assert no page 2 chunk contains text from page 1
    for c in p2_chunks:
        assert "RECITALS" not in c.text
        assert "WHEREAS" not in c.text


def test_chunker_infinite_loop_guard_on_long_unpunctuated_text() -> None:
    """Ensure hard-slicing handles long text without punctuation or spaces without looping."""
    # 2,000 characters of unbroken text
    unbroken_text = "ABCDEFGHIJ" * 200
    block = ParsedBlock(
        text=unbroken_text,
        doc_name="opaque.txt",
        doc_type="txt",
        page_number=None,
    )

    chunker = LegalChunker(target_chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_block(block)

    assert len(chunks) >= 6
    # Verify strict forward progression
    total_reconstructed = "".join(c.text for c in chunks)
    assert len(total_reconstructed) >= len(unbroken_text)


def test_parse_document_file_convenience(tmp_path: Path) -> None:
    """Test reading from local disk using parse_document_file."""
    test_file = tmp_path / "statute_excerpt.txt"
    test_file.write_text(
        "SECTION 10: WHAT AGREEMENTS ARE CONTRACTS.\n"
        "All agreements are contracts if they are made by the free consent of parties competent to contract.",
        encoding="utf-8",
    )

    doc = parse_document_file(test_file)
    assert doc.doc_name == "statute_excerpt.txt"
    assert "SECTION 10" in doc.full_text
    assert doc.total_characters > 0

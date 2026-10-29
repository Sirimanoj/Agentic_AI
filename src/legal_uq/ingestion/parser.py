"""Document parsing engine for legal PDF, DOCX, and TXT files.

Extracts text while preserving provenance metadata (page numbers, paragraph indices,
headings, and source locators) for downstream grounded citation.
"""

import io
import os
import re
from pathlib import Path
from typing import List, Optional, Union

import docx
import pypdf

from legal_uq.ingestion.models import (
    CorruptDocumentError,
    DocumentType,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    UnsupportedFormatError,
)

# Recognized legal heading patterns in text/docx
_LEGAL_HEADING_REGEX = re.compile(
    r"^(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|ANNEXURE|APPENDIX)\s+[0-9IVXLCDM]+[:.]?\s*.*|"
    r"^[0-9]+(?:\.[0-9]+)*\s+[A-Z][A-Za-z0-9\s,–—\-]{2,80}$|"
    r"^[A-Z\s]{4,80}$",
    re.IGNORECASE,
)


def _detect_legal_heading(text: str) -> Optional[str]:
    """Identify if a single line represents a formal legal heading or section title."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return None
    # Avoid treating full sentences ending in periods as headings
    if stripped.endswith(".") and not re.match(r"^(?:ARTICLE|SECTION|CLAUSE)\s+\d+\.", stripped, re.IGNORECASE):
        return None
    if _LEGAL_HEADING_REGEX.match(stripped):
        return stripped
    return None


def parse_pdf_bytes(filename: str, content_bytes: bytes) -> ParsedDocument:
    """Extract page-by-page text from PDF bytes with 1-based page metadata."""
    if not content_bytes:
        raise EmptyDocumentError(f"PDF file '{filename}' contains 0 bytes.")

    try:
        stream = io.BytesIO(content_bytes)
        reader = pypdf.PdfReader(stream)
    except Exception as exc:
        raise CorruptDocumentError(f"Failed to read PDF '{filename}': {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypt_res = reader.decrypt("")
            if decrypt_res == 0:
                raise CorruptDocumentError(f"PDF '{filename}' is encrypted with a password.")
        except Exception as exc:
            raise CorruptDocumentError(f"Encrypted PDF '{filename}' could not be decrypted: {exc}") from exc

    if len(reader.pages) == 0:
        raise EmptyDocumentError(f"PDF file '{filename}' contains 0 pages.")

    blocks: List[ParsedBlock] = []
    total_chars = 0
    current_heading: Optional[str] = None

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-based page indexing
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise CorruptDocumentError(f"Error extracting text from page {page_num} of '{filename}': {exc}") from exc

        page_text_clean = page_text.strip()
        if not page_text_clean:
            continue

        # Look for potential leading section headings on the page
        lines = page_text_clean.splitlines()
        for line in lines[:3]:
            detected = _detect_legal_heading(line)
            if detected:
                current_heading = detected
                break

        block = ParsedBlock(
            text=page_text_clean,
            doc_name=filename,
            doc_type=DocumentType.PDF.value,
            page_number=page_num,
            paragraph_index=page_idx,
            section_title=current_heading,
            source_locator=f"{filename}:page_{page_num}" + (f" [{current_heading}]" if current_heading else ""),
            metadata={"raw_page_number": page_num, "char_len": len(page_text_clean)},
        )
        blocks.append(block)
        total_chars += len(page_text_clean)

    if not blocks or total_chars == 0:
        raise EmptyDocumentError(f"PDF file '{filename}' contains no extractable text (may be image-only scan).")

    return ParsedDocument(
        doc_name=filename,
        doc_type=DocumentType.PDF.value,
        blocks=blocks,
        total_characters=total_chars,
        metadata={"page_count": len(reader.pages), "extracted_pages": len(blocks)},
    )


def parse_docx_bytes(filename: str, content_bytes: bytes) -> ParsedDocument:
    """Extract paragraphs and headings from DOCX bytes.

    Note: DOCX page numbers cannot be reliably extracted by python-docx,
    so page_number is explicitly set to None (never fabricated).
    Paragraph and heading metadata are preserved instead.
    """
    if not content_bytes:
        raise EmptyDocumentError(f"DOCX file '{filename}' contains 0 bytes.")

    try:
        stream = io.BytesIO(content_bytes)
        doc = docx.Document(stream)
    except Exception as exc:
        raise CorruptDocumentError(f"Failed to read DOCX '{filename}': {exc}") from exc

    blocks: List[ParsedBlock] = []
    total_chars = 0
    current_heading: Optional[str] = None
    p_counter = 0

    # Extract standard paragraphs
    for p in doc.paragraphs:
        raw_text = p.text.strip()
        if not raw_text:
            continue

        style_name = getattr(p.style, "name", "")
        if style_name and ("Heading" in style_name or "Title" in style_name):
            current_heading = raw_text
            # Store heading itself as a block for section context
            block = ParsedBlock(
                text=raw_text,
                doc_name=filename,
                doc_type=DocumentType.DOCX.value,
                page_number=None,  # Not available in DOCX
                paragraph_index=p_counter,
                section_title=current_heading,
                source_locator=f"{filename}:heading_{p_counter} [{current_heading}]",
                metadata={"is_heading": True, "style": style_name},
            )
            blocks.append(block)
            total_chars += len(raw_text)
            p_counter += 1
            continue

        # Check regex heading in regular paragraph text
        detected = _detect_legal_heading(raw_text)
        if detected:
            current_heading = detected

        block = ParsedBlock(
            text=raw_text,
            doc_name=filename,
            doc_type=DocumentType.DOCX.value,
            page_number=None,
            paragraph_index=p_counter,
            section_title=current_heading,
            source_locator=f"{filename}:para_{p_counter}" + (f" [{current_heading}]" if current_heading else ""),
            metadata={"is_heading": False, "style": style_name},
        )
        blocks.append(block)
        total_chars += len(raw_text)
        p_counter += 1

    # Extract tables if present
    for t_idx, table in enumerate(doc.tables):
        for r_idx, row in enumerate(table.rows):
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if not row_cells:
                continue
            row_text = " | ".join(row_cells)
            block = ParsedBlock(
                text=row_text,
                doc_name=filename,
                doc_type=DocumentType.DOCX.value,
                page_number=None,
                paragraph_index=p_counter,
                section_title=current_heading,
                source_locator=f"{filename}:table_{t_idx}_row_{r_idx}" + (f" [{current_heading}]" if current_heading else ""),
                metadata={"is_table": True, "table_index": t_idx, "row_index": r_idx},
            )
            blocks.append(block)
            total_chars += len(row_text)
            p_counter += 1

    if not blocks or total_chars == 0:
        raise EmptyDocumentError(f"DOCX file '{filename}' contains no extractable text.")

    return ParsedDocument(
        doc_name=filename,
        doc_type=DocumentType.DOCX.value,
        blocks=blocks,
        total_characters=total_chars,
        metadata={"paragraph_count": p_counter, "table_count": len(doc.tables)},
    )


def parse_txt_bytes(filename: str, content_bytes: bytes) -> ParsedDocument:
    """Extract paragraphs and clauses from TXT bytes with safe fallback encodings."""
    if not content_bytes:
        raise EmptyDocumentError(f"TXT file '{filename}' contains 0 bytes.")

    decoded_text: Optional[str] = None
    encodings_to_try = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

    for enc in encodings_to_try:
        try:
            decoded_text = content_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if decoded_text is None:
        raise CorruptDocumentError(f"TXT file '{filename}' could not be decoded with any standard encoding.")

    clean_text = decoded_text.strip()
    if not clean_text:
        raise EmptyDocumentError(f"TXT file '{filename}' contains only empty whitespace.")

    # Split into paragraphs by double newlines or single newlines
    raw_paras = re.split(r"\n\s*\n+", clean_text)
    blocks: List[ParsedBlock] = []
    total_chars = 0
    current_heading: Optional[str] = None

    for p_idx, para in enumerate(raw_paras):
        para_clean = para.strip()
        if not para_clean:
            continue

        detected = _detect_legal_heading(para_clean)
        if detected:
            current_heading = detected

        block = ParsedBlock(
            text=para_clean,
            doc_name=filename,
            doc_type=DocumentType.TXT.value,
            page_number=None,
            paragraph_index=p_idx,
            section_title=current_heading,
            source_locator=f"{filename}:block_{p_idx}" + (f" [{current_heading}]" if current_heading else ""),
            metadata={"encoding": enc},
        )
        blocks.append(block)
        total_chars += len(para_clean)

    if not blocks or total_chars == 0:
        raise EmptyDocumentError(f"TXT file '{filename}' has no extractable text content.")

    return ParsedDocument(
        doc_name=filename,
        doc_type=DocumentType.TXT.value,
        blocks=blocks,
        total_characters=total_chars,
        metadata={"block_count": len(blocks)},
    )


def parse_document_bytes(filename: str, content_bytes: bytes) -> ParsedDocument:
    """Public parser dispatching by filename extension.

    Accepts raw bytes, making it directly compatible with Streamlit's UploadedFile.read()
    as well as standard Python byte streams.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return parse_pdf_bytes(filename, content_bytes)
    elif ext == ".docx":
        return parse_docx_bytes(filename, content_bytes)
    elif ext == ".txt":
        return parse_txt_bytes(filename, content_bytes)
    else:
        raise UnsupportedFormatError(
            f"Unsupported document format '{ext}' for file '{filename}'. "
            f"Supported extensions are: .pdf, .docx, .txt"
        )


def parse_document_file(file_path: Union[str, Path]) -> ParsedDocument:
    """Convenience function to parse a file from the local filesystem."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    filename = p.name
    with open(p, "rb") as f:
        content_bytes = f.read()
    return parse_document_bytes(filename, content_bytes)

"""Data models and exception definitions for legal document ingestion and chunking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentType(str, Enum):
    """Supported document formats for legal document ingestion."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""
    pass


class UnsupportedFormatError(DocumentIngestionError):
    """Raised when an uploaded file extension or mime type is not supported."""
    pass


class EmptyDocumentError(DocumentIngestionError):
    """Raised when an uploaded document has zero pages, zero bytes, or only whitespace."""
    pass


class CorruptDocumentError(DocumentIngestionError):
    """Raised when a document cannot be parsed due to corrupt formatting or decryption failure."""
    pass


@dataclass(frozen=True)
class ParsedBlock:
    """Represents a coherent segment extracted from a document (e.g. a PDF page or DOCX paragraph).

    Attributes:
        text: Cleaned text content of the block.
        doc_name: Original filename of the document.
        doc_type: Ingestion document type ('pdf', 'docx', 'txt').
        page_number: 1-based page number for paginated documents (PDF).
                     Set to None for non-paginated documents (DOCX, TXT) to prevent
                     inventing fabricated page numbers.
        paragraph_index: 0-based index of the paragraph or block within the document.
        section_title: Title or heading of the surrounding legal section/clause if available.
        source_locator: Human-readable citation anchor, e.g. 'contract.pdf:page_3'
                        or 'agreement.docx:para_12 [Section 4: Indemnity]'.
        metadata: Additional diagnostic key-values (e.g. font styles, word count).
    """
    text: str
    doc_name: str
    doc_type: str
    page_number: Optional[int] = None
    paragraph_index: Optional[int] = None
    section_title: Optional[str] = None
    source_locator: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_locator:
            if self.page_number is not None:
                loc = f"{self.doc_name}:p{self.page_number}"
            elif self.paragraph_index is not None:
                loc = f"{self.doc_name}:para_{self.paragraph_index}"
            else:
                loc = self.doc_name
            if self.section_title:
                loc = f"{loc} [{self.section_title}]"
            object.__setattr__(self, "source_locator", loc)


@dataclass
class ParsedDocument:
    """Complete structured representation of an ingested document."""
    doc_name: str
    doc_type: str
    blocks: List[ParsedBlock] = field(default_factory=list)
    total_characters: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenated text of all parsed blocks."""
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @property
    def page_count(self) -> Optional[int]:
        """Total distinct pages if document is paginated, else None."""
        pages = {b.page_number for b in self.blocks if b.page_number is not None}
        return len(pages) if pages else None


@dataclass(frozen=True)
class DocumentChunk:
    """A retrievable text passage with exact provenance metadata for citation.

    Attributes:
        chunk_id: Stable identifier, formatted as '{doc_name}::p{page_no or "na"}::c{chunk_index}'.
        doc_name: Original filename.
        doc_type: Document format ('pdf', 'docx', 'txt').
        page_number: 1-based page number if available (PDF), else None. Never invented.
        section_title: Section or clause heading active when chunk was constructed.
        source_locator: Exact traceability citation string.
        text: The textual passage content.
        character_count: Total length in Unicode characters.
        token_count: Approximate token count, computed as len(text.split())
                     (standard whitespace word token proxy for LLM context estimation).
        chunk_index: 0-based sequential chunk index within the document.
        metadata: Extra attributes (e.g. overlap flags, parent block indices).
    """
    chunk_id: str
    doc_name: str
    doc_type: str
    page_number: Optional[int]
    section_title: Optional[str]
    source_locator: str
    text: str
    character_count: int
    token_count: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


"""Document Ingestion, Parsing, and Legal-Aware Chunking Module."""

from legal_uq.ingestion.models import (
    DocumentType,
    ParsedBlock,
    ParsedDocument,
    DocumentChunk,
    DocumentIngestionError,
    UnsupportedFormatError,
    EmptyDocumentError,
    CorruptDocumentError,
)
from legal_uq.ingestion.parser import parse_document_bytes, parse_document_file
from legal_uq.ingestion.chunker import LegalChunker

__all__ = [
    "DocumentType",
    "ParsedBlock",
    "ParsedDocument",
    "DocumentChunk",
    "DocumentIngestionError",
    "UnsupportedFormatError",
    "EmptyDocumentError",
    "CorruptDocumentError",
    "parse_document_bytes",
    "parse_document_file",
    "LegalChunker",
]


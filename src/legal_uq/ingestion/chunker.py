"""Legal-aware document chunker with clause and sentence boundary preservation.

Ensures strict citation traceability:
- Never blends text across PDF page boundaries so page citations remain 100% exact.
- Respects legal clause and sentence boundaries with hard fallback for unstructured text.
- Guaranteed loop termination on arbitrarily long paragraphs with no punctuation.
"""

import re
from typing import List, Optional

from legal_uq.ingestion.models import DocumentChunk, ParsedBlock, ParsedDocument

# Common legal clause patterns: (a), (i), 1.1, (1), ARTICLE, SECTION
_CLAUSE_BOUNDARY_REGEX = re.compile(
    r"(?=(?:^|\n)\s*(?:\([a-zA-Z0-9]+\)|[0-9]+(?:\.[0-9]+)+\.?|[A-Z][A-Za-z0-9\s–—\-]{2,50}:))",
    re.MULTILINE,
)

# Standard sentence split regex (lookbehind for terminal punctuation followed by whitespace)
_SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“(\[])")


class LegalChunker:
    """Configurable chunker tailored for legal documents and contracts.

    Attributes:
        target_chunk_size: Target maximum characters per chunk (default: 1,200).
        chunk_overlap: Overlap characters between consecutive chunks (default: 150).
        min_chunk_size: Minimum characters required to retain a trailing chunk (default: 80).
    """

    def __init__(
        self,
        target_chunk_size: int = 1200,
        chunk_overlap: int = 150,
        min_chunk_size: int = 80,
    ) -> None:
        if target_chunk_size <= 0:
            raise ValueError(f"target_chunk_size must be positive, got {target_chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= target_chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be non-negative and strictly less than target_chunk_size ({target_chunk_size})"
            )
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text by legal clause markers or sentence punctuation."""
        # First split by paragraphs
        paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        units: List[str] = []

        for p in paras:
            # Check for clause list markers within paragraph
            clause_parts = _CLAUSE_BOUNDARY_REGEX.split(p)
            for cp in clause_parts:
                cp_clean = cp.strip()
                if not cp_clean:
                    continue
                # Split clauses into sentences
                sentences = _SENTENCE_BOUNDARY_REGEX.split(cp_clean)
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        units.append(s_clean)
        return units

    def _hard_slice(self, text: str) -> List[str]:
        """Fallback slicer for huge sentences or unpunctuated runs to prevent infinite loops."""
        slices: List[str] = []
        step = max(1, self.target_chunk_size - self.chunk_overlap)
        pos = 0
        text_len = len(text)
        while pos < text_len:
            end = min(pos + self.target_chunk_size, text_len)
            sub = text[pos:end].strip()
            if sub:
                slices.append(sub)
            if end >= text_len:
                break
            pos += step
        return slices

    def chunk_block(self, block: ParsedBlock, start_chunk_idx: int = 0) -> List[DocumentChunk]:
        """Chunk a single parsed block while preserving its page number and section."""
        raw_text = block.text.strip()
        if not raw_text:
            return []

        # If block fits in one chunk, return immediately
        if len(raw_text) <= self.target_chunk_size:
            chunk_id = f"{block.doc_name}::p{block.page_number or 'na'}::c{start_chunk_idx}"
            tokens = len(raw_text.split())
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_name=block.doc_name,
                    doc_type=block.doc_type,
                    page_number=block.page_number,
                    section_title=block.section_title,
                    source_locator=block.source_locator,
                    text=raw_text,
                    character_count=len(raw_text),
                    token_count=tokens,
                    chunk_index=start_chunk_idx,
                    metadata=dict(block.metadata),
                )
            ]

        # Break into clause/sentence units
        units = self._split_into_sentences(raw_text)
        if not units:
            units = [raw_text]

        # Further subdivide any unit that alone exceeds target_chunk_size
        refined_units: List[str] = []
        for u in units:
            if len(u) > self.target_chunk_size:
                refined_units.extend(self._hard_slice(u))
            else:
                refined_units.append(u)

        chunks: List[DocumentChunk] = []
        chunk_counter = start_chunk_idx
        idx = 0
        n_units = len(refined_units)

        while idx < n_units:
            current_pieces: List[str] = []
            curr_len = 0
            start_unit_idx = idx

            while idx < n_units:
                unit = refined_units[idx]
                unit_len = len(unit)
                # If adding this unit exceeds target and we already have content, stop
                if current_pieces and (curr_len + unit_len + 1 > self.target_chunk_size):
                    break
                current_pieces.append(unit)
                curr_len += unit_len + 1
                idx += 1

            chunk_text = " ".join(current_pieces).strip()
            if chunk_text:
                chunk_id = f"{block.doc_name}::p{block.page_number or 'na'}::c{chunk_counter}"
                tokens = len(chunk_text.split())
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_name=block.doc_name,
                        doc_type=block.doc_type,
                        page_number=block.page_number,
                        section_title=block.section_title,
                        source_locator=block.source_locator,
                        text=chunk_text,
                        character_count=len(chunk_text),
                        token_count=tokens,
                        chunk_index=chunk_counter,
                        metadata=dict(block.metadata),
                    )
                )
                chunk_counter += 1

            if idx >= n_units:
                break

            # Calculate overlap: rewind idx to include overlap characters
            if self.chunk_overlap > 0:
                overlap_accum = 0
                rewind_idx = idx
                while rewind_idx > start_unit_idx:
                    prev_unit_len = len(refined_units[rewind_idx - 1])
                    if overlap_accum + prev_unit_len > self.chunk_overlap:
                        break
                    overlap_accum += prev_unit_len
                    rewind_idx -= 1
                # Ensure guaranteed forward progress (at least 1 unit advance)
                if rewind_idx == start_unit_idx:
                    idx = start_unit_idx + 1
                else:
                    idx = rewind_idx
            else:
                # No overlap: next chunk starts where current ended
                pass

        return chunks

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Convert an entire ParsedDocument into an ordered list of DocumentChunks.

        Crucial Citation Traceability Rule:
        PDF pages are processed independently so a chunk NEVER merges text from
        Page N and Page N+1 into a single chunk. Citations remain 100% faithful.
        """
        all_chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # Group blocks by page if paginated (PDF), or process sequentially
        for block in doc.blocks:
            block_chunks = self.chunk_block(block, start_chunk_idx=chunk_idx)
            all_chunks.extend(block_chunks)
            chunk_idx += len(block_chunks)

        return all_chunks


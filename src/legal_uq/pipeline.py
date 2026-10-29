"""End-to-end orchestration pipeline for Legal UQ.

Integrates:
1. Ingestion: Document parsing (PDF, DOCX, TXT) and legal-aware chunking.
2. Retrieval: FAISS cosine similarity vector store with exact citation preservation.
3. Generation: Grounded multi-candidate response generation (Live Gemini or Offline Mock).
4. Uncertainty Estimation: Lexical, semantic, and claim-level alignment aggregation.
5. Guardrails: Abstention and warning triage engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from legal_uq.generation.generator import (
    CandidateResponse,
    GenerationRun,
    GroundedGenerator,
    LEGAL_DISCLAIMER,
)
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionDecision, AbstentionEngine
from legal_uq.ingestion.chunker import LegalChunker
from legal_uq.ingestion.models import DocumentChunk, ParsedDocument
from legal_uq.ingestion.parser import parse_document_bytes, parse_document_file
from legal_uq.retrieval.vector_store import (
    CachedSentenceTransformerEmbedder,
    DeterministicFakeEmbedder,
    Embedder,
    RetrievalResult,
    VectorStore,
)
from legal_uq.uncertainty.aggregator import UncertaintyAggregator, UncertaintyMetrics
from legal_uq.uncertainty.claim_aligner import ClaimAlignmentResult
from legal_uq.uncertainty.similarity import SimilarityMatrix


@dataclass
class DocumentSummary:
    """Summary record for an indexed legal document."""
    doc_name: str
    doc_type: str
    num_blocks: int
    num_chunks: int
    total_characters: int


@dataclass
class PipelineQueryResult:
    """Consolidated outcome of an end-to-end legal QA query."""
    question: str
    retrieval_results: List[RetrievalResult]
    candidate_responses: List[CandidateResponse]
    primary_answer: CandidateResponse
    similarity_matrix: Optional[SimilarityMatrix]
    claim_alignment: Optional[ClaimAlignmentResult]
    uncertainty_metrics: UncertaintyMetrics
    abstention_decision: AbstentionDecision
    generation_mode: str
    duration_seconds: float
    disclaimer: str = LEGAL_DISCLAIMER
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer_allowed(self) -> bool:
        return self.abstention_decision.answer_allowed

    @property
    def decision_level(self) -> str:
        return self.abstention_decision.decision_level

    @property
    def confidence_score(self) -> float:
        return self.uncertainty_metrics.confidence_score

    @property
    def uncertainty_score(self) -> float:
        return self.uncertainty_metrics.uncertainty_score


class LegalUQPipeline:
    """High-level facade orchestrating all stages of the legal QA & uncertainty pipeline."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        default_abstention_config: Optional[AbstentionConfig] = None,
        target_chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> None:
        """Initialize the pipeline.

        If embedder is not supplied, attempts to load CachedSentenceTransformerEmbedder.
        If the cached model is unavailable, gracefully falls back to DeterministicFakeEmbedder.
        """
        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                self.embedder = CachedSentenceTransformerEmbedder()
            except Exception:
                self.embedder = DeterministicFakeEmbedder(dimension=384)

        self.vector_store = VectorStore(embedder=self.embedder)
        self.chunker = LegalChunker(
            target_chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.generator = GroundedGenerator(api_key=api_key, model_name=model_name)
        self.aggregator = UncertaintyAggregator(embedder=self.embedder)
        self.default_abstention_config = default_abstention_config or AbstentionConfig()
        self.abstention_engine = AbstentionEngine(config=self.default_abstention_config)

        self._indexed_docs: Dict[str, DocumentSummary] = {}
        self._all_chunks: List[DocumentChunk] = []

    @property
    def indexed_documents(self) -> List[DocumentSummary]:
        """List of currently indexed document summaries."""
        return list(self._indexed_docs.values())

    @property
    def total_chunk_count(self) -> int:
        """Total chunks in the vector index."""
        return len(self._all_chunks)

    def set_api_key(
        self,
        api_key: Optional[str],
        model_name: Optional[str] = None,
        force_mock: bool = False,
        max_concurrency: Optional[int] = None,
    ) -> None:
        """Update the LLM generator credentials dynamically."""
        self.generator = GroundedGenerator(
            api_key=api_key,
            model_name=model_name,
            force_mock=force_mock,
            require_live=not force_mock,
            max_concurrency=max_concurrency,
        )

    def update_abstention_config(self, config: AbstentionConfig) -> None:
        """Update abstention guardrail policy."""
        self.default_abstention_config = config
        self.abstention_engine = AbstentionEngine(config=config)

    def clear(self) -> None:
        """Reset the vector store and clear all indexed document data."""
        self.vector_store.clear()
        self._indexed_docs.clear()
        self._all_chunks.clear()

    def ingest_bytes(self, filename: str, file_bytes: bytes) -> DocumentSummary:
        """Parse raw bytes, chunk the document, and add it to the vector index.

        Args:
            filename: Original filename with valid extension (.pdf, .docx, .txt).
            file_bytes: Binary payload of the document.

        Returns:
            DocumentSummary record.
        """
        parsed_doc = parse_document_bytes(filename, file_bytes)
        chunks = self.chunker.chunk_document(parsed_doc)

        # Remove existing chunks for this filename if re-uploading
        self._all_chunks = [c for c in self._all_chunks if c.doc_name != parsed_doc.doc_name]
        self._all_chunks.extend(chunks)

        # Rebuild vector store with consolidated chunk collection
        self.vector_store.build_index(self._all_chunks)

        total_chars = sum(len(b.text) for b in parsed_doc.blocks)
        summary = DocumentSummary(
            doc_name=parsed_doc.doc_name,
            doc_type=parsed_doc.doc_type,
            num_blocks=len(parsed_doc.blocks),
            num_chunks=len(chunks),
            total_characters=total_chars,
        )
        self._indexed_docs[parsed_doc.doc_name] = summary
        return summary

    def ingest_file(self, file_path: str | Path) -> DocumentSummary:
        """Parse and index a document from a local filesystem path."""
        path = Path(file_path)
        with open(path, "rb") as f:
            data = f.read()
        return self.ingest_bytes(path.name, data)

    def query(
        self,
        question: str,
        top_k: int = 4,
        custom_config: Optional[AbstentionConfig] = None,
        simulate_contradiction: bool = False,
    ) -> PipelineQueryResult:
        """Execute the end-to-end QA, uncertainty evaluation, and guardrail check.

        Args:
            question: User's legal question.
            top_k: Number of relevant evidence chunks to retrieve.
            custom_config: Optional override of the default AbstentionConfig.
            simulate_contradiction: If True, introduces conflicting answers across candidates
                for testing and interactive demonstration of the contradiction guardrail.

        Returns:
            Consolidated PipelineQueryResult.
        """
        t0 = time.perf_counter()
        engine = AbstentionEngine(config=custom_config) if custom_config else self.abstention_engine

        # Handle empty index
        if not self._all_chunks or self.vector_store.total_chunks == 0:
            mode = "LIVE" if self.generator.is_live else "MOCK"
            empty_cand = CandidateResponse(
                candidate_index=0,
                generation_mode=mode,
                answer_text="No legal documents are currently indexed. Please upload at least one document.",
                cited_sources=[],
            )
            empty_metrics = self.aggregator.aggregate([empty_cand], evidence=[])
            decision = engine.evaluate(empty_metrics)
            return PipelineQueryResult(
                question=question,
                retrieval_results=[],
                candidate_responses=[empty_cand],
                primary_answer=empty_cand,
                similarity_matrix=None,
                claim_alignment=None,
                uncertainty_metrics=empty_metrics,
                abstention_decision=decision,
                generation_mode=mode,
                duration_seconds=time.perf_counter() - t0,
            )

        # 1. Retrieve evidence
        retrieval_results = self.vector_store.retrieve(question, k=top_k)

        # 2. Generate candidate answers
        gen_run = self.generator.generate_candidates(question, retrieval_results)
        candidates = list(gen_run.candidates)

        # Optional contradiction injection for interactive demonstration
        if simulate_contradiction and len(candidates) >= 2:
            c0_words = [w for w in re.findall(r"\b\w+\b", candidates[0].answer_text) if len(w) > 3]
            anchor_words = " ".join(c0_words[:4]) if c0_words else "terms and obligations"
            # Ensure candidate 0 has explicit positive polarity and a duration
            candidates[0] = CandidateResponse(
                candidate_index=0,
                generation_mode=candidates[0].generation_mode,
                answer_text=f"{candidates[0].answer_text} This provision is valid and enforceable, and parties shall comply within 30 days.",
                cited_sources=candidates[0].cited_sources,
                metadata={"polarity_anchor": "positive"},
            )
            # Candidate 1 asserts direct opposing polarity and conflicting duration
            candidates[1] = CandidateResponse(
                candidate_index=1,
                generation_mode=candidates[1].generation_mode,
                answer_text=(
                    f"Contrary to peer interpretation regarding {anchor_words}, this provision is void and unenforceable. "
                    f"Such actions are prohibited and parties shall not comply, requiring 990 days advance notice."
                ),
                cited_sources=candidates[1].cited_sources,
                metadata={"injected_contradiction": True},
            )

        # 3. Aggregate uncertainty and agreement metrics
        metrics = self.aggregator.aggregate(candidates, evidence=retrieval_results, query=question)

        # 4. Evaluate guardrails
        decision = engine.evaluate(metrics)

        # 5. Select primary answer
        primary_candidate = candidates[0]

        duration = time.perf_counter() - t0

        return PipelineQueryResult(
            question=question,
            retrieval_results=retrieval_results,
            candidate_responses=candidates,
            primary_answer=primary_candidate,
            similarity_matrix=metrics.similarity_matrix,
            claim_alignment=metrics.claim_alignment,
            uncertainty_metrics=metrics,
            abstention_decision=decision,
            generation_mode=gen_run.mode,
            duration_seconds=duration,
        )

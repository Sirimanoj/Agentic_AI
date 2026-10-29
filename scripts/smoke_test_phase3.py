"""Phase 3 demonstration: Candidate consensus, disagreement, and abstention guardrails."""

from legal_uq.generation.generator import CandidateResponse
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionEngine
from legal_uq.ingestion.models import DocumentChunk
from legal_uq.retrieval.vector_store import DeterministicFakeEmbedder, RetrievalResult
from legal_uq.uncertainty.aggregator import UncertaintyAggregator


def make_chunk_result(locator: str, text: str, score: float = 0.82) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chk_{abs(hash(locator)) % 1000}",
        doc_name="commercial_lease.pdf",
        doc_type="pdf",
        page_number=3,
        section_title="ARTICLE 5: TERMINATION AND NOTICE",
        source_locator=locator,
        text=text,
        character_count=len(text),
        token_count=len(text.split()),
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=1)


def main() -> None:
    embedder = DeterministicFakeEmbedder(dimension=128)
    aggregator = UncertaintyAggregator(embedder=embedder)
    engine = AbstentionEngine(AbstentionConfig(block_on_contradiction=True))

    ev = make_chunk_result(
        "commercial_lease.pdf:page_3 [ARTICLE 5]",
        "Either party may terminate this Lease upon giving not less than sixty (60) days advance written notice.",
    )

    print("=================================================================")
    print("SCENARIO 1: CANDIDATE CONTRADICTION -> GUARDRAIL HARD ABSTENTION")
    print("=================================================================")
    # Two candidates conflicting on notice period
    c0 = CandidateResponse(
        candidate_index=0,
        generation_mode="LIVE",
        answer_text="Based on [S1], either party may terminate the lease upon providing 30 days written notice.",
        cited_sources=["commercial_lease.pdf:page_3"],
    )
    c1 = CandidateResponse(
        candidate_index=1,
        generation_mode="LIVE",
        answer_text="Under [S1], the lease agreement strictly requires 60 days advance notice to effectuate termination.",
        cited_sources=["commercial_lease.pdf:page_3"],
    )
    c2 = CandidateResponse(
        candidate_index=2,
        generation_mode="LIVE",
        answer_text="Article 5 specifies that termination notice of 90 days must be served to the other party.",
        cited_sources=["commercial_lease.pdf:page_3"],
    )

    metrics_conflicted = aggregator.aggregate([c0, c1, c2], evidence=[ev])
    decision_conflicted = engine.evaluate(metrics_conflicted)

    print(f"Confidence Score    : {metrics_conflicted.confidence_score:.4f} (Penalized due to contradiction)")
    print(f"Uncertainty Score   : {metrics_conflicted.uncertainty_score:.4f}")
    print(f"Contradiction Found : {metrics_conflicted.contradiction_detected}")
    print(f"Reason Codes        : {metrics_conflicted.reason_codes}")
    print(f"Guardrail Decision  : {decision_conflicted.decision_level}")
    print(f"Answer Permitted    : {decision_conflicted.answer_allowed}")
    print(f"User-Facing Notice  :\n  \"{decision_conflicted.user_explanation}\"")

    print("\n=================================================================")
    print("SCENARIO 2: WELL-GROUNDED CONSENSUS -> GUARDRAIL PERMISSION")
    print("=================================================================")
    c_agree_0 = CandidateResponse(
        candidate_index=0,
        generation_mode="LIVE",
        answer_text="According to [S1], termination requires not less than 60 days advance written notice by either party.",
        cited_sources=["commercial_lease.pdf:page_3"],
    )
    c_agree_1 = CandidateResponse(
        candidate_index=1,
        generation_mode="LIVE",
        answer_text="Under Article 5 [S1], either party can terminate the lease by delivering sixty (60) days advance notice.",
        cited_sources=["commercial_lease.pdf:page_3"],
    )
    c_agree_2 = CandidateResponse(
        candidate_index=2,
        generation_mode="LIVE",
        answer_text="The commercial lease specifies a mandatory 60 days written notice period for either party seeking termination [S1].",
        cited_sources=["commercial_lease.pdf:page_3"],
    )

    metrics_agree = aggregator.aggregate([c_agree_0, c_agree_1, c_agree_2], evidence=[ev])
    decision_agree = engine.evaluate(metrics_agree)

    print(f"Confidence Score    : {metrics_agree.confidence_score:.4f}")
    print(f"Uncertainty Score   : {metrics_agree.uncertainty_score:.4f}")
    print(f"Support Ratio       : {metrics_agree.support_ratio * 100:.1f}%")
    print(f"Guardrail Decision  : {decision_agree.decision_level}")
    print(f"Answer Permitted    : {decision_agree.answer_allowed}")
    print(f"User-Facing Notice  :\n  \"{decision_agree.user_explanation}\"")


if __name__ == "__main__":
    main()


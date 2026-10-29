"""Deterministic unit tests for uncertainty estimation, claim alignment, and abstention guardrails."""

import pytest

from legal_uq.generation.generator import CandidateResponse
from legal_uq.guardrails.abstention import (
    AbstentionConfig,
    AbstentionDecision,
    AbstentionEngine,
)
from legal_uq.ingestion.models import DocumentChunk
from legal_uq.retrieval.vector_store import (
    DeterministicFakeEmbedder,
    RetrievalResult,
)
from legal_uq.uncertainty.aggregator import UncertaintyAggregator, UncertaintyMetrics
from legal_uq.uncertainty.claim_aligner import (
    ClaimAlignmentResult,
    ClaimVerdict,
    HeuristicClaimAligner,
)
from legal_uq.uncertainty.similarity import (
    compute_candidate_similarities,
    jaccard_similarity,
    rouge_l_similarity,
    semantic_similarity,
)


def make_cand(idx: int, text: str) -> CandidateResponse:
    return CandidateResponse(
        candidate_index=idx,
        generation_mode="MOCK",
        answer_text=text,
        cited_sources=[],
    )


def make_ev(locator: str, text: str, score: float = 0.85) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chk_{abs(hash(locator)) % 1000}",
        doc_name="contract.pdf",
        doc_type="pdf",
        page_number=1,
        section_title="TERMS",
        source_locator=locator,
        text=text,
        character_count=len(text),
        token_count=len(text.split()),
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=1)


@pytest.fixture
def fake_embedder() -> DeterministicFakeEmbedder:
    return DeterministicFakeEmbedder(dimension=128)


# ============================================================================
# Similarity Tests
# ============================================================================


def test_identical_vs_dissimilar_answers(fake_embedder: DeterministicFakeEmbedder) -> None:
    text_a = "All agreements are contracts if they are made by free consent of competent parties."
    text_b = "All agreements are contracts if they are made by free consent of competent parties."
    text_c = "Apples oranges bananas grapes watermelon pineapple fruits."

    # Identical
    assert jaccard_similarity(text_a, text_b) == 1.0
    assert rouge_l_similarity(text_a, text_b) == 1.0
    assert pytest.approx(semantic_similarity(text_a, text_b, fake_embedder), abs=1e-3) == 1.0

    # Completely dissimilar
    assert jaccard_similarity(text_a, text_c) == 0.0
    assert rouge_l_similarity(text_a, text_c) == 0.0


def test_similarity_matrix_computation(fake_embedder: DeterministicFakeEmbedder) -> None:
    c0 = make_cand(0, "The supplier shall deliver software within 30 days.")
    c1 = make_cand(1, "The supplier must deliver software within 30 days.")
    c2 = make_cand(2, "Court hearings are postponed indefinitely due to weather.")

    matrix = compute_candidate_similarities([c0, c1, c2], embedder=fake_embedder)
    assert matrix.num_candidates == 3
    assert len(matrix.pairs) == 3

    # Pair (0, 1) should have much higher agreement than (0, 2)
    pair_01 = next(p for p in matrix.pairs if p.idx1 == 0 and p.idx2 == 1)
    pair_02 = next(p for p in matrix.pairs if p.idx1 == 0 and p.idx2 == 2)

    assert pair_01.composite > pair_02.composite
    assert 0.0 <= matrix.mean_pairwise_agreement <= 1.0


# ============================================================================
# Claim Alignment & Contradiction Tests
# ============================================================================


def test_opposing_negation_contradiction_detection() -> None:
    c0 = make_cand(0, "The receiving party shall disclose confidential information to any requesting third party.")
    c1 = make_cand(1, "The receiving party shall not disclose confidential information to any requesting third party.")

    aligner = HeuristicClaimAligner()
    res = aligner.align_and_verify([c0, c1], evidence=[])

    assert res.contradiction_detected is True
    assert any("opposing legal polarity" in d.lower() for d in res.contradiction_details)


def test_conflicting_quantities_detection() -> None:
    c0 = make_cand(0, "The tenant must pay rent within 30 days after receiving the landlord invoice.")
    c1 = make_cand(1, "The tenant must pay rent within 90 days after receiving the landlord invoice.")

    aligner = HeuristicClaimAligner()
    res = aligner.align_and_verify([c0, c1], evidence=[])

    assert res.contradiction_detected is True
    assert any("incompatible quantities" in d.lower() for d in res.contradiction_details)
    assert any("30 days" in d and "90 days" in d for d in res.contradiction_details)


def test_unsupported_claims_detection() -> None:
    c0 = make_cand(0, "The server architecture relies on quantum computing and satellite telemetry.")
    ev = make_ev("lease.txt", "The monthly rent is 10,000 rupees payable on the first of each month.")

    aligner = HeuristicClaimAligner()
    res = aligner.align_and_verify([c0], evidence=[ev])

    assert res.supported_claims == 0
    assert res.unsupported_claims > 0
    assert res.support_ratio == 0.0
    assert len(res.unsupported_claim_texts) > 0


# ============================================================================
# Aggregator & Grounding Tests
# ============================================================================


def test_agreement_cannot_override_missing_evidence(fake_embedder: DeterministicFakeEmbedder) -> None:
    """Critical property: Identical candidate answers with zero evidence must produce 0 confidence."""
    # 3 candidates agree 100% on identical text
    identical_text = "The vendor provides a lifetime warranty on all manufactured parts."
    candidates = [make_cand(0, identical_text), make_cand(1, identical_text), make_cand(2, identical_text)]

    aggregator = UncertaintyAggregator(embedder=fake_embedder)
    metrics = aggregator.aggregate(candidates, evidence=[], similarity_matrix=None)

    assert metrics.confidence_score == 0.0
    assert metrics.uncertainty_score == 1.0
    assert "NO_RETRIEVED_EVIDENCE" in metrics.reason_codes
    assert "CONFIDENCE_ZEROED_NO_GROUNDING" in metrics.reason_codes

    # Abstention guardrail must trigger hard abstention
    engine = AbstentionEngine()
    decision = engine.evaluate(metrics)
    assert decision.answer_allowed is False
    assert decision.decision_level == "ABSTAIN"


def test_well_grounded_candidates_produce_high_confidence(fake_embedder: DeterministicFakeEmbedder) -> None:
    text = "Section 10 requires free consent of competent parties for a lawful object."
    ev = make_ev("contract_act.pdf:p2", text, score=0.88)
    candidates = [
        make_cand(0, f"Based on [S1], {text}"),
        make_cand(1, f"Under the Indian Contract Act, {text}"),
    ]

    sim_mat = compute_candidate_similarities(candidates, embedder=fake_embedder)
    aggregator = UncertaintyAggregator(embedder=fake_embedder)
    metrics = aggregator.aggregate(candidates, evidence=[ev], similarity_matrix=sim_mat)

    assert metrics.confidence_score > 0.45
    assert metrics.uncertainty_score < 0.55
    assert metrics.support_ratio > 0.50
    assert metrics.contradiction_detected is False

    engine = AbstentionEngine()
    decision = engine.evaluate(metrics)
    assert decision.answer_allowed is True


def test_edge_cases_zero_and_one_candidate(fake_embedder: DeterministicFakeEmbedder) -> None:
    aggregator = UncertaintyAggregator(embedder=fake_embedder)

    # 0 candidates
    m0 = aggregator.aggregate([], evidence=[])
    assert m0.confidence_score == 0.0
    assert m0.uncertainty_score == 1.0
    assert "NO_CANDIDATES_GENERATED" in m0.reason_codes

    # 1 candidate
    ev = make_ev("doc.pdf", "Some legal evidence.")
    m1 = aggregator.aggregate([make_cand(0, "Some legal evidence.")], evidence=[ev])
    assert 0.0 <= m1.confidence_score <= 1.0
    assert "SINGLE_CANDIDATE_RUN" in m1.reason_codes


# ============================================================================
# Abstention Guardrail Decision Tests
# ============================================================================


def test_abstention_on_low_retrieval_relevance(fake_embedder: DeterministicFakeEmbedder) -> None:
    # Retrieved evidence has very low relevance (score = 0.15)
    ev_weak = make_ev("tangential.txt", "Weather report for New Delhi.", score=0.15)
    cand = [make_cand(0, "Weather report in the city.")]

    aggregator = UncertaintyAggregator(embedder=fake_embedder)
    metrics = aggregator.aggregate(cand, evidence=[ev_weak])

    engine = AbstentionEngine(AbstentionConfig(min_evidence_relevance=0.30, min_confidence_to_answer=0.25))
    decision = engine.evaluate(metrics)

    assert decision.warning_required is True
    assert any("Evidence relevance" in r for r in decision.reasons)


def test_abstention_on_severe_contradiction(fake_embedder: DeterministicFakeEmbedder) -> None:
    c0 = make_cand(0, "Payment is due in 30 days from date of invoice.")
    c1 = make_cand(1, "Payment is due in 90 days from date of invoice.")
    ev = make_ev("contract.pdf", "Payment shall be rendered in accordance with invoice.")

    aggregator = UncertaintyAggregator(embedder=fake_embedder)
    metrics = aggregator.aggregate([c0, c1], evidence=[ev])

    # With block_on_contradiction = True, direct contradiction triggers hard abstention
    engine = AbstentionEngine(AbstentionConfig(block_on_contradiction=True))
    decision = engine.evaluate(metrics)

    assert decision.answer_allowed is False
    assert decision.decision_level == "ABSTAIN"
    assert any("contradictions" in r.lower() for r in decision.reasons)


# ============================================================================
# Threshold Boundary Tests (Below, At, Above)
# ============================================================================


def _make_metric_fixture(
    conf: float = 0.65,
    unc: float = 0.35,
    r_rel: float = 0.50,
    support: float = 0.80,
    contradiction: bool = False,
    reasons: list | None = None,
) -> UncertaintyMetrics:
    return UncertaintyMetrics(
        confidence_score=conf,
        uncertainty_score=unc,
        s_top=conf,
        s_mean=conf,
        r_rel=r_rel,
        support_ratio=support,
        contradiction_detected=contradiction,
        reason_codes=reasons or [],
    )


def test_boundary_support_ratio_below_at_above() -> None:
    """Validate behavior immediately below (0.39), at (0.40), and above (0.41) min_support_ratio."""
    engine = AbstentionEngine(AbstentionConfig(min_support_ratio=0.40))

    # Below: 0.39 -> Mandatory hard abstention and answer suppression
    m_below = _make_metric_fixture(support=0.39)
    d_below = engine.evaluate(m_below)
    assert d_below.decision_level == "ABSTAIN"
    assert d_below.answer_allowed is False
    assert any("minimum required: 40.0%" in r for r in d_below.reasons)

    # At: 0.40 -> Meets threshold, answer allowed
    m_at = _make_metric_fixture(support=0.40)
    d_at = engine.evaluate(m_at)
    assert d_at.decision_level == "SAFE"
    assert d_at.answer_allowed is True

    # Above: 0.41 -> Well within boundary, answer allowed
    m_above = _make_metric_fixture(support=0.41)
    d_above = engine.evaluate(m_above)
    assert d_above.decision_level == "SAFE"
    assert d_above.answer_allowed is True


def test_boundary_confidence_cutoff_below_at_above() -> None:
    """Validate behavior immediately below (0.29), at (0.30), and above (0.31) min_confidence_to_answer."""
    engine = AbstentionEngine(AbstentionConfig(min_confidence_to_answer=0.30))

    # Below: 0.29 -> Hard ABSTAIN
    m_below = _make_metric_fixture(conf=0.29, unc=0.71)
    d_below = engine.evaluate(m_below)
    assert d_below.decision_level == "ABSTAIN"
    assert d_below.answer_allowed is False

    # At: 0.30 -> Answer allowed with WARNING (since conf < 0.55)
    m_at = _make_metric_fixture(conf=0.30, unc=0.70)
    d_at = engine.evaluate(m_at)
    assert d_at.decision_level == "WARNING"
    assert d_at.answer_allowed is True

    # Above: 0.31 -> Answer allowed with WARNING (since conf < 0.55)
    m_above = _make_metric_fixture(conf=0.31, unc=0.69)
    d_above = engine.evaluate(m_above)
    assert d_above.decision_level == "WARNING"
    assert d_above.answer_allowed is True


def test_boundary_confidence_warning_threshold() -> None:
    """Validate behavior immediately below (0.54), at (0.55), and above (0.56) confidence_warning_threshold."""
    engine = AbstentionEngine(AbstentionConfig(confidence_warning_threshold=0.55, max_uncertainty_threshold=0.55))

    # Below: 0.54 -> Triggers WARNING
    m_below = _make_metric_fixture(conf=0.54, unc=0.46)
    d_below = engine.evaluate(m_below)
    assert d_below.decision_level == "WARNING"
    assert any("Confidence is moderate (0.54)" in r for r in d_below.reasons)

    # At: 0.55 -> Meets threshold, SAFE
    m_at = _make_metric_fixture(conf=0.55, unc=0.45)
    d_at = engine.evaluate(m_at)
    assert d_at.decision_level == "SAFE"
    assert d_at.warning_required is False

    # Above: 0.56 -> SAFE
    m_above = _make_metric_fixture(conf=0.56, unc=0.44)
    d_above = engine.evaluate(m_above)
    assert d_above.decision_level == "SAFE"
    assert d_above.warning_required is False


def test_boundary_uncertainty_alert_threshold() -> None:
    """Validate behavior immediately below (0.54), at (0.55), and above (0.56) max_uncertainty_threshold."""
    engine = AbstentionEngine(AbstentionConfig(max_uncertainty_threshold=0.55, confidence_warning_threshold=0.50))

    # Below: 0.54 -> No uncertainty alert
    m_below = _make_metric_fixture(conf=0.60, unc=0.54)
    d_below = engine.evaluate(m_below)
    assert d_below.decision_level == "SAFE"

    # At: 0.55 -> No uncertainty alert
    m_at = _make_metric_fixture(conf=0.60, unc=0.55)
    d_at = engine.evaluate(m_at)
    assert d_at.decision_level == "SAFE"

    # Above: 0.56 -> Triggers WARNING
    m_above = _make_metric_fixture(conf=0.60, unc=0.56)
    d_above = engine.evaluate(m_above)
    assert d_above.decision_level == "WARNING"
    assert any("Estimated uncertainty (0.56) exceeds the alert limit" in r for r in d_above.reasons)


def test_zero_evidence_causes_hard_abstention() -> None:
    """Validate that zero evidence strictly causes hard abstention with zero speculation."""
    engine = AbstentionEngine()

    # Zero r_rel
    m_zero_r = _make_metric_fixture(r_rel=0.0)
    d_zero_r = engine.evaluate(m_zero_r)
    assert d_zero_r.decision_level == "ABSTAIN"
    assert d_zero_r.answer_allowed is False

    # Explicit NO_RETRIEVED_EVIDENCE reason code
    m_no_ev = _make_metric_fixture(reasons=["NO_RETRIEVED_EVIDENCE"])
    d_no_ev = engine.evaluate(m_no_ev)
    assert d_no_ev.decision_level == "ABSTAIN"
    assert d_no_ev.answer_allowed is False
    assert "The system has abstained" in d_no_ev.user_explanation


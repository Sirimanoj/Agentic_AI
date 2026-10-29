"""Deterministic regression tests for semantic answerability, missing-information detection, and abstention."""

from unittest.mock import MagicMock
import pytest

from legal_uq.generation.generator import (
    CandidateResponse,
    GenerationRun,
    GroundedGenerator,
    check_evidence_answerability,
    detect_missing_information,
)
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionDecision, AbstentionEngine
from legal_uq.ingestion.models import DocumentChunk
from legal_uq.pipeline import LegalUQPipeline
from legal_uq.retrieval.vector_store import DeterministicFakeEmbedder, RetrievalResult
from legal_uq.uncertainty.aggregator import UncertaintyAggregator, UncertaintyMetrics

VINEETA_SHARMA_EXCERPT = (
    "IN THE SUPREME COURT OF INDIA\n"
    "CIVIL APPEAL NO. 32601 OF 2018\n"
    "VINEETA SHARMA ... APPELLANT\n"
    "VERSUS\n"
    "RAKESH SHARMA & ORS. ... RESPONDENTS\n\n"
    "HELD: Section 6 of the Hindu Succession Act, 1956 confers coparcenary rights on a daughter born before or after "
    "the amendment of 2005. The daughter is conferred the status of coparcener in the same manner as a son, "
    "with the same rights and liabilities. The rights can be claimed by a daughter born prior to the amendment, "
    "even if the father had passed away before September 9, 2005. It is an unobstructed heritage by birth."
)


def _make_vineeta_evidence() -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id="vineeta_sharma::p1::c0",
        doc_name="vineeta_sharma_judgment.txt",
        doc_type="txt",
        page_number=1,
        section_title="CIVIL APPEAL NO. 32601 OF 2018 - VINEETA SHARMA",
        source_locator="vineeta_sharma_judgment.txt:para_1 [JUDGMENT]",
        text=VINEETA_SHARMA_EXCERPT,
        character_count=len(VINEETA_SHARMA_EXCERPT),
        token_count=len(VINEETA_SHARMA_EXCERPT.split()),
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=0.88, rank=1)


# =============================================================================
# SCENARIO A: Answer Present in the Judgment
# =============================================================================
def test_scenario_a_answer_present_in_judgment():
    """Verify that when the requested legal fact is present in the document, it is answered with high confidence."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    pipeline.ingest_bytes("vineeta_sharma_judgment.txt", VINEETA_SHARMA_EXCERPT.encode("utf-8"))

    # Question is directly addressed by Section 6 of the Hindu Succession Act in the judgment
    res = pipeline.query("Does Section 6 of the Hindu Succession Act confer coparcenary rights on a daughter?")

    assert res.uncertainty_metrics.is_answerable
    assert not res.uncertainty_metrics.missing_information_detected
    assert res.decision_level in ("SAFE", "WARNING")
    assert res.answer_allowed
    assert "vineeta_sharma_judgment.txt" in res.primary_answer.cited_sources[0]


# =============================================================================
# SCENARIO B: Trademark Fee Absent from the Judgment (Offline & Live Modes)
# =============================================================================
def test_scenario_b_trademark_fee_absent_offline_demo_mode():
    """Verify that in Offline Demo Mode, an unanswerable query produces a clear refusal and strictly ABSTAINS."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    pipeline.ingest_bytes("vineeta_sharma_judgment.txt", VINEETA_SHARMA_EXCERPT.encode("utf-8"))

    # Query for trademark registration fee (completely absent from Hindu Succession inheritance judgment)
    res = pipeline.query("What is the current government trademark registration fee?")

    # 1. Must NOT be SAFE!
    assert res.decision_level == "ABSTAIN", f"Expected ABSTAIN, got: {res.decision_level}"
    assert not res.answer_allowed, "Substantive answer must not be allowed for absent legal fact!"

    # 2. Must detect missing information
    assert not res.uncertainty_metrics.is_answerable
    assert res.uncertainty_metrics.missing_information_detected
    assert "REQUESTED_FACT_ABSENT_FROM_EVIDENCE" in res.uncertainty_metrics.reason_codes

    # 3. Primary answer in mock mode must be a clear refusal, NOT an irrelevant quoted passage
    answer_text = res.primary_answer.answer_text
    assert "does not contain" in answer_text.lower() or "not addressed" in answer_text.lower() or "no provisions" in answer_text.lower()
    assert res.primary_answer.generation_mode == "MOCK"


def test_scenario_b_trademark_fee_absent_live_gemini_mode():
    """Verify that in Live Mode, when Gemini reports missing information, the pipeline strictly ABSTAINS."""
    mock_client = MagicMock()
    del mock_client.interactions
    gemini_refusal_text = (
        "The provided judgment in Vineeta Sharma v. Rakesh Sharma concerns inheritance rights and coparcenary "
        "provisions under Section 6 of the Hindu Succession Act, 1956 [S1]. It contains no information regarding "
        "government trademark registration fees or intellectual property filings."
    )

    mock_resp = MagicMock()
    mock_resp.text = gemini_refusal_text
    mock_client.models.generate_content.return_value = mock_resp

    embedder = DeterministicFakeEmbedder(dimension=64)
    generator = GroundedGenerator(api_key="fake_key_for_test", client=mock_client)
    pipeline = LegalUQPipeline(embedder=embedder)
    pipeline.generator = generator

    pipeline.ingest_bytes("vineeta_sharma_judgment.txt", VINEETA_SHARMA_EXCERPT.encode("utf-8"))

    res = pipeline.query("What is the current government trademark registration fee?")

    # Must strictly ABSTAIN, NEVER SAFE!
    assert res.decision_level == "ABSTAIN", f"Expected ABSTAIN, got {res.decision_level}"
    assert not res.answer_allowed, "Substantive answer must be withheld when fact is absent!"
    assert res.uncertainty_metrics.missing_information_detected
    assert not res.uncertainty_metrics.is_answerable
    assert "REQUESTED_FACT_ABSENT_FROM_EVIDENCE" in res.uncertainty_metrics.reason_codes

    # Primary answer text retains the helpful missing-information explanation
    assert "Vineeta Sharma" in res.primary_answer.answer_text
    assert "contains no information" in res.primary_answer.answer_text


# =============================================================================
# SCENARIO C: Highly Agreeing Candidates Lacking Evidence Cannot Receive SAFE
# =============================================================================
def test_scenario_c_agreeing_candidates_on_absent_fact_cannot_receive_safe():
    """Verify that even when 3 candidates 100% agree that information is absent, the verdict is ABSTAIN, not SAFE."""
    c0 = CandidateResponse(
        candidate_index=0,
        generation_mode="LIVE",
        answer_text="The provided judgment does not contain information regarding trademark registration fees.",
        cited_sources=["vineeta_sharma:p1"],
        metadata={"missing_information_detected": True, "is_answerable": False},
    )
    c1 = CandidateResponse(
        candidate_index=1,
        generation_mode="LIVE",
        answer_text="The provided text does not contain any information regarding government trademark registration fees.",
        cited_sources=["vineeta_sharma:p1"],
        metadata={"missing_information_detected": True, "is_answerable": False},
    )
    c2 = CandidateResponse(
        candidate_index=2,
        generation_mode="LIVE",
        answer_text="The indexed document does not contain details regarding trademark registration fees.",
        cited_sources=["vineeta_sharma:p1"],
        metadata={"missing_information_detected": True, "is_answerable": False},
    )

    ev = [_make_vineeta_evidence()]
    fake_embedder = DeterministicFakeEmbedder(dimension=64)
    aggregator = UncertaintyAggregator(embedder=fake_embedder)
    metrics = aggregator.aggregate(
        [c0, c1, c2],
        evidence=ev,
        query="What is the current government trademark registration fee?",
    )

    # Candidate consensus on refusal is high
    assert metrics.s_mean >= 0.70, f"Expected high consensus on refusal, got: {metrics.s_mean}"

    # But document answerability is False and substantive confidence is zeroed!
    assert not metrics.is_answerable
    assert metrics.missing_information_detected
    assert metrics.confidence_score == 0.0

    # Guardrail evaluation must strictly ABSTAIN
    engine = AbstentionEngine()
    decision = engine.evaluate(metrics)

    assert decision.decision_level == "ABSTAIN", f"Expected ABSTAIN, got {decision.decision_level}"
    assert not decision.answer_allowed
    assert "INSUFFICIENT EVIDENCE" in decision.user_explanation


# =============================================================================
# SCENARIO D: Explicitly Detected Missing Information Patterns
# =============================================================================
def test_scenario_d_detect_missing_information_patterns():
    """Verify that explicit missing information phrasing is reliably detected."""
    # Positive missing-info detections
    assert detect_missing_information("The provided judgment contains no information regarding trademark fees.")
    assert detect_missing_information("Based on the provided excerpts, there is no information regarding government trademark registration fees.")
    assert detect_missing_information("The retrieved document does not mention the statutory filing fee.")
    assert detect_missing_information("The uploaded text does not provide registration fee details.")
    assert detect_missing_information("The available passages lack sufficient information to answer this inquiry.")
    assert detect_missing_information("There is no retrieved evidence addressing the fee structure.")
    assert detect_missing_information("No relevant clauses to substantiate an answer.")

    # Negative detections (actual substantive assertions)
    assert not detect_missing_information("The employee shall receive an annual base salary of $175,000 payable semi-monthly.")
    assert not detect_missing_information("Section 6 confers coparcenary rights on a daughter with the same liabilities as a son.")
    assert not detect_missing_information("The obligations of confidentiality shall survive for 3 years from disclosure.")


def test_scenario_d_check_evidence_answerability():
    """Verify topical keyword containment between query and evidence."""
    ev = [_make_vineeta_evidence()]

    # Relevant query (Section 6, daughter, coparcenary) -> True
    is_ans, overlap, missing = check_evidence_answerability(
        "Does Section 6 confer coparcenary rights on a daughter?",
        ev,
    )
    assert is_ans
    assert overlap >= 0.50

    # Unrelated query (trademark registration fee) -> False
    is_ans_tm, overlap_tm, missing_tm = check_evidence_answerability(
        "What is the current government trademark registration fee?",
        ev,
    )
    assert not is_ans_tm
    assert "trademark" in missing_tm
    assert "fee" in missing_tm


# =============================================================================
# SCENARIO E: Genuinely Empty Retrieval
# =============================================================================
def test_scenario_e_genuinely_empty_retrieval_abstains():
    """Verify that empty retrieval triggers hard abstention with zero relevance."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    # Empty index
    res = pipeline.query("What is the trademark registration fee?")

    assert res.decision_level == "ABSTAIN"
    assert not res.answer_allowed
    assert len(res.retrieval_results) == 0
    assert res.uncertainty_metrics.r_rel == 0.0
    assert "No legal documents are currently indexed" in res.primary_answer.answer_text

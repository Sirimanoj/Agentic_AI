"""Deterministic unit and integration tests covering the 7 Agentic AI scenarios:

Scenario 1: Question with strong supporting evidence -> SAFE/WARNING, high grounding.
Scenario 2: Question with weak or missing evidence -> Bounded retry, ABSTAIN.
Scenario 3: Question requiring retrieval refinement -> Bounded retry loop (2 attempts max).
Scenario 4: Generated answer containing unsupported claim -> Bounded revision loop (1 revision max).
Scenario 5: Retrieval tool failure -> Graceful handling, RETRIEVAL_TOOL_ERROR log, ABSTAIN, no crash.
Scenario 6: LLM/API failure -> Graceful handling, GENERATION_API_ERROR log, ABSTAIN, no crash.
Scenario 7: Offline Demo Mode without API credentials -> Deterministic, zero API calls, full trace.
"""

from unittest.mock import MagicMock
import pytest

from legal_uq.agents.agent_workflow import AgentWorkflowOrchestrator
from legal_uq.agents.models import StepStatus
from legal_uq.agents.retrieval_agent import RetrievalAgent
from legal_uq.agents.analysis_agent import LegalAnalysisAgent
from legal_uq.agents.verification_agent import VerificationAgent
from legal_uq.agents.decision_controller import DecisionController
from legal_uq.generation.generator import CandidateResponse, GroundedGenerator
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionEngine
from legal_uq.ingestion.models import DocumentChunk
from legal_uq.pipeline import LegalUQPipeline
from legal_uq.retrieval.vector_store import (
    DeterministicFakeEmbedder,
    RetrievalResult,
    VectorStore,
)
from legal_uq.uncertainty.aggregator import UncertaintyAggregator
from legal_uq.uncertainty.claim_aligner import HeuristicClaimAligner


def _build_test_chunk(doc_name: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_name}::p{page}::c0",
        doc_name=doc_name,
        doc_type="pdf",
        page_number=page,
        section_title="TEST CLAUSE",
        source_locator=f"{doc_name}:p{page} [SECTION]",
        text=text,
        character_count=len(text),
        token_count=len(text.split()),
        chunk_index=0,
    )


@pytest.fixture
def mock_pipeline_with_docs():
    """Create a fully offline LegalUQPipeline with sample contract clauses."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    pipeline.ingest_bytes(
        "sample_nda.txt",
        b"The confidentiality obligations under Section 4 shall survive for a period of three (3) years following termination. "
        b"Within fourteen (14) calendar days of written notice, Receiving Party shall return or destroy all confidential materials. "
        b"The parties submit to the exclusive jurisdiction of the Court of Chancery of the State of Delaware.",
    )
    return pipeline


# -----------------------------------------------------------------------------
# SCENARIO 1: Strong Supporting Evidence
# -----------------------------------------------------------------------------
def test_scenario_1_strong_supporting_evidence(mock_pipeline_with_docs):
    """Test scenario 1: Question with strong evidence yields plan, tools invoked, and verified decision."""
    pipeline = mock_pipeline_with_docs
    query = "What is the duration of confidentiality obligations following termination?"

    res = pipeline.agentic_query(query)

    # 1. Orchestrator produces execution plan with 6 steps
    assert res.plan is not None
    assert len(res.plan.steps) == 6
    assert res.plan.steps[0].assigned_agent == "RetrievalAgent"
    assert "Retrieval" in res.plan.steps[0].title
    assert res.plan.steps[0].status == StepStatus.COMPLETED

    # 2. Retrieval and analysis tools are actually invoked
    assert len(res.retrieval_history) >= 1
    primary_retrieval = res.retrieval_history[0]
    assert len(primary_retrieval.retrieved_results) > 0
    assert primary_retrieval.top_relevance_score > 0.35

    assert len(res.analysis_history) >= 1
    draft = res.analysis_history[0]
    assert len(draft.candidate_responses) == 3
    assert draft.primary_candidate is not None
    assert len(draft.primary_candidate.cited_sources) > 0

    # 3. Verification agent checks evidence support
    assert len(res.verification_history) >= 1
    verif = res.verification_history[0]
    assert verif.support_ratio > 0.0

    # 4. Decision controller returns safe/warning with answer allowed
    assert res.controller_decision is not None
    assert res.controller_decision.decision.decision_level in ("SAFE", "WARNING")
    assert res.controller_decision.decision.answer_allowed is True

    # 5. Agent action trace contains explicit logged actions
    action_names = [a.action for a in res.execution_trace]
    assert "PLAN_SYNTHESIS" in action_names
    assert "RETRIEVAL_PRIMARY" in action_names
    assert "GENERATION_DRAFT" in action_names
    assert "VERIFICATION_AUDIT" in action_names
    assert "GUARDRAIL_ARBITRATION" in action_names


# -----------------------------------------------------------------------------
# SCENARIO 2: Weak or Missing Evidence
# -----------------------------------------------------------------------------
def test_scenario_2_weak_or_missing_evidence(mock_pipeline_with_docs):
    """Test scenario 2: Question with missing evidence triggers low score and ABSTAIN."""
    pipeline = mock_pipeline_with_docs
    query = "What is the statutory filing fee for trademark registration in Delaware?"

    res = pipeline.agentic_query(query)

    # Decision controller must enforce ABSTAIN or INSUFFICIENT EVIDENCE
    assert res.controller_decision.decision.decision_level == "ABSTAIN"
    assert res.controller_decision.decision.answer_allowed is False

    # Execution trace records low evidence relevance or missing information
    metrics = res.controller_decision.uncertainty_metrics
    assert (
        metrics.support_ratio == 0.0
        or metrics.missing_information_detected is True
        or "LOW_EVIDENCE_RELEVANCE" in metrics.reason_codes
        or "NO_RETRIEVED_EVIDENCE" in metrics.reason_codes
    )


# -----------------------------------------------------------------------------
# SCENARIO 3: Retrieval Refinement (Bounded Retry Loop)
# -----------------------------------------------------------------------------
def test_scenario_3_retrieval_refinement_bounded_retry():
    """Test scenario 3: Query with low initial score triggers reformulation retry up to max 2 attempts."""
    mock_store = MagicMock()
    chunk_weak = _build_test_chunk("doc.pdf", "General preamble text.", page=1)
    chunk_strong = _build_test_chunk("doc.pdf", "Specific governing law clause in Section 6.", page=3)

    # Pass 1 returns weak score (< 0.35)
    result_weak = [RetrievalResult(chunk=chunk_weak, score=0.20, rank=1)]
    # Pass 2 returns improved score (>= 0.35)
    result_strong = [RetrievalResult(chunk=chunk_strong, score=0.85, rank=1)]

    mock_store.retrieve.side_effect = [result_weak, result_strong]

    agent = RetrievalAgent(vector_store=mock_store, min_sufficiency_score=0.35, max_attempts=2)
    output, logs = agent.retrieve_with_retry(
        "Please tell me under what conditions does governing law apply?"
    )

    # Exactly 2 attempts invoked (bounded)
    assert mock_store.retrieve.call_count == 2
    assert output.attempt_number == 2
    assert output.reformulation_applied is True
    assert output.top_relevance_score == 0.85
    assert output.is_sufficient is True

    # Logs record reformulation action
    action_types = [l.action for l in logs]
    assert "RETRIEVAL_PRIMARY" in action_types
    assert "RETRIEVAL_REFORMULATION_TRIGGERED" in action_types
    assert "RETRIEVAL_REFINED_EVALUATED" in action_types


def test_scenario_3_retrieval_retry_enforces_hard_cap():
    """Ensure RetrievalAgent never executes more than 2 attempts even if configured higher."""
    mock_store = MagicMock()
    chunk = _build_test_chunk("doc.pdf", "Arbitrary text.")
    weak_res = [RetrievalResult(chunk=chunk, score=0.10, rank=1)]
    mock_store.retrieve.return_value = weak_res

    # Attempt to initialize with max_attempts=10
    agent = RetrievalAgent(vector_store=mock_store, min_sufficiency_score=0.35, max_attempts=10)
    assert agent.max_attempts == 2

    output, _ = agent.retrieve_with_retry("sample query")
    assert mock_store.retrieve.call_count <= 2
    assert output.attempt_number <= 2


# -----------------------------------------------------------------------------
# SCENARIO 4: Unsupported Claim Triggers Bounded Revision Loop
# -----------------------------------------------------------------------------
def test_scenario_4_unsupported_claim_triggers_revision():
    """Test scenario 4: An ungrounded draft answer triggers 1 revision cycle and logs guidance."""
    aligner = HeuristicClaimAligner()
    v_agent = VerificationAgent(aligner=aligner, min_acceptable_support_ratio=0.60)

    chunk = _build_test_chunk("doc.pdf", "Confidentiality lasts for 3 years following termination.", page=2)
    evidence = [RetrievalResult(chunk=chunk, score=0.88, rank=1)]

    # Candidate with unsupported claim (e.g. inventing 10 years and criminal penalties)
    unsupported_candidate = CandidateResponse(
        candidate_index=0,
        generation_mode="MOCK",
        answer_text="Confidentiality lasts for 10 years, and unauthorized disclosure triggers mandatory criminal imprisonment.",
        cited_sources=["doc.pdf:p2 [SECTION]"],
    )

    verif_out, v_log = v_agent.verify_candidates(
        primary_candidate=unsupported_candidate,
        evidence=evidence,
        allow_revision_request=True,
    )

    # Verification must flag need for revision
    assert verif_out.needs_revision is True
    assert verif_out.support_ratio < 0.60
    assert verif_out.revision_guidance is not None

    # AnalysisAgent executes targeted revision
    generator = GroundedGenerator(force_mock=True)
    analysis_agent = LegalAnalysisAgent(generator=generator)
    from legal_uq.agents.models import AnalysisOutput

    initial_draft = AnalysisOutput(
        candidate_responses=[unsupported_candidate],
        primary_candidate=unsupported_candidate,
        generation_mode="MOCK",
        revision_count=0,
    )

    revised_draft, rev_log = analysis_agent.revise_draft(
        question="How long does confidentiality last?",
        evidence=evidence,
        current_draft=initial_draft,
        revision_guidance=verif_out.revision_guidance,
    )

    # Revision is performed and logged
    assert revised_draft.revision_count == 1
    assert rev_log.action == "GENERATION_REVISED"
    assert "REVISED" in revised_draft.primary_candidate.answer_text


# -----------------------------------------------------------------------------
# SCENARIO 5: Retrieval Tool Failure Resilience
# -----------------------------------------------------------------------------
def test_scenario_5_retrieval_tool_failure_gracefully_handled():
    """Test scenario 5: When the vector store throws an exception, workflow logs it and issues ABSTAIN without crashing."""
    mock_store = MagicMock()
    mock_store.total_chunks = 10
    mock_store.retrieve.side_effect = RuntimeError("FAISS internal C++ index read error: file truncated")

    embedder = DeterministicFakeEmbedder(dimension=64)
    aggregator = UncertaintyAggregator(embedder=embedder)
    generator = GroundedGenerator(force_mock=True)
    orchestrator = AgentWorkflowOrchestrator(
        vector_store=mock_store,
        generator=generator,
        aggregator=aggregator,
    )

    # Must NOT raise unhandled exception
    res = orchestrator.execute_workflow(question="What is the governing jurisdiction?")

    assert res is not None
    assert res.controller_decision.decision.decision_level == "ABSTAIN"
    assert res.controller_decision.decision.answer_allowed is False

    # Check that RETRIEVAL_TOOL_ERROR was logged
    actions = [a.action for a in res.execution_trace]
    assert "RETRIEVAL_TOOL_ERROR" in actions
    error_log = next(a for a in res.execution_trace if a.action == "RETRIEVAL_TOOL_ERROR")
    assert "FAISS" in error_log.details.get("error", "")


# -----------------------------------------------------------------------------
# SCENARIO 6: LLM / API Failure Resilience
# -----------------------------------------------------------------------------
def test_scenario_6_llm_api_failure_gracefully_handled():
    """Test scenario 6: When generation service raises API error, workflow logs it and issues ABSTAIN without crashing."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    store = VectorStore(embedder=embedder)
    chunk = _build_test_chunk("contract.txt", "Disputes are resolved under Delaware law.")
    store.build_index([chunk])

    # Create generator with mock that raises live API error
    mock_generator = GroundedGenerator(force_mock=True)
    mock_generator.generate_candidates = MagicMock(
        side_effect=RuntimeError("Google GenAI 503 Service Unavailable: overloaded backends")
    )

    aggregator = UncertaintyAggregator(embedder=embedder)
    orchestrator = AgentWorkflowOrchestrator(
        vector_store=store,
        generator=mock_generator,
        aggregator=aggregator,
    )

    # Must NOT raise unhandled exception
    res = orchestrator.execute_workflow(question="What is the governing jurisdiction?")

    assert res is not None
    assert res.controller_decision.decision.decision_level == "ABSTAIN"
    assert res.controller_decision.decision.answer_allowed is False

    # Check that GENERATION_API_ERROR was logged
    actions = [a.action for a in res.execution_trace]
    assert "GENERATION_API_ERROR" in actions
    error_log = next(a for a in res.execution_trace if a.action == "GENERATION_API_ERROR")
    assert "503" in error_log.details.get("error", "") or "RuntimeError" in error_log.reason


# -----------------------------------------------------------------------------
# SCENARIO 7: Offline Demo Mode without API Credentials
# -----------------------------------------------------------------------------
def test_scenario_7_offline_demo_mode_without_credentials():
    """Test scenario 7: Offline Demo Mode operates 100% locally with zero external network or credentials."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)
    pipeline.ingest_bytes(
        "test_policy.txt",
        b"Standard notice period is 30 days prior to written termination.",
    )

    # GroundedGenerator in mock mode
    pipeline.generator = GroundedGenerator(force_mock=True)

    res = pipeline.agentic_query("What is the standard notice period for termination?")

    assert res is not None
    assert res.plan is not None
    assert len(res.retrieval_history) >= 1
    assert len(res.analysis_history) >= 1
    assert res.analysis_history[0].generation_mode == "MOCK"
    assert len(res.analysis_history[0].candidate_responses) == 3
    assert len(res.verification_history) >= 1
    assert res.controller_decision is not None
    assert len(res.execution_trace) >= 5

    # Check that mock warning exists in disclaimer and no crash occurred
    assert "MOCK" in res.analysis_history[0].disclaimer.upper()

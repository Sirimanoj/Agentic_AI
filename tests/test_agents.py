"""Comprehensive unit and integration tests for the multi-agent orchestration architecture.

Validates:
1. PlanningAgent task decomposition and structured plan synthesis.
2. RetrievalAgent with bounded query reformulation retry (max 2 attempts).
3. LegalAnalysisAgent candidate generation and bounded revision (max 1 revision).
4. VerificationAgent claim audit, polarity conflict detection, and revision guidance.
5. DecisionController uncertainty aggregation and guardrail arbitration.
6. End-to-end AgentWorkflowOrchestrator execution trace and action logging.
"""

from unittest.mock import MagicMock
import pytest

from legal_uq.agents.analysis_agent import LegalAnalysisAgent
from legal_uq.agents.decision_controller import DecisionController
from legal_uq.agents.models import (
    AgentActionLog,
    AgenticWorkflowResult,
    AnalysisOutput,
    ExecutionPlan,
    PlanStep,
    RetrievalOutput,
    StepStatus,
    VerificationOutput,
)
from legal_uq.agents.orchestrator import PlanningAgent
from legal_uq.agents.retrieval_agent import RetrievalAgent, _refine_query_keywords
from legal_uq.agents.verification_agent import VerificationAgent
from legal_uq.agents.agent_workflow import (
    AgentWorkflowOrchestrator,
    convert_workflow_result_to_pipeline_result,
)
from legal_uq.generation.generator import CandidateResponse, GroundedGenerator
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionDecision, AbstentionEngine
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


# =============================================================================
# 1. Planning Agent Tests
# =============================================================================
def test_planning_agent_generates_structured_plan():
    planner = PlanningAgent()
    plan, log = planner.create_plan(
        question="What is the survival term of confidentiality?",
        total_chunks=10,
        is_live_mode=False,
    )

    assert isinstance(plan, ExecutionPlan)
    assert plan.question == "What is the survival term of confidentiality?"
    assert len(plan.steps) == 6
    assert plan.steps[0].assigned_agent == "RetrievalAgent"
    assert plan.steps[2].assigned_agent == "LegalAnalysisAgent"
    assert plan.steps[3].assigned_agent == "VerificationAgent"
    assert plan.steps[5].assigned_agent == "DecisionController"

    # Verify action logging
    assert isinstance(log, AgentActionLog)
    assert log.agent_name == "PlanningAgent"
    assert log.action == "PLAN_SYNTHESIS"
    assert "strategy" in log.details


def test_planning_agent_empty_index_strategy():
    planner = PlanningAgent()
    plan, log = planner.create_plan(
        question="What is the base salary?",
        total_chunks=0,
        is_live_mode=False,
    )

    assert len(plan.steps) == 2
    assert "Empty Index Strategy" in plan.strategy_rationale
    assert plan.steps[0].status == StepStatus.COMPLETED


# =============================================================================
# 2. Retrieval Agent Tests
# =============================================================================
def test_query_refinement_helper():
    q1 = "What is the duration of the confidentiality obligations under the NDA?"
    refined1 = _refine_query_keywords(q1)
    assert "duration" in refined1.lower()
    assert "confidentiality" in refined1.lower()
    assert not refined1.lower().startswith("what is the")

    q2 = "Please tell me the notice period required for termination"
    refined2 = _refine_query_keywords(q2)
    assert "notice period" in refined2.lower()
    assert not refined2.lower().startswith("please tell me")


def test_retrieval_agent_bounded_retry():
    embedder = DeterministicFakeEmbedder(dimension=64)
    vstore = VectorStore(embedder=embedder)

    chunk = _build_test_chunk(
        "employment.pdf",
        "Executive base compensation is $200,000 per year paid semi-monthly.",
    )
    vstore.build_index([chunk])

    # Case A: Sufficient query
    agent = RetrievalAgent(vector_store=vstore, min_sufficiency_score=0.10, max_attempts=2)
    out_a, logs_a = agent.retrieve_with_retry("compensation base executive", top_k=1)

    assert out_a.attempt_number == 1
    assert not out_a.reformulation_applied
    assert len(out_a.retrieved_results) == 1
    assert logs_a[0].action == "RETRIEVAL_PRIMARY"

    # Case B: Low initial score triggers attempt 2 (reformulation), capped at 2 attempts
    agent_strict = RetrievalAgent(vector_store=vstore, min_sufficiency_score=0.99, max_attempts=2)
    out_b, logs_b = agent_strict.retrieve_with_retry("What is the compensation?", top_k=1)

    # Must execute exactly 2 attempts, never more
    assert out_b.attempt_number == 2
    assert out_b.reformulation_applied
    assert any(l.action == "RETRIEVAL_REFORMULATION_TRIGGERED" for l in logs_b)
    assert any(l.action == "RETRIEVAL_REFINED_EVALUATED" for l in logs_b)


# =============================================================================
# 3. Legal Analysis Agent & Bounded Revision Tests
# =============================================================================
def test_legal_analysis_agent_draft_and_bounded_revision():
    gen = GroundedGenerator(force_mock=True)
    agent = LegalAnalysisAgent(generator=gen)

    chunk = _build_test_chunk(
        "nda.pdf",
        "Confidential information shall be kept secret for 3 years following termination.",
    )
    ev = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

    # Draft generation
    draft_out, draft_log = agent.generate_draft(
        question="What is the survival term?",
        evidence=ev,
        num_candidates=3,
    )
    assert draft_out.revision_count == 0
    assert len(draft_out.candidate_responses) == 3
    assert draft_log.action == "GENERATION_DRAFT"

    # Revision (bounded to 1)
    guidance = "Restrict answer strictly to cited survival provisions."
    rev_out, rev_log = agent.revise_draft(
        question="What is the survival term?",
        evidence=ev,
        current_draft=draft_out,
        revision_guidance=guidance,
    )
    assert rev_out.revision_count == 1
    assert rev_log.action == "GENERATION_REVISED"
    assert "REVISED FINDINGS" in rev_out.primary_candidate.answer_text


# =============================================================================
# 4. Verification Agent Tests
# =============================================================================
def test_verification_agent_audit_and_contradiction():
    aligner = HeuristicClaimAligner()
    verifier = VerificationAgent(aligner=aligner, min_acceptable_support_ratio=0.60)

    chunk = _build_test_chunk(
        "nda.pdf",
        "Confidentiality shall survive for 3 years following agreement termination.",
    )
    ev = [RetrievalResult(chunk=chunk, score=0.85, rank=1)]

    # Clean candidate
    cand_good = CandidateResponse(
        candidate_index=0,
        generation_mode="MOCK",
        answer_text="Confidentiality obligations survive for 3 years following termination under nda.pdf:p1.",
        cited_sources=["nda.pdf:p1"],
    )

    out_good, log_good = verifier.verify_candidates(
        primary_candidate=cand_good,
        evidence=ev,
        allow_revision_request=True,
    )
    assert log_good.action == "VERIFICATION_AUDIT"
    assert not out_good.contradiction_detected
    assert out_good.support_ratio >= 0.50

    # Conflicting candidates trigger contradiction detection
    cand_clash_1 = CandidateResponse(
        candidate_index=0,
        generation_mode="MOCK",
        answer_text="This agreement is valid and enforceable and parties shall comply within 30 days.",
        cited_sources=["nda.pdf:p1"],
    )
    cand_clash_2 = CandidateResponse(
        candidate_index=1,
        generation_mode="MOCK",
        answer_text="This agreement is void and unenforceable and parties shall not comply, requiring 900 days notice.",
        cited_sources=["nda.pdf:p1"],
    )

    out_clash, log_clash = verifier.verify_candidates(
        primary_candidate=cand_clash_1,
        evidence=ev,
        all_candidates=[cand_clash_1, cand_clash_2],
    )
    assert out_clash.contradiction_detected
    assert len(out_clash.contradiction_details) > 0


# =============================================================================
# 5. Decision Controller Tests
# =============================================================================
def test_decision_controller_arbitration():
    embedder = DeterministicFakeEmbedder(dimension=64)
    aggregator = UncertaintyAggregator(embedder=embedder)
    cfg = AbstentionConfig(min_confidence_to_answer=0.30)
    controller = DecisionController(aggregator=aggregator, abstention_engine=AbstentionEngine(config=cfg))

    chunk = _build_test_chunk("agreement.pdf", "Notice period is 30 days.")
    ev = [RetrievalResult(chunk=chunk, score=0.90, rank=1)]

    candidates = [
        CandidateResponse(0, "MOCK", "Notice period is 30 days.", ["agreement.pdf:p1"]),
        CandidateResponse(1, "MOCK", "The required notice is 30 days.", ["agreement.pdf:p1"]),
        CandidateResponse(2, "MOCK", "Notice is 30 days.", ["agreement.pdf:p1"]),
    ]

    ctrl_decision, ctrl_log = controller.arbitrate(
        question="What is the notice period?",
        candidates=candidates,
        evidence=ev,
    )

    assert ctrl_log.action == "GUARDRAIL_ARBITRATION"
    assert ctrl_decision.decision.decision_level in ("SAFE", "WARNING")
    assert ctrl_decision.final_answer == candidates[0]
    assert ctrl_decision.uncertainty_metrics.confidence_score > 0.30


# =============================================================================
# 6. End-to-End Multi-Agent Orchestration & Action Logging Tests
# =============================================================================
def test_workflow_orchestrator_full_trace_and_bounded_limits():
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    pipeline.ingest_bytes(
        "statute.txt",
        b"UCC Section 2-201 requires a writing for contracts for the sale of goods for $500 or more.",
    )

    # Execute agentic workflow via pipeline
    wf_result = pipeline.agentic_query(
        question="What is the dollar threshold under UCC Section 2-201?",
        top_k=2,
    )

    assert isinstance(wf_result, AgenticWorkflowResult)
    assert wf_result.plan is not None
    assert len(wf_result.plan.steps) == 6

    # Verify bounded limits
    assert 1 <= wf_result.retrieval_attempts <= 2
    assert 0 <= wf_result.answer_revisions <= 1

    # Verify action logging
    trace = wf_result.execution_trace
    assert len(trace) >= 4

    agents_in_trace = {log.agent_name for log in trace}
    assert "PlanningAgent" in agents_in_trace
    assert "RetrievalAgent" in agents_in_trace
    assert "LegalAnalysisAgent" in agents_in_trace
    assert "VerificationAgent" in agents_in_trace
    assert "DecisionController" in agents_in_trace

    # Verify every log record has required fields
    for log in trace:
        assert log.agent_name
        assert log.action
        assert log.reason
        assert log.result_summary

    # Test conversion to PipelineQueryResult
    p_res = convert_workflow_result_to_pipeline_result(wf_result)
    assert p_res.primary_answer is not None
    assert p_res.decision_level in ("SAFE", "WARNING")
    assert p_res.metadata["retrieval_attempts"] == wf_result.retrieval_attempts
    assert p_res.agentic_workflow == wf_result
    assert p_res.execution_trace == wf_result.execution_trace


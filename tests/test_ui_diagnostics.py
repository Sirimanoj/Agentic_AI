"""Deterministic regression tests for UI diagnostics rendering and claim alignment schemas."""

from unittest.mock import MagicMock, patch
import pytest

from app import render_agentic_workflow_tab, render_claim_alignment_tab
from legal_uq.agents.models import (
    AgentActionLog,
    AgenticWorkflowResult,
    ExecutionPlan,
    PlanStep,
    StepStatus,
)
from legal_uq.uncertainty.claim_aligner import (
    ClaimAlignmentResult,
    ClaimEvidenceMatch,
    ClaimVerdict,
    ExtractedClaim,
)


def test_claim_alignment_result_schema_and_backward_compatibility():
    """Verify ClaimAlignmentResult defines canonical contradiction_detected and backward-compatible contradiction_found."""
    # Default case: no contradiction
    res_clean = ClaimAlignmentResult(contradiction_detected=False)
    assert res_clean.contradiction_detected is False
    assert res_clean.contradiction_found is False

    # Contradiction detected
    res_clash = ClaimAlignmentResult(
        contradiction_detected=True,
        contradiction_details=["Clashing payment deadlines"],
    )
    assert res_clash.contradiction_detected is True
    assert res_clash.contradiction_found is True

    # Mutability check: changing canonical field reflects in backward-compat property
    res_clean.contradiction_detected = True
    assert res_clean.contradiction_found is True


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_claim_alignment_tab_no_contradiction(
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify clean claim alignment renders success notice and no contradiction error."""
    claim = ExtractedClaim(
        claim_id="c1",
        candidate_index=0,
        text="The agreement survives for 3 years.",
        cited_sources=["contract.pdf:p1"],
    )
    match = ClaimEvidenceMatch(
        claim=claim,
        verdict=ClaimVerdict.SUPPORTED,
        matched_chunk_id="chk_1",
        matched_source_locator="contract.pdf:p1",
        overlap_score=0.85,
        rationale="Exact clause match in Section 1.",
    )
    alignment = ClaimAlignmentResult(
        total_claims=1,
        supported_claims=1,
        unsupported_claims=0,
        contradicted_claims=0,
        support_ratio=1.0,
        contradiction_detected=False,
        contradiction_details=[],
        matches=[match],
        unsupported_claim_texts=[],
    )

    render_claim_alignment_tab(alignment)

    # Must call st.success and not st.error
    mock_succ.assert_called_once()
    assert "No direct inter-candidate contradictions" in mock_succ.call_args[0][0]
    mock_err.assert_not_called()
    mock_warn.assert_not_called()


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_claim_alignment_tab_detected_contradiction(
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify detected contradiction renders prominent error banner with details."""
    alignment = ClaimAlignmentResult(
        total_claims=2,
        supported_claims=1,
        unsupported_claims=0,
        contradicted_claims=1,
        support_ratio=0.5,
        contradiction_detected=True,
        contradiction_details=["Candidate 0 asserts 'shall not disclose' while Candidate 1 asserts 'shall disclose'."],
        matches=[],
        unsupported_claim_texts=["Conflicting disclosure requirement"],
    )

    render_claim_alignment_tab(alignment)

    # Must call st.error and must NOT call st.success
    mock_err.assert_called_once()
    assert "MATERIAL CONTRADICTIONS DETECTED" in mock_err.call_args[0][0]
    mock_succ.assert_not_called()


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_claim_alignment_tab_empty_alignment_result(
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify empty/default ClaimAlignmentResult executes safely without AttributeError."""
    empty_alignment = ClaimAlignmentResult()

    render_claim_alignment_tab(empty_alignment)

    mock_succ.assert_called_once()
    mock_err.assert_not_called()
    mock_info.assert_called_once_with("No atomic claims extracted from primary response.")


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_claim_alignment_tab_missing_none_alignment(
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify None alignment handles missing data gracefully without exception."""
    render_claim_alignment_tab(None)

    mock_info.assert_called_once_with("No claim alignment data available for this query.")
    mock_err.assert_not_called()
    mock_succ.assert_not_called()


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_claim_alignment_tab_unknown_contradiction_status_never_falsely_reports_success(
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify when contradiction status is indeterminate, the UI does NOT falsely declare safety."""
    # Mock object where contradiction fields are None
    ambiguous_alignment = MagicMock(spec=[])
    ambiguous_alignment.matches = []
    ambiguous_alignment.unsupported_claim_texts = []

    render_claim_alignment_tab(ambiguous_alignment)

    # Must warn that status is unknown, and NEVER declare success
    mock_warn.assert_called_once()
    assert "status unknown or unavailable" in mock_warn.call_args[0][0]
    mock_succ.assert_not_called()


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_claim_alignment_tab_catches_unexpected_exception_and_presents_clean_error(
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify unexpected internal rendering errors do not dump a raw traceback to the user."""
    # Object that raises an unexpected exception when iterating matches
    class CrashingAlignment:
        contradiction_detected = False
        @property
        def matches(self):
            raise RuntimeError("Simulated internal rendering corruption")

    render_claim_alignment_tab(CrashingAlignment())

    # Must display a clean diagnostic error rather than raising an unhandled exception
    mock_err.assert_called_once()
    err_text = mock_err.call_args[0][0]
    assert "Diagnostic Rendering Notice" in err_text
    assert "RuntimeError" in err_text
    mock_cap.assert_called_once()


@patch("streamlit.caption")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.success")
@patch("streamlit.error")
@patch("streamlit.markdown")
@patch("streamlit.dataframe")
def test_render_agentic_workflow_tab_active_trace(
    mock_df: MagicMock,
    mock_md: MagicMock,
    mock_err: MagicMock,
    mock_succ: MagicMock,
    mock_warn: MagicMock,
    mock_info: MagicMock,
    mock_cap: MagicMock,
):
    """Verify render_agentic_workflow_tab displays execution plan, agent badges, and action table."""
    plan = ExecutionPlan(
        plan_id="plan_test123",
        question="What is the notice period?",
        steps=[
            PlanStep(1, "Evidence Retrieval", "RetrievalAgent", "Search vector store", StepStatus.COMPLETED),
            PlanStep(2, "Sufficiency Assessment", "RetrievalAgent", "Check relevance", StepStatus.COMPLETED),
        ],
        strategy_rationale="Active test strategy",
    )
    trace = [
        AgentActionLog(agent_name="PlanningAgent", action="PLAN_SYNTHESIS", reason="Plan inquiry", result_summary="Done"),
        AgentActionLog(agent_name="RetrievalAgent", action="RETRIEVAL_PRIMARY", reason="Search index", result_summary="Found 1"),
    ]
    wf = AgenticWorkflowResult(
        question="What is the notice period?",
        plan=plan,
        execution_trace=trace,
        retrieval_attempts=1,
        answer_revisions=0,
    )
    mock_res = MagicMock()
    mock_res.agentic_workflow = wf
    mock_res.execution_trace = trace
    mock_res.decision_level = "SAFE"
    mock_res.confidence_score = 0.88
    mock_res.metadata = {"agentic_workflow": wf, "agentic_trace": trace}

    render_agentic_workflow_tab(mock_res)

    # Must render dataframe for action log
    mock_df.assert_called_once()
    mock_err.assert_not_called()


@patch("streamlit.info")
@patch("streamlit.markdown")
def test_render_agentic_workflow_tab_empty_trace(mock_md: MagicMock, mock_info: MagicMock):
    """Verify render_agentic_workflow_tab handles missing trace gracefully."""
    mock_res = MagicMock()
    mock_res.agentic_workflow = None
    mock_res.execution_trace = []
    mock_res.metadata = {}

    render_agentic_workflow_tab(mock_res)
    mock_info.assert_called_once()
    assert "No agentic workflow trace" in mock_info.call_args[0][0]


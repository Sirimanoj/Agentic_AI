"""Structured data contracts exchanged between agents in the Legal UQ workflow.

Defines schemas for:
- Agent action logging (action, reason, result, details)
- Execution planning and step tracking
- Retrieval, Analysis, Verification, and Decision outputs
- Complete agentic workflow results
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from legal_uq.generation.generator import CandidateResponse
from legal_uq.guardrails.abstention import AbstentionDecision
from legal_uq.retrieval.vector_store import RetrievalResult
from legal_uq.uncertainty.aggregator import UncertaintyMetrics
from legal_uq.uncertainty.claim_aligner import ClaimAlignmentResult
from legal_uq.uncertainty.similarity import SimilarityMatrix


class StepStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"
    REVISING = "REVISING"
    FAILED = "FAILED"


@dataclass
class AgentActionLog:
    """Individual record of a tool invocation, agent handoff, or internal decision."""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    agent_name: str = ""
    action: str = ""
    reason: str = ""
    result_summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    """A discrete step in the Orchestrator's execution plan."""
    step_id: int
    title: str
    assigned_agent: str
    description: str
    status: StepStatus = StepStatus.PENDING
    status_notes: str = ""


@dataclass
class ExecutionPlan:
    """Structured execution plan formulated by the Planning Agent."""
    plan_id: str
    question: str
    steps: List[PlanStep] = field(default_factory=list)
    strategy_rationale: str = ""


@dataclass
class RetrievalOutput:
    """Structured output emitted by the Retrieval Agent."""
    query_used: str
    attempt_number: int  # 1 or 2 (max 2 attempts)
    retrieved_results: List[RetrievalResult] = field(default_factory=list)
    top_relevance_score: float = 0.0
    is_sufficient: bool = True
    sufficiency_reason: str = ""
    reformulation_applied: bool = False
    original_query: str = ""


@dataclass
class AnalysisOutput:
    """Structured output emitted by the Legal Analysis Agent."""
    candidate_responses: List[CandidateResponse] = field(default_factory=list)
    primary_candidate: Optional[CandidateResponse] = None
    generation_mode: str = "MOCK"  # "LIVE" or "MOCK"
    revision_count: int = 0  # 0 (initial) or 1 (revised)
    generation_rationale: str = ""
    disclaimer: str = ""


@dataclass
class VerificationOutput:
    """Structured output emitted by the Verification Agent."""
    claim_alignment: Optional[ClaimAlignmentResult] = None
    support_ratio: float = 0.0
    contradiction_detected: bool = False
    contradiction_details: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    needs_revision: bool = False
    revision_guidance: Optional[str] = None
    verification_summary: str = ""


@dataclass
class ControllerDecision:
    """Structured decision emitted by the Decision Controller."""
    decision: AbstentionDecision
    uncertainty_metrics: UncertaintyMetrics
    similarity_matrix: Optional[SimilarityMatrix]
    final_answer: CandidateResponse
    risk_summary: str = ""


@dataclass
class AgenticWorkflowResult:
    """Consolidated outcome of the multi-agent legal QA orchestration workflow."""
    question: str
    plan: ExecutionPlan
    retrieval_history: List[RetrievalOutput] = field(default_factory=list)
    analysis_history: List[AnalysisOutput] = field(default_factory=list)
    verification_history: List[VerificationOutput] = field(default_factory=list)
    controller_decision: Optional[ControllerDecision] = None
    execution_trace: List[AgentActionLog] = field(default_factory=list)
    retrieval_attempts: int = 1
    answer_revisions: int = 0
    duration_seconds: float = 0.0


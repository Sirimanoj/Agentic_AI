"""Multi-agent orchestration module for Legal UQ.

Provides explicit, structured agent interactions with bounded retry loops:
- PlanningAgent
- RetrievalAgent
- LegalAnalysisAgent
- VerificationAgent
- DecisionController
- AgentWorkflowOrchestrator
"""

from legal_uq.agents.analysis_agent import LegalAnalysisAgent
from legal_uq.agents.decision_controller import DecisionController
from legal_uq.agents.models import (
    AgentActionLog,
    AgenticWorkflowResult,
    AnalysisOutput,
    ControllerDecision,
    ExecutionPlan,
    PlanStep,
    RetrievalOutput,
    StepStatus,
    VerificationOutput,
)
from legal_uq.agents.orchestrator import PlanningAgent
from legal_uq.agents.retrieval_agent import RetrievalAgent
from legal_uq.agents.verification_agent import VerificationAgent
from legal_uq.agents.agent_workflow import (
    AgentWorkflowOrchestrator,
    convert_workflow_result_to_pipeline_result,
)

__all__ = [
    "AgentActionLog",
    "AgenticWorkflowResult",
    "AgentWorkflowOrchestrator",
    "AnalysisOutput",
    "ControllerDecision",
    "DecisionController",
    "ExecutionPlan",
    "LegalAnalysisAgent",
    "PlanStep",
    "PlanningAgent",
    "RetrievalAgent",
    "RetrievalOutput",
    "StepStatus",
    "VerificationAgent",
    "VerificationOutput",
    "convert_workflow_result_to_pipeline_result",
]


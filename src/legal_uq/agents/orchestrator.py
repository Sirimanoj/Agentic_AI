"""Orchestrator & Planning Agent.

Decides the sequential steps and strategies required to answer a legal inquiry,
considering document repository status, query phrasing, and guardrail constraints.
"""

from typing import List, Optional
import uuid

from legal_uq.agents.models import AgentActionLog, ExecutionPlan, PlanStep, StepStatus


class PlanningAgent:
    """Agent responsible for task decomposition and execution plan synthesis."""

    def __init__(self, name: str = "PlanningAgent") -> None:
        self.name = name

    def create_plan(
        self,
        question: str,
        total_chunks: int,
        is_live_mode: bool = False,
    ) -> tuple[ExecutionPlan, AgentActionLog]:
        """Formulate a structured execution plan for answering the legal inquiry.

        Args:
            question: The user's input legal question.
            total_chunks: Total number of indexed vector chunks available.
            is_live_mode: Whether live Gemini API mode or offline mock mode is active.

        Returns:
            Tuple of (ExecutionPlan, AgentActionLog).
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        clean_q = question.strip()

        if total_chunks == 0:
            strategy = "Empty Index Strategy: Directly report absence of documents and abstain safely."
            steps = [
                PlanStep(
                    step_id=1,
                    title="Index Verification",
                    assigned_agent="PlanningAgent",
                    description="Verify presence of indexed legal instruments.",
                    status=StepStatus.COMPLETED,
                    status_notes="Repository is empty. Immediate abstention triggered.",
                ),
                PlanStep(
                    step_id=2,
                    title="Safety Guardrail Abstention",
                    assigned_agent="DecisionController",
                    description="Issue hard ABSTAIN decision due to zero evidence availability.",
                    status=StepStatus.PENDING,
                ),
            ]
        else:
            mode_desc = "Live Gemini LLM" if is_live_mode else "Deterministic Offline Generator"
            strategy = (
                f"Active Corpus Strategy ({total_chunks} chunks indexed): "
                f"1) Semantic FAISS retrieval with relevance check; "
                f"2) Reformulation retry if relevance < 0.35 (max 2 attempts); "
                f"3) Grounded legal drafting ({mode_desc}); "
                f"4) Claim-level containment & polarity verification; "
                f"5) Targeted revision if unsupported claims detected (max 1 revision); "
                f"6) Consensus matrix & abstention risk triage."
            )
            steps = [
                PlanStep(
                    step_id=1,
                    title="Evidence Retrieval",
                    assigned_agent="RetrievalAgent",
                    description="Search FAISS vector store for relevant legal clauses with exact citations.",
                    status=StepStatus.PENDING,
                ),
                PlanStep(
                    step_id=2,
                    title="Evidence Sufficiency Evaluation",
                    assigned_agent="RetrievalAgent",
                    description="Evaluate top cosine score and query term overlap; retry with refined query if weak (max 2 attempts).",
                    status=StepStatus.PENDING,
                ),
                PlanStep(
                    step_id=3,
                    title="Grounded Legal Drafting",
                    assigned_agent="LegalAnalysisAgent",
                    description=f"Generate multi-candidate candidate answers strictly grounded in retrieved excerpts ({mode_desc}).",
                    status=StepStatus.PENDING,
                ),
                PlanStep(
                    step_id=4,
                    title="Claim Verification & Contradiction Audit",
                    assigned_agent="VerificationAgent",
                    description="Decompose candidate claims, audit against excerpts, and detect opposing legal polarities or numbers.",
                    status=StepStatus.PENDING,
                ),
                PlanStep(
                    step_id=5,
                    title="Conditional Answer Revision",
                    assigned_agent="LegalAnalysisAgent",
                    description="If verification identifies unsupported claims, revise primary draft with targeted feedback (max 1 revision).",
                    status=StepStatus.PENDING,
                ),
                PlanStep(
                    step_id=6,
                    title="Uncertainty & Guardrail Triage",
                    assigned_agent="DecisionController",
                    description="Compute pairwise candidate similarity matrices and determine SAFE, WARNING, or ABSTAIN decision.",
                    status=StepStatus.PENDING,
                ),
            ]

        plan = ExecutionPlan(
            plan_id=plan_id,
            question=clean_q,
            steps=steps,
            strategy_rationale=strategy,
        )

        log = AgentActionLog(
            agent_name=self.name,
            action="PLAN_SYNTHESIS",
            reason=f"Formulate step-by-step strategy for inquiry: '{clean_q[:60]}...'",
            result_summary=f"Synthesized plan {plan_id} with {len(steps)} steps.",
            details={
                "plan_id": plan_id,
                "step_count": len(steps),
                "total_chunks_available": total_chunks,
                "strategy": strategy,
            },
        )
        return plan, log


"""Agent Workflow Orchestrator.

Implements the multi-agent orchestration lifecycle with bounded retry loops:
- Orchestrator/Planning Agent: task decomposition and strategy
- Retrieval Agent: FAISS tool invocation and bounded query reformulation (max 2 attempts)
- Legal Analysis Agent: grounded candidate drafting
- Verification Agent: claim containment and contradiction audit
- Bounded Revision: targeted draft revision if unsupported claims found (max 1 revision)
- Decision Controller: consensus matrix and SAFE/WARNING/ABSTAIN triage
- Full audit logging: action, reason, result, details for every step
"""

import re
import time
from typing import List, Optional, Sequence

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
from legal_uq.generation.generator import CandidateResponse, GroundedGenerator
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionDecision, AbstentionEngine
from legal_uq.pipeline import PipelineQueryResult
from legal_uq.retrieval.vector_store import RetrievalResult, VectorStore
from legal_uq.uncertainty.aggregator import UncertaintyAggregator
from legal_uq.uncertainty.claim_aligner import HeuristicClaimAligner


class AgentWorkflowOrchestrator:
    """Master orchestrator executing the multi-agent legal QA protocol."""

    def __init__(
        self,
        vector_store: VectorStore,
        generator: GroundedGenerator,
        aggregator: UncertaintyAggregator,
        abstention_engine: Optional[AbstentionEngine] = None,
        claim_aligner: Optional[HeuristicClaimAligner] = None,
        max_retrieval_attempts: int = 2,
        max_answer_revisions: int = 1,
    ) -> None:
        self.vector_store = vector_store
        self.generator = generator
        self.aggregator = aggregator
        self.abstention_engine = abstention_engine or AbstentionEngine()
        self.claim_aligner = claim_aligner or HeuristicClaimAligner()

        # Bounded limits (enforced per Requirements)
        self.max_retrieval_attempts = min(max_retrieval_attempts, 2)
        self.max_answer_revisions = min(max_answer_revisions, 1)

        # Agent instances
        self.planning_agent = PlanningAgent()
        self.retrieval_agent = RetrievalAgent(
            vector_store=self.vector_store,
            max_attempts=self.max_retrieval_attempts,
        )
        self.analysis_agent = LegalAnalysisAgent(generator=self.generator)
        self.verification_agent = VerificationAgent(aligner=self.claim_aligner)
        self.decision_controller = DecisionController(
            aggregator=self.aggregator,
            abstention_engine=self.abstention_engine,
        )

    def execute_workflow(
        self,
        question: str,
        top_k: int = 4,
        custom_config: Optional[AbstentionConfig] = None,
        simulate_contradiction: bool = False,
    ) -> AgenticWorkflowResult:
        """Run the end-to-end multi-agent orchestration workflow.

        Args:
            question: The user's legal question.
            top_k: Target number of retrieved passages.
            custom_config: Optional override of AbstentionConfig.
            simulate_contradiction: If True, injects conflicting candidate outputs for contradiction guardrail testing.

        Returns:
            AgenticWorkflowResult containing complete structured history and audit log.
        """
        start_time = time.perf_counter()
        execution_trace: List[AgentActionLog] = []
        retrieval_history: List[RetrievalOutput] = []
        analysis_history: List[AnalysisOutput] = []
        verification_history: List[VerificationOutput] = []

        total_chunks = self.vector_store.total_chunks
        is_live = self.generator.is_live

        # ---------------------------------------------------------------------
        # STEP 1: Planning / Orchestrator Agent
        # ---------------------------------------------------------------------
        plan, plan_log = self.planning_agent.create_plan(
            question=question,
            total_chunks=total_chunks,
            is_live_mode=is_live,
        )
        execution_trace.append(plan_log)

        # Handle empty index
        if total_chunks == 0:
            return self._handle_empty_index(
                question=question,
                plan=plan,
                execution_trace=execution_trace,
                custom_config=custom_config,
                start_time=start_time,
            )

        # Mark step 1 in progress
        plan.steps[0].status = StepStatus.IN_PROGRESS

        # ---------------------------------------------------------------------
        # STEP 2: Retrieval Agent with bounded retry loop (max 2 attempts)
        # ---------------------------------------------------------------------
        retrieval_output, ret_logs = self.retrieval_agent.retrieve_with_retry(
            query=question,
            top_k=top_k,
        )
        execution_trace.extend(ret_logs)
        retrieval_history.append(retrieval_output)

        plan.steps[0].status = StepStatus.COMPLETED
        plan.steps[0].status_notes = f"Retrieved {len(retrieval_output.retrieved_results)} chunks (Attempt {retrieval_output.attempt_number})."

        plan.steps[1].status = StepStatus.COMPLETED
        plan.steps[1].status_notes = (
            f"Evidence {'Sufficient' if retrieval_output.is_sufficient else 'Insufficient'}; "
            f"Score: {retrieval_output.top_relevance_score:.3f}."
        )

        evidence = retrieval_output.retrieved_results

        # ---------------------------------------------------------------------
        # STEP 3: Legal Analysis Agent (Grounded Candidate Drafting)
        # ---------------------------------------------------------------------
        plan.steps[2].status = StepStatus.IN_PROGRESS
        draft_output, draft_log = self.analysis_agent.generate_draft(
            question=question,
            evidence=evidence,
            num_candidates=3,
        )
        execution_trace.append(draft_log)
        analysis_history.append(draft_output)

        candidates = list(draft_output.candidate_responses)

        # Optional contradiction injection for interactive testing / demo
        if simulate_contradiction and len(candidates) >= 2:
            c0_words = [w for w in re.findall(r"\b\w+\b", candidates[0].answer_text) if len(w) > 3]
            anchor_words = " ".join(c0_words[:4]) if c0_words else "terms and obligations"
            candidates[0] = CandidateResponse(
                candidate_index=0,
                generation_mode=candidates[0].generation_mode,
                answer_text=f"{candidates[0].answer_text} This provision is valid and enforceable, and parties shall comply within 30 days.",
                cited_sources=candidates[0].cited_sources,
                metadata={"polarity_anchor": "positive"},
            )
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
            draft_output.candidate_responses = candidates
            draft_output.primary_candidate = candidates[0]
            execution_trace.append(
                AgentActionLog(
                    agent_name="Orchestrator",
                    action="CONTRADICTION_SIMULATION_INJECTED",
                    reason="User enabled simulate_contradiction flag for interactive demonstration.",
                    result_summary="Injected opposing polarity and conflicting durations into candidates 0 and 1.",
                )
            )

        plan.steps[2].status = StepStatus.COMPLETED
        plan.steps[2].status_notes = f"Generated {len(candidates)} candidates in {draft_output.generation_mode} mode."

        # ---------------------------------------------------------------------
        # STEP 4: Verification Agent (Claim-Level Grounding & Contradictions)
        # ---------------------------------------------------------------------
        plan.steps[3].status = StepStatus.IN_PROGRESS
        verification_output, verify_log = self.verification_agent.verify_candidates(
            primary_candidate=candidates[0],
            evidence=evidence,
            all_candidates=candidates,
            allow_revision_request=True,
        )
        execution_trace.append(verify_log)
        verification_history.append(verification_output)

        plan.steps[3].status = StepStatus.COMPLETED
        plan.steps[3].status_notes = (
            f"Support: {verification_output.support_ratio:.1%}, "
            f"Contradictions: {'Yes' if verification_output.contradiction_detected else 'No'}."
        )

        # ---------------------------------------------------------------------
        # STEP 5: Bounded Revision Loop (Max 1 Revision Attempt)
        # ---------------------------------------------------------------------
        answer_revisions = 0
        if verification_output.needs_revision and self.max_answer_revisions > 0:
            plan.steps[4].status = StepStatus.REVISING
            plan.steps[4].status_notes = "Executing bounded draft revision based on verification guidance."

            revised_output, rev_log = self.analysis_agent.revise_draft(
                question=question,
                evidence=evidence,
                current_draft=draft_output,
                revision_guidance=verification_output.revision_guidance or "Ensure strict factual adherence.",
            )
            execution_trace.append(rev_log)
            analysis_history.append(revised_output)
            answer_revisions = 1
            candidates = list(revised_output.candidate_responses)

            # Re-verify revised output
            post_rev_verif, post_rev_log = self.verification_agent.verify_candidates(
                primary_candidate=candidates[0],
                evidence=evidence,
                all_candidates=candidates,
                allow_revision_request=False,  # Revisions bounded to 1
            )
            execution_trace.append(post_rev_log)
            verification_history.append(post_rev_verif)
            final_verification = post_rev_verif

            plan.steps[4].status = StepStatus.COMPLETED
            plan.steps[4].status_notes = f"Completed 1 revision. Post-revision support: {post_rev_verif.support_ratio:.1%}."
        else:
            plan.steps[4].status = StepStatus.SKIPPED
            plan.steps[4].status_notes = "No draft revision required (support ratio acceptable or no evidence)."
            final_verification = verification_output

        # ---------------------------------------------------------------------
        # STEP 6: Decision Controller (Guardrail Triage & Uncertainty Arbitration)
        # ---------------------------------------------------------------------
        plan.steps[5].status = StepStatus.IN_PROGRESS
        controller_decision, ctrl_log = self.decision_controller.arbitrate(
            question=question,
            candidates=candidates,
            evidence=evidence,
            verified_alignment=final_verification.claim_alignment,
            custom_config=custom_config,
        )
        execution_trace.append(ctrl_log)

        plan.steps[5].status = StepStatus.COMPLETED
        plan.steps[5].status_notes = f"Arbitration complete: {controller_decision.decision.decision_level}."

        total_duration = time.perf_counter() - start_time

        return AgenticWorkflowResult(
            question=question,
            plan=plan,
            retrieval_history=retrieval_history,
            analysis_history=analysis_history,
            verification_history=verification_history,
            controller_decision=controller_decision,
            execution_trace=execution_trace,
            retrieval_attempts=retrieval_output.attempt_number,
            answer_revisions=answer_revisions,
            duration_seconds=total_duration,
        )

    def _handle_empty_index(
        self,
        question: str,
        plan: ExecutionPlan,
        execution_trace: List[AgentActionLog],
        custom_config: Optional[AbstentionConfig],
        start_time: float,
    ) -> AgenticWorkflowResult:
        """Handle zero-document state safely with explicit agent logging."""
        engine = AbstentionEngine(config=custom_config) if custom_config else self.abstention_engine
        mode = "LIVE" if self.generator.is_live else "MOCK"

        empty_cand = CandidateResponse(
            candidate_index=0,
            generation_mode=mode,
            answer_text="No legal documents are currently indexed. Please upload at least one document.",
            cited_sources=[],
        )

        metrics = self.aggregator.aggregate([empty_cand], evidence=[])
        decision = engine.evaluate(metrics)

        controller_decision = ControllerDecision(
            decision=decision,
            uncertainty_metrics=metrics,
            similarity_matrix=None,
            final_answer=empty_cand,
            risk_summary="Empty repository: Immediate ABSTAIN decision issued.",
        )

        execution_trace.append(
            AgentActionLog(
                agent_name="DecisionController",
                action="GUARDRAIL_EMPTY_INDEX_ABSTAIN",
                reason="No indexed chunks found in FAISS repository.",
                result_summary="Issued hard ABSTAIN decision.",
                details={"decision_level": "ABSTAIN", "answer_allowed": False},
            )
        )

        plan.steps[1].status = StepStatus.COMPLETED
        plan.steps[1].status_notes = "Hard ABSTAIN issued for empty index."

        return AgenticWorkflowResult(
            question=question,
            plan=plan,
            retrieval_history=[],
            analysis_history=[],
            verification_history=[],
            controller_decision=controller_decision,
            execution_trace=execution_trace,
            retrieval_attempts=0,
            answer_revisions=0,
            duration_seconds=time.perf_counter() - start_time,
        )


def convert_workflow_result_to_pipeline_result(
    wf: AgenticWorkflowResult,
) -> PipelineQueryResult:
    """Map AgenticWorkflowResult into backward-compatible PipelineQueryResult."""
    ctrl = wf.controller_decision
    assert ctrl is not None

    latest_analysis = wf.analysis_history[-1] if wf.analysis_history else None
    latest_retrieval = wf.retrieval_history[-1] if wf.retrieval_history else None
    latest_verif = wf.verification_history[-1] if wf.verification_history else None

    cands = latest_analysis.candidate_responses if latest_analysis else [ctrl.final_answer]
    ev = latest_retrieval.retrieved_results if latest_retrieval else []
    mode = latest_analysis.generation_mode if latest_analysis else "MOCK"

    res = PipelineQueryResult(
        question=wf.question,
        retrieval_results=ev,
        candidate_responses=cands,
        primary_answer=ctrl.final_answer,
        similarity_matrix=ctrl.similarity_matrix,
        claim_alignment=latest_verif.claim_alignment if latest_verif else None,
        uncertainty_metrics=ctrl.uncertainty_metrics,
        abstention_decision=ctrl.decision,
        generation_mode=mode,
        duration_seconds=wf.duration_seconds,
        metadata={
            "agentic_trace": wf.execution_trace,
            "agentic_workflow": wf,
            "retrieval_attempts": wf.retrieval_attempts,
            "answer_revisions": wf.answer_revisions,
        },
    )
    return res

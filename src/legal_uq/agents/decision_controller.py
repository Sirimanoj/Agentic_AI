"""Decision Controller Agent.

Computes candidate consensus matrices, aggregates multi-faceted uncertainty metrics,
and issues the final safety guardrail verdict (SAFE, WARNING, ABSTAIN).
"""

from typing import List, Optional, Sequence, Tuple

from legal_uq.agents.models import AgentActionLog, ControllerDecision
from legal_uq.generation.generator import CandidateResponse
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionDecision, AbstentionEngine
from legal_uq.retrieval.vector_store import RetrievalResult
from legal_uq.uncertainty.aggregator import UncertaintyAggregator, UncertaintyMetrics
from legal_uq.uncertainty.claim_aligner import ClaimAlignmentResult


class DecisionController:
    """Agent responsible for uncertainty aggregation and final guardrail arbitration."""

    def __init__(
        self,
        aggregator: UncertaintyAggregator,
        abstention_engine: Optional[AbstentionEngine] = None,
        name: str = "DecisionController",
    ) -> None:
        self.aggregator = aggregator
        self.abstention_engine = abstention_engine or AbstentionEngine()
        self.name = name

    def arbitrate(
        self,
        question: str,
        candidates: Sequence[CandidateResponse],
        evidence: Sequence[RetrievalResult],
        verified_alignment: Optional[ClaimAlignmentResult] = None,
        custom_config: Optional[AbstentionConfig] = None,
    ) -> Tuple[ControllerDecision, AgentActionLog]:
        """Aggregate consensus, evaluate risk thresholds, and determine whether the answer is permitted.

        Args:
            question: The user query.
            candidates: Final candidate responses.
            evidence: Retrieved evidence passages.
            verified_alignment: Optional pre-computed ClaimAlignmentResult from VerificationAgent.
            custom_config: Optional override of AbstentionConfig.

        Returns:
            Tuple of (ControllerDecision, AgentActionLog).
        """
        engine = AbstentionEngine(config=custom_config) if custom_config else self.abstention_engine

        # Aggregate uncertainty metrics
        metrics = self.aggregator.aggregate(
            candidates=candidates,
            evidence=evidence,
            query=question,
        )

        # If VerificationAgent already performed multi-candidate contradiction audit, preserve detected contradictions
        if verified_alignment and verified_alignment.contradiction_detected:
            if not metrics.claim_alignment.contradiction_detected:
                # Re-evaluate with contradiction flag set
                metrics = self.aggregator.aggregate(
                    candidates=candidates,
                    evidence=evidence,
                    query=question,
                )

        decision: AbstentionDecision = engine.evaluate(metrics)
        final_answer = candidates[0] if candidates else CandidateResponse(
            candidate_index=0,
            generation_mode="MOCK",
            answer_text="No answer generated.",
            cited_sources=[],
        )

        risk_summary = (
            f"Verdict: {decision.decision_level} (Answer Allowed: {decision.answer_allowed}) | "
            f"Confidence: {metrics.confidence_score:.3f} | Uncertainty: {metrics.uncertainty_score:.3f} | "
            f"Support Ratio: {metrics.support_ratio:.1%}"
        )

        log = AgentActionLog(
            agent_name=self.name,
            action="GUARDRAIL_ARBITRATION",
            reason=(
                f"Evaluate uncertainty metrics (C={metrics.confidence_score:.2f}, U={metrics.uncertainty_score:.2f}) "
                f"against safety guardrail thresholds."
            ),
            result_summary=risk_summary,
            details={
                "decision_level": decision.decision_level,
                "answer_allowed": decision.answer_allowed,
                "confidence_score": float(metrics.confidence_score),
                "uncertainty_score": float(metrics.uncertainty_score),
                "support_ratio": float(metrics.support_ratio),
                "reasons": decision.reasons,
                "user_explanation": decision.user_explanation,
            },
        )

        controller_decision = ControllerDecision(
            decision=decision,
            uncertainty_metrics=metrics,
            similarity_matrix=metrics.similarity_matrix,
            final_answer=final_answer,
            risk_summary=risk_summary,
        )
        return controller_decision, log


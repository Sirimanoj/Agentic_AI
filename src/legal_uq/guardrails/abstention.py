"""Abstention guardrails and confidence-aware triage engine.

Enforces:
1. Hard abstention when retrieved evidence is absent or tangential.
2. Contradiction suppression when candidate responses materially disagree.
3. User-facing warnings when uncertainty exceeds configured risk limits.

METHODOLOGICAL DISCLAIMER:
    Configured thresholds represent baseline engineering heuristics for risk triage.
    They are not formal legal guarantees and must be calibrated against human ground truth.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from legal_uq.uncertainty.aggregator import UncertaintyMetrics


@dataclass
class AbstentionConfig:
    """Configurable heuristics governing abstention and warning triggers.

    Attributes:
        min_evidence_relevance: Minimum top passage similarity (r_rel) required (default: 0.30).
        min_confidence_to_answer: Absolute lower bound on confidence to permit an answer (default: 0.30).
        confidence_warning_threshold: Boundary below which an answer requires a warning banner (default: 0.55).
        max_uncertainty_threshold: Upper limit on uncertainty before issuing a warning (default: 0.55).
        min_support_ratio: Mandatory minimum proportion of claims substantiated in text (default: 0.40).
            Falling below this threshold triggers hard ABSTAIN and suppresses the substantive answer.
        block_on_contradiction: If True, direct contradiction triggers hard abstention (default: False, issues high warning).
    """
    min_evidence_relevance: float = 0.30
    min_confidence_to_answer: float = 0.30
    confidence_warning_threshold: float = 0.55
    max_uncertainty_threshold: float = 0.55
    min_support_ratio: float = 0.40
    block_on_contradiction: bool = False


@dataclass(frozen=True)
class AbstentionDecision:
    """The formal verdict determining if and how an answer may be presented."""
    answer_allowed: bool
    warning_required: bool
    decision_level: str  # "SAFE", "WARNING", "ABSTAIN"
    reasons: List[str]
    user_explanation: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AbstentionEngine:
    """Applies policy guardrails to UncertaintyMetrics to prevent misleading legal answers."""

    def __init__(self, config: Optional[AbstentionConfig] = None) -> None:
        self.config = config if config is not None else AbstentionConfig()

    def evaluate(self, metrics: UncertaintyMetrics) -> AbstentionDecision:
        """Evaluate uncertainty and grounding metrics against policy thresholds.

        Args:
            metrics: The aggregated UncertaintyMetrics for the current QA run.

        Returns:
            AbstentionDecision detailing answer permission, warning status, and user explanation.
        """
        reasons: List[str] = []
        conf = metrics.confidence_score
        unc = metrics.uncertainty_score
        r_rel = metrics.r_rel
        support = metrics.support_ratio
        contradiction = metrics.contradiction_detected

        # Condition 1: Complete lack of evidence -> MANDATORY HARD ABSTENTION
        if "NO_RETRIEVED_EVIDENCE" in metrics.reason_codes or r_rel == 0.0:
            return AbstentionDecision(
                answer_allowed=False,
                warning_required=True,
                decision_level="ABSTAIN",
                reasons=["NO_RETRIEVED_EVIDENCE"],
                user_explanation=(
                    "The system has abstained from answering: The uploaded documents do not contain "
                    "retrieved passages relevant to your question. To prevent misinformation, "
                    "no speculative legal advice is provided."
                ),
                diagnostics={"confidence": conf, "r_rel": r_rel, "support_ratio": support},
            )

        # Condition 2: Tangential evidence (below minimum relevance threshold)
        if r_rel < self.config.min_evidence_relevance:
            reasons.append(
                f"Evidence relevance ({r_rel:.2f}) is below the required threshold ({self.config.min_evidence_relevance:.2f})."
            )

        # Condition 3: Excessive unsupported claims
        if support < self.config.min_support_ratio:
            reasons.append(
                f"Only {support * 100:.1f}% of claims could be verified in the uploaded text "
                f"(minimum required: {self.config.min_support_ratio * 100:.1f}%)."
            )

        # Condition 4: Severe inter-candidate contradiction
        if contradiction:
            if self.config.block_on_contradiction:
                reasons.append("Direct contradictions were detected across candidate interpretations.")
            else:
                reasons.append("Candidate answers present materially conflicting provisions or dates.")

        # Condition 5: Low confidence / High uncertainty
        if conf < self.config.min_confidence_to_answer:
            reasons.append(
                f"Overall confidence ({conf:.2f}) is below minimum threshold ({self.config.min_confidence_to_answer:.2f})."
            )
        elif conf < self.config.confidence_warning_threshold:
            reasons.append(
                f"Confidence is moderate ({conf:.2f}), requiring verification against the primary document."
            )

        if unc > self.config.max_uncertainty_threshold:
            reasons.append(f"Estimated uncertainty ({unc:.2f}) exceeds the alert limit ({self.config.max_uncertainty_threshold:.2f}).")

        # Determine Decision Level
        # Hard Abstention (Answer Suppressed, answer_allowed=False) if:
        # 1. Overall confidence is below absolute cutoff (conf < min_confidence_to_answer), OR
        # 2. Direct contradiction occurs when block_on_contradiction is enabled, OR
        # 3. Grounded claim support is below mandatory minimum (support < min_support_ratio)
        if (
            (conf < self.config.min_confidence_to_answer)
            or (contradiction and self.config.block_on_contradiction)
            or (support < self.config.min_support_ratio)
        ):
            decision_level = "ABSTAIN"
            answer_allowed = False
            warning_required = True
            explanation = (
                "The system cannot provide a reliable answer based on the uploaded documents. "
                "The retrieved evidence is either insufficient, contains contradictory obligations, "
                "or fails to substantiate key legal claims. "
                + " | ".join(reasons)
            )
        elif len(reasons) > 0:
            decision_level = "WARNING"
            answer_allowed = True
            warning_required = True
            explanation = (
                "CAUTION: This answer is provided with moderate confidence. "
                "Potential ambiguities, partial evidence support, or candidate variances were detected. "
                "Review the cited primary passages directly before relying on this information. "
                + " | ".join(reasons)
            )
        else:
            decision_level = "SAFE"
            answer_allowed = True
            warning_required = False
            explanation = (
                "The response is well-grounded in the retrieved text with high multi-candidate consensus. "
                "Always verify with a qualified legal professional for binding matters."
            )

        return AbstentionDecision(
            answer_allowed=answer_allowed,
            warning_required=warning_required,
            decision_level=decision_level,
            reasons=reasons,
            user_explanation=explanation,
            diagnostics={
                "confidence_score": conf,
                "uncertainty_score": unc,
                "r_rel": r_rel,
                "support_ratio": support,
                "contradiction_detected": contradiction,
                "thresholds": self.config.__dict__,
            },
        )


"""Verification Agent.

Decomposes candidate responses into atomic legal assertions, audits containment
against retrieved evidence passages, checks for contradictory polarities or numbers,
and generates targeted revision guidance if claims lack grounding.
"""

from typing import List, Optional, Sequence, Tuple

from legal_uq.agents.models import AgentActionLog, VerificationOutput
from legal_uq.generation.generator import CandidateResponse, detect_missing_information
from legal_uq.retrieval.vector_store import RetrievalResult
from legal_uq.uncertainty.claim_aligner import ClaimAlignmentResult, HeuristicClaimAligner


class VerificationAgent:
    """Agent responsible for claim-level verification and contradiction detection."""

    def __init__(
        self,
        aligner: Optional[HeuristicClaimAligner] = None,
        min_acceptable_support_ratio: float = 0.60,
        name: str = "VerificationAgent",
    ) -> None:
        self.aligner = aligner or HeuristicClaimAligner()
        self.min_acceptable_support_ratio = min_acceptable_support_ratio
        self.name = name

    def verify_candidates(
        self,
        primary_candidate: CandidateResponse,
        evidence: Sequence[RetrievalResult],
        all_candidates: Optional[Sequence[CandidateResponse]] = None,
        allow_revision_request: bool = True,
    ) -> Tuple[VerificationOutput, AgentActionLog]:
        """Audit primary candidate response and inter-candidate consistency.

        Args:
            primary_candidate: The focal answer to verify.
            evidence: Retrieved evidence passages.
            all_candidates: Optional set of all candidate answers (to check inter-candidate polarity clash).
            allow_revision_request: Whether a revision recommendation is allowed (bounded loop control).

        Returns:
            Tuple of (VerificationOutput, AgentActionLog).
        """
        # 1. Run claim alignment and contradiction check against evidence passages
        candidates_to_verify = list(all_candidates) if all_candidates else [primary_candidate]
        alignment: ClaimAlignmentResult = self.aligner.align_and_verify(
            candidates=candidates_to_verify,
            evidence=evidence,
        )

        contradiction_detected = bool(alignment.contradiction_detected)
        contradiction_details = list(alignment.contradiction_details)
        support_ratio = float(alignment.support_ratio)
        unsupported = list(alignment.unsupported_claim_texts)

        # 3. Determine if revision should be requested
        # Check if candidate is already a verified refusal indicating missing info
        is_missing_info = (
            primary_candidate.metadata.get("missing_information_detected", False)
            or primary_candidate.metadata.get("is_answerable") is False
            or detect_missing_information(primary_candidate.answer_text)
        )

        needs_revision = False
        revision_guidance = None

        if allow_revision_request and evidence and not is_missing_info:
            if unsupported and support_ratio < self.min_acceptable_support_ratio:
                needs_revision = True
                claims_bullet = "; ".join(f"'{c}'" for c in unsupported[:2])
                revision_guidance = (
                    f"Identified {len(unsupported)} ungrounded claim(s) ({claims_bullet}). "
                    f"Overall support ratio is {support_ratio:.1%}, below acceptable threshold {self.min_acceptable_support_ratio:.0%}. "
                    f"Please confine the answer strictly to terms articulated in cited passages."
                )

        summary = (
            f"Verification complete: Support Ratio = {support_ratio:.1%}, "
            f"Unsupported Claims = {len(unsupported)}, "
            f"Contradictions = {'Detected' if contradiction_detected else 'None'}."
        )

        log = AgentActionLog(
            agent_name=self.name,
            action="VERIFICATION_AUDIT",
            reason="Verify candidate factual claims against retrieved evidence and scan for contradictions.",
            result_summary=summary,
            details={
                "support_ratio": support_ratio,
                "unsupported_claim_count": len(unsupported),
                "unsupported_samples": unsupported[:2],
                "contradiction_detected": contradiction_detected,
                "contradiction_details": contradiction_details,
                "needs_revision": needs_revision,
                "revision_guidance": revision_guidance,
            },
        )

        output = VerificationOutput(
            claim_alignment=alignment,
            support_ratio=support_ratio,
            contradiction_detected=contradiction_detected,
            contradiction_details=contradiction_details,
            unsupported_claims=unsupported,
            needs_revision=needs_revision,
            revision_guidance=revision_guidance,
            verification_summary=summary,
        )
        return output, log

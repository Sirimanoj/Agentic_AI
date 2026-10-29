"""Confidence and uncertainty aggregation engine for legal document QA.

Combines:
1. Inter-candidate agreement (s_top, s_mean from SimilarityMatrix).
2. Evidence relevance (r_rel from top retrieval scores).
3. Grounding integrity (claim support ratio and contradiction penalties).

METHODOLOGICAL NOTICE:
    The resulting confidence score is a deterministic, heuristic ranking and triage metric.
    It is NOT a calibrated Bayesian posterior or objective probability of legal correctness.
    A score of 0.85 indicates high textual agreement and strong passage alignment;
    it does not guarantee that the legal conclusion is legally valid under all jurisdictions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from legal_uq.generation.generator import CandidateResponse
from legal_uq.retrieval.vector_store import RetrievalResult
from legal_uq.uncertainty.claim_aligner import ClaimAlignmentResult, HeuristicClaimAligner
from legal_uq.uncertainty.similarity import SimilarityMatrix, compute_candidate_similarities


@dataclass
class UncertaintyMetrics:
    """Consolidated confidence, uncertainty, and diagnostic breakdown."""
    confidence_score: float  # [0.0, 1.0] - higher = higher estimated evidence grounding & consensus
    uncertainty_score: float  # [0.0, 1.0] - higher = higher risk / disagreement / ungrounded claims
    s_top: float  # Highest candidate mean agreement with peer candidates
    s_mean: float  # Mean pairwise composite agreement across all candidate pairs
    r_rel: float  # Bounded evidence retrieval relevance indicator
    support_ratio: float  # Fraction of claims grounded in retrieved evidence
    contradiction_detected: bool  # True if inter-candidate or evidence clash was found
    reason_codes: List[str] = field(default_factory=list)
    similarity_matrix: Optional[SimilarityMatrix] = None
    claim_alignment: Optional[ClaimAlignmentResult] = None
    is_answerable: bool = True  # True if evidence contains facts addressing the query
    missing_information_detected: bool = False  # True if candidate responses explicitly state info is absent
    answerability_verdict: str = "ANSWERABLE"  # "ANSWERABLE", "INSUFFICIENT_EVIDENCE", "UNANSWERABLE_ABSENT"
    disclaimer: str = (
        "HEURISTIC METRIC NOTICE: Confidence scores reflect deterministic textual alignment "
        "and multi-candidate agreement heuristics. They do not constitute certified probabilities "
        "of objective legal correctness."
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class UncertaintyAggregator:
    """Aggregates similarity, retrieval relevance, and claim grounding into confidence & uncertainty."""

    def __init__(
        self,
        weight_s_top: float = 0.50,
        weight_s_mean: float = 0.30,
        weight_r_rel: float = 0.20,
        contradiction_penalty: float = 0.35,
        embedder: Optional[Any] = None,
    ) -> None:
        """Initialize UncertaintyAggregator.

        Args:
            weight_s_top: Weight for top candidate consensus (default: 0.50).
            weight_s_mean: Weight for average pairwise candidate consensus (default: 0.30).
            weight_r_rel: Weight for retrieval evidence relevance (default: 0.20).
            contradiction_penalty: Absolute deduction applied when contradictions are detected.
            embedder: Optional injected Embedder instance.
        """
        total = weight_s_top + weight_s_mean + weight_r_rel
        self.w_top = weight_s_top / total
        self.w_mean = weight_s_mean / total
        self.w_rel = weight_r_rel / total
        self.contradiction_penalty = contradiction_penalty
        self.embedder = embedder
        self.claim_aligner = HeuristicClaimAligner()

    def aggregate(
        self,
        candidates: Sequence[CandidateResponse],
        evidence: Sequence[RetrievalResult],
        similarity_matrix: Optional[SimilarityMatrix] = None,
        claim_alignment: Optional[ClaimAlignmentResult] = None,
        query: Optional[str] = None,
    ) -> UncertaintyMetrics:
        """Compute the final confidence and uncertainty estimates.

        Formula:
            Raw Baseline = (0.50 * s_top) + (0.30 * s_mean) + (0.20 * r_rel)
            Grounded Confidence = Raw Baseline * support_ratio - contradiction_penalty (if any)
            Uncertainty = 1.0 - Grounded Confidence

        Args:
            candidates: Generated candidate responses.
            evidence: Retrieved passages.
            similarity_matrix: Optional precomputed SimilarityMatrix.
            claim_alignment: Optional precomputed ClaimAlignmentResult.
            query: Optional user question string to verify evidence answerability.

        Returns:
            UncertaintyMetrics instance with bounded [0.0, 1.0] scores.
        """
        reasons: List[str] = []
        K = len(candidates)

        # 1. Edge Case: Zero candidates
        if K == 0:
            return UncertaintyMetrics(
                confidence_score=0.0,
                uncertainty_score=1.0,
                s_top=0.0,
                s_mean=0.0,
                r_rel=0.0,
                support_ratio=0.0,
                contradiction_detected=False,
                reason_codes=["NO_CANDIDATES_GENERATED"],
                is_answerable=False,
                missing_information_detected=True,
                answerability_verdict="INSUFFICIENT_EVIDENCE",
            )

        # 2. Retrieval Relevance Indicator (r_rel)
        if not evidence or all(not e.text.strip() for e in evidence):
            r_rel = 0.0
            reasons.append("NO_RETRIEVED_EVIDENCE")
        else:
            top_score = evidence[0].score
            # Clamp cosine similarity from [-1, 1] to [0, 1]
            r_rel = max(0.0, min(1.0, float(top_score)))
            if r_rel < 0.35:
                reasons.append("LOW_EVIDENCE_RELEVANCE")
            else:
                reasons.append("ADEQUATE_EVIDENCE_RELEVANCE")

        # 3. Inter-Candidate Agreement (s_top, s_mean)
        sim_mat = similarity_matrix
        if sim_mat is None:
            sim_mat = compute_candidate_similarities(candidates, embedder=self.embedder)

        if K == 1:
            # Single candidate cannot measure consensus across independent runs
            s_top = 0.50
            s_mean = 0.50
            reasons.append("SINGLE_CANDIDATE_RUN")
        else:
            s_top = max(sim_mat.candidate_mean_agreements) if sim_mat.candidate_mean_agreements else 0.0
            s_mean = sim_mat.mean_pairwise_agreement
            if s_mean >= 0.70:
                reasons.append("HIGH_CANDIDATE_CONSENSUS")
            elif s_mean <= 0.40:
                reasons.append("HIGH_CANDIDATE_DIVERGENCE")

        # 4. Claim Alignment & Contradictions
        align_res = claim_alignment
        if align_res is None:
            align_res = self.claim_aligner.align_and_verify(candidates, evidence)

        support_ratio = align_res.support_ratio
        contradiction_detected = align_res.contradiction_detected

        if not evidence:
            support_ratio = 0.0

        if contradiction_detected:
            reasons.append("CONTRADICTION_DETECTED")
        if support_ratio < 0.40 and evidence:
            reasons.append("SUBSTANTIAL_UNSUPPORTED_CLAIMS")
        elif support_ratio >= 0.75:
            reasons.append("STRONG_CLAIM_GROUNDING")

        # 5. Core Formulation
        raw_formula = (self.w_top * s_top) + (self.w_mean * s_mean) + (self.w_rel * r_rel)

        # Apply grounding modulation: agreement on hallucinated/unsupported facts is penalized
        # If support_ratio is 0 (e.g. no evidence), confidence must collapse to 0
        if not evidence or support_ratio == 0.0:
            final_conf = 0.0
            reasons.append("CONFIDENCE_ZEROED_NO_GROUNDING")
        else:
            # Scale by grounding factor: 50% fixed weight, 50% modulated by support_ratio
            grounding_multiplier = 0.40 + 0.60 * support_ratio
            conf_grounded = raw_formula * grounding_multiplier

            # Subtract penalty if direct contradictions or conflicting dates/obligations exist
            if contradiction_detected:
                conf_grounded = max(0.0, conf_grounded - self.contradiction_penalty)

            final_conf = conf_grounded

        # Bound strictly to [0.0, 1.0]
        final_conf = round(max(0.0, min(1.0, float(final_conf))), 4)
        final_uncertainty = round(1.0 - final_conf, 4)

        # 6. Semantic Answerability & Missing Information Check
        from legal_uq.generation.generator import detect_missing_information, check_evidence_answerability

        missing_info_declared = (
            getattr(align_res, "missing_information_detected", False)
            or any(
                c.metadata.get("missing_information_detected", False) or detect_missing_information(c.answer_text)
                for c in candidates
            )
        )

        query_unanswerable = False
        if query and evidence:
            is_ans, _, _ = check_evidence_answerability(query, evidence)
            if not is_ans:
                query_unanswerable = True

        is_unanswerable_missing = missing_info_declared or query_unanswerable

        if is_unanswerable_missing:
            reasons.append("REQUESTED_FACT_ABSENT_FROM_EVIDENCE")
            if missing_info_declared:
                reasons.append("EXPLICIT_MISSING_INFORMATION_REPORTED")
            is_answerable = False
            missing_information_detected = True
            answerability_verdict = "INSUFFICIENT_EVIDENCE"
            support_ratio = 0.0
            final_conf = 0.0
            final_uncertainty = 1.0
        else:
            is_answerable = True
            missing_information_detected = False
            answerability_verdict = "ANSWERABLE"

        return UncertaintyMetrics(
            confidence_score=final_conf,
            uncertainty_score=final_uncertainty,
            s_top=round(s_top, 4),
            s_mean=round(s_mean, 4),
            r_rel=round(r_rel, 4),
            support_ratio=round(support_ratio, 4),
            contradiction_detected=contradiction_detected,
            is_answerable=is_answerable,
            missing_information_detected=missing_information_detected,
            answerability_verdict=answerability_verdict,
            reason_codes=reasons,
            similarity_matrix=sim_mat,
            claim_alignment=align_res,
            metadata={
                "formula": "0.50*s_top + 0.30*s_mean + 0.20*r_rel (with grounding modulation)",
                "raw_formula_value": round(raw_formula, 4),
                "num_candidates": K,
                "contradiction_details": align_res.contradiction_details,
                "candidate_consensus": round(s_mean, 4),
                "is_answerable": is_answerable,
                "missing_information_detected": missing_information_detected,
            },
        )

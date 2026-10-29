"""Uncertainty and Confidence Estimation Module for Legal Assistance."""

from legal_uq.uncertainty.similarity import (
    PairwiseSimilarity,
    SimilarityMatrix,
    compute_candidate_similarities,
    jaccard_similarity,
    rouge_l_similarity,
)
from legal_uq.uncertainty.claim_aligner import (
    ClaimAlignmentResult,
    ClaimVerdict,
    ExtractedClaim,
    HeuristicClaimAligner,
)
from legal_uq.uncertainty.aggregator import (
    UncertaintyAggregator,
    UncertaintyMetrics,
)

__all__ = [
    "PairwiseSimilarity",
    "SimilarityMatrix",
    "compute_candidate_similarities",
    "jaccard_similarity",
    "rouge_l_similarity",
    "ExtractedClaim",
    "ClaimVerdict",
    "ClaimAlignmentResult",
    "HeuristicClaimAligner",
    "UncertaintyAggregator",
    "UncertaintyMetrics",
]


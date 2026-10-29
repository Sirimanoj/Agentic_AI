"""Lexical and semantic similarity analysis across candidate responses.

Computes:
1. Lexical Jaccard token set similarity in [0.0, 1.0].
2. ROUGE-L sequence alignment score based on Longest Common Subsequence (LCS) F1 in [0.0, 1.0].
3. Semantic cosine similarity using the shared Embedder interface, clamped to [0.0, 1.0].
4. Pairwise similarity matrices and candidate-level mean agreement scores.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Sequence

import numpy as np

from legal_uq.generation.generator import CandidateResponse
from legal_uq.retrieval.vector_store import (
    CachedSentenceTransformerEmbedder,
    Embedder,
)

_TOKEN_REGEX = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    """Extract lowercased word tokens."""
    return _TOKEN_REGEX.findall(text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity in [0.0, 1.0].

    J(A, B) = |A ∩ B| / |A ∪ B|.
    If both strings are empty, returns 1.0. If only one is empty, returns 0.0.
    """
    tokens1 = set(_tokenize(text1))
    tokens2 = set(_tokenize(text2))

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return round(float(intersection / union), 4)


def rouge_l_similarity(text1: str, text2: str) -> float:
    """Compute ROUGE-L F1 similarity score based on Longest Common Subsequence.

    Returns value bounded in [0.0, 1.0].
    If both are empty, returns 1.0. If one is empty, returns 0.0.
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    m = len(tokens1)
    n = len(tokens2)

    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    # Memory-efficient 2-row LCS dynamic programming
    prev_row = [0] * (n + 1)
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        t1 = tokens1[i - 1]
        for j in range(1, n + 1):
            if t1 == tokens2[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
            else:
                curr_row[j] = max(prev_row[j], curr_row[j - 1])
        prev_row = list(curr_row)

    lcs_len = curr_row[n]

    recall = lcs_len / m
    precision = lcs_len / n

    if recall + precision == 0:
        return 0.0

    f1 = (2.0 * recall * precision) / (recall + precision)
    return round(float(f1), 4)


def semantic_similarity(text1: str, text2: str, embedder: Embedder) -> float:
    """Compute semantic cosine similarity using the provided Embedder.

    Transformation & Range:
        Raw cosine similarity of unit vectors lies in [-1.0, 1.0].
        To ensure higher values monotonically indicate greater agreement and avoid
        negative similarity artifacts, we clamp the score to [0.0, 1.0] using:
        score = max(0.0, float(dot_product)).
        A score of 1.0 denotes identical semantic orientation; 0.0 denotes orthogonal
        or opposing direction.
    """
    clean1 = text1.strip()
    clean2 = text2.strip()

    if not clean1 and not clean2:
        return 1.0
    if not clean1 or not clean2:
        return 0.0

    vectors = embedder.embed_texts([clean1, clean2])
    v1 = vectors[0]
    v2 = vectors[1]

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0

    raw_cosine = float(np.dot(v1, v2) / (n1 * n2))
    # Non-negative clamping for agreement representation
    clamped = max(0.0, min(1.0, raw_cosine))
    return round(clamped, 4)


@dataclass(frozen=True)
class PairwiseSimilarity:
    """Detailed similarity metrics between two candidate responses."""
    idx1: int
    idx2: int
    jaccard: float
    rouge_l: float
    semantic: float
    composite: float  # Weighted combination in [0.0, 1.0]


@dataclass
class SimilarityMatrix:
    """Complete agreement analysis across all generated candidates."""
    num_candidates: int
    pairs: List[PairwiseSimilarity] = field(default_factory=list)
    jaccard_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rouge_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    semantic_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    composite_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mean_pairwise_agreement: float = 0.0
    candidate_mean_agreements: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def compute_candidate_similarities(
    candidates: Sequence[CandidateResponse],
    embedder: Optional[Embedder] = None,
    weight_jaccard: float = 0.30,
    weight_rouge: float = 0.35,
    weight_semantic: float = 0.35,
) -> SimilarityMatrix:
    """Compute all pairwise lexical and semantic agreement metrics across candidates.

    Args:
        candidates: Sequence of generated CandidateResponse instances.
        embedder: Embedding model instance. Defaults to CachedSentenceTransformerEmbedder.
        weight_jaccard: Weight for token Jaccard similarity.
        weight_rouge: Weight for ROUGE-L sequence alignment.
        weight_semantic: Weight for semantic cosine similarity.

    Returns:
        SimilarityMatrix containing pairwise and aggregate agreement statistics.
    """
    K = len(candidates)
    if K == 0:
        return SimilarityMatrix(
            num_candidates=0,
            mean_pairwise_agreement=0.0,
            candidate_mean_agreements=[],
            metadata={"status": "NO_CANDIDATES"},
        )

    if K == 1:
        # A single candidate has perfect self-agreement
        identity = np.ones((1, 1), dtype=np.float32)
        return SimilarityMatrix(
            num_candidates=1,
            jaccard_matrix=identity,
            rouge_matrix=identity,
            semantic_matrix=identity,
            composite_matrix=identity,
            mean_pairwise_agreement=1.0,
            candidate_mean_agreements=[1.0],
            metadata={"status": "SINGLE_CANDIDATE"},
        )

    resolved_embedder = embedder if embedder is not None else CachedSentenceTransformerEmbedder()

    # Pre-compute embeddings for all candidates in one batch
    texts = [c.answer_text for c in candidates]
    embeddings = resolved_embedder.embed_texts(texts)

    # Normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    jaccard_mat = np.eye(K, dtype=np.float32)
    rouge_mat = np.eye(K, dtype=np.float32)
    semantic_mat = np.eye(K, dtype=np.float32)
    composite_mat = np.eye(K, dtype=np.float32)

    total_w = weight_jaccard + weight_rouge + weight_semantic
    w_j = weight_jaccard / total_w
    w_r = weight_rouge / total_w
    w_s = weight_semantic / total_w

    pairs: List[PairwiseSimilarity] = []
    pairwise_composites = []

    for i in range(K):
        for j in range(i + 1, K):
            text_i = candidates[i].answer_text
            text_j = candidates[j].answer_text

            j_score = jaccard_similarity(text_i, text_j)
            r_score = rouge_l_similarity(text_i, text_j)

            # Cosine from batch dot product
            cos_sim = float(np.dot(embeddings[i], embeddings[j]))
            s_score = round(max(0.0, min(1.0, cos_sim)), 4)

            comp_score = round(w_j * j_score + w_r * r_score + w_s * s_score, 4)

            jaccard_mat[i, j] = jaccard_mat[j, i] = j_score
            rouge_mat[i, j] = rouge_mat[j, i] = r_score
            semantic_mat[i, j] = semantic_mat[j, i] = s_score
            composite_mat[i, j] = composite_mat[j, i] = comp_score

            pair_res = PairwiseSimilarity(
                idx1=i,
                idx2=j,
                jaccard=j_score,
                rouge_l=r_score,
                semantic=s_score,
                composite=comp_score,
            )
            pairs.append(pair_res)
            pairwise_composites.append(comp_score)

    mean_pairwise = float(np.mean(pairwise_composites)) if pairwise_composites else 0.0

    # Mean agreement of candidate i with all other candidates j != i
    cand_means = []
    for i in range(K):
        other_scores = [composite_mat[i, j] for j in range(K) if j != i]
        cand_means.append(round(float(np.mean(other_scores)), 4) if other_scores else 1.0)

    return SimilarityMatrix(
        num_candidates=K,
        pairs=pairs,
        jaccard_matrix=jaccard_mat,
        rouge_matrix=rouge_mat,
        semantic_matrix=semantic_mat,
        composite_matrix=composite_mat,
        mean_pairwise_agreement=round(mean_pairwise, 4),
        candidate_mean_agreements=cand_means,
        metadata={"weights": {"jaccard": w_j, "rouge": w_r, "semantic": w_s}},
    )


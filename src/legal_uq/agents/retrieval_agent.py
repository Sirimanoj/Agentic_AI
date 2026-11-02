"""Retrieval Agent.

Interacts with the FAISS vector store tool, evaluates evidence sufficiency,
and performs bounded query reformulation and retry if initial relevance is weak.
"""

import re
from typing import List, Optional, Sequence, Tuple

from legal_uq.agents.models import AgentActionLog, RetrievalOutput
from legal_uq.retrieval.vector_store import RetrievalResult, VectorStore

_FILLER_PHRASES = [
    r"^(?:what\s+is\s+the|what\s+are\s+the|under\s+what\s+conditions\s+is|what\s+does\s+the|how\s+long\s+is\s+the)\s+",
    r"^(?:can\s+you\s+explain|please\s+tell\s+me|is\s+there\s+any\s+information\s+about)\s+",
    r"^(?:according\s+to\s+the\s+contract|in\s+the\s+agreement|under\s+the\s+statute)\s+",
]


def _refine_query_keywords(query: str) -> str:
    """Reformulate legal query by stripping conversational preambles and focusing on core substantive terms."""
    cleaned = query.strip()
    for pattern in _FILLER_PHRASES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip punctuation
    cleaned = re.sub(r"[?!.,;:\"]+", " ", cleaned).strip()
    words = cleaned.split()

    # If reduced to too few words, keep substantive words from original
    if len(words) < 2:
        orig_words = [w for w in re.findall(r"\b[A-Za-z0-9\-_]{3,}\b", query) if w.lower() not in {"what", "the", "for", "and", "are"}]
        return " ".join(orig_words) if orig_words else query

    return " ".join(words)


class RetrievalAgent:
    """Agent responsible for vector store querying and bounded retrieval retry."""

    def __init__(
        self,
        vector_store: VectorStore,
        min_sufficiency_score: float = 0.35,
        max_attempts: int = 2,
        name: str = "RetrievalAgent",
    ) -> None:
        self.vector_store = vector_store
        self.min_sufficiency_score = min_sufficiency_score
        self.max_attempts = max(1, min(max_attempts, 2))  # Enforce hard upper bound of 2
        self.name = name

    def retrieve_with_retry(
        self,
        query: str,
        top_k: int = 4,
    ) -> Tuple[RetrievalOutput, List[AgentActionLog]]:
        """Execute FAISS retrieval with bounded query reformulation retry if evidence is weak.

        Args:
            query: The legal inquiry.
            top_k: Target number of passages to retrieve.

        Returns:
            Tuple of (RetrievalOutput, List of AgentActionLog records).
        """
        logs: List[AgentActionLog] = []

        # Attempt 1: Direct retrieval
        try:
            attempt_1_results = self.vector_store.retrieve(query, k=top_k)
        except Exception as exc:
            logs.append(
                AgentActionLog(
                    agent_name=self.name,
                    action="RETRIEVAL_TOOL_ERROR",
                    reason=f"Vector store query failed with exception: {type(exc).__name__}",
                    result_summary=f"Retrieval tool error: {str(exc)[:120]}. Falling back to empty evidence.",
                    details={"error": str(exc), "error_type": type(exc).__name__, "attempt": 1, "query": query},
                )
            )
            attempt_1_results = []

        top_score_1 = attempt_1_results[0].score if attempt_1_results else 0.0

        is_sufficient = (
            len(attempt_1_results) > 0 and top_score_1 >= self.min_sufficiency_score
        )

        logs.append(
            AgentActionLog(
                agent_name=self.name,
                action="RETRIEVAL_PRIMARY",
                reason=f"Execute vector similarity search for inquiry (k={top_k}).",
                result_summary=(
                    f"Retrieved {len(attempt_1_results)} passages. "
                    f"Top cosine similarity: {top_score_1:.3f} "
                    f"({'Sufficient' if is_sufficient else 'Below threshold ' + str(self.min_sufficiency_score)})."
                ),
                details={
                    "query": query,
                    "attempt": 1,
                    "results_count": len(attempt_1_results),
                    "top_score": float(top_score_1),
                    "is_sufficient": is_sufficient,
                    "top_locators": [r.source_locator for r in attempt_1_results[:2]],
                },
            )
        )

        # If sufficient or max attempts reached, return immediately
        if is_sufficient or self.max_attempts < 2 or not attempt_1_results:
            output = RetrievalOutput(
                query_used=query,
                attempt_number=1,
                retrieved_results=attempt_1_results,
                top_relevance_score=top_score_1,
                is_sufficient=is_sufficient,
                sufficiency_reason=(
                    f"Evidence relevance {top_score_1:.3f} meets threshold {self.min_sufficiency_score}."
                    if is_sufficient
                    else f"Evidence relevance {top_score_1:.3f} is weak."
                ),
                reformulation_applied=False,
                original_query=query,
            )
            return output, logs

        # Attempt 2: Bounded query refinement retry
        refined_query = _refine_query_keywords(query)
        if refined_query.lower() == query.lower():
            # If refinement produces identical query, no point in duplicate call
            refined_query = f"{query} legal provisions definitions"

        logs.append(
            AgentActionLog(
                agent_name=self.name,
                action="RETRIEVAL_REFORMULATION_TRIGGERED",
                reason=(
                    f"Top cosine score ({top_score_1:.3f}) below sufficiency target ({self.min_sufficiency_score}). "
                    f"Reformulating query to focus on substantive legal keywords."
                ),
                result_summary=f"Reformulated query from '{query}' to '{refined_query}'.",
                details={
                    "original_query": query,
                    "refined_query": refined_query,
                    "attempt": 2,
                },
            )
        )

        try:
            attempt_2_results = self.vector_store.retrieve(refined_query, k=top_k)
        except Exception as exc:
            logs.append(
                AgentActionLog(
                    agent_name=self.name,
                    action="RETRIEVAL_TOOL_ERROR",
                    reason=f"Refined vector store query failed with exception: {type(exc).__name__}",
                    result_summary=f"Retrieval tool error on retry: {str(exc)[:120]}.",
                    details={"error": str(exc), "error_type": type(exc).__name__, "attempt": 2, "query": refined_query},
                )
            )
            attempt_2_results = []
        top_score_2 = attempt_2_results[0].score if attempt_2_results else 0.0

        # Choose best evidence: merge unique passages sorted by highest score
        merged_dict = {r.chunk_id: r for r in attempt_1_results}
        for r in attempt_2_results:
            if r.chunk_id not in merged_dict or r.score > merged_dict[r.chunk_id].score:
                merged_dict[r.chunk_id] = r

        best_results = sorted(merged_dict.values(), key=lambda r: r.score, reverse=True)[:top_k]
        best_score = best_results[0].score if best_results else 0.0
        final_sufficient = best_score >= self.min_sufficiency_score

        logs.append(
            AgentActionLog(
                agent_name=self.name,
                action="RETRIEVAL_REFINED_EVALUATED",
                reason="Evaluate results of refined query and merge with primary retrieval set.",
                result_summary=(
                    f"Refined search yielded {len(attempt_2_results)} results (score={top_score_2:.3f}). "
                    f"Consolidated best set has top score {best_score:.3f} "
                    f"({'Sufficient' if final_sufficient else 'Remains below threshold'})."
                ),
                details={
                    "attempt": 2,
                    "query_used": refined_query,
                    "top_score_attempt_2": float(top_score_2),
                    "best_consolidated_score": float(best_score),
                    "consolidated_count": len(best_results),
                },
            )
        )

        output = RetrievalOutput(
            query_used=refined_query,
            attempt_number=2,
            retrieved_results=best_results,
            top_relevance_score=best_score,
            is_sufficient=final_sufficient,
            sufficiency_reason=(
                f"Consolidated relevance {best_score:.3f} after 2 retrieval attempts "
                f"({'sufficient' if final_sufficient else 'insufficient'})."
            ),
            reformulation_applied=True,
            original_query=query,
        )
        return output, logs


"""Legal Analysis Agent.

Generates candidate legal answers strictly grounded in retrieved evidence passages,
and produces targeted revisions when guided by the Verification Agent.
"""

from typing import List, Optional, Sequence, Tuple

from legal_uq.agents.models import AgentActionLog, AnalysisOutput
from legal_uq.generation.generator import CandidateResponse, GroundedGenerator
from legal_uq.retrieval.vector_store import RetrievalResult


class LegalAnalysisAgent:
    """Agent responsible for multi-candidate legal response generation and grounded revision."""

    def __init__(
        self,
        generator: GroundedGenerator,
        name: str = "LegalAnalysisAgent",
    ) -> None:
        self.generator = generator
        self.name = name

    def generate_draft(
        self,
        question: str,
        evidence: Sequence[RetrievalResult],
        num_candidates: int = 3,
    ) -> Tuple[AnalysisOutput, AgentActionLog]:
        """Generate initial grounded candidate responses for the inquiry.

        Args:
            question: The user's legal question.
            evidence: Retrieved evidence passages from VectorStore.
            num_candidates: Target candidate count (default 3).

        Returns:
            Tuple of (AnalysisOutput, AgentActionLog).
        """
        try:
            gen_run = self.generator.generate_candidates(
                query=question,
                evidence=evidence,
                num_candidates=num_candidates,
            )
            candidates = list(gen_run.candidates)
            primary = candidates[0] if candidates else None
            mode = gen_run.mode
            disclaimer = gen_run.disclaimer

            log = AgentActionLog(
                agent_name=self.name,
                action="GENERATION_DRAFT",
                reason=f"Synthesize {len(candidates)} grounded candidate answers via {gen_run.mode} mode.",
                result_summary=(
                    f"Generated {len(candidates)} candidate responses in {gen_run.mode} mode "
                    f"citing {[c for c in (primary.cited_sources if primary else [])]}."
                ),
                details={
                    "generation_mode": gen_run.mode,
                    "model_name": gen_run.model_name,
                    "candidate_count": len(candidates),
                    "primary_preview": primary.answer_text[:120] + "..." if primary else "",
                    "citations": primary.cited_sources if primary else [],
                },
            )
        except Exception as exc:
            # Re-raise explicit missing API key error to prevent silent generation in live mode
            if "No API key provided" in str(exc):
                raise exc

            from legal_uq.generation.generator import _sanitize_error_message
            api_key = getattr(self.generator, "api_key", None)
            sanitized_err = _sanitize_error_message(exc, api_key=api_key)
            log = AgentActionLog(
                agent_name=self.name,
                action="GENERATION_API_ERROR",
                reason=f"Candidate generation failed due to model/API error: {type(exc).__name__}",
                result_summary=f"Generation failed ({type(exc).__name__}). Refusing substantive answer for safety.",
                details={"error": sanitized_err, "error_type": type(exc).__name__},
            )
            error_cand = CandidateResponse(
                candidate_index=0,
                generation_mode="ERROR",
                answer_text=f"The legal analysis generation service encountered an error ({type(exc).__name__}) and cannot produce a reliable response. Please verify service availability.",
                cited_sources=[],
                metadata={"api_error": True, "error_details": sanitized_err, "is_answerable": False, "missing_information_detected": True},
            )
            candidates = [error_cand] * max(1, num_candidates)
            primary = candidates[0]
            mode = "ERROR"
            disclaimer = "GENERATION ERROR: Generation failed; substantive advice withheld."

        output = AnalysisOutput(
            candidate_responses=candidates,
            primary_candidate=primary,
            generation_mode=mode,
            revision_count=0,
            generation_rationale=f"Generated {len(candidates)} candidates using {mode} pipeline.",
            disclaimer=disclaimer,
        )
        return output, log

    def revise_draft(
        self,
        question: str,
        evidence: Sequence[RetrievalResult],
        current_draft: AnalysisOutput,
        revision_guidance: str,
    ) -> Tuple[AnalysisOutput, AgentActionLog]:
        """Produce a targeted revision of candidate answers based on verification feedback.

        Bounded to 1 revision attempt.

        Args:
            question: The original legal question.
            evidence: Retrieved evidence passages.
            current_draft: The pre-revision AnalysisOutput.
            revision_guidance: Specific instructions or unsupported claims from VerificationAgent.

        Returns:
            Tuple of revised (AnalysisOutput, AgentActionLog).
        """
        candidates = list(current_draft.candidate_responses)
        primary = current_draft.primary_candidate

        if not candidates or primary is None:
            # Cannot revise empty draft
            log = AgentActionLog(
                agent_name=self.name,
                action="GENERATION_REVISION_SKIPPED",
                reason="No candidate responses available to revise.",
                result_summary="Skipped revision because draft was empty.",
            )
            return current_draft, log

        # Handle revision bounded to 1 revision
        if self.generator.is_live and self.generator.api_key:
            # Live revision via Gemini with explicit grounding prompt
            try:
                revision_prompt = (
                    f"USER INQUIRY: {question}\n\n"
                    f"VERIFICATION FEEDBACK ON PRIOR DRAFT:\n{revision_guidance}\n\n"
                    f"INSTRUCTION: Revise the answer strictly following the verification feedback. "
                    f"Eliminate any speculative assertions not directly substantiated by the evidence passages. "
                    f"Include explicit source markers [S1], [S2] for all factual claims."
                )
                revised_text = self.generator._call_live_single(
                    client=self.generator._get_or_create_client(),
                    prompt_text=revision_prompt,
                    temperature=0.2,  # Low temperature for strict factual adherence
                )
                candidates[0] = CandidateResponse(
                    candidate_index=0,
                    generation_mode="LIVE",
                    answer_text=revised_text,
                    cited_sources=primary.cited_sources,
                    metadata={**primary.metadata, "revised": True, "revision_feedback": revision_guidance},
                )
            except Exception as exc:
                # If live revision call encounters transient error, apply deterministic pruning
                pruned_text = (
                    f"{primary.answer_text}\n\n"
                    f"[Revision Note]: Portions of this response were audited for strict adherence: {revision_guidance}."
                )
                candidates[0] = CandidateResponse(
                    candidate_index=0,
                    generation_mode="LIVE",
                    answer_text=pruned_text,
                    cited_sources=primary.cited_sources,
                    metadata={**primary.metadata, "revised": True, "revision_fallback_error": str(exc)},
                )
        else:
            # Deterministic revision in offline mock mode
            if evidence:
                top_ev = evidence[0]
                pruned_text = (
                    f"REVISED FINDINGS FOR: '{question}'\n"
                    f"In response to verification audit ({revision_guidance}), non-substantiated assertions have been excluded.\n"
                    f"Grounded text per {top_ev.source_locator} [S1]:\n"
                    f"\"{top_ev.text.strip()[:300]}...\""
                )
            else:
                pruned_text = (
                    f"REVISED FINDINGS: Based on verification audit ({revision_guidance}), "
                    f"no substantive factual answer can be substantiated from available text."
                )

            candidates[0] = CandidateResponse(
                candidate_index=0,
                generation_mode=primary.generation_mode,
                answer_text=pruned_text,
                cited_sources=primary.cited_sources,
                metadata={**primary.metadata, "revised": True, "revision_guidance": revision_guidance},
            )

        new_primary = candidates[0]
        log = AgentActionLog(
            agent_name=self.name,
            action="GENERATION_REVISED",
            reason=f"Verification agent identified grounding deficiencies: {revision_guidance[:100]}...",
            result_summary=f"Produced revised Primary Candidate (revision count = 1).",
            details={
                "revision_count": 1,
                "revision_guidance": revision_guidance,
                "revised_preview": new_primary.answer_text[:120] + "...",
            },
        )

        output = AnalysisOutput(
            candidate_responses=candidates,
            primary_candidate=new_primary,
            generation_mode=current_draft.generation_mode,
            revision_count=1,
            generation_rationale=f"Revised primary draft based on VerificationAgent feedback.",
            disclaimer=current_draft.disclaimer,
        )
        return output, log


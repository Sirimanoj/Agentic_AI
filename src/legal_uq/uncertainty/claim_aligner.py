"""Claim-level alignment and heuristic evidence grounding verification.

Decomposes candidate answers into constituent legal claims and verifies whether
each claim is substantiated by retrieved document passages. Detects direct contradictions
(opposing negations, clashing quantities/dates, and incompatible obligations).

IMPORTANT METHODOLOGICAL DISCLOSURE:
    As no local NLI model (e.g. DeBERTa-v3-NLI) is pre-cached on this system, this module
    operates in explicit HEURISTIC mode. It utilizes conservative syntactic, lexical-overlap,
    and semantic polarity rules. Heuristic alignment serves as a transparent baseline;
    lexical overlap alone does NOT prove formal semantic entailment.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from legal_uq.generation.generator import CandidateResponse
from legal_uq.retrieval.vector_store import RetrievalResult

# Patterns for extracting numbers, durations, and financial amounts
_QUANTITY_REGEX = re.compile(
    r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(days?|months?|years?|weeks?|hours?|rupees?|inr|usd|dollars?|percent|%)\b",
    re.IGNORECASE,
)

# Negation and prohibition markers common in legal instruments
# Direct antonym/negation pairs in legal assertions
_OPPOSING_POLARITY_PAIRS = [
    (re.compile(r"\bshall\s+not\b", re.I), re.compile(r"\bshall(?!\s+not)\b", re.I)),
    (re.compile(r"\bmust\s+not\b", re.I), re.compile(r"\bmust(?!\s+not)\b", re.I)),
    (re.compile(r"\bmay\s+not\b", re.I), re.compile(r"\bmay(?!\s+not)\b", re.I)),
    (re.compile(r"\bprohibited\b", re.I), re.compile(r"\bpermitted\b", re.I)),
    (re.compile(r"\bvoid\b", re.I), re.compile(r"\bvalid\b", re.I)),
    (re.compile(r"\bunenforceable\b", re.I), re.compile(r"\benforceable\b", re.I)),
    (re.compile(r"\bnot\s+liable\b", re.I), re.compile(r"\bliable\b", re.I)),
    (re.compile(r"\binvalid\b", re.I), re.compile(r"\bvalid\b", re.I)),
]


class ClaimVerdict(str, Enum):
    """Categorical verdict for a claim against retrieved evidence."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


@dataclass(frozen=True)
class ExtractedClaim:
    """An individual atomic clause or proposition asserted in an answer."""
    claim_id: str
    candidate_index: int
    text: str
    cited_sources: List[str]


@dataclass(frozen=True)
class ClaimEvidenceMatch:
    """Detailed provenance match between an extracted claim and an evidence passage."""
    claim: ExtractedClaim
    verdict: ClaimVerdict
    matched_chunk_id: Optional[str]
    matched_source_locator: Optional[str]
    overlap_score: float
    rationale: str


@dataclass
class ClaimAlignmentResult:
    """Aggregate claim-level alignment and contradiction report."""
    status: str = "HEURISTIC"  # Transparently declared as HEURISTIC
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    support_ratio: float = 0.0  # Supported / max(1, total_claims)
    contradiction_detected: bool = False
    contradiction_details: List[str] = field(default_factory=list)
    matches: List[ClaimEvidenceMatch] = field(default_factory=list)
    unsupported_claim_texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    missing_information_detected: bool = False
    is_answerable: bool = True

    @property
    def contradiction_found(self) -> bool:
        """Backward-compatibility alias for contradiction_detected."""
        return self.contradiction_detected


class HeuristicClaimAligner:
    """Rule-based legal claim decomposer and evidence alignment verifier.

    Evaluates grounding via asymmetric vocabulary containment: the fraction of
    content tokens in the extracted candidate claim that appear in the retrieved passage.
    """

    def __init__(
        self,
        support_overlap_threshold: float = 0.55,
        partial_support_threshold: float = 0.35,
    ) -> None:
        self.support_overlap_threshold = support_overlap_threshold
        self.partial_support_threshold = partial_support_threshold

    def _split_into_claims(self, text: str, candidate_index: int) -> List[ExtractedClaim]:
        """Decompose an answer into discrete factual/legal propositions."""
        clean = text.strip()
        if not clean:
            return []

        # Remove boilerplate disclaimers and meta-notices before evaluating claims
        filtered_lines = []
        for line in clean.splitlines():
            l_str = line.strip()
            if not l_str:
                continue
            if l_str.startswith("DISCLAIMER:") or l_str.startswith("MOCK/DEMO MODE NOTICE:"):
                continue
            if l_str.startswith("Scope Limitation:") or l_str.startswith("Note:"):
                continue
            filtered_lines.append(l_str)

        filtered_text = " ".join(filtered_lines)
        if not filtered_text:
            return []

        # Split on legal enumerations, semicolons, and terminal punctuation
        raw_units = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“(\[])|;\s*(?=\([a-zA-Z0-9]+\))|\n+", filtered_text)
        claims: List[ExtractedClaim] = []
        c_counter = 0

        for unit in raw_units:
            u_clean = unit.strip()
            # Ignore tiny punctuation fragments or pure headers
            if len(u_clean) < 20 or len(u_clean.split()) < 4:
                continue

            # Extract source markers [S1], [S2] if referenced in claim
            sources = re.findall(r"\[(S\d+)\]", u_clean)

            claims.append(
                ExtractedClaim(
                    claim_id=f"c{candidate_index}_clm_{c_counter}",
                    candidate_index=candidate_index,
                    text=u_clean,
                    cited_sources=sources,
                )
            )
            c_counter += 1

        return claims

    def _check_inter_candidate_contradictions(
        self,
        candidates: Sequence[CandidateResponse],
    ) -> Tuple[bool, List[str]]:
        """Inspect pairs of candidates for incompatible statements or opposing polarities."""
        if len(candidates) < 2:
            return False, []

        contradictions: List[str] = []

        for i in range(len(candidates)):
            text_i = candidates[i].answer_text.lower()
            tokens_i = set(re.findall(r"\b\w+\b", text_i))
            quantities_i = dict(_QUANTITY_REGEX.findall(text_i))  # e.g. {'30': 'days'}

            for j in range(i + 1, len(candidates)):
                text_j = candidates[j].answer_text.lower()
                tokens_j = set(re.findall(r"\b\w+\b", text_j))
                quantities_j = dict(_QUANTITY_REGEX.findall(text_j))

                # 1. Quantity clash: same unit, conflicting numbers
                # e.g. Cand A says 30 days, Cand B says 60 days
                for val_i, unit_i in quantities_i.items():
                    for val_j, unit_j in quantities_j.items():
                        if unit_i == unit_j and val_i != val_j:
                            # Verify context overlap to avoid comparing unrelated numbers
                            overlap = len(tokens_i & tokens_j) / max(1, len(tokens_i | tokens_j))
                            shared_count = len(tokens_i & tokens_j)
                            if overlap > 0.15 or shared_count >= 4:
                                msg = (
                                    f"Incompatible quantities between Candidate {i} ({val_i} {unit_i}) "
                                    f"and Candidate {j} ({val_j} {unit_j})."
                                )
                                contradictions.append(msg)

                # 2. Opposing legal polarity on overlapping legal subject matter
                # Check explicit pairs: shall vs shall not, prohibited vs permitted, valid vs void
                shared_content_words = {
                    w for w in (tokens_i & tokens_j)
                    if len(w) > 3 and w not in {"with", "from", "that", "this", "under", "party", "shall", "must"}
                }
                if len(shared_content_words) >= 3:
                    for neg_pat, pos_pat in _OPPOSING_POLARITY_PAIRS:
                        has_neg_i = bool(neg_pat.search(text_i))
                        has_pos_j = bool(pos_pat.search(text_j))
                        has_neg_j = bool(neg_pat.search(text_j))
                        has_pos_i = bool(pos_pat.search(text_i))

                        if (has_neg_i and has_pos_j and not has_neg_j) or (has_neg_j and has_pos_i and not has_neg_i):
                            msg = (
                                f"Opposing legal polarity detected between Candidate {i} and Candidate {j} "
                                f"regarding shared terms: {list(shared_content_words)[:4]}."
                            )
                            contradictions.append(msg)
                            break

        return (len(contradictions) > 0), contradictions

    def align_and_verify(
        self,
        candidates: Sequence[CandidateResponse],
        evidence: Sequence[RetrievalResult],
    ) -> ClaimAlignmentResult:
        """Analyze claim support against evidence and detect inter-candidate conflicts.

        Args:
            candidates: Generated candidate responses.
            evidence: Retrieved passages from VectorStore.

        Returns:
            ClaimAlignmentResult detailing support counts, ratios, and contradictions.
        """
        has_contradiction, contradiction_details = self._check_inter_candidate_contradictions(candidates)

        # Check if candidate responses explicitly declare that requested information is absent
        from legal_uq.generation.generator import detect_missing_information
        is_missing_declared = any(
            c.metadata.get("missing_information_detected", False) or detect_missing_information(c.answer_text)
            for c in candidates
        )

        # Extract claims primarily from candidate 0 (primary response), plus others
        all_claims: List[ExtractedClaim] = []
        for c in candidates:
            all_claims.extend(self._split_into_claims(c.answer_text, c.candidate_index))

        if not all_claims:
            return ClaimAlignmentResult(
                status="HEURISTIC",
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                support_ratio=0.0,
                contradiction_detected=has_contradiction,
                contradiction_details=contradiction_details,
                metadata={"info": "No claims extracted from candidates"},
                missing_information_detected=is_missing_declared,
                is_answerable=not is_missing_declared,
            )

        # Case 0: Candidate explicitly identifies that requested information is missing/absent
        if is_missing_declared:
            matches = [
                ClaimEvidenceMatch(
                    claim=clm,
                    verdict=ClaimVerdict.UNSUPPORTED,
                    matched_chunk_id=evidence[0].chunk_id if evidence else None,
                    matched_source_locator=evidence[0].source_locator if evidence else None,
                    overlap_score=0.0,
                    rationale="Candidate response explicitly indicates that requested legal information is absent from the indexed text.",
                )
                for clm in all_claims
            ]
            return ClaimAlignmentResult(
                status="HEURISTIC",
                total_claims=len(all_claims),
                supported_claims=0,
                unsupported_claims=len(all_claims),
                contradicted_claims=0,
                support_ratio=0.0,
                contradiction_detected=has_contradiction,
                contradiction_details=contradiction_details,
                matches=matches,
                unsupported_claim_texts=[
                    "Requested legal information was not found in the indexed documents."
                ],
                metadata={"reason": "MISSING_INFORMATION_DECLARED", "is_answerable": False},
                missing_information_detected=True,
                is_answerable=False,
            )

        # Case 1: Empty evidence -> all claims are strictly UNSUPPORTED
        if not evidence or all(not e.text.strip() for e in evidence):
            matches = [
                ClaimEvidenceMatch(
                    claim=clm,
                    verdict=ClaimVerdict.UNSUPPORTED,
                    matched_chunk_id=None,
                    matched_source_locator=None,
                    overlap_score=0.0,
                    rationale="No evidence passages available in uploaded documents.",
                )
                for clm in all_claims
            ]
            return ClaimAlignmentResult(
                status="HEURISTIC",
                total_claims=len(all_claims),
                supported_claims=0,
                unsupported_claims=len(all_claims),
                contradicted_claims=0,
                support_ratio=0.0,
                contradiction_detected=has_contradiction,
                contradiction_details=contradiction_details,
                matches=matches,
                unsupported_claim_texts=[c.text for c in all_claims],
                metadata={"reason": "NO_EVIDENCE_PROVIDED"},
            )

        # Case 2: Evidence exists -> measure grounding per claim
        matches: List[ClaimEvidenceMatch] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0
        unsupported_texts: List[str] = []

        evidence_token_sets = [
            (e, set(re.findall(r"\b\w+\b", e.text.lower())))
            for e in evidence
            if e.text.strip()
        ]

        for claim in all_claims:
            claim_tokens = set(re.findall(r"\b\w+\b", claim.text.lower()))
            if not claim_tokens:
                continue

            best_overlap = 0.0
            best_ev: Optional[RetrievalResult] = None

            for ev, ev_tokens in evidence_token_sets:
                if not ev_tokens:
                    continue
                intersection = len(claim_tokens & ev_tokens)
                # Asymmetric containment: fraction of claim vocabulary verified in the retrieved evidence passage
                containment = intersection / len(claim_tokens) if claim_tokens else 0.0
                if containment > best_overlap:
                    best_overlap = containment
                    best_ev = ev

            # Determine verdict
            if best_ev is not None and best_overlap >= self.support_overlap_threshold:
                # Check for direct negation clash with best matching passage
                clm_text = claim.text.lower()
                ev_text = best_ev.text.lower()
                is_contradicted = False

                for neg_pat, pos_pat in _OPPOSING_POLARITY_PAIRS:
                    if (neg_pat.search(clm_text) and pos_pat.search(ev_text) and not neg_pat.search(ev_text)) or \
                       (neg_pat.search(ev_text) and pos_pat.search(clm_text) and not neg_pat.search(clm_text)):
                        is_contradicted = True
                        break

                if is_contradicted:
                    verdict = ClaimVerdict.CONTRADICTED
                    contradicted_count += 1
                    rationale = f"Opposing legal polarity relative to evidence {best_ev.source_locator}."
                else:
                    verdict = ClaimVerdict.SUPPORTED
                    supported_count += 1
                    rationale = f"Sufficient textual grounding in {best_ev.source_locator} (overlap={best_overlap:.2f})."

            elif best_ev is not None and best_overlap >= self.partial_support_threshold:
                verdict = ClaimVerdict.PARTIALLY_SUPPORTED
                supported_count += 1  # Count partial support toward grounded claims
                rationale = f"Partial textual overlap with {best_ev.source_locator} (overlap={best_overlap:.2f})."

            else:
                verdict = ClaimVerdict.UNSUPPORTED
                unsupported_count += 1
                unsupported_texts.append(claim.text)
                rationale = f"Insufficient overlap with retrieved evidence (max overlap={best_overlap:.2f})."

            matches.append(
                ClaimEvidenceMatch(
                    claim=claim,
                    verdict=verdict,
                    matched_chunk_id=best_ev.chunk_id if best_ev else None,
                    matched_source_locator=best_ev.source_locator if best_ev else None,
                    overlap_score=round(best_overlap, 4),
                    rationale=rationale,
                )
            )

        total = len(all_claims)
        support_ratio = round(float(supported_count / max(1, total)), 4)

        if contradicted_count > 0:
            has_contradiction = True
            contradiction_details.append(
                f"{contradicted_count} claim(s) contradict retrieved source evidence."
            )

        return ClaimAlignmentResult(
            status="HEURISTIC",
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            support_ratio=support_ratio,
            contradiction_detected=has_contradiction,
            contradiction_details=contradiction_details,
            matches=matches,
            unsupported_claim_texts=unsupported_texts,
            metadata={
                "support_overlap_threshold": self.support_overlap_threshold,
                "partial_support_threshold": self.partial_support_threshold,
            },
        )

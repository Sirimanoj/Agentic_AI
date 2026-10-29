"""Grounded multi-candidate answer generation for legal document Q&A.

Supports:
1. Live Mode: Uses Google Gen AI SDK (Gemini) with configurable temperature (default 0.7)
   and bounded retries for transient errors.
2. Offline Mock Mode: Deterministic demonstration generator producing 3 structured candidate
   answers derived strictly from retrieved passages without hallucinations.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

if "PYTEST_CURRENT_TEST" not in os.environ:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

from legal_uq.retrieval.vector_store import RetrievalResult

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
SUPPORTED_GEMINI_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
)

LEGAL_DISCLAIMER = (
    "DISCLAIMER: This system is an informational research and evaluation prototype, "
    "not a substitute for qualified legal counsel. It does not provide legal advice or create "
    "an attorney-client relationship. Findings are grounded solely in the uploaded documents."
)

MOCK_AGREEMENT_WARNING = (
    "MOCK/DEMO MODE NOTICE: Generated candidate responses are deterministic demonstration outputs "
    "constructed from retrieved document passages because no GEMINI_API_KEY was provided. "
    "Agreement across these synthetic candidates MUST NEVER be interpreted as empirical proof "
    "of genuine LLM certainty or calibration."
)


@dataclass(frozen=True)
class CandidateResponse:
    """A single candidate response record."""
    candidate_index: int
    generation_mode: str  # "LIVE" or "MOCK"
    answer_text: str
    cited_sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRun:
    """A collection of candidate responses for a legal query."""
    query: str
    mode: str  # "LIVE" or "MOCK"
    model_name: str
    candidates: List[CandidateResponse]
    disclaimer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _build_evidence_context(evidence: Sequence[RetrievalResult]) -> str:
    """Format retrieved evidence passages into structured prompt context."""
    if not evidence:
        return "NO RETRIEVED EVIDENCE AVAILABLE."

    lines = []
    for idx, item in enumerate(evidence, start=1):
        page_str = f"Page {item.page_number}" if item.page_number is not None else "Page N/A"
        sec_str = f"Section: {item.section_title}" if item.section_title else "Section: N/A"
        lines.append(
            f"[S{idx}] Source: {item.source_locator} ({page_str}, {sec_str})\n"
            f"Passage Text:\n{item.text.strip()}\n"
        )
    return "\n".join(lines)


def _sanitize_error_message(error: Any, api_key: Optional[str] = None) -> str:
    """Ensure API key strings, query params, and bearer tokens are never exposed in error text."""
    msg = str(error)
    if api_key and api_key.strip() and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED_API_KEY]")
    # Redact common Google API key pattern (AIzaSy...)
    msg = re.sub(r"AIza[0-9A-Za-z\-_]{20,50}", "[REDACTED_API_KEY]", msg)
    # Redact Authorization: Bearer tokens
    msg = re.sub(r"Bearer\s+[A-Za-z0-9\-\._~+/]+=*", "Bearer [REDACTED_TOKEN]", msg)
    return msg


_QUERY_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", 
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", 
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", 
    "with", "would", "you", "your", "yours", "yourself", "yourselves", "current", "provide", 
    "provided", "state", "states", "document", "documents", "tell", "explain", "give", "please",
}

_MISSING_INFO_REGEX = re.compile(
    r"\b("
    r"contains?\s+no\s+information|"
    r"does\s+not\s+contain\s+(?:any\s+)?information|"
    r"does\s+not\s+(?:mention|provide|disclose|state|specify|address|reference)|"
    r"do\s+not\s+(?:contain|mention|provide|disclose|state|specify|address|reference)|"
    r"(?:there\s+is\s+)?no\s+information\s+(?:regarding|concerning|about|on|is|was|are|found|available|provided|contained|given)|"
    r"there\s+is\s+no\s+(?:mention|information|reference|record|provision|evidence|detail)|"
    r"not\s+(?:found|addressed|mentioned|provided|contained)\s+in\s+the\s+(?:provided|retrieved|uploaded|available|indexed)|"
    r"the\s+(?:provided|retrieved|uploaded|available|indexed)\s+(?:text|passages?|documents?|excerpts?|judgment|contract)\s+(?:does\s+not|do\s+not|contains?\s+no|lacks?)|"
    r"lacks?\s+(?:sufficient\s+)?(?:information|evidence|details?)\s+to\s+answer|"
    r"there\s+is\s+no\s+(?:retrieved\s+)?evidence\s+addressing|"
    r"no\s+relevant\s+(?:clauses?|terms?|provisions?)\s+to\s+substantiate|"
    r"unable\s+to\s+answer\s+based\s+on\s+the\s+provided|"
    r"cannot\s+be\s+answered\s+from\s+the\s+provided"
    r")\b",
    re.IGNORECASE,
)


def detect_missing_information(text: str) -> bool:
    """Return True if response explicitly indicates that requested information is absent from evidence."""
    if not text or not text.strip():
        return True
    return bool(_MISSING_INFO_REGEX.search(text))


def check_evidence_answerability(
    query: str,
    evidence: Sequence[RetrievalResult],
) -> Tuple[bool, float, List[str]]:
    """Determine whether retrieved passages contain sufficient topical vocabulary to answer the query.

    Returns:
        Tuple of (is_answerable, overlap_ratio, missing_core_terms)
    """
    if not evidence or all(not e.text.strip() for e in evidence):
        return False, 0.0, []

    # Extract non-stopword query tokens (min length 3)
    q_words = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", query)
        if w.lower() not in _QUERY_STOPWORDS
    ]
    if not q_words:
        top_score = evidence[0].score if evidence else 0.0
        return top_score >= 0.25, 1.0, []

    # Evidence vocabulary
    ev_text = " ".join(e.text.lower() for e in evidence if e.text.strip())
    ev_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", ev_text))

    matched_words = [w for w in q_words if w in ev_words]
    missing_words = [w for w in q_words if w not in ev_words]
    overlap_ratio = len(matched_words) / max(1, len(q_words))

    top_score = evidence[0].score if evidence else 0.0
    # If none of the query content words appear in evidence, or if overlap is minimal with poor relevance:
    if len(matched_words) == 0:
        return False, overlap_ratio, missing_words

    if overlap_ratio < 0.25 and top_score < 0.35:
        return False, overlap_ratio, missing_words

    return True, overlap_ratio, missing_words


def _build_system_prompt() -> str:
    return (
        "You are an expert legal document analysis assistant providing evidence-grounded answers.\n"
        "Answer the user's legal question STRICTLY and SOLELY based on the provided evidence passages.\n\n"
        "MANDATORY OPERATING RULES:\n"
        "1. STRICT EVIDENCE GROUNDING: State only what is directly and unequivocally substantiated in the provided passages.\n"
        "2. EXPLICIT CITATIONS: Attribute facts using the source labels [S1], [S2] corresponding to the retrieved passages.\n"
        "3. DISTINGUISH QUOTES FROM INTERPRETATION: Clearly distinguish verbatim document quotations from your own analytical explanation or summary.\n"
        "4. IDENTIFY MISSING INFORMATION: If the retrieved evidence is incomplete, tangential, or does not answer the question, "
        "explicitly state what information is missing from the uploaded documents rather than speculating or extrapolating.\n"
        "5. ZERO FABRICATION: NEVER invent legal provisions, statute numbers, case citations, court decisions, quotations, or page numbers.\n"
        "6. INFORMATIONAL ONLY: Provide factual document findings; do not offer definitive legal counsel or establish an attorney-client relationship."
    )


class GroundedGenerator:
    """Multi-candidate response generator for legal questions.

    Operates in Live Mode when GEMINI_API_KEY is supplied and force_mock is False,
    or Offline Mock Mode when configured or when no API key is present.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
        max_retries: int = 2,
        force_mock: bool = False,
        require_live: bool = False,
        max_concurrency: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
    ) -> None:
        """Initialize the generator.

        Args:
            api_key: Optional Gemini API key. If omitted, checks GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            model_name: Model identifier. Defaults to GEMINI_MODEL or 'gemini-3.8-flash'.
            client: Optional injected GenAI client instance (useful for unit tests and mocks).
            max_retries: Maximum attempts on transient generation failures.
            force_mock: If True, forces deterministic mock mode regardless of environment variables.
            require_live: If True, marks live mode as mandatory and refuses to fall back to mock even if key is missing.
            max_concurrency: Maximum parallel requests for candidate generation (default: 3 or GEMINI_MAX_CONCURRENCY).
            request_timeout_seconds: Timeout per candidate request in seconds (default: 90.0 or GEMINI_REQUEST_TIMEOUT).
        """
        self.force_mock = force_mock
        raw_model = model_name or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.model_name: str = raw_model.strip()
        self.client: Optional[Any] = None if force_mock else client
        self.max_retries: int = max(0, max_retries)
        self.require_live: bool = bool(require_live and not force_mock)

        # Configurable bounded concurrency & timeouts
        raw_conc = max_concurrency or os.environ.get("GEMINI_MAX_CONCURRENCY", "3")
        try:
            self.max_concurrency: int = max(1, int(raw_conc))
        except (ValueError, TypeError):
            self.max_concurrency = 3

        raw_timeout = request_timeout_seconds or os.environ.get("GEMINI_REQUEST_TIMEOUT", "90.0")
        try:
            self.request_timeout_seconds: float = max(5.0, float(raw_timeout))
        except (ValueError, TypeError):
            self.request_timeout_seconds = 90.0

        if force_mock:
            self.api_key = None
        else:
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self.api_key = resolved_key.strip() if (resolved_key and resolved_key.strip()) else None

    @property
    def is_live(self) -> bool:
        """Returns True if live LLM generation is configured or required."""
        if self.force_mock:
            return False
        return bool(self.require_live or self.client is not None or self.api_key)

    def generate_candidates(
        self,
        query: str,
        evidence: Sequence[RetrievalResult],
        num_candidates: int = 3,
        temperature: float = 0.7,
    ) -> GenerationRun:
        """Generate multiple candidate answers for uncertainty estimation.

        Args:
            query: The legal question asked by the user.
            evidence: Top retrieved passages from VectorStore.
            num_candidates: Target count of candidate responses (default 3).
            temperature: Sampling temperature for candidate variation (default 0.7).

        Returns:
            GenerationRun containing candidate responses and provenance metadata.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty.")

        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {num_candidates}")

        if self.is_live:
            return self._generate_live(clean_query, evidence, num_candidates, temperature)
        else:
            return self._generate_mock(clean_query, evidence, num_candidates)

    def _generate_mock(
        self,
        query: str,
        evidence: Sequence[RetrievalResult],
        num_candidates: int,
    ) -> GenerationRun:
        """Generate deterministic demonstration candidate answers based solely on retrieved passages."""
        candidates: List[CandidateResponse] = []

        if not evidence or all(not e.text.strip() for e in evidence):
            # No evidence case: explicitly abstain from answering without fabricating legal rules
            no_evidence_msg = (
                f"Based on the uploaded documents, there is no retrieved evidence addressing: '{query}'. "
                f"The available text does not contain relevant clauses or terms to substantiate an answer. "
                f"Please consult the original legal instrument directly."
            )
            for i in range(num_candidates):
                candidates.append(
                    CandidateResponse(
                        candidate_index=i,
                        generation_mode="MOCK",
                        answer_text=no_evidence_msg,
                        cited_sources=[],
                        metadata={"has_evidence": False, "style": f"mock_abstention_{i}"},
                    )
                )
            return GenerationRun(
                query=query,
                mode="MOCK",
                model_name="offline-deterministic-mock",
                candidates=candidates,
                disclaimer=f"{LEGAL_DISCLAIMER}\n\n{MOCK_AGREEMENT_WARNING}",
                metadata={"total_evidence_passages": 0, "mock_warning": MOCK_AGREEMENT_WARNING},
            )

        top_ev = evidence[0]
        cited_locators = [e.source_locator for e in evidence[:2]]

        # Evaluate whether the retrieved passages actually contain vocabulary to answer the query
        is_answerable, overlap, missing_terms = check_evidence_answerability(query, evidence)

        if not is_answerable:
            missing_topic = ", ".join(f"'{m}'" for m in missing_terms[:3]) if missing_terms else f"'{query}'"
            refusal_0 = (
                f"Based on the indexed document ({top_ev.doc_name}), the retrieved text does not contain "
                f"information addressing '{query}'. The available excerpts ({top_ev.source_locator} [S1]) "
                f"pertain to '{top_ev.section_title or top_ev.doc_name}' and lack clauses relating to {missing_topic}. "
                f"No substantive factual answer can be substantiated from the uploaded text."
            )
            refusal_1 = (
                f"The uploaded document ({top_ev.doc_name}) does not mention or provide details for '{query}'. "
                f"Retrieved excerpt {top_ev.source_locator} [S1] contains no provisions addressing {missing_topic}."
            )
            refusal_2 = (
                f"A review of {top_ev.source_locator} [S1] indicates that '{query}' is not addressed in this instrument. "
                f"The indexed legal text pertains to different subject matter and contains no information for {missing_topic}."
            )
            refusals = [refusal_0, refusal_1, refusal_2]

            for i in range(num_candidates):
                r_text = refusals[i % len(refusals)]
                candidates.append(
                    CandidateResponse(
                        candidate_index=i,
                        generation_mode="MOCK",
                        answer_text=r_text,
                        cited_sources=[top_ev.source_locator],
                        metadata={
                            "is_answerable": False,
                            "missing_information_detected": True,
                            "missing_terms": missing_terms,
                            "style": f"mock_unanswerable_refusal_{i}",
                        },
                    )
                )

            return GenerationRun(
                query=query,
                mode="MOCK",
                model_name="offline-deterministic-mock",
                candidates=candidates,
                disclaimer=f"{LEGAL_DISCLAIMER}\n\n{MOCK_AGREEMENT_WARNING}",
                metadata={
                    "is_answerable": False,
                    "missing_information_detected": True,
                    "missing_terms": missing_terms,
                    "total_evidence_passages": len(evidence),
                },
            )

        # Candidate 0: Direct passage quotation and textual synthesis
        text_preview_0 = top_ev.text.strip().replace("\n", " ")
        if len(text_preview_0) > 300:
            text_preview_0 = text_preview_0[:300] + "..."
        c0_text = (
            f"Based on {top_ev.source_locator} [S1], the document provides:\n"
            f"\"{text_preview_0}\"\n\n"
            f"This passage directly defines the legal requirements governing '{query}'. "
            f"No extraneous or conflicting terms were observed in the primary retrieved provision."
        )
        candidates.append(
            CandidateResponse(
                candidate_index=0,
                generation_mode="MOCK",
                answer_text=c0_text,
                cited_sources=[top_ev.source_locator],
                metadata={"style": "direct_quotation", "primary_source": top_ev.source_locator},
            )
        )

        # Candidate 1: Multi-clause analytical breakdown (incorporating S2 if available)
        if len(evidence) > 1:
            second_ev = evidence[1]
            text_preview_1 = second_ev.text.strip().replace("\n", " ")
            if len(text_preview_1) > 200:
                text_preview_1 = text_preview_1[:200] + "..."
            c1_text = (
                f"A review of the uploaded text reveals two relevant provisions:\n"
                f"1. Primary provision at {top_ev.source_locator} [S1] establishes the baseline rule: \"{text_preview_0[:200]}...\".\n"
                f"2. Further provision at {second_ev.source_locator} [S2] supplements this: \"{text_preview_1}\".\n\n"
                f"In combination, these clauses govern obligations relating to '{query}'."
            )
            c1_citations = [top_ev.source_locator, second_ev.source_locator]
        else:
            c1_text = (
                f"Under the structured terms of {top_ev.source_locator} [S1], '{query}' is addressed as follows:\n"
                f"- Extracted Provision: \"{text_preview_0}\".\n"
                f"- Traceability: Document '{top_ev.doc_name}', locator '{top_ev.source_locator}'."
            )
            c1_citations = [top_ev.source_locator]

        candidates.append(
            CandidateResponse(
                candidate_index=1,
                generation_mode="MOCK",
                answer_text=c1_text,
                cited_sources=c1_citations,
                metadata={"style": "clause_breakdown"},
            )
        )

        # Candidate 2: Cautious scope and limitation assessment
        c2_text = (
            f"Regarding '{query}', the available evidence from {top_ev.source_locator} [S1] indicates:\n"
            f"\"{text_preview_0[:240]}...\".\n\n"
            f"Scope Limitation: This interpretation is bounded strictly by the text of {top_ev.doc_name}. "
            f"Any rights or obligations not explicitly articulated in this provision remain subject "
            f"to applicable governing law and full document context."
        )
        candidates.append(
            CandidateResponse(
                candidate_index=2,
                generation_mode="MOCK",
                answer_text=c2_text,
                cited_sources=[top_ev.source_locator],
                metadata={"style": "cautious_limitation"},
            )
        )

        # Pad or slice to exact num_candidates requested
        while len(candidates) < num_candidates:
            idx = len(candidates)
            candidates.append(
                CandidateResponse(
                    candidate_index=idx,
                    generation_mode="MOCK",
                    answer_text=f"Summary of {top_ev.source_locator} [S1] regarding '{query}': \"{text_preview_0[:180]}...\".",
                    cited_sources=[top_ev.source_locator],
                    metadata={"style": f"mock_variant_{idx}"},
                )
            )

        return GenerationRun(
            query=query,
            mode="MOCK",
            model_name="offline-deterministic-mock",
            candidates=candidates[:num_candidates],
            disclaimer=f"{LEGAL_DISCLAIMER}\n\n{MOCK_AGREEMENT_WARNING}",
            metadata={"total_evidence_passages": len(evidence), "mock_warning": MOCK_AGREEMENT_WARNING},
        )

    def _get_or_create_client(self) -> Any:
        """Instantiate the Gemini client using modern google.genai SDK or fallback."""
        if self.client is not None:
            return self.client

        if not self.api_key:
            raise RuntimeError(
                "Cannot instantiate live Gemini client: No API key provided. "
                "Set GEMINI_API_KEY environment variable or enter it in the application sidebar."
            )

        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except Exception as e1:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                return legacy_genai.GenerativeModel(self.model_name)
            except Exception as e2:
                err1 = _sanitize_error_message(e1, self.api_key)
                err2 = _sanitize_error_message(e2, self.api_key)
                raise RuntimeError(
                    f"Failed to initialize Google GenAI client ({self.model_name}): {err1} | {err2}"
                )

    def _call_live_single(
        self,
        client: Any,
        prompt_text: str,
        temperature: float,
    ) -> str:
        """Execute a single live model call with bounded retries and sanitized errors."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # 1. Modern google.genai Client.models.generate_content interface (installed SDK 1.53.0)
                if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=_build_system_prompt(),
                    )
                    resp = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_text,
                        config=config,
                    )
                    return str(resp.text or "").strip()

                # 2. Google GenAI Interactions API (if supported by client/SDK)
                elif hasattr(client, "interactions") and hasattr(client.interactions, "create"):
                    resp = client.interactions.create(
                        model=self.model_name,
                        input=prompt_text,
                        generation_config={
                            "temperature": temperature,
                            "system_instruction": _build_system_prompt(),
                        },
                    )
                    output_text = getattr(resp, "output_text", None) or getattr(resp, "text", "")
                    return str(output_text).strip()

                # 3. Legacy google.generativeai GenerativeModel interface
                elif hasattr(client, "generate_content"):
                    generation_config = {"temperature": temperature}
                    resp = client.generate_content(prompt_text, generation_config=generation_config)
                    return str(resp.text or "").strip()

                else:
                    raise TypeError(f"Unrecognized client interface for: {type(client)}")

            except Exception as exc:
                last_error = _sanitize_error_message(exc, self.api_key)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                continue

        raise RuntimeError(
            f"Live Gemini generation failed for model '{self.model_name}' after {self.max_retries + 1} attempts: {last_error}"
        )

    def _generate_live(
        self,
        query: str,
        evidence: Sequence[RetrievalResult],
        num_candidates: int,
        temperature: float,
    ) -> GenerationRun:
        """Execute live LLM generation calls. Never silently falls back to mock."""
        client = self._get_or_create_client()
        evidence_context = _build_evidence_context(evidence)

    def _generate_candidate_worker(
        self,
        candidate_idx: int,
        client: Any,
        user_prompt: str,
        temperature: float,
        locator_map: Dict[str, str],
        evidence: Sequence[RetrievalResult],
    ) -> CandidateResponse:
        """Worker executing a single live candidate generation with per-request latency tracking."""
        t_start = time.perf_counter()
        raw_answer = self._call_live_single(client, user_prompt, temperature)
        latency_sec = round(time.perf_counter() - t_start, 3)

        # Extract source labels [S1], [S2] cited in answer
        found_citations = []
        for label, loc in locator_map.items():
            if label in raw_answer:
                found_citations.append(loc)

        # If no [S#] was explicitly written, include top evidence locator if evidence was present
        if not found_citations and evidence:
            found_citations.append(evidence[0].source_locator)

        # Detect explicit missing-information declaration by the model
        is_missing = detect_missing_information(raw_answer)

        return CandidateResponse(
            candidate_index=candidate_idx,
            generation_mode="LIVE",
            answer_text=raw_answer,
            cited_sources=found_citations,
            metadata={
                "temperature": temperature,
                "model": self.model_name,
                "latency_seconds": latency_sec,
                "missing_information_detected": is_missing,
                "is_answerable": not is_missing,
            },
        )

    def _generate_live(
        self,
        query: str,
        evidence: Sequence[RetrievalResult],
        num_candidates: int,
        temperature: float,
    ) -> GenerationRun:
        """Execute live LLM generation calls. Never silently falls back to mock."""
        client = self._get_or_create_client()
        evidence_context = _build_evidence_context(evidence)

        user_prompt = (
            f"LEGAL INQUIRY:\n{query}\n\n"
            f"RETRIEVED DOCUMENT EVIDENCE PASSAGES:\n{evidence_context}\n\n"
            f"INSTRUCTIONS FOR YOUR RESPONSE:\n"
            f"- Answer strictly based on the passages above, citing [S1], [S2] etc. where each fact appears.\n"
            f"- Distinguish direct quotations from your synthesis.\n"
            f"- If the passages lack sufficient information to answer fully, explicitly identify what is missing.\n"
            f"- Do not include outside knowledge not contained in the text."
        )

        locator_map = {f"[S{i}]": e.source_locator for i, e in enumerate(evidence, start=1)}

        candidates: List[CandidateResponse] = []

        # Execute concurrently if concurrency > 1, otherwise sequential
        if self.max_concurrency > 1 and num_candidates > 1:
            workers = min(self.max_concurrency, num_candidates)
            candidates_by_idx: Dict[int, CandidateResponse] = {}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self._generate_candidate_worker,
                        i,
                        client,
                        user_prompt,
                        temperature,
                        locator_map,
                        evidence,
                    ): i
                    for i in range(num_candidates)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        cand_resp = future.result(timeout=self.request_timeout_seconds)
                        candidates_by_idx[idx] = cand_resp
                    except Exception as exc:
                        # Cancel remaining futures if any
                        for f in future_to_idx:
                            f.cancel()
                        sanitized_err = _sanitize_error_message(exc, self.api_key)
                        raise RuntimeError(
                            f"Live Gemini generation failed for model '{self.model_name}' on candidate {idx}: {sanitized_err}"
                        ) from None

            candidates = [candidates_by_idx[i] for i in range(num_candidates)]
        else:
            for i in range(num_candidates):
                candidates.append(
                    self._generate_candidate_worker(
                        i, client, user_prompt, temperature, locator_map, evidence
                    )
                )

        missing_count = sum(1 for c in candidates if c.metadata.get("missing_information_detected", False))
        is_run_answerable = (missing_count < len(candidates) / 2)
        missing_info_detected = (missing_count > 0)

        return GenerationRun(
            query=query,
            mode="LIVE",
            model_name=self.model_name,
            candidates=candidates,
            disclaimer=LEGAL_DISCLAIMER,
            metadata={
                "total_evidence_passages": len(evidence),
                "temperature": temperature,
                "concurrency": min(self.max_concurrency, num_candidates),
                "is_answerable": is_run_answerable,
                "missing_information_detected": missing_info_detected,
            },
        )


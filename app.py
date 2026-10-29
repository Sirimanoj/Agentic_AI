"""Streamlit Web Application: Legal UQ.

Similarity-Based Confidence and Uncertainty Estimation for LLM-Based Legal Assistance.
Features:
- Document upload (.pdf, .docx, .txt) with exact citation preservation.
- Pre-packaged 1-click legal demonstration corpus.
- Dual-mode generation: Live Gemini or Deterministic Grounded Mock.
- Heuristic claim alignment, lexical & semantic similarity matrices.
- Confidence gauge, uncertainty breakdown, and risk guardrails (SAFE, WARNING, ABSTAIN).
- Traceable source metadata display (never invents page numbers).
"""

import io
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure src directory is in sys.path so legal_uq can be imported directly
_src_dir = str(Path(__file__).resolve().parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pandas as pd
import streamlit as st

from legal_uq.generation.generator import (
    DEFAULT_GEMINI_MODEL,
    LEGAL_DISCLAIMER,
    SUPPORTED_GEMINI_MODELS,
)
from legal_uq.guardrails.abstention import AbstentionConfig, AbstentionDecision
from legal_uq.pipeline import DocumentSummary, LegalUQPipeline, PipelineQueryResult
from legal_uq.retrieval.vector_store import (
    CachedSentenceTransformerEmbedder,
    DeterministicFakeEmbedder,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LegalUQ - Legal Confidence & Uncertainty Estimation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished legal tech aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Georgia', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .badge-safe {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-abstain {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .citation-tag {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: monospace;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing embedding model...")
def get_pipeline() -> LegalUQPipeline:
    """Cached pipeline initialization."""
    try:
        embedder = CachedSentenceTransformerEmbedder()
    except Exception:
        embedder = DeterministicFakeEmbedder(dimension=384)
    return LegalUQPipeline(embedder=embedder)


# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = get_pipeline()

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ LegalUQ Controller")
    st.caption("Similarity-Based Uncertainty & Confidence Triage")

    st.markdown("---")
    st.markdown("#### 📁 Document Repository")

    uploaded_files = st.file_uploader(
        "Upload Legal Documents (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload contracts, court judgments, statutes, or legal notices.",
    )

    if uploaded_files:
        for f in uploaded_files:
            file_bytes = f.read()
            # Check if not already indexed
            if not any(d.doc_name == f.name for d in st.session_state.pipeline.indexed_documents):
                with st.spinner(f"Indexing {f.name}..."):
                    summary = st.session_state.pipeline.ingest_bytes(f.name, file_bytes)
                    st.success(f"Indexed {summary.doc_name} ({summary.num_chunks} chunks)")

    # 1-Click Demo Corpus Button
    sample_dir = Path("data/sample_documents")
    if sample_dir.exists():
        if st.button("📂 Load Built-In Legal Corpus (3 Docs)", use_container_width=True):
            with st.spinner("Loading sample corpus (NDA, Employment, Statute)..."):
                sample_files = [
                    sample_dir / "sample_nda.pdf",
                    sample_dir / "sample_employment_agreement.docx",
                    sample_dir / "sample_statute_excerpts.txt",
                ]
                for sf in sample_files:
                    if sf.exists():
                        st.session_state.pipeline.ingest_file(sf)
                st.success("Loaded 3 sample legal documents successfully!")
                st.rerun()

    if st.button("🗑️ Clear Indexed Documents", use_container_width=True):
        st.session_state.pipeline.clear()
        st.session_state.query_result = None
        st.info("Vector index cleared.")
        st.rerun()

    # Display indexed document statistics
    docs = st.session_state.pipeline.indexed_documents
    if docs:
        st.markdown(f"**Indexed Documents ({len(docs)}):**")
        for doc in docs:
            st.markdown(
                f"- **{doc.doc_name}** (`{doc.doc_type}`)\n"
                f"  └ *{doc.num_chunks} chunks, {doc.total_characters:,} chars*"
            )
        st.caption(f"Total vector chunks: **{st.session_state.pipeline.total_chunk_count}**")
    else:
        st.warning("No documents indexed. Upload files or load demo corpus.")

    st.markdown("---")
    st.markdown("#### ⚙️ LLM Generation Mode")

    mode_options = [
        "⚡ Offline Demo Mode (Deterministic Mock)",
        "🟢 Live Gemini API Mode",
    ]
    selected_mode = st.radio(
        "Operating Mode:",
        mode_options,
        index=0,
        help="Select whether to run completely offline without API calls, or use Live Gemini API.",
    )
    is_live_selected = (selected_mode == mode_options[1])

    # Model configuration: respects GEMINI_MODEL env var or defaults to DEFAULT_GEMINI_MODEL (gemini-3.8-flash)
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    base_models = list(SUPPORTED_GEMINI_MODELS)
    if env_model:
        model_candidates = [env_model] + [m for m in base_models if m != env_model]
    else:
        model_candidates = base_models

    def _format_model_label(model_id: str) -> str:
        if model_id == "gemini-3.8-flash":
            return "gemini-3.8-flash (Default - Recommended)"
        elif model_id == "gemini-3.6-flash":
            return "gemini-3.6-flash (Fast Workhorse)"
        elif model_id == "gemini-3.1-pro-preview":
            return "gemini-3.1-pro-preview (Deep Reasoning Preview)"
        elif model_id == "gemini-2.5-flash":
            return "gemini-2.5-flash (Legacy - Historical Access Only)"
        return f"{model_id} (Custom / Override)"

    model_choice = st.selectbox(
        "Gemini Model Identifier:",
        model_candidates,
        index=0,
        format_func=_format_model_label,
        help=f"Supported text models. Default is {DEFAULT_GEMINI_MODEL}. Configurable via $env:GEMINI_MODEL.",
    )

    active_key: Optional[str] = None
    if is_live_selected:
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and env_key.strip():
            st.caption("🔑 API key detected from environment (`$env:GEMINI_API_KEY`).")
            key_override = st.text_input(
                "Override API Key (Optional):",
                type="password",
                help="Leave blank to use the detected environment variable key.",
            )
            active_key = key_override.strip() if key_override.strip() else env_key.strip()
        else:
            active_key = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="Key is held in memory for this session only; never logged, printed, or saved.",
            ).strip()

        if active_key:
            st.session_state.pipeline.set_api_key(active_key, model_name=model_choice, force_mock=False)
            st.sidebar.success(f"🟢 Live Gemini Active: `{model_choice}`")
        else:
            # Preserve Live mode intent without falling back to mock; missing key is flagged clearly
            st.session_state.pipeline.set_api_key(None, model_name=model_choice, force_mock=False)
            st.sidebar.warning("⚠️ Live Mode selected, but API key is missing. Enter a key above or set $env:GEMINI_API_KEY.")
    else:
        # Enforce deterministic mock even if environment variables exist
        st.session_state.pipeline.set_api_key(None, model_name=model_choice, force_mock=True)
        st.sidebar.info("⚡ Offline Mock Mode Active (Deterministic)")

    st.markdown("---")
    with st.expander("🛡️ Guardrail Thresholds"):
        min_conf = st.slider("Min Confidence to Answer", 0.0, 1.0, 0.30, 0.05)
        warn_conf = st.slider("Confidence Warning Level", 0.0, 1.0, 0.55, 0.05)
        max_uncert = st.slider("Max Uncertainty Limit", 0.0, 1.0, 0.55, 0.05)
        min_relevance = st.slider("Min Evidence Cosine Sim", 0.0, 1.0, 0.30, 0.05)
        block_contradictions = st.checkbox("Block on Candidate Contradiction", value=True)

        # Update pipeline configuration
        cfg = AbstentionConfig(
            min_confidence_to_answer=min_conf,
            confidence_warning_threshold=warn_conf,
            max_uncertainty_threshold=max_uncert,
            min_evidence_relevance=min_relevance,
            block_on_contradiction=block_contradictions,
        )
        st.session_state.pipeline.update_abstention_config(cfg)


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Similarity-Based Legal Uncertainty Estimation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-candidate consensus, semantic similarity, and claim-level grounding for transparent legal assistance.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
    <strong>⚖️ Legal Research & Evaluation Prototype:</strong> This software is an informational research tool for legal document analysis.
    Findings and answers are derived strictly and solely from uploaded files with traceable citations.
    The system is not an attorney and does not provide formal legal advice.
    </div>
    """,
    unsafe_allow_html=True,
)

# Question Input Section
st.markdown("### 🔍 Ask a Question About Ingested Legal Documents")

col_q1, col_q2 = st.columns([3, 1])

# Preset Question Selectbox
sample_prompts = [
    "--- Select a Quick Demo Question ---",
    "What is the duration of the confidentiality obligations following termination of the NDA?",
    "What is the annualized base salary for the Chief Technology Officer under the employment contract?",
    "What is the notice period required for termination without cause in the employment agreement?",
    "Under what conditions is disclosure of confidential information permitted under the NDA?",
    "What is the dollar threshold for the statute of frauds to apply under UCC Section 2-201?",
    "Under the UCC, what is the legal effect of fixing unreasonably large liquidated damages?",
    "What are the criminal penalties and prison sentences for violating the NDA? (Out-of-Domain)",
    "What are the patent registration filing fees required by the USPTO? (Out-of-Domain)",
]

with col_q1:
    selected_preset = st.selectbox("Quick-Select Sample Inquiries:", sample_prompts)
    default_text = ""
    if selected_preset != sample_prompts[0]:
        default_text = selected_preset

    query_input = st.text_input(
        "Enter legal inquiry:",
        value=default_text,
        placeholder="e.g., What are the permitted disclosures under Section 3 of the agreement?",
    )

with col_q2:
    top_k_val = st.slider("Evidence Chunks (top-k)", min_value=1, max_value=8, value=3)
    simulate_conflict = st.checkbox(
        "⚠️ Simulate Contradiction",
        value=False,
        help="Injects conflicting candidate assertions to demonstrate automated contradiction detection and abstention.",
    )

submit_col1, submit_col2 = st.columns([1, 4])
with submit_col1:
    analyze_btn = st.button("🚀 Analyze & Verify", type="primary", use_container_width=True)

if analyze_btn:
    if not query_input.strip():
        st.error("Please enter a question to analyze.")
    elif st.session_state.pipeline.total_chunk_count == 0:
        st.warning("Please upload a document or click 'Load Built-In Legal Corpus' in the sidebar first.")
    elif is_live_selected and not active_key:
        st.error(
            "❌ Live Gemini mode is selected, but no API key was provided. "
            "Please enter your API key in the sidebar or switch to 'Offline Demo Mode'."
        )
    else:
        with st.spinner("Retrieving evidence, generating candidates, and evaluating uncertainty..."):
            try:
                result = st.session_state.pipeline.query(
                    question=query_input.strip(),
                    top_k=top_k_val,
                    simulate_contradiction=simulate_conflict,
                )
                st.session_state.query_result = result
                st.session_state.last_query = query_input.strip()
            except Exception as exc:
                st.session_state.query_result = None
                st.error(f"❌ Execution Failure: {exc}")



def render_similarity_matrix_tab(res: PipelineQueryResult) -> None:
    """Render the Inter-Candidate Agreement Matrices tab safely."""
    st.markdown("##### Inter-Candidate Agreement Matrices")
    st.caption("Pairwise candidate lexical Jaccard, ROUGE-L, and SentenceTransformer semantic cosine similarity.")
    try:
        sim = res.similarity_matrix
        if sim is not None and getattr(sim, "num_candidates", 0) >= 2:
            labels = [f"Cand {i}" for i in range(sim.num_candidates)]

            c_mat1, c_mat2, c_mat3 = st.columns(3)
            with c_mat1:
                st.markdown("**Lexical Jaccard**")
                df_jaccard = pd.DataFrame(sim.jaccard_matrix, index=labels, columns=labels)
                st.dataframe(df_jaccard.style.format("{:.2f}"))

            with c_mat2:
                st.markdown("**ROUGE-L Overlap**")
                df_rouge = pd.DataFrame(sim.rouge_matrix, index=labels, columns=labels)
                st.dataframe(df_rouge.style.format("{:.2f}"))

            with c_mat3:
                st.markdown("**Semantic Cosine Agreement**")
                df_sem = pd.DataFrame(sim.semantic_matrix, index=labels, columns=labels)
                st.dataframe(df_sem.style.format("{:.2f}"))

            s_top = getattr(res.uncertainty_metrics, "s_top", 0.0)
            s_mean = getattr(res.uncertainty_metrics, "s_mean", 0.0)
            r_rel = getattr(res.uncertainty_metrics, "r_rel", 0.0)
            st.markdown(
                f"**Composite Mean Pairwise Agreement:** `{sim.mean_pairwise_agreement:.4f}` | "
                f"**Formula Breakdown:** `0.50 * s_top ({s_top:.2f}) + "
                f"0.30 * s_mean ({s_mean:.2f}) + "
                f"0.20 * r_rel ({r_rel:.2f})`"
            )
        else:
            st.info("Insufficient candidates to compute pairwise similarity matrix.")
    except Exception as exc:
        logging.getLogger("legal_uq.app").exception("Diagnostic rendering error in similarity matrix tab: %s", exc)
        st.error(f"⚠️ Diagnostic Rendering Notice: Unable to display similarity matrices ({type(exc).__name__}: {exc}).")
        st.caption("A UI rendering exception occurred while formatting pairwise matrices.")


def render_claim_alignment_tab(alignment: Optional[Any]) -> None:
    """Render the Claim Alignment & Contradictions diagnostic tab safely.

    Uses canonical ClaimAlignmentResult.contradiction_detected with property fallback,
    and safely handles empty, missing, or malformed alignment inputs without exposing
    raw tracebacks to normal users.
    """
    st.markdown("##### Legal Claim Decomposition & Contradiction Detection")
    try:
        if alignment is None:
            st.info("No claim alignment data available for this query.")
            return

        # Canonical schema field is contradiction_detected; fallback to backward-compatibility property
        contradiction_flag = getattr(
            alignment, "contradiction_detected", getattr(alignment, "contradiction_found", None)
        )

        if contradiction_flag is True:
            st.error("🚨 **MATERIAL CONTRADICTIONS DETECTED BETWEEN CANDIDATE RESPONSES:**")
            details = getattr(alignment, "contradiction_details", [])
            if details:
                for err in details:
                    st.markdown(f"- ❌ {err}")
            else:
                st.markdown("- ❌ Material inter-candidate contradictions or opposing polarities detected.")
        elif contradiction_flag is False:
            st.success("✅ No direct inter-candidate contradictions or polarity clashes detected.")
        else:
            # If safety/contradiction status cannot be determined, never falsely declare success
            st.warning("⚠️ Contradiction assessment status unknown or unavailable for this result.")

        st.markdown("---")
        st.markdown("##### Extracted Legal Claims & Evidence Verification:")
        matches = getattr(alignment, "matches", [])
        if matches:
            for match in matches:
                verdict_val = getattr(getattr(match, "verdict", None), "value", str(getattr(match, "verdict", "UNKNOWN")))
                verdict_icon = "🟢" if verdict_val == "SUPPORTED" else "🟡" if verdict_val == "PARTIALLY_SUPPORTED" else "🔴"
                claim_text = getattr(getattr(match, "claim", None), "text", "Unknown claim")
                locator = getattr(match, "matched_source_locator", None) or "None"
                overlap = getattr(match, "overlap_score", 0.0)
                rationale = getattr(match, "rationale", "No rationale provided.")
                st.markdown(
                    f"**{verdict_icon} [{verdict_val}]** *\"{claim_text}\"*\n"
                    f"- **Matched Locator:** `{locator}` (Overlap: `{overlap:.2f}`)\n"
                    f"- **Rationale:** {rationale}"
                )
        else:
            st.info("No atomic claims extracted from primary response.")

        unsupported = getattr(alignment, "unsupported_claim_texts", [])
        if unsupported:
            st.markdown("##### ⚠️ Claims Lacking Sufficient Document Grounding:")
            for unc in unsupported:
                st.markdown(f"- *\"{unc}\"*")

    except Exception as exc:
        logging.getLogger("legal_uq.app").exception("Diagnostic rendering error in claim alignment tab: %s", exc)
        st.error(f"⚠️ Diagnostic Rendering Notice: Unable to display claim alignment details ({type(exc).__name__}: {exc}).")
        st.caption("A UI rendering exception occurred while formatting alignment diagnostics. The underlying guardrail assessment was performed independently.")


# -----------------------------------------------------------------------------
# Results Presentation
# -----------------------------------------------------------------------------
if st.session_state.query_result is not None:
    res: PipelineQueryResult = st.session_state.query_result

    st.markdown("---")
    st.markdown(f"### 📋 Analysis Results for: *\"{res.question}\"*")

    # Mode Indicator
    if res.generation_mode == "LIVE":
        st.caption(
            f"🟢 **Generation Mode:** LIVE GEMINI (`{res.primary_answer.metadata.get('model', model_choice)}`) "
            f"• Responses generated via live model API calls"
        )
    else:
        st.caption(
            "⚡ **Generation Mode:** OFFLINE DEMO (Deterministic synthetic responses derived from retrieved passages)"
        )
        st.markdown(
            """
            <div style="background-color: #fefce8; border: 1px solid #fef08a; padding: 0.6rem 0.8rem; border-radius: 4px; font-size: 0.84rem; color: #854d0e; margin-bottom: 0.8rem;">
            ⚠️ <strong>MOCK/DEMO MODE NOTICE:</strong> Candidate responses are synthetic outputs derived from retrieved document passages. 
            Agreement across these demonstration candidates must not be interpreted as empirical proof of genuine LLM certainty or calibration.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Guardrail Verdict Banner
    dec = res.abstention_decision
    if dec.decision_level == "SAFE":
        st.success(
            f"**SAFE (Prototype Risk Assessment — Answer Permitted):** {dec.user_explanation}\n\n"
            f"*Notice: A 'SAFE' assessment reflects high textual candidate consensus and passage alignment within this prototype. "
            f"It is not a legal certification of substantive correctness.*"
        )
    elif dec.decision_level == "WARNING":
        st.warning(
            f"**WARNING (Moderate Uncertainty — Human Verification Recommended):** {dec.user_explanation}\n\n"
            f"*Notice: Discrepancies, partial evidence, or elevated uncertainty were detected. Cross-examine primary source passages.*"
        )
    else:  # ABSTAIN
        st.error(
            f"**ABSTAIN (Answer Suppressed by Safety Guardrail):** {dec.user_explanation}\n\n"
            f"*Notice: The system withheld the generated answer due to critical evidence deficiency or material candidate contradiction.*"
        )

    # 2. Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    conf_pct = res.confidence_score * 100
    uncert_pct = res.uncertainty_score * 100
    top_sim = res.retrieval_results[0].score if res.retrieval_results else 0.0
    support_pct = res.uncertainty_metrics.support_ratio * 100

    with m1:
        st.metric(
            label="Confidence Score",
            value=f"{res.confidence_score:.2f}",
            delta=f"{conf_pct:.0f}% Agreement & Grounding",
        )
        st.progress(res.confidence_score)

    with m2:
        st.metric(
            label="Uncertainty Score",
            value=f"{res.uncertainty_score:.2f}",
            delta=f"{uncert_pct:.0f}% Estimated Risk",
            delta_color="inverse",
        )
        st.progress(res.uncertainty_score)

    with m3:
        st.metric(
            label="Top Passage Cosine Sim",
            value=f"{top_sim:.3f}",
            delta="Evidence Relevance",
        )
        st.progress(max(0.0, min(1.0, top_sim)))

    with m4:
        st.metric(
            label="Claim Grounding Ratio",
            value=f"{support_pct:.0f}%",
            delta=f"{len(res.uncertainty_metrics.claim_alignment.unsupported_claim_texts if res.uncertainty_metrics.claim_alignment else [])} Unsupported",
        )
        st.progress(max(0.0, min(1.0, res.uncertainty_metrics.support_ratio)))

    # 3. Primary Answer Display
    st.markdown("#### 💬 Primary Grounded Response")
    if not res.answer_allowed:
        st.error(
            "⚠️ **RESPONSE WITHHELD BY POLICY**: The confidence and grounding thresholds were not met, "
            "or material contradictions were detected across candidate outputs. "
            "Please review the underlying evidence passages below directly."
        )
    else:
        st.info(res.primary_answer.answer_text)

        # Show cited sources badge row
        if res.primary_answer.cited_sources:
            citations_html = " ".join(
                f'<span class="citation-tag">📌 {cite}</span>' for cite in res.primary_answer.cited_sources
            )
            st.markdown(f"**Referenced Document Citations:** {citations_html}", unsafe_allow_html=True)

    # 4. Diagnostic Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Candidate Agreement (3 Candidates)",
        "📊 Similarity & Agreement Matrices",
        "⚖️ Claim-Level Alignment & Contradictions",
        "📄 Retrieved Evidence Passages",
    ])

    # Tab 1: Candidates
    with tab1:
        st.markdown("##### Multi-Candidate Generation Outputs")
        st.caption("Comparing multiple independent candidate answers highlights consensus and surfaces variance.")
        cand_cols = st.columns(len(res.candidate_responses))
        for idx, (col, cand) in enumerate(zip(cand_cols, res.candidate_responses)):
            with col:
                st.markdown(f"**Candidate {idx}** (`{cand.generation_mode}`)")
                st.text_area(
                    label=f"Candidate {idx} Text",
                    value=cand.answer_text,
                    height=200,
                    key=f"cand_view_{idx}",
                    disabled=True,
                )
                if cand.cited_sources:
                    st.caption("Citations: " + ", ".join(cand.cited_sources))

    # Tab 2: Matrices
    with tab2:
        render_similarity_matrix_tab(res)

    # Tab 3: Claim Alignment & Contradictions
    with tab3:
        render_claim_alignment_tab(res.claim_alignment)

    # Tab 4: Retrieved Passages
    with tab4:
        st.markdown("##### Retrieved Evidence Passages from Index")
        st.caption("FAISS L2-normalized cosine similarity retrieval preserves exact source metadata.")

        if not res.retrieval_results:
            st.warning("No evidence passages retrieved.")
        else:
            for idx, r in enumerate(res.retrieval_results, start=1):
                page_info = f"Page {r.page_number}" if r.page_number is not None else "Page N/A (Paragraph/Heading Locator)"
                sec_info = f"Section: {r.section_title}" if r.section_title else "Section: N/A"

                with st.expander(f"[S{idx}] {r.doc_name} — {page_info} | Cosine Sim: {r.score:.3f}"):
                    st.markdown(f"**Source Locator:** `{r.source_locator}`")
                    st.markdown(f"**Section Heading:** `{sec_info}`")
                    st.markdown(f"**Chunk ID:** `{r.chunk_id}`")
                    st.markdown("**Passage Text:**")
                    st.code(r.text, language="markdown")


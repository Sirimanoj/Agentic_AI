"""Unit tests for LegalUQPipeline end-to-end orchestration."""

import io
from docx import Document
from reportlab.pdfgen import canvas

from legal_uq.guardrails.abstention import AbstentionConfig
from legal_uq.pipeline import LegalUQPipeline
from legal_uq.retrieval.vector_store import DeterministicFakeEmbedder


def _create_sample_pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(50, 750, "CONFIDENTIALITY AGREEMENT")
    p.drawString(50, 720, "1. Term and Survival: The obligations of confidentiality shall survive for 3 years.")
    p.drawString(50, 700, "2. Permitted Disclosures: Disclosures required by law or judicial order are permitted.")
    p.showPage()
    p.drawString(50, 750, "3. Governing Law: This Agreement is governed by the laws of the State of Delaware.")
    p.drawString(50, 720, "4. Return of Materials: Promptly upon written demand, recipient shall return all copies.")
    p.showPage()
    p.save()
    return buffer.getvalue()


def _create_sample_docx_bytes() -> bytes:
    doc = Document()
    doc.add_heading("EMPLOYMENT AGREEMENT", level=0)
    doc.add_heading("SECTION 1: COMPENSATION", level=1)
    doc.add_paragraph("The Employee shall receive an annual base salary of $175,000 payable semi-monthly.")
    doc.add_heading("SECTION 2: TERMINATION NOTICE", level=1)
    doc.add_paragraph("Either party may terminate employment by giving at least 30 days prior written notice.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_pipeline_empty_index_abstains_safely():
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    res = pipeline.query("What is the base salary?")
    assert not res.answer_allowed
    assert res.decision_level == "ABSTAIN"
    assert len(res.retrieval_results) == 0
    assert "No legal documents are currently indexed" in res.primary_answer.answer_text


def test_pipeline_ingest_and_grounded_query():
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    # Ingest PDF
    pdf_bytes = _create_sample_pdf_bytes()
    summary_pdf = pipeline.ingest_bytes("nda.pdf", pdf_bytes)
    assert summary_pdf.doc_name == "nda.pdf"
    assert summary_pdf.num_chunks > 0

    # Ingest DOCX
    docx_bytes = _create_sample_docx_bytes()
    summary_docx = pipeline.ingest_bytes("employment.docx", docx_bytes)
    assert summary_docx.doc_name == "employment.docx"
    assert len(pipeline.indexed_documents) == 2

    # Query for salary
    res = pipeline.query("What is the base salary under the employment agreement?", top_k=2)
    assert len(res.retrieval_results) > 0
    assert len(res.candidate_responses) == 3
    assert res.primary_answer is not None
    assert res.confidence_score >= 0.0
    assert res.uncertainty_score <= 1.0
    assert res.abstention_decision is not None
    assert res.primary_answer.generation_mode == "MOCK"

    # Verify retrieval retained metadata
    top_chunk = res.retrieval_results[0]
    assert top_chunk.doc_name in ("employment.docx", "nda.pdf")


def test_pipeline_simulated_contradiction_trips_guardrail():
    embedder = DeterministicFakeEmbedder(dimension=64)
    cfg = AbstentionConfig(block_on_contradiction=True)
    pipeline = LegalUQPipeline(embedder=embedder, default_abstention_config=cfg)

    pdf_bytes = _create_sample_pdf_bytes()
    pipeline.ingest_bytes("nda.pdf", pdf_bytes)

    # Normal query
    normal_res = pipeline.query("What is the governing law of the agreement?")
    assert not normal_res.uncertainty_metrics.contradiction_detected

    # Simulated contradiction query
    clash_res = pipeline.query(
        "What is the governing law of the agreement?",
        simulate_contradiction=True,
    )
    assert clash_res.uncertainty_metrics.contradiction_detected
    assert clash_res.decision_level in ("WARNING", "ABSTAIN")


def test_pipeline_clear_and_reindex():
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    pipeline.ingest_bytes("nda.pdf", _create_sample_pdf_bytes())
    assert len(pipeline.indexed_documents) == 1
    assert pipeline.total_chunk_count > 0

    pipeline.clear()
    assert len(pipeline.indexed_documents) == 0
    assert pipeline.total_chunk_count == 0

    # Querying after clear returns empty-index abstention
    res = pipeline.query("What is the governing law?")
    assert res.decision_level == "ABSTAIN"


def test_pipeline_uploaded_bytes_multi_format_traceability_and_abstention():
    """Verify raw uploaded bytes for PDF, DOCX, and TXT retain exact citation invariants."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    # 1. Custom PDF Upload (2 pages)
    pdf_buf = io.BytesIO()
    p = canvas.Canvas(pdf_buf)
    p.drawString(50, 750, "TITLE: RESEARCH COLLABORATION CONTRACT")
    p.drawString(50, 720, "Page 1 states: Confidential proprietary algorithms shall not be exported.")
    p.showPage()
    p.drawString(50, 750, "SECTION: DISPUTE RESOLUTION AND ARBITRATION")
    p.drawString(50, 720, "Page 2 states: All controversies shall be resolved by mandatory binding arbitration in Dover.")
    p.showPage()
    p.save()
    pdf_bytes = pdf_buf.getvalue()

    # 2. Custom DOCX Upload
    docx_doc = Document()
    docx_doc.add_heading("CONSULTING SERVICES AGREEMENT", level=0)
    docx_doc.add_heading("HOURLY RATE AND BILLING", level=1)
    docx_doc.add_paragraph("The Consultant hourly rate is fixed at 150 dollars per hour billed monthly.")
    docx_buf = io.BytesIO()
    docx_doc.save(docx_buf)
    docx_bytes = docx_buf.getvalue()

    # 3. Custom TXT Upload
    txt_bytes = (
        "STATUTORY LIMITATIONS CLAUSE\n"
        "Claims for breach of contract must be commenced within four years after the cause of action accrued.\n"
    ).encode("utf-8")

    # Ingest all 3 files via raw byte interfaces
    sum_pdf = pipeline.ingest_bytes("uploaded_contract.pdf", pdf_bytes)
    sum_docx = pipeline.ingest_bytes("uploaded_consulting.docx", docx_bytes)
    sum_txt = pipeline.ingest_bytes("uploaded_statute.txt", txt_bytes)

    assert sum_pdf.num_chunks >= 2
    assert sum_docx.num_chunks >= 1
    assert sum_txt.num_chunks >= 1
    assert len(pipeline.indexed_documents) == 3

    # Query PDF: Must return 1-based page number (integer)
    res_pdf = pipeline.query("What are the dispute resolution and arbitration requirements?", top_k=2)
    pdf_match = next((r for r in res_pdf.retrieval_results if r.doc_name == "uploaded_contract.pdf"), None)
    assert pdf_match is not None
    assert pdf_match.page_number in (1, 2)
    assert isinstance(pdf_match.page_number, int)
    assert "uploaded_contract.pdf:p" in pdf_match.source_locator

    # Query DOCX: Must NOT fabricate a page number (page_number must be None)
    res_docx = pipeline.query("What is the consultant hourly rate?", top_k=2)
    docx_match = next((r for r in res_docx.retrieval_results if r.doc_name == "uploaded_consulting.docx"), None)
    assert docx_match is not None
    assert docx_match.page_number is None  # Strictly None, never fabricated!
    assert "uploaded_consulting.docx:para_" in docx_match.source_locator or "HOURLY" in docx_match.source_locator

    # Query TXT: Must NOT fabricate a page number
    res_txt = pipeline.query("Within how many years must a breach claim be commenced?", top_k=2)
    txt_match = next((r for r in res_txt.retrieval_results if r.doc_name == "uploaded_statute.txt"), None)
    assert txt_match is not None
    assert txt_match.page_number is None  # Strictly None!

    # Query Missing / Out-of-domain evidence: Must trigger warning or abstention
    res_missing = pipeline.query(
        "What are the federal aerospace regulations for commercial orbital launch rocket telemetry?",
        top_k=2,
    )
    assert res_missing.decision_level in ("WARNING", "ABSTAIN")
    assert res_missing.uncertainty_score > 0.40


def test_uploaded_document_isolation_and_citation_survival():
    """Verify uploaded documents remain isolated from built-in corpus, source locators survive to primary answer citations, and missing evidence triggers hard abstention."""
    embedder = DeterministicFakeEmbedder(dimension=64)
    pipeline = LegalUQPipeline(embedder=embedder)

    # 1. Pipeline starts completely empty (no built-in corpus pre-loaded)
    assert len(pipeline.indexed_documents) == 0
    assert pipeline.total_chunk_count == 0

    # 2. Ingest a single user-uploaded PDF
    pdf_bytes = _create_sample_pdf_bytes()
    summary = pipeline.ingest_bytes("user_custom_nda.pdf", pdf_bytes)
    assert summary.doc_name == "user_custom_nda.pdf"

    # Verify isolation: ONLY the uploaded document is present
    assert len(pipeline.indexed_documents) == 1
    assert pipeline.indexed_documents[0].doc_name == "user_custom_nda.pdf"
    assert not any("sample_" in d.doc_name for d in pipeline.indexed_documents)

    # 3. Grounded query on the uploaded PDF
    res = pipeline.query("What is the governing law of the agreement?", top_k=2)
    assert res.decision_level in ("SAFE", "WARNING")
    assert res.primary_answer is not None
    assert len(res.primary_answer.cited_sources) > 0

    # Verify citation survival: cited source points strictly to the uploaded document locator
    primary_citation = res.primary_answer.cited_sources[0]
    assert "user_custom_nda.pdf" in primary_citation
    assert ":p" in primary_citation  # PDF locator with page number

    # 4. Out-of-domain query produces elevated uncertainty and warning/abstention
    res_unsupported = pipeline.query(
        "What is the punitive damages cap under maritime admiralty law?",
        top_k=2,
    )
    assert res_unsupported.decision_level in ("WARNING", "ABSTAIN")
    assert res_unsupported.uncertainty_score > 0.50

    # 5. Missing-evidence abstention: Pipeline clear resets index completely
    pipeline.clear()
    assert len(pipeline.indexed_documents) == 0
    assert pipeline.total_chunk_count == 0

    # Querying with missing evidence forces hard abstention (answer_allowed=False)
    res_cleared = pipeline.query("What is the governing law?")
    assert not res_cleared.answer_allowed
    assert res_cleared.decision_level == "ABSTAIN"
    assert len(res_cleared.retrieval_results) == 0
    assert "No legal documents are currently indexed" in res_cleared.primary_answer.answer_text




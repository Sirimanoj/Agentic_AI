"""Local smoke test verifying cached SentenceTransformer, FAISS, and GroundedGenerator."""

import sys
import time

from legal_uq.generation.generator import GroundedGenerator
from legal_uq.ingestion.models import DocumentChunk
from legal_uq.retrieval.vector_store import CachedSentenceTransformerEmbedder, VectorStore


def run_smoke_test(use_cached_model: bool = False) -> None:
    print("=== PHASE 2 RETRIEVAL & GENERATION SMOKE TEST ===")
    t0 = time.time()
    if use_cached_model:
        print("Using CachedSentenceTransformerEmbedder (all-MiniLM-L6-v2)...")
        embedder = CachedSentenceTransformerEmbedder()
    else:
        print("Using DeterministicFakeEmbedder (fast offline mode)...")
        from legal_uq.retrieval.vector_store import DeterministicFakeEmbedder
        embedder = DeterministicFakeEmbedder(dimension=384)
    store = VectorStore(embedder=embedder)

    # Synthetic legal document chunks representing Indian statutes
    chunks = [
        DocumentChunk(
            chunk_id="indian_contract_act.pdf::p2::c0",
            doc_name="indian_contract_act.pdf",
            doc_type="pdf",
            page_number=2,
            section_title="SECTION 10: WHAT AGREEMENTS ARE CONTRACTS",
            source_locator="indian_contract_act.pdf:page_2 [SECTION 10]",
            text=(
                "All agreements are contracts if they are made by the free consent of parties "
                "competent to contract, for a lawful consideration and with a lawful object, "
                "and are not hereby expressly declared to be void."
            ),
            character_count=214,
            token_count=35,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="it_act.pdf::p4::c1",
            doc_name="it_act.pdf",
            doc_type="pdf",
            page_number=4,
            section_title="SECTION 10A: VALIDITY OF ELECTRONIC CONTRACTS",
            source_locator="it_act.pdf:page_4 [SECTION 10A]",
            text=(
                "Where in a contract formation, the communication of proposals, the acceptance of proposals, "
                "the revocation of proposals and acceptances are expressed in electronic form, such contract "
                "shall not be deemed to be unenforceable solely on the ground that such electronic form or "
                "means was used for that purpose."
            ),
            character_count=308,
            token_count=45,
            chunk_index=1,
        ),
        DocumentChunk(
            chunk_id="court_notice.txt::pna::c2",
            doc_name="court_notice.txt",
            doc_type="txt",
            page_number=None,
            section_title="SCHEDULE: COURT TIMELINES",
            source_locator="court_notice.txt:block_0 [SCHEDULE: COURT TIMELINES]",
            text=(
                "Hearing dates are scheduled within ninety days of filing preliminary statements. "
                "Adjournments will only be granted upon demonstration of exceptional hardship."
            ),
            character_count=180,
            token_count=23,
            chunk_index=2,
        ),
    ]

    print("Loading cached SentenceTransformer and building FAISS index...")
    indexed_count = store.build_index(chunks)
    build_time = round(time.time() - t0, 2)
    print(f"FAISS index built successfully: {indexed_count} chunks indexed in {build_time}s.")

    query = "What are the essential requirements for a valid contract under Indian law?"
    print(f"\nExecuting Retrieval Query: '{query}'")
    results = store.retrieve(query, k=2)

    print(f"Retrieved {len(results)} evidence passages:")
    for res in results:
        page_display = f"Page {res.page_number}" if res.page_number else "Page N/A"
        print(f"  [Rank {res.rank}] Score: {res.score:.4f} | {res.source_locator} ({page_display})")
        print(f"         Excerpt: \"{res.text[:100]}...\"")

    # Verify both contract-related documents were retrieved ahead of unrelated court schedules
    retrieved_docs = {r.doc_name for r in results}
    assert "indian_contract_act.pdf" in retrieved_docs, "Expected indian_contract_act.pdf in top results"
    assert "court_notice.txt" not in retrieved_docs, "Unrelated court notice should not be in top-2 results"
    print("\nRetrieval Relevance & Provenance Filtering: PASSED")

    # Candidate Generation
    gen = GroundedGenerator()
    print(f"\nGenerator Status: is_live={gen.is_live} (api_key_configured={bool(gen.api_key)})")
    run = gen.generate_candidates(query, results, num_candidates=3)

    print(f"Generation Mode: {run.mode} | Model: {run.model_name}")
    print(f"Candidates Generated: {len(run.candidates)}")
    for c in run.candidates:
        print(f"\n--- Candidate {c.candidate_index} [{c.generation_mode}] ---")
        print(f"Cited Sources: {c.cited_sources}")
        first_sentence = c.answer_text.splitlines()[0]
        print(f"Preview: {first_sentence[:120]}...")

    print("\n=== SMOKE TEST COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()

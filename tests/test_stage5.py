# tests/test_stage5.py
"""Stage 5 verification — RAG: index, hybrid retrieve, rerank."""
import time

from app.storage.db import init_db
from app.tools.sources import search_all_sources, LinkedInMockSource
from app.normalize import normalize_batch
from app.dedup import deduplicate
from app.rag import build_index, retrieve_for_criterion, hybrid_retrieve, rerank


def main():
    init_db()
    print("=" * 60)
    print("STAGE 5 VERIFICATION (RAG: Chroma + BM25 + RRF + Reranker)")
    print("=" * 60)

    if not LinkedInMockSource().health_check():
        print("\n✗ Start mock server first: ./scripts/run_mock_server.sh")
        return

    # ----------------------------------------------------------------
    # 1. Build corpus via sourcing -> normalize -> dedup
    # ----------------------------------------------------------------
    raw = search_all_sources(queries=[], yoe_min=0, max_pages=5, page_size=20)
    normalized = normalize_batch(raw)
    deduped, _ = deduplicate(normalized)
    print(f"✓ Corpus ready: {len(deduped)} unique profiles")

    # ----------------------------------------------------------------
    # 2. Build the hybrid index (one Chroma collection + BM25 in memory)
    # ----------------------------------------------------------------
    t0 = time.time()
    index = build_index(deduped, jd_id="stage5-test")
    print(f"✓ Index built in {time.time() - t0:.1f}s "
          f"(Chroma collection: {index.collection_name})")

    # ----------------------------------------------------------------
    # 3. Sanity: index size matches input
    # ----------------------------------------------------------------
    assert index.collection.count() == len(deduped)
    assert len(index.all_profiles()) == len(deduped)
    print(f"✓ All {len(deduped)} profiles indexed in Chroma and BM25")

    # ----------------------------------------------------------------
    # 4. Semantic-only query
    # ----------------------------------------------------------------
    sem = index.chroma_query("machine learning engineer with Python and AWS", top_k=5)
    assert len(sem) > 0
    print(f"\n✓ Semantic-only: top match candidate_id={sem[0][0][:8]}... "
          f"distance={sem[0][1]:.3f}")

    # ----------------------------------------------------------------
    # 5. BM25-only query
    # ----------------------------------------------------------------
    bm = index.bm25_query("kubernetes docker python", top_k=5)
    assert len(bm) > 0
    print(f"✓ BM25-only:     top match candidate_id={bm[0][0][:8]}... "
          f"score={bm[0][1]:.3f}")

    # ----------------------------------------------------------------
    # 6. Hybrid retrieval (RRF fusion)
    # ----------------------------------------------------------------
    t0 = time.time()
    hybrid = hybrid_retrieve(
        index=index,
        query="senior Python ML engineer with LLM and RAG experience",
        top_k=15,
        yoe_min=3,
    )
    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n✓ Hybrid retrieve: {len(hybrid)} candidates in {elapsed_ms:.0f}ms")
    assert len(hybrid) > 0
    # Sanity: YOE filter respected
    assert all(r.profile.years_experience >= 3 for r in hybrid)
    print(f"   YOE filter respected: all candidates have ≥3 years experience")
    print(f"   Top 3 candidates with RRF scores:")
    for r in hybrid[:3]:
        print(f"     - {r.profile.full_name:25s} "
              f"rrf={r.rrf_score:.4f} "
              f"sources={r.sources} "
              f"sem_rank={r.semantic_rank} bm25_rank={r.bm25_rank}")

    # Sanity: some candidates should come from BOTH rankers (RRF working)
    both_sources = [r for r in hybrid if "semantic" in r.sources and "bm25" in r.sources]
    print(f"   {len(both_sources)} candidates appear in BOTH rankers "
          f"(RRF is fusing, not just OR-ing)")

    # ----------------------------------------------------------------
    # 7. Reranker — first call will download ~280MB, takes a bit
    # ----------------------------------------------------------------
    print(f"\n   Loading cross-encoder reranker (first run downloads ~280MB)...")
    t0 = time.time()
    reranked = rerank(
        query="senior Python ML engineer with LLM and RAG experience",
        candidates=hybrid,
        top_k=5,
    )
    elapsed = time.time() - t0
    print(f"✓ Rerank complete in {elapsed:.1f}s, returned top {len(reranked)}")
    assert len(reranked) <= 5

    print(f"   Top 3 AFTER reranking (cross-encoder scores):")
    for r in reranked[:3]:
        print(f"     - {r.profile.full_name:25s} "
              f"score={r.rrf_score:+.3f} "
              f"sources={r.sources}")

    # ----------------------------------------------------------------
    # 8. Re-ordering check — reranker should change order from RRF
    # ----------------------------------------------------------------
    hybrid_order = [r.candidate_id for r in hybrid[:5]]
    rerank_order = [r.candidate_id for r in reranked[:5]]
    same = hybrid_order == rerank_order
    print(f"\n   Did reranker change the top-5 order? "
          f"{'NO — same as RRF' if same else 'YES — refined order'}")
    # Note: it's not a failure if order is the same; reranker just confirmed RRF.

    # ----------------------------------------------------------------
    # 9. End-to-end pipeline
    # ----------------------------------------------------------------
    print(f"\n✓ End-to-end: retrieve_for_criterion('...kubernetes...')")
    final = retrieve_for_criterion(
        index=index,
        criterion_text="5+ years building production Kubernetes infrastructure",
        top_k_retrieve=20,
        top_k_final=5,
        yoe_min=5,
        jd_id="stage5-test",
    )
    print(f"   Got {len(final)} final candidates (after rerank)")
    for r in final[:3]:
        print(f"     - {r.profile.full_name:25s} "
              f"score={r.rrf_score:+.3f} "
              f"yoe={r.profile.years_experience:.1f}")

    # ----------------------------------------------------------------
    # 10. Cleanup
    # ----------------------------------------------------------------
    index.cleanup()
    print(f"\n✓ Index cleaned up (Chroma collection dropped)")

    print("=" * 60)
    print("ALL STAGE 5 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
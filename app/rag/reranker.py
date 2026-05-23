# app/rag/reranker.py
"""Cross-encoder reranker — refines the hybrid top-K to a final ordering.

Two-stage retrieval is the standard production pattern:
  Stage 1: bi-encoder (Chroma embeddings) + BM25 — fast, broad recall
  Stage 2: cross-encoder — slower, more accurate, only on top candidates

The bi-encoder embeds query and doc independently. The cross-encoder feeds
(query, doc) jointly through a transformer — better precision but 100x slower,
which is why it only runs on the ~20 candidates we already shortlisted.

Model: BAAI/bge-reranker-base (~280MB, CPU-friendly, downloads on first use).
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.obs.events import log_event
from app.rag.retriever import RetrievalResult


RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Lazy-load. First call downloads ~280MB; subsequent calls are instant."""
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 10,
    jd_id: str | None = None,
) -> list[RetrievalResult]:
    """Re-score and re-order candidates with a cross-encoder.

    Returns the top_k candidates with updated `rrf_score` field replaced
    by the cross-encoder relevance score, sorted descending."""
    if not candidates:
        return []

    log_event(jd_id, "rag.reranker", "rerank_start",
              query=query[:80], n_input=len(candidates), top_k=top_k)

    model = _get_model()

    # The cross-encoder eats (query, doc) pairs and outputs a relevance score.
    # We pass each candidate's raw_text, capped to keep latency reasonable.
    pairs = [(query, c.profile.raw_text[:2000]) for c in candidates]
    scores = model.predict(pairs)

    # Pair scores back, sort desc, truncate
    scored: list[tuple[RetrievalResult, float]] = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    reranked: list[RetrievalResult] = []
    for c, s in scored[:top_k]:
        # Replace the rrf_score with the reranker score so downstream consumers
        # see the latest, most reliable relevance signal.
        reranked.append(RetrievalResult(
            candidate_id=c.candidate_id,
            profile=c.profile,
            rrf_score=float(s),
            sources=c.sources + ["reranker"],
            semantic_rank=c.semantic_rank,
            bm25_rank=c.bm25_rank,
        ))

    log_event(jd_id, "rag.reranker", "rerank_end",
              n_output=len(reranked),
              top3=[(r.candidate_id, round(r.rrf_score, 3)) for r in reranked[:3]])

    return reranked
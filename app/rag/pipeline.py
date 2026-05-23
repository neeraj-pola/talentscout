# app/rag/pipeline.py
"""High-level RAG pipeline. Index once, then retrieve+rerank per criterion."""
from __future__ import annotations

from app.models import CommonProfile
from app.obs.events import log_event
from app.rag.indexer import HybridIndex
from app.rag.retriever import hybrid_retrieve, RetrievalResult
from app.rag.reranker import rerank


def build_index(profiles: list[CommonProfile], jd_id: str) -> HybridIndex:
    """Build the per-JD hybrid index. Call once after dedup."""
    log_event(jd_id, "rag.pipeline", "build_index_start", n_profiles=len(profiles))
    idx = HybridIndex(jd_id=jd_id)
    idx.index(profiles)
    log_event(jd_id, "rag.pipeline", "build_index_end")
    return idx


def retrieve_for_criterion(
    index: HybridIndex,
    criterion_text: str,
    top_k_retrieve: int = 20,
    top_k_final: int = 10,
    yoe_min: int = 0,
    yoe_max: int | None = None,
    location: str | None = None,
    jd_id: str | None = None,
) -> list[RetrievalResult]:
    """One end-to-end retrieval for one criterion.

    Steps:
      1. Hybrid retrieve (semantic + BM25 + RRF) -> top 20
      2. Cross-encoder rerank -> top 10
    """
    hybrid = hybrid_retrieve(
        index=index,
        query=criterion_text,
        top_k=top_k_retrieve,
        yoe_min=yoe_min,
        yoe_max=yoe_max,
        location=location,
        jd_id=jd_id,
    )
    reranked = rerank(
        query=criterion_text,
        candidates=hybrid,
        top_k=top_k_final,
        jd_id=jd_id,
    )
    return reranked
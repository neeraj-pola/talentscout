# app/rag/retriever.py
"""Hybrid retrieval = semantic + BM25 fused via Reciprocal Rank Fusion.

Why RRF (not weighted sum):
- No score calibration needed across heterogeneous rankers
- Robust to outlier scores
- Used in production at major search engines
- One line of math

For each candidate i in any ranker's top-k:
    score_i = sum_over_rankers( 1 / (k + rank_in_ranker(i)) )
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.models import CommonProfile
from app.obs.events import log_event
from app.rag.indexer import HybridIndex


RRF_K = 60  # standard constant from the original RRF paper


@dataclass
class RetrievalResult:
    """One retrieved candidate with the reasoning behind it."""
    candidate_id: str
    profile: CommonProfile
    rrf_score: float
    sources: list[str]          # ["semantic", "bm25"]
    semantic_rank: int | None
    bm25_rank: int | None


def _build_chroma_filter(
    location: str | None,
    yoe_min: int,
    yoe_max: int | None,
) -> dict | None:
    """Build a ChromaDB `where` clause. Chroma requires $and for multi-filter."""
    clauses: list[dict] = []

    if yoe_min and yoe_min > 0:
        clauses.append({"years_experience": {"$gte": float(yoe_min)}})

    if yoe_max:
        clauses.append({"years_experience": {"$lte": float(yoe_max)}})

    if location and location.lower() not in ("", "remote", "any"):
        # Loose location match — Chroma only does exact equality on metadata,
        # so we don't filter on location here; the seeded profiles include
        # 'Remote' people too. Real production code would expand to city aliases.
        pass

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def hybrid_retrieve(
    index: HybridIndex,
    query: str,
    top_k: int = 20,
    yoe_min: int = 0,
    yoe_max: int | None = None,
    location: str | None = None,
    jd_id: str | None = None,
) -> list[RetrievalResult]:
    """Run semantic + BM25, fuse via RRF, return top_k candidates."""

    where = _build_chroma_filter(location, yoe_min, yoe_max)

    log_event(jd_id, "rag.retriever", "hybrid_start",
              query=query[:80], top_k=top_k, where=where)

    # Pull more than top_k from each ranker; RRF re-orders the union
    per_ranker_k = max(top_k * 2, 30)

    # Semantic
    semantic = index.chroma_query(query, top_k=per_ranker_k, where=where)
    semantic_ranks = {cid: i + 1 for i, (cid, _) in enumerate(semantic)}

    # BM25 (no metadata filter — we'll filter post-hoc by intersecting
    # with what Chroma allowed, so YOE constraint applies to both)
    bm25 = index.bm25_query(query, top_k=per_ranker_k)
    bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(bm25)}

    # If metadata filter was applied to Chroma, restrict BM25 to those candidates
    if where is not None:
        allowed = set(semantic_ranks.keys()) | {
            cid for cid in bm25_ranks
            if (p := index.get_profile(cid)) is not None and (
                (yoe_min == 0 or p.years_experience >= yoe_min) and
                (yoe_max is None or p.years_experience <= yoe_max)
            )
        }
        bm25_ranks = {cid: r for cid, r in bm25_ranks.items() if cid in allowed}

    # ---------- RRF ----------
    rrf_scores: dict[str, float] = defaultdict(float)
    sources_for: dict[str, list[str]] = defaultdict(list)
    for cid, r in semantic_ranks.items():
        rrf_scores[cid] += 1.0 / (RRF_K + r)
        sources_for[cid].append("semantic")
    for cid, r in bm25_ranks.items():
        rrf_scores[cid] += 1.0 / (RRF_K + r)
        sources_for[cid].append("bm25")

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Materialize results
    results: list[RetrievalResult] = []
    for cid, score in fused[:top_k]:
        prof = index.get_profile(cid)
        if prof is None:
            continue
        results.append(RetrievalResult(
            candidate_id=cid,
            profile=prof,
            rrf_score=score,
            sources=sources_for[cid],
            semantic_rank=semantic_ranks.get(cid),
            bm25_rank=bm25_ranks.get(cid),
        ))

    log_event(jd_id, "rag.retriever", "hybrid_end",
              n_semantic=len(semantic_ranks), n_bm25=len(bm25_ranks),
              n_fused=len(results),
              top3_ids=[r.candidate_id for r in results[:3]])

    return results
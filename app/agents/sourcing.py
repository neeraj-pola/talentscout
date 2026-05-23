# app/agents/sourcing.py
"""Sourcing agent: turn a ParsedJD into a deduped, indexed candidate pool.

This is the bridge between "the JD is parsed" and "we can score candidates".
Steps:
  1. Call all three sources in parallel using derived_search_queries
  2. Normalize source-shaped dicts -> CommonProfile
  3. Deduplicate (blocking + Jaro-Winkler)
  4. Build the per-JD hybrid RAG index (Chroma + BM25)

Returns the deduped profile list + the live HybridIndex (passed to Screening).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models import ParsedJD, CommonProfile
from app.tools.sources import search_all_sources
from app.normalize import normalize_batch
from app.dedup import deduplicate
from app.rag import build_index, HybridIndex
from app.obs.events import log_event


@dataclass
class SourcingResult:
    """Everything the Screening Agent needs to start scoring."""
    profiles: list[CommonProfile]            # deduped CommonProfiles
    index: HybridIndex                        # ready-to-query RAG index
    raw_counts: dict[str, int]                # {"linkedin": 28, "naukri": 38, ...}
    n_normalized: int
    n_after_dedup: int
    n_merges: int
    merge_audit: list[dict]                   # for the UI dedup panel


def run_sourcing(parsed: ParsedJD) -> SourcingResult:
    """End-to-end: search all sources, normalize, dedup, build RAG index."""
    jd_id = str(parsed.jd_id)
    log_event(jd_id, "sourcing_agent", "start",
              n_queries=len(parsed.derived_search_queries),
              queries=parsed.derived_search_queries,
              yoe_min=parsed.yoe_min)

    # ----------------------------------------------------------------
    # 1. Multi-source parallel search
    # ----------------------------------------------------------------
    # We pass the derived_search_queries — the JD Intake agent already
    # distilled the JD into 3-5 short, source-friendly queries.
    # Location: if remote_ok, don't filter by city — many remote candidates
    # have varied locations. If onsite-only, pass the city.
    location_filter = None
    if not parsed.location_constraint.remote_ok and parsed.location_constraint.city:
        location_filter = parsed.location_constraint.city

    raw_by_source = search_all_sources(
        queries=parsed.derived_search_queries,
        location=location_filter,
        yoe_min=parsed.yoe_min,
        max_pages=3,
        page_size=20,
        jd_id=jd_id,
    )
    raw_counts = {k: len(v) for k, v in raw_by_source.items()}

    # ----------------------------------------------------------------
    # 2. Normalize all raw profiles to CommonProfile
    # ----------------------------------------------------------------
    normalized = normalize_batch(raw_by_source, jd_id=jd_id)

    # ----------------------------------------------------------------
    # 3. Deduplicate across sources
    # ----------------------------------------------------------------
    deduped, merge_audit = deduplicate(normalized, jd_id=jd_id)

    # ----------------------------------------------------------------
    # 4. Build the hybrid RAG index (Chroma + BM25) for this JD
    # ----------------------------------------------------------------
    index = build_index(deduped, jd_id=jd_id)

    result = SourcingResult(
        profiles=deduped,
        index=index,
        raw_counts=raw_counts,
        n_normalized=len(normalized),
        n_after_dedup=len(deduped),
        n_merges=len(merge_audit),
        merge_audit=merge_audit,
    )

    log_event(jd_id, "sourcing_agent", "end",
              raw_counts=raw_counts,
              n_normalized=result.n_normalized,
              n_after_dedup=result.n_after_dedup,
              n_merges=result.n_merges)

    return result
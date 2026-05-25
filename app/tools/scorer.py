# app/tools/scorer.py
"""Candidate scoring tool.

Thin wrapper around the screening agent. Exposes a uniform tool interface
so all 6 spec-required operations live under app/tools/ with consistent
contracts.

Public API:
    score_candidates(parsed_jd, index, top_k=None, jd_id=None)
        -> list[ScoredCandidate]

Design note:
    The pipeline's screening node calls `run_screening` directly — that's
    the heavy path that builds the index and runs the full RAG-driven
    scoring loop.

    This tool exists for callers who already have a built hybrid index
    in hand (e.g. refinement's `find_similar` after re-retrieval) and
    want to drive the scoring with a uniform tool interface.

    We do NOT build the index here — index construction depends on the
    RAG module's specific API, which is owned by sourcing. Callers pass
    the index in. Keeps this tool's contract narrow and stable.
"""
from __future__ import annotations

from typing import Any

from app.agents.screening import run_screening
from app.models import ParsedJD, ScoredCandidate
from app.obs.events import log_event


def score_candidates(
    parsed_jd: ParsedJD,
    index: Any,
    top_k_per_criterion: int = 6,
    max_concurrency: int = 8,
    jd_id: str | None = None,
) -> list[ScoredCandidate]:
    """Score candidates against the parsed JD using an existing hybrid index.

    Args:
        parsed_jd:           ParsedJD with criteria list
        index:               Pre-built hybrid index (Chroma + BM25 + reranker).
                             Caller owns construction and cleanup.
        top_k_per_criterion: How many candidates to surface per criterion
                             before scoring (default 6, matches pipeline)
        max_concurrency:     Parallel (criterion, candidate) scoring calls
        jd_id:               Optional JD UUID for event correlation

    Returns:
        List of ScoredCandidate with per-criterion scores + coverage.
        Empty list if the index has no profiles.

    Resilience:
      - LLM retries: handled inside `run_screening` via the instrumented
        chat() wrapper
      - Per-pair scoring failures fall back to score=0 with no-evidence
        flag (handled inside screening)
    """
    jd_id_str = jd_id or str(parsed_jd.jd_id)

    if index is None:
        log_event(jd_id_str, "tool.scorer", "skip", reason="no_index")
        return []

    log_event(jd_id_str, "tool.scorer", "score_start",
              n_criteria=len(parsed_jd.criteria),
              top_k_per_criterion=top_k_per_criterion)

    scored = run_screening(
        parsed=parsed_jd,
        index=index,
        top_k_per_criterion=top_k_per_criterion,
        max_concurrency=max_concurrency,
    )

    log_event(jd_id_str, "tool.scorer", "score_end", n_scored=len(scored))
    return scored
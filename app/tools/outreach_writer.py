# app/tools/outreach_writer.py
"""Outreach drafting tool.

Thin wrapper around `run_outreach`. Used by the pipeline's outreach node
and by the refinement agent's `regenerate_outreach` action. Centralizes
the call site so observability is uniform across all 6 spec-required
tools.

Public API:
    draft_outreach_for_candidate(candidate, profile_text, parsed_jd, jd_title)
        -> OutreachDraft
    draft_outreach_for_shortlist(shortlist, profiles_by_id, parsed_jd, jd_title, n=3)
        -> list[OutreachDraft]

Resilience:
  - LLM retries: handled inside `run_outreach` via the instrumented chat()
  - Empty profile_text: returns None (caller logs + skips that candidate)
  - Empty shortlist: returns []
"""
from __future__ import annotations

from uuid import UUID

from app.agents.outreach import run_outreach, run_outreach_for_top_n
from app.models import ScoredCandidate, ParsedJD, OutreachDraft
from app.obs.events import log_event


def draft_outreach_for_candidate(
    candidate: ScoredCandidate,
    profile_text: str,
    parsed_jd: ParsedJD,
    jd_title: str,
) -> OutreachDraft | None:
    """Draft outreach for a single candidate.

    Returns None when profile_text is empty (the agent needs source
    material to personalize). Caller should log + skip.
    """
    jd_id = str(parsed_jd.jd_id)
    if not profile_text:
        log_event(jd_id, "tool.outreach_writer", "skip",
                  reason="empty_profile_text",
                  candidate=candidate.candidate_name)
        return None

    log_event(jd_id, "tool.outreach_writer", "draft_start",
              candidate=candidate.candidate_name)
    draft = run_outreach(candidate, profile_text, parsed_jd, jd_title)
    log_event(jd_id, "tool.outreach_writer", "draft_end",
              candidate=candidate.candidate_name,
              n_hooks=len(draft.personalization_hooks))
    return draft


def draft_outreach_for_shortlist(
    shortlist: list[ScoredCandidate],
    profiles_by_id: dict[UUID, str],
    parsed_jd: ParsedJD,
    jd_title: str,
    n: int = 3,
) -> list[OutreachDraft]:
    """Draft outreach for the top N candidates in the shortlist.

    Delegates to the existing batch function but logs at the tool boundary
    for observability uniformity. Empty shortlist returns [] cleanly.
    """
    jd_id = str(parsed_jd.jd_id)
    if not shortlist:
        log_event(jd_id, "tool.outreach_writer", "skip",
                  reason="empty_shortlist")
        return []

    log_event(jd_id, "tool.outreach_writer", "batch_start", n=min(n, len(shortlist)))
    drafts = run_outreach_for_top_n(
        shortlist=shortlist,
        profiles_by_id=profiles_by_id,
        parsed=parsed_jd,
        jd_title=jd_title,
        n=n,
    )
    log_event(jd_id, "tool.outreach_writer", "batch_end", n_drafts=len(drafts))
    return drafts
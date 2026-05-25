# app/tools/state_updater.py
"""JD state mutation tool — single point of entry for all DB writes.

Consolidates the various save_*/update_* functions in jd_repo behind a
uniform tool interface. Pipeline nodes (sourcing, screening, ranking,
top_pick, outreach) and the refinement agent all call into here rather
than reaching into jd_repo directly. This gives us:

  - One observability boundary: every state write logs a tool event
  - One swap point: if we change storage backend (SQLite → Postgres),
    we change one module
  - Spec compliance: "updating JD state" lives as a named tool with
    documented contract

Public API:
    update_status(jd_id, status)
    save_parsed_jd_tool(jd_id, parsed)
    save_sourcing_tool(jd_id, summary, merge_audit)
    save_profiles_tool(jd_id, profiles)
    save_shortlist_tool(jd_id, shortlist)
    save_top_pick_tool(jd_id, top_pick)
    save_outreach_tool(jd_id, drafts)
    save_guardrail_verdict_tool(jd_id, verdict)
    save_refinement_state_tool(jd_id, history, filter_stack, total_cost)

All functions are idempotent — calling twice with the same args is safe.
All return None (writes-only).
"""
from __future__ import annotations

from uuid import UUID

from app.models import (
    ParsedJD, GuardrailResult, ScoredCandidate,
    TopPickRecommendation, OutreachDraft, CommonProfile,
)
from app.models.jd import JDStatus
from app.storage import jd_repo
from app.obs.events import log_event


# ============================================================
# Status transition
# ============================================================

def update_status(jd_id: str | UUID, status: JDStatus) -> None:
    """Update the JD's lifecycle status. Idempotent."""
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "update_status",
              status=status.value if hasattr(status, "value") else str(status))
    jd_repo.update_jd_status(jd_id_str, status)


# ============================================================
# Pipeline output writes
# ============================================================

def save_parsed_jd_tool(jd_id: str | UUID, parsed: ParsedJD) -> None:
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_parsed_jd",
              n_criteria=len(parsed.criteria))
    jd_repo.save_parsed_jd(jd_id_str, parsed)


def save_sourcing_tool(
    jd_id: str | UUID,
    sourcing_summary: dict,
    merge_audit: list[dict],
) -> None:
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_sourcing",
              n_after_dedup=sourcing_summary.get("n_after_dedup", 0),
              n_merges=sourcing_summary.get("n_merges", 0))
    jd_repo.save_sourcing(jd_id_str, sourcing_summary=sourcing_summary,
                          merge_audit=merge_audit)


def save_profiles_tool(
    jd_id: str | UUID,
    profiles: list[CommonProfile],
) -> None:
    jd_id_str = str(jd_id)
    n_summarized = sum(1 for p in profiles if (p.summary or "").strip())
    log_event(jd_id_str, "tool.state_updater", "save_profiles",
              n=len(profiles), n_with_summary=n_summarized)
    jd_repo.save_profiles(jd_id_str, profiles)


def save_shortlist_tool(
    jd_id: str | UUID,
    shortlist: list[ScoredCandidate],
) -> None:
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_shortlist",
              n=len(shortlist))
    jd_repo.save_shortlist(jd_id_str, shortlist)


def save_top_pick_tool(
    jd_id: str | UUID,
    top_pick: TopPickRecommendation,
) -> None:
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_top_pick",
              candidate=top_pick.candidate_name)
    jd_repo.save_top_pick(jd_id_str, top_pick)


def save_outreach_tool(
    jd_id: str | UUID,
    drafts: list[OutreachDraft],
) -> None:
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_outreach",
              n_drafts=len(drafts))
    jd_repo.save_outreach(jd_id_str, drafts)


def save_guardrail_verdict_tool(
    jd_id: str | UUID,
    verdict: GuardrailResult,
) -> None:
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_guardrail_verdict",
              is_discriminatory=verdict.is_discriminatory)
    jd_repo.save_guardrail_verdict(jd_id_str, verdict)


# ============================================================
# Refinement state
# ============================================================

def save_refinement_state_tool(
    jd_id: str | UUID,
    conversation_history: list[dict],
    filter_stack: list[dict],
    total_refinement_cost_usd: float,
) -> None:
    """Persist refinement state for cross-session reload.

    State shape:
      {
        "conversation_history": [{role, content, timestamp, ...}, ...],
        "filter_stack":         [{type, params, ...}, ...],
        "total_refinement_cost_usd": float
      }
    """
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.state_updater", "save_refinement_state",
              n_turns=len(conversation_history),
              n_filters=len(filter_stack),
              total_cost_usd=round(total_refinement_cost_usd, 6))
    jd_repo.save_refinement_state(
        jd_id_str,
        conversation_history=conversation_history,
        filter_stack=filter_stack,
        total_refinement_cost_usd=total_refinement_cost_usd,
    )
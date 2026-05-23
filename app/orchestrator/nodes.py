# app/orchestrator/nodes.py
"""Graph nodes. Each one wraps an agent and updates state.

Pattern:
    def node_x(state) -> dict:
        # read what we need from state
        # call the agent (pure, doesn't know about LangGraph)
        # return a dict of fields to merge into state
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from app.models import (
    JD, ParsedJD, GuardrailResult, ScoredCandidate,
    TopPickRecommendation, OutreachDraft,
)
from app.agents import (
    screen_jd, parse_jd, run_sourcing, run_screening,
    run_ranking, run_top_pick, run_outreach_for_top_n,
)
from app.storage.jd_repo import (
    update_jd_status, save_parsed_jd, save_shortlist,
    save_top_pick, save_outreach,
)
from app.models.jd import JDStatus
from app.obs.events import log_event
from app.orchestrator.state import TalentScoutState


# ============================================================
# Per-JD cache for non-serializable objects (Chroma index, profiles)
#
# LangGraph checkpointer can only persist JSON-serializable state. The live
# Chroma index and CommonProfile objects with their indexed embeddings stay
# in this in-process cache, keyed by jd_id. If the process restarts, we'd
# rebuild the index from the deduped_profiles dicts that ARE persisted.
# ============================================================

_index_cache: dict[str, Any] = {}              # jd_id -> HybridIndex
_profiles_cache: dict[str, list[Any]] = {}     # jd_id -> [CommonProfile, ...]


def _profile_to_dict(p) -> dict:
    """Serialize a CommonProfile for state storage."""
    return p.model_dump(mode="json")


# ============================================================
# Node: guardrails
# ============================================================

def node_guardrails(state: TalentScoutState) -> dict:
    """Reject discriminatory JDs before any work happens."""
    jd = JD.model_validate(state["jd"])
    log_event(state["jd_id"], "graph", "node_start", node="guardrails")

    verdict: GuardrailResult = screen_jd(jd)

    log_event(state["jd_id"], "graph", "node_end", node="guardrails",
              is_discriminatory=verdict.is_discriminatory)

    if verdict.is_discriminatory:
        update_jd_status(state["jd_id"], JDStatus.REJECTED_GUARDRAIL)
        return {
            "guardrail_verdict": verdict.model_dump(mode="json"),
            "status": "rejected_guardrail",
            "halt_reason": f"JD failed guardrails: {'; '.join(verdict.reasons[:3])}",
        }

    return {"guardrail_verdict": verdict.model_dump(mode="json")}


def route_after_guardrails(state: TalentScoutState) -> str:
    """If guardrails rejected the JD, jump straight to END."""
    if state.get("status") == "rejected_guardrail":
        return "halt"
    return "continue"


# ============================================================
# Node: jd_intake
# ============================================================

def node_jd_intake(state: TalentScoutState) -> dict:
    jd = JD.model_validate(state["jd"])
    log_event(state["jd_id"], "graph", "node_start", node="jd_intake")

    parsed = parse_jd(jd)

    save_parsed_jd(state["jd_id"], parsed)
    update_jd_status(state["jd_id"], JDStatus.PARSED)

    log_event(state["jd_id"], "graph", "node_end", node="jd_intake",
              n_criteria=len(parsed.criteria))

    return {"parsed_jd": parsed.model_dump(mode="json")}


# ============================================================
# Node: sourcing
# ============================================================

def node_sourcing(state: TalentScoutState) -> dict:
    parsed = ParsedJD.model_validate(state["parsed_jd"])
    log_event(state["jd_id"], "graph", "node_start", node="sourcing")

    update_jd_status(state["jd_id"], JDStatus.SOURCING)

    src = run_sourcing(parsed)

    # Stash the live objects in module cache (not serializable)
    _index_cache[state["jd_id"]] = src.index
    _profiles_cache[state["jd_id"]] = src.profiles

    log_event(state["jd_id"], "graph", "node_end", node="sourcing",
              n_normalized=src.n_normalized, n_after_dedup=src.n_after_dedup)

    return {
        "sourcing_result": {
            "raw_counts": src.raw_counts,
            "n_normalized": src.n_normalized,
            "n_after_dedup": src.n_after_dedup,
            "n_merges": src.n_merges,
        },
        "deduped_profiles": [_profile_to_dict(p) for p in src.profiles],
        "merge_audit": src.merge_audit,
    }


# ============================================================
# Node: screening
# ============================================================

def node_screening(state: TalentScoutState) -> dict:
    parsed = ParsedJD.model_validate(state["parsed_jd"])
    index = _index_cache.get(state["jd_id"])
    if index is None:
        return {"status": "failed",
                "error": "No live index found for screening — sourcing didn't run."}

    log_event(state["jd_id"], "graph", "node_start", node="screening")
    update_jd_status(state["jd_id"], JDStatus.SCREENING)

    scored = run_screening(
        parsed=parsed,
        index=index,
        top_k_per_criterion=6,
        max_concurrency=8,
    )

    log_event(state["jd_id"], "graph", "node_end", node="screening",
              n_scored=len(scored))

    return {
        "scored_candidates": [sc.model_dump(mode="json") for sc in scored],
    }


# ============================================================
# Node: ranking
# ============================================================

def node_ranking(state: TalentScoutState) -> dict:
    parsed = ParsedJD.model_validate(state["parsed_jd"])
    jd = JD.model_validate(state["jd"])
    scored = [ScoredCandidate.model_validate(d) for d in state["scored_candidates"]]

    log_event(state["jd_id"], "graph", "node_start", node="ranking")

    shortlist = run_ranking(
        scored=scored,
        parsed=parsed,
        jd_title=jd.title,
    )

    save_shortlist(state["jd_id"], shortlist)
    update_jd_status(state["jd_id"], JDStatus.SHORTLISTED)

    log_event(state["jd_id"], "graph", "node_end", node="ranking",
              n_shortlist=len(shortlist))

    return {"shortlist": [sc.model_dump(mode="json") for sc in shortlist]}


# ============================================================
# Node: top_pick
# ============================================================

def node_top_pick(state: TalentScoutState) -> dict:
    parsed = ParsedJD.model_validate(state["parsed_jd"])
    jd = JD.model_validate(state["jd"])
    shortlist = [ScoredCandidate.model_validate(d) for d in state["shortlist"]]

    log_event(state["jd_id"], "graph", "node_start", node="top_pick")

    pick = run_top_pick(shortlist=shortlist, parsed=parsed, jd_title=jd.title)
    if pick is None:
        return {"top_pick": None,
                "status": "failed",
                "error": "Top-pick agent returned no recommendation."}

    save_top_pick(state["jd_id"], pick)

    log_event(state["jd_id"], "graph", "node_end", node="top_pick",
              recommended=pick.candidate_name)

    return {"top_pick": pick.model_dump(mode="json")}


# ============================================================
# Node: outreach
# ============================================================

def node_outreach(state: TalentScoutState) -> dict:
    parsed = ParsedJD.model_validate(state["parsed_jd"])
    jd = JD.model_validate(state["jd"])
    shortlist = [ScoredCandidate.model_validate(d) for d in state["shortlist"]]

    if not state.get("top_pick"):
        return {"outreach_drafts": []}

    top_pick = TopPickRecommendation.model_validate(state["top_pick"])

    # Reorder shortlist so the top pick is first
    recommended = next(
        (c for c in shortlist if c.profile_id == top_pick.recommended_candidate_id),
        None,
    )
    if recommended is None:
        return {"outreach_drafts": []}

    others = [c for c in shortlist if c.profile_id != top_pick.recommended_candidate_id]
    outreach_order = [recommended] + others

    # Look up raw_text for each shortlisted candidate from the cache
    profiles = _profiles_cache.get(state["jd_id"], [])
    profiles_by_id = {p.id: p.raw_text for p in profiles}

    log_event(state["jd_id"], "graph", "node_start", node="outreach")

    drafts = run_outreach_for_top_n(
        shortlist=outreach_order,
        profiles_by_id=profiles_by_id,
        parsed=parsed,
        jd_title=jd.title,
        n=1,  # just the top pick — bump up in production
    )

    save_outreach(state["jd_id"], drafts)

    log_event(state["jd_id"], "graph", "node_end", node="outreach",
              n_drafts=len(drafts))

    return {
        "outreach_drafts": [d.model_dump(mode="json") for d in drafts],
        "status": "completed",
    }


# ============================================================
# Cleanup helpers (called by the public run_pipeline)
# ============================================================

def cleanup_caches(jd_id: str) -> None:
    """Drop the per-JD live objects. Call after pipeline completes or fails."""
    idx = _index_cache.pop(jd_id, None)
    _profiles_cache.pop(jd_id, None)
    if idx is not None:
        try:
            idx.cleanup()
        except Exception:
            pass
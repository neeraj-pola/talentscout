# app/orchestrator/state.py
"""LangGraph state for the TalentScout pipeline.

The state is a TypedDict because LangGraph requires it (so it can serialize
for checkpointing). Pydantic objects inside the state are converted to dicts
when persisted to SQLite, and back when resumed.

Every node:
  1. Reads the fields it needs from `state`
  2. Returns a partial dict — LangGraph merges this into the running state
  3. Does NOT mutate `state` in place (the framework relies on immutability)
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class TalentScoutState(TypedDict, total=False):
    """The single piece of state that flows through the pipeline graph.

    `total=False` means every field is optional — nodes set the fields they
    produce. This is the LangGraph convention.
    """

    # ----- input -----
    jd_id: str
    jd: dict                          # JD pydantic serialized

    # ----- guardrails -----
    guardrail_verdict: dict | None    # GuardrailResult serialized

    # ----- jd_intake -----
    parsed_jd: dict | None            # ParsedJD serialized

    # ----- sourcing -----
    sourcing_result: dict | None      # SourcingResult summary fields (not the full index)
    deduped_profiles: list[dict]      # CommonProfile list serialized
    merge_audit: list[dict]

    # ----- screening -----
    scored_candidates: list[dict]     # ScoredCandidate list serialized

    # ----- ranking -----
    shortlist: list[dict]             # ScoredCandidate enriched with rationale

    # ----- top pick -----
    top_pick: dict | None             # TopPickRecommendation serialized

    # ----- outreach -----
    outreach_drafts: list[dict]       # OutreachDraft list serialized

    # ----- control / status -----
    status: str                       # "running" | "rejected_guardrail" | "completed" | "failed"
    error: str | None
    halt_reason: str | None

    # ----- conversational refinement (chat messages between user and pipeline) -----
    messages: Annotated[list[dict], add_messages]


def empty_state(jd_id: str, jd_dict: dict) -> TalentScoutState:
    """Initial state when a fresh JD enters the graph."""
    return {
        "jd_id": jd_id,
        "jd": jd_dict,
        "status": "running",
        "error": None,
        "halt_reason": None,
        "messages": [],
        "deduped_profiles": [],
        "merge_audit": [],
        "scored_candidates": [],
        "shortlist": [],
        "outreach_drafts": [],
    }
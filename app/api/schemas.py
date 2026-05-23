# app/api/schemas.py
"""Request and response Pydantic schemas for the REST API.

Kept separate from app/models/* (domain models) so API contracts can evolve
independently from internal data structures.
"""
from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================
# Requests
# ============================================================

class CreateJDRequest(BaseModel):
    """POST /jds payload — matches the intake form fields the UI will collect."""
    title: str = Field(min_length=2, max_length=200)
    description: str = Field(min_length=20, max_length=10_000)
    must_have_skills: list[str] = Field(min_length=1)
    nice_to_have_skills: list[str] = []
    min_years_experience: int = Field(ge=0, le=40)
    max_years_experience: int | None = Field(default=None, ge=0, le=40)
    location: str
    remote_ok: bool = False
    employment_type: Literal["full_time", "contract", "intern"]
    target_hiring_date: date


class CloseJDRequest(BaseModel):
    """POST /jds/{id}/close payload."""
    closed_by: str = Field(min_length=2, description="Username/email of recruiter")
    candidate_id: UUID = Field(description="Candidate to close the JD with")


# ============================================================
# Responses
# ============================================================

class JDSummary(BaseModel):
    """Lightweight JD record for listing pages."""
    id: UUID
    title: str
    location: str
    status: str
    created_at: str
    target_hiring_date: str
    closed_at: str | None = None
    closed_by: str | None = None


class PipelineRunResponse(BaseModel):
    """Returned by POST /jds — the full pipeline state after running."""
    jd_id: UUID
    status: str
    halt_reason: str | None = None
    guardrail_verdict: dict | None = None
    parsed_jd: dict | None = None
    sourcing_result: dict | None = None
    shortlist: list[dict] = []
    top_pick: dict | None = None
    outreach_drafts: list[dict] = []


class JDDetailResponse(BaseModel):
    """Returned by GET /jds/{id} — everything the UI needs to render a JD page."""
    jd: dict
    status: str
    parsed_jd: dict | None
    shortlist: list[dict]
    top_pick: dict | None
    outreach_drafts: list[dict]
    merge_audit: list[dict] = []
    sourcing_summary: dict | None = None
    cost_summary: dict
    events: list[dict]


class AuditRecordResponse(BaseModel):
    jd_id: str
    candidate_id: str
    closed_by: str
    closed_at: str | None
    justification: str
    final_ranking_snapshot: list[str]
    total_cost_usd: float
    total_tokens: int
    total_llm_calls: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mock_server_reachable: bool
    db_initialized: bool
# app/models/scoring.py
from uuid import UUID

from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    """Per-criterion score with evidence — the core auditable unit."""
    criterion_id: str
    criterion_text: str
    score: float = Field(ge=0.0, le=1.0)
    evidence: str  # direct quote from profile (or "No evidence found")
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    has_evidence: bool = True  # False when "No evidence found." or scoring failed


class ScoredCandidate(BaseModel):
    profile_id: UUID
    candidate_name: str
    criterion_scores: list[CriterionScore]
    must_have_score: float
    nice_to_have_score: float
    must_have_coverage: float = 1.0       # fraction of must-haves with evidence
    nice_to_have_coverage: float = 1.0    # fraction of nice-to-haves with evidence
    overall_score: float
    overall_rationale: str = ""
    red_flags: list[str] = []
    has_must_have_gap: bool = False


class TopPickRecommendation(BaseModel):
    recommended_candidate_id: UUID
    candidate_name: str
    justification: str
    key_tradeoff_vs_runner_up: str
    runner_up_id: UUID | None = None


class OutreachDraft(BaseModel):
    candidate_id: UUID
    subject: str
    linkedin_inmail: str
    email_body: str
    personalization_hooks: list[str]


class AuditRecord(BaseModel):
    jd_id: UUID
    candidate_id: UUID
    closed_by: str
    closed_at: str  # ISO datetime
    justification: str
    final_ranking_snapshot: list[UUID]
    total_cost_usd: float
    total_tokens: int
    total_llm_calls: int
# app/models/jd.py
from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JDStatus(str, Enum):
    DRAFT = "draft"
    PARSED = "parsed"
    SOURCING = "sourcing"
    SCREENING = "screening"
    SHORTLISTED = "shortlisted"
    CLOSED = "closed"
    REJECTED_GUARDRAIL = "rejected_guardrail"
    FAILED = "failed"


class JD(BaseModel):
    """The raw JD as captured from the intake form."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str] = []
    min_years_experience: int
    max_years_experience: int | None = None
    location: str
    remote_ok: bool = False
    employment_type: Literal["full_time", "contract", "intern"]
    target_hiring_date: date
    status: JDStatus = JDStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    closed_by: str | None = None
    closed_with_candidate_id: UUID | None = None


class Criterion(BaseModel):
    """A single scoreable requirement extracted from the JD."""
    id: str  # e.g. "must_001"
    text: str
    weight: float  # 1.0 for must, 0.4 for nice
    category: Literal["skill", "experience", "domain", "education", "location"]
    is_must_have: bool


class LocationConstraint(BaseModel):
    city: str | None = None
    country: str | None = None
    remote_ok: bool = False
    hybrid_ok: bool = True


class ParsedJD(BaseModel):
    """The structured form produced by the JD Intake Agent."""
    jd_id: UUID
    seniority: Literal["junior", "mid", "senior", "staff", "principal"]
    criteria: list[Criterion]
    location_constraint: LocationConstraint
    yoe_min: int
    yoe_max: int | None = None
    derived_search_queries: list[str]  # 3-5 queries for sourcing


class GuardrailResult(BaseModel):
    is_discriminatory: bool
    reasons: list[str] = []
    flagged_phrases: list[str] = []
    severity: Literal["none", "low", "medium", "high"] = "none"
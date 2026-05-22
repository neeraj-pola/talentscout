# app/models/profile.py
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str
    company: str
    duration_months: int
    description: str = ""
    start_year: int | None = None
    end_year: int | None = None  # None means current


class Education(BaseModel):
    degree: str
    institution: str
    field: str = ""
    graduation_year: int | None = None


class CommonProfile(BaseModel):
    """The normalized schema all sources are mapped to."""
    id: UUID = Field(default_factory=uuid4)
    source: Literal["linkedin", "naukri", "ats"]
    source_id: str  # original ID from the source
    full_name: str
    headline: str = ""
    location: str
    years_experience: float
    skills: list[str]  # normalized lowercase
    experiences: list[Experience] = []
    education: list[Education] = []
    raw_text: str  # for embedding
    contact_email: str | None = None
    metadata: dict = {}  # source-specific extras
    merged_from: list[UUID] = []  # if this is a deduped record


class RawProfileBatch(BaseModel):
    """What a source tool returns."""
    source: Literal["linkedin", "naukri", "ats"]
    profiles: list[dict]  # raw, source-shaped dicts
    next_page: int | None = None
    total_count: int | None = None
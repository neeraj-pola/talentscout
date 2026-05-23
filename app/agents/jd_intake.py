# app/agents/jd_intake.py
"""JD Intake agent: turn the free-text JD into a structured ParsedJD.

What it produces:
  - seniority (from YOE + title heuristics, confirmed by LLM)
  - explicit list of Criterion objects (each scoreable, with must/nice weight)
  - location constraint (city, remote-ok, hybrid-ok)
  - 3-5 derived search queries used by the Sourcing Agent

Every downstream agent operates on ParsedJD, NEVER on the raw JD text.
This is the single point where natural language becomes structured logic.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.config import settings
from app.models import JD, ParsedJD, Criterion, LocationConstraint
from app.obs.events import log_event
from app.obs.llm_client import chat


# ============================================================
# Wire-format schemas (what the LLM returns)
#
# We use a slightly different shape from the domain ParsedJD because the
# LLM is better at producing flat fields than nested objects. The agent
# function below converts wire -> domain.
# ============================================================

class _WireCriterion(BaseModel):
    text: str = Field(description="The criterion phrased as a check, e.g. "
                                  "'5+ years of production Python experience'")
    category: Literal["skill", "experience", "domain", "education", "location"]
    is_must_have: bool


class _WireLocation(BaseModel):
    city: str | None = Field(description="City if specified, else null")
    country: str | None = Field(description="Country if specified, else null")
    remote_ok: bool
    hybrid_ok: bool


class _WireParsedJD(BaseModel):
    seniority: Literal["junior", "mid", "senior", "staff", "principal"]
    yoe_min: int = Field(ge=0, le=40)
    yoe_max: int | None = Field(default=None, ge=0, le=40)
    criteria: list[_WireCriterion] = Field(
        description="At LEAST 4 criteria total. Cover must-have skills, "
                    "core experience, and any nice-to-haves. Each one is a "
                    "single, atomic, scoreable requirement."
    )
    location_constraint: _WireLocation
    derived_search_queries: list[str] = Field(
        description="3-5 short queries (3-7 words each) that would surface "
                    "matching candidates from LinkedIn/Naukri/ATS-style search. "
                    "Bias toward concrete role+skill combinations."
    )


# ============================================================
# Prompt
# ============================================================

_INTAKE_PROMPT = """\
You are a senior recruiter parsing a job description into structured criteria \
that a downstream scoring agent will evaluate against candidate profiles.

Your output must:
1. Identify seniority from YOE + title (junior 0-3, mid 3-6, senior 6-10, \
staff 10-14, principal 14+).
2. Break the JD into ATOMIC criteria — each one a single check.
   - GOOD: "5+ years of production Python experience"
   - GOOD: "Hands-on experience with AWS or Azure"
   - BAD:  "Strong technical skills" (not scoreable — too vague)
   - BAD:  "Python and AWS and Kubernetes" (3 checks in one — split them)
3. Mark each criterion as must-have (true) or nice-to-have (false).
   - The recruiter explicitly lists must-haves and nice-to-haves. Honor that.
   - Phrases like "required", "must have", "essential" → must-have.
   - Phrases like "plus", "bonus", "preferred" → nice-to-have.
4. Categorize each criterion:
   - "skill" — a specific technology, tool, framework, language
   - "experience" — years, scope, leadership, scale
   - "domain" — industry knowledge (e.g., fintech, healthcare)
   - "education" — degree, university, certification
   - "location" — geographic requirement (rare; usually goes in location_constraint)
5. Extract the location constraint as city/country/remote_ok/hybrid_ok.
6. Produce 3-5 derived_search_queries that would surface matching candidates \
on platforms like LinkedIn or Naukri. Each is 3-7 words. Bias toward concrete \
role+skill combinations. Example: ["senior python ML engineer", \
"machine learning AWS", "LLM RAG production"].

Do NOT invent requirements that aren't in the JD. Do NOT add diversity preferences. \
Stick to what the recruiter wrote."""


# ============================================================
# Public API
# ============================================================

def parse_jd(jd: JD) -> ParsedJD:
    """Run the JD Intake agent. Returns a domain-shaped ParsedJD."""
    jd_id = str(jd.id)
    log_event(jd_id, "jd_intake", "parse_start", title=jd.title)

    user_msg = _format_jd_for_prompt(jd)

    wire: _WireParsedJD = chat(
        messages=[
            {"role": "system", "content": _INTAKE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        model=settings.openai_model_heavy,   # parsing is foundational — spend on it
        jd_id=jd_id,
        agent="jd_intake",
        response_format=_WireParsedJD,
        temperature=0.1,
        max_tokens=1500,
    )

    parsed = _wire_to_domain(wire, jd)

    log_event(jd_id, "jd_intake", "parse_end",
              seniority=parsed.seniority,
              n_criteria=len(parsed.criteria),
              n_must=sum(1 for c in parsed.criteria if c.is_must_have),
              n_nice=sum(1 for c in parsed.criteria if not c.is_must_have),
              search_queries=parsed.derived_search_queries)

    return parsed


# ============================================================
# Helpers
# ============================================================

def _format_jd_for_prompt(jd: JD) -> str:
    return f"""\
Title: {jd.title}

Description:
{jd.description}

Recruiter-flagged must-have skills: {', '.join(jd.must_have_skills) or '(none listed)'}
Recruiter-flagged nice-to-have skills: {', '.join(jd.nice_to_have_skills) or '(none listed)'}

Minimum years of experience: {jd.min_years_experience}
Maximum years of experience: {jd.max_years_experience or '(none specified)'}

Location: {jd.location}
Remote OK: {jd.remote_ok}
Employment type: {jd.employment_type}
Target hiring date: {jd.target_hiring_date.isoformat()}
"""


def _wire_to_domain(wire: _WireParsedJD, jd: JD) -> ParsedJD:
    """Convert wire schema -> domain ParsedJD with IDs and weights."""
    criteria: list[Criterion] = []
    must_idx = 0
    nice_idx = 0
    for wc in wire.criteria:
        if wc.is_must_have:
            must_idx += 1
            cid = f"must_{must_idx:03d}"
            weight = settings.must_have_weight
        else:
            nice_idx += 1
            cid = f"nice_{nice_idx:03d}"
            weight = settings.nice_have_weight
        criteria.append(Criterion(
            id=cid,
            text=wc.text,
            weight=weight,
            category=wc.category,
            is_must_have=wc.is_must_have,
        ))

    return ParsedJD(
        jd_id=jd.id,
        seniority=wire.seniority,
        criteria=criteria,
        location_constraint=LocationConstraint(
            city=wire.location_constraint.city,
            country=wire.location_constraint.country,
            remote_ok=wire.location_constraint.remote_ok,
            hybrid_ok=wire.location_constraint.hybrid_ok,
        ),
        yoe_min=wire.yoe_min,
        yoe_max=wire.yoe_max,
        derived_search_queries=wire.derived_search_queries,
    )
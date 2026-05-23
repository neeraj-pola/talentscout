# app/normalize/linkedin.py
"""Map a raw LinkedIn profile dict -> CommonProfile."""
from app.models import CommonProfile, Experience, Education
from app.normalize.skills import canonicalize_skills


def normalize_linkedin(raw: dict) -> CommonProfile:
    """Map LinkedIn-shaped dict to the common schema.

    LinkedIn uses:
      - firstName + lastName for the name
      - headline as the one-line title
      - summary as the bio
      - location: {"name": "..."} (nested object)
      - skills: list of {"name": "..."} dicts
      - positions: list of {"title", "companyName", "durationMonths", "description"}
      - educations: list of {"schoolName", "degreeName", "graduationYear"}
      - yearsOfExperience (top-level)
    """
    full_name = f"{raw.get('firstName', '').strip()} {raw.get('lastName', '').strip()}".strip()

    skills_raw = [s.get("name", "") for s in raw.get("skills", []) if isinstance(s, dict)]
    skills = canonicalize_skills(skills_raw)

    experiences = [
        Experience(
            title=p.get("title", ""),
            company=p.get("companyName", ""),
            duration_months=int(p.get("durationMonths", 0)),
            description=p.get("description", ""),
        )
        for p in raw.get("positions", [])
    ]
    education = [
        Education(
            degree=e.get("degreeName", ""),
            institution=e.get("schoolName", ""),
            graduation_year=e.get("graduationYear"),
        )
        for e in raw.get("educations", [])
    ]

    raw_text = _build_raw_text(
        name=full_name,
        headline=raw.get("headline", ""),
        summary=raw.get("summary", ""),
        skills=skills,
        experiences=experiences,
        education=education,
    )

    return CommonProfile(
        source="linkedin",
        source_id=raw["linkedin_id"],
        full_name=full_name,
        headline=raw.get("headline", ""),
        location=raw.get("location", {}).get("name", ""),
        years_experience=float(raw.get("yearsOfExperience", 0)),
        skills=skills,
        experiences=experiences,
        education=education,
        raw_text=raw_text,
        metadata={
            "publicProfileUrl": raw.get("publicProfileUrl"),
            "connections": raw.get("connections"),
            "industry": raw.get("industry"),
            "_canonical_id": raw.get("_canonical_id"),  # ground truth for tests
        },
    )


def _build_raw_text(name, headline, summary, skills, experiences, education) -> str:
    """Build the text blob that will get embedded. Order matters for retrieval."""
    parts = [
        f"Name: {name}",
        f"Headline: {headline}",
        f"Summary: {summary}",
        f"Skills: {', '.join(skills)}",
        "Experience:",
        *[
            f"  - {e.title} at {e.company} ({e.duration_months} months): {e.description}"
            for e in experiences
        ],
        "Education:",
        *[f"  - {ed.degree}, {ed.institution} ({ed.graduation_year})" for ed in education],
    ]
    return "\n".join(parts)
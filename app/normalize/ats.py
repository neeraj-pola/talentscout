# app/normalize/ats.py
"""Map a raw internal-ATS profile dict -> CommonProfile."""
from app.models import CommonProfile, Experience, Education
from app.normalize.skills import canonicalize_skills


def normalize_ats(raw: dict) -> CommonProfile:
    """Map ATS-shaped dict to the common schema.

    ATS uses:
      - full_name (single field)
      - role for the headline
      - bio for the summary
      - city (flat string)
      - tags: list of strings (flat, no wrapping objects)
      - work_history: list of {"role", "employer", "months", "summary"}
      - academics: list of {"qualification", "school", "year"}
      - tenure_years (float)
    """
    name = raw.get("full_name", "").strip()
    skills = canonicalize_skills(raw.get("tags", []))

    experiences = [
        Experience(
            title=w.get("role", ""),
            company=w.get("employer", ""),
            duration_months=int(w.get("months", 0)),
            description=w.get("summary", ""),
        )
        for w in raw.get("work_history", [])
    ]
    education = [
        Education(
            degree=e.get("qualification", ""),
            institution=e.get("school", ""),
            graduation_year=e.get("year"),
        )
        for e in raw.get("academics", [])
    ]

    raw_text = _build_raw_text(
        name=name,
        headline=raw.get("role", ""),
        summary=raw.get("bio", ""),
        skills=skills,
        experiences=experiences,
        education=education,
    )

    return CommonProfile(
        source="ats",
        source_id=raw["ats_id"],
        full_name=name,
        headline=raw.get("role", ""),
        location=raw.get("city", ""),
        years_experience=float(raw.get("tenure_years", 0)),
        skills=skills,
        experiences=experiences,
        education=education,
        raw_text=raw_text,
        contact_email=raw.get("email"),
        metadata={
            "phone_number": raw.get("phone_number"),
            "source_channel": raw.get("source_channel"),
            "_canonical_id": raw.get("_canonical_id"),
        },
    )


def _build_raw_text(name, headline, summary, skills, experiences, education) -> str:
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
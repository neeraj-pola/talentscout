# app/normalize/naukri.py
"""Map a raw Naukri profile dict -> CommonProfile."""
from app.models import CommonProfile, Experience, Education
from app.normalize.skills import canonicalize_skills, parse_skill_string


def _parse_years(s: str) -> float:
    """Parse '5.2 years' -> 5.2"""
    try:
        return float(s.split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0.0


def _parse_months(s: str) -> int:
    """Parse '24 months' -> 24"""
    try:
        return int(s.split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0


def normalize_naukri(raw: dict) -> CommonProfile:
    """Map Naukri-shaped dict to the common schema.

    Naukri uses:
      - candidateName for the full name
      - currentDesignation for the headline
      - aboutSelf for the bio
      - currentLocation (flat string, not nested)
      - keySkills: COMMA-SEPARATED STRING (not a list!)
      - workEx: list of {"designation", "organization", "tenure", "responsibilities"}
      - education: list of {"course", "university", "passingYear"}
      - totalExp: "5.2 years" (string, needs parsing)
    """
    name = raw.get("candidateName", "").strip()
    skills_raw = parse_skill_string(raw.get("keySkills", ""))
    skills = canonicalize_skills(skills_raw)

    experiences = [
        Experience(
            title=w.get("designation", ""),
            company=w.get("organization", ""),
            duration_months=_parse_months(w.get("tenure", "0 months")),
            description=w.get("responsibilities", ""),
        )
        for w in raw.get("workEx", [])
    ]
    education = [
        Education(
            degree=e.get("course", ""),
            institution=e.get("university", ""),
            graduation_year=int(e["passingYear"]) if e.get("passingYear", "").isdigit() else None,
        )
        for e in raw.get("education", [])
    ]

    raw_text = _build_raw_text(
        name=name,
        headline=raw.get("currentDesignation", ""),
        summary=raw.get("aboutSelf", ""),
        skills=skills,
        experiences=experiences,
        education=education,
    )

    return CommonProfile(
        source="naukri",
        source_id=raw["naukri_id"],
        full_name=name,
        headline=raw.get("currentDesignation", ""),
        location=raw.get("currentLocation", ""),
        years_experience=_parse_years(raw.get("totalExp", "0 years")),
        skills=skills,
        experiences=experiences,
        education=education,
        raw_text=raw_text,
        contact_email=raw.get("emailId"),
        metadata={
            "mobile": raw.get("mobile"),
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
# app/agents/outreach.py
"""Outreach agent: draft a personalized message for the recommended candidate.

Produces:
  - subject line
  - LinkedIn InMail (≤1500 chars, more direct)
  - longer email body (with company context)
  - list of personalization hooks (specific things from the profile)

Guardrails: outreach references work, never personal attributes.
"""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.config import settings
from app.models import ScoredCandidate, ParsedJD, OutreachDraft
from app.obs.events import log_event
from app.obs.llm_client import chat


# ============================================================
# Wire schema
# ============================================================

class _WireOutreach(BaseModel):
    subject: str = Field(description="Email subject line. Max ~80 chars.")
    linkedin_inmail: str = Field(
        description="LinkedIn InMail. Max ~1500 chars. More direct and shorter. "
                    "Open with a specific reference to their work, not 'I came across "
                    "your profile'. End with a clear call to action.")
    email_body: str = Field(
        description="Longer outreach email. ~200-300 words. Include role context "
                    "and the company hiring pitch. Sign off as 'TalentScout Recruiting'.")
    personalization_hooks: list[str] = Field(
        description="2-4 specific things from the candidate's profile that the "
                    "message draws on (e.g., 'led RAG pipeline at Acme AI', "
                    "'5 years Python production'). These prove the message is "
                    "actually personalized, not boilerplate.")


# ============================================================
# Prompt
# ============================================================

_OUTREACH_PROMPT_TEMPLATE = """\
You are drafting outreach for a candidate the recruiter wants to engage about \
a specific role.

THE ROLE:
  Title: {jd_title}
  Seniority: {seniority}
  Key must-haves the candidate matched on:
{must_have_strengths}

THE CANDIDATE'S PROFILE EXCERPT:
{profile_text}

WHY WE'RE REACHING OUT (from the ranking rationale):
{rationale}

WRITE three artifacts:
1. subject: an attention-getting subject line
2. linkedin_inmail: max 1500 chars, conversational and direct
3. email_body: 200-300 words, professional and warm

STRICT RULES:
- Reference SPECIFIC things from their profile (a project, a tech, a role).
- Do NOT mention name (the recruiter system will insert it), age, gender, race, \
religion, nationality, marital status, appearance, or any protected attribute.
- Do NOT claim accomplishments they don't have.
- Do NOT use generic phrases ("rockstar", "ninja", "10x engineer", "passionate").
- Be honest about being a recruiter — no fake urgency, no false friendship.
- Sign off as 'TalentScout Recruiting' (a generic recruiter signature).

Also list 2-4 personalization_hooks — the specific profile elements you drew on.
"""


def _top_strengths(scored: ScoredCandidate, n: int = 3) -> str:
    """Pull the top N must-have criterion scores as 'why they matched'."""
    musts = [cs for cs in scored.criterion_scores
             if cs.criterion_id.startswith("must")]
    musts.sort(key=lambda cs: cs.score, reverse=True)
    lines = []
    for cs in musts[:n]:
        ev = cs.evidence[:120] if cs.evidence and cs.has_evidence else "(skill match)"
        lines.append(f"    - {cs.criterion_text} (score {cs.score:.2f}): {ev!r}")
    return "\n".join(lines) if lines else "    (none)"


# ============================================================
# Public API
# ============================================================

def run_outreach(
    candidate: ScoredCandidate,
    profile_text: str,
    parsed: ParsedJD,
    jd_title: str,
) -> OutreachDraft:
    """Draft outreach for one candidate. Returns an OutreachDraft."""
    jd_id = str(parsed.jd_id)
    log_event(jd_id, "outreach_agent", "start", candidate=candidate.candidate_name)

    prompt = _OUTREACH_PROMPT_TEMPLATE.format(
        jd_title=jd_title,
        seniority=parsed.seniority,
        must_have_strengths=_top_strengths(candidate),
        profile_text=profile_text[:2500],
        rationale=candidate.overall_rationale or "(no rationale yet)",
    )

    wire: _WireOutreach = chat(
        messages=[
            {"role": "system", "content": "You write personalized, professional, "
                                          "bias-free recruiter outreach."},
            {"role": "user", "content": prompt},
        ],
        model=settings.openai_model_heavy,
        jd_id=jd_id,
        agent="outreach",
        response_format=_WireOutreach,
        temperature=0.4,    # slight creativity in tone, still grounded
        max_tokens=800,
    )

    draft = OutreachDraft(
        candidate_id=candidate.profile_id,
        subject=wire.subject,
        linkedin_inmail=wire.linkedin_inmail,
        email_body=wire.email_body,
        personalization_hooks=wire.personalization_hooks,
    )

    log_event(jd_id, "outreach_agent", "end",
              candidate=candidate.candidate_name,
              n_hooks=len(draft.personalization_hooks),
              inmail_len=len(draft.linkedin_inmail))
    return draft


def run_outreach_for_top_n(
    shortlist: list[ScoredCandidate],
    profiles_by_id: dict[UUID, str],
    parsed: ParsedJD,
    jd_title: str,
    n: int = 1,
) -> list[OutreachDraft]:
    """Draft outreach for the top N candidates. Default n=1 (just the top pick).

    `profiles_by_id`: map candidate_id -> raw_text (from the deduped profiles).
    """
    jd_id = str(parsed.jd_id)
    log_event(jd_id, "outreach_agent", "batch_start", n=min(n, len(shortlist)))

    drafts: list[OutreachDraft] = []
    for c in shortlist[:n]:
        profile_text = profiles_by_id.get(c.profile_id, "")
        if not profile_text:
            log_event(jd_id, "outreach_agent", "missing_profile_text",
                      candidate=c.candidate_name)
            continue
        drafts.append(run_outreach(c, profile_text, parsed, jd_title))

    log_event(jd_id, "outreach_agent", "batch_end", n_drafts=len(drafts))
    return drafts
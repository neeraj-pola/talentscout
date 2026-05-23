# app/agents/guardrails.py
"""Guardrails agent: reject discriminatory JDs before any work happens.

Two-layer defense:
  1. Regex layer — fast keyword/pattern scan for obvious violations
  2. LLM classifier — catches paraphrased / subtle discrimination

Why two layers:
  Regex alone misses paraphrases ("looking for young energy" doesn't say "young").
  LLM alone is slower and costlier per JD, and can be jailbroken on edge cases.
  Both together = defense in depth.

Runs at TWO points in the pipeline (see ranking_agent in Stage 7):
  - At JD intake (this file) — reject the JD entirely
  - At ranking — verify no protected attribute leaked into the ranking rationale
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.models import GuardrailResult, JD
from app.obs.events import log_event
from app.obs.llm_client import chat


# ============================================================
# Regex layer — fast, deterministic, catches obvious patterns
# ============================================================

# Pattern -> reason. Case-insensitive matches with word boundaries.
_REGEX_RULES: dict[str, str] = {
    # Age
    r"\b(young|youthful|fresh|energetic young)\b": "age discrimination (favors young)",
    r"\b(under|below|max(imum)?)\s*\d{1,2}\s*(years|yrs)?\s*(of\s*age|old)\b": "explicit age limit",
    r"\b\d{2}\s*[-–to]+\s*\d{2}\s*(years|yrs)?\s*(of\s*age|old)\b": "explicit age range",
    r"\bdigital natives?\b": "age discrimination (coded language)",
    r"\brecent graduates? only\b": "age discrimination via grad year",

    # Gender
    r"\b(male|female|man|woman|boys?|girls?)\s+(only|preferred|candidate|applicant)\b": "gender discrimination",
    r"\b(only|prefer(ably)?)\s+(male|female|men|women)\b": "gender discrimination",
    r"\b(he|she)\s+(must|should)\b": "gendered pronouns implying gender preference",
    r"\bsalesman\b|\bsaleswoman\b": "gendered job title",

    # Marital / family
    r"\b(single|unmarried|married|widowed)\s+(only|preferred|candidate)\b": "marital status discrimination",
    r"\b(no\s+kids|childless|no\s+children)\s+(preferred|required)\b": "family-status discrimination",

    # Religion / nationality / race
    r"\b(hindu|muslim|christian|sikh|jewish|buddhist)\s+(only|preferred|candidates?)\b": "religious discrimination",
    r"\b(indian|american|british|chinese|white|black|asian|caucasian)\s+(only|nationals?\s+only|candidates?\s+only)\b": "racial / national-origin discrimination",
    r"\bnative english speaker\b": "potentially discriminatory (national-origin proxy)",

    # Physical / disability
    r"\b(able[-\s]?bodied|no\s+disabilities|physically\s+fit)\s+only\b": "disability discrimination",
    r"\b(good[-\s]?looking|attractive|presentable)\b": "appearance-based discrimination",
}


def _regex_scan(text: str) -> list[tuple[str, str]]:
    """Return list of (matched_phrase, reason) for any rule that fires."""
    findings = []
    for pattern, reason in _REGEX_RULES.items():
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            findings.append((m.group(0), reason))
    return findings


# ============================================================
# LLM classifier layer — catches subtle paraphrased patterns
# ============================================================

class _LLMGuardrailVerdict(BaseModel):
    """Wire-format schema for the LLM classifier."""
    is_discriminatory: bool = Field(
        description="True if the JD contains criteria based on protected attributes "
                    "(age, gender, race, religion, marital status, disability, national origin, "
                    "sexual orientation, pregnancy, appearance)."
    )
    severity: Literal["none", "low", "medium", "high"] = Field(
        description="'high' for explicit discrimination, 'medium' for coded language, "
                    "'low' for ambiguous phrasing, 'none' if clean."
    )
    reasons: list[str] = Field(
        description="One short bullet per problem found. Empty list if clean."
    )
    flagged_phrases: list[str] = Field(
        description="Verbatim phrases from the JD that triggered concern. "
                    "Empty list if clean."
    )


_GUARDRAIL_PROMPT = """\
You are a fairness-and-compliance reviewer for job descriptions.

Flag a JD as discriminatory ONLY if it requires or strongly favors candidates \
based on a protected attribute: age, gender, race, religion, marital or family \
status, disability, national origin, sexual orientation, pregnancy, or appearance.

Do NOT flag legitimate professional requirements:
- "5+ years of experience" is OK (experience-based, not age)
- "must be authorized to work in X country" is OK (legal requirement, not nationality)
- "fluent in English" is OK (job skill, not national origin)
- "willing to travel" is OK
- "must be able to lift 50 lbs" may be OK if it's a genuine job requirement

DO flag:
- "young, energetic team" — coded age discrimination
- "looking for a salesman" — gendered job title
- "Hindu candidates preferred" — religious discrimination
- "20-30 years of age" — explicit age range
- "native English speaker" — national-origin proxy
- "good-looking" — appearance

Output severity:
- "high": explicit and direct (e.g., "men only")
- "medium": coded but clear (e.g., "young energetic team")
- "low": ambiguous, might be innocent (e.g., "fast-paced startup vibe")
- "none": no concerns

Be strict but fair. False positives waste reviewer time; false negatives let \
discriminatory JDs through. Err toward strict on protected attributes."""


def _llm_classify(jd_text: str, jd_id: str | None) -> _LLMGuardrailVerdict:
    """Run the LLM classifier on the full JD text."""
    return chat(
        messages=[
            {"role": "system", "content": _GUARDRAIL_PROMPT},
            {"role": "user", "content": f"Job description to review:\n\n{jd_text}"},
        ],
        model=settings.openai_model_light,   # cheap model — this is a classifier
        jd_id=jd_id,
        agent="guardrails",
        response_format=_LLMGuardrailVerdict,
        temperature=0.0,                      # deterministic for a classifier
        max_tokens=400,
    )


# ============================================================
# Public API
# ============================================================

def screen_jd(jd: JD) -> GuardrailResult:
    """Run both layers on a JD. Returns a GuardrailResult.

    The orchestrator decides what to do with the result (Stage 8).
    If is_discriminatory, the JD should be rejected and NOT proceed.
    """
    jd_id = str(jd.id)
    log_event(jd_id, "guardrails", "screen_start",
              title=jd.title, desc_len=len(jd.description))

    # Build the full text the guardrails will examine
    full_text = "\n".join([
        f"Title: {jd.title}",
        f"Description: {jd.description}",
        f"Must-have skills: {', '.join(jd.must_have_skills)}",
        f"Nice-to-have skills: {', '.join(jd.nice_to_have_skills)}",
    ])

    # Layer 1: regex
    regex_hits = _regex_scan(full_text)
    log_event(jd_id, "guardrails", "regex_done", n_hits=len(regex_hits))

    # Layer 2: LLM (always runs — regex hits don't short-circuit, so we get
    # the full picture in one verdict)
    llm_verdict = _llm_classify(full_text, jd_id)
    log_event(jd_id, "guardrails", "llm_done",
              is_discriminatory=llm_verdict.is_discriminatory,
              severity=llm_verdict.severity)

    # Combine
    combined_reasons = list({
        *[r for _, r in regex_hits],
        *llm_verdict.reasons,
    })
    combined_phrases = list({
        *[p for p, _ in regex_hits],
        *llm_verdict.flagged_phrases,
    })

    # Final verdict: discriminatory if EITHER layer flagged it
    is_discriminatory = bool(regex_hits) or llm_verdict.is_discriminatory

    # Severity: pick the more serious of the two
    if regex_hits and llm_verdict.severity in ("none", "low"):
        # Regex caught something the LLM missed — bump to medium minimum
        severity = "medium"
    else:
        severity = llm_verdict.severity

    result = GuardrailResult(
        is_discriminatory=is_discriminatory,
        reasons=combined_reasons,
        flagged_phrases=combined_phrases,
        severity=severity,
    )

    log_event(jd_id, "guardrails", "screen_end",
              is_discriminatory=is_discriminatory,
              severity=severity,
              n_reasons=len(combined_reasons))
    return result


def check_ranking_for_bias(rationale: str, jd_id: str | None = None) -> bool:
    """Lightweight check: did a protected attribute leak into ranking rationale?

    Called by the Ranking Agent in Stage 7 as a defensive depth measure.
    Returns True if rationale looks clean, False if it contains bias indicators."""
    hits = _regex_scan(rationale)
    if hits:
        log_event(jd_id, "guardrails", "ranking_bias_detected",
                  rationale_preview=rationale[:200],
                  hits=[h[1] for h in hits])
        return False
    return True
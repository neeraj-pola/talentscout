# app/agents/profile_summary.py
"""Profile summary agent.

Produces a short, bias-blind professional summary for each candidate.
Runs after dedup and before screening. Output is stored on the
CommonProfile.summary field and reused downstream by the ranking and
outreach agents (saves tokens and makes ranking decisions more
interpretable).

Design choices:
  - Async fan-out: each profile is independent, so we parallelize via
    asyncio.gather, capped at max_concurrency to respect OpenAI rate limits.
    Uses the shared AsyncOpenAI client from llm_client.get_async_client().
  - Bias-blind by construction: the prompt explicitly instructs the model
    to omit name, location, gender, age, and other protected attributes.
    The summary becomes "what this candidate brings to a role" rather
    than "who this candidate is."
  - Cheap: uses gpt-4o-mini, ~50-80 input tokens x ~80 output tokens
    per candidate. ~$0.0001 per profile, ~$0.002 per JD of 20 candidates.
  - Robust to failures: if a single summary call errors out, we log it
    and proceed with an empty summary for that candidate. Downstream
    agents already handle empty summaries gracefully.
  - Cost-tracked: we call record_cost() manually for each LLM call so
    profile_summary appears in the per-agent cost breakdown alongside
    screening, ranking, etc. The sync chat() does this automatically;
    the async client does not, so we mirror it here.
"""
from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.models import CommonProfile
from app.obs.cost import record_cost
from app.obs.events import log_event
from app.obs.llm_client import get_async_client


# ============================================================
# Wire schema for structured output
# ============================================================

class _WireSummary(BaseModel):
    summary: str = Field(
        description="2-3 sentence bias-blind professional summary. "
                    "Focus on: years of experience, key skills, notable "
                    "employers or projects, evident gaps. Do NOT mention "
                    "name, location, gender, age, ethnicity, religion, or "
                    "any other protected attribute."
    )


# ============================================================
# Prompt
# ============================================================

_SUMMARY_PROMPT_TEMPLATE = """\
You are writing a brief, bias-blind professional summary of a candidate \
that downstream recruiting agents will use to make screening, ranking, \
and outreach decisions.

CANDIDATE PROFILE EXCERPT:
{profile_text}

YEARS OF EXPERIENCE: {years_experience}
KEY SKILLS: {skills}

WRITE a 2-3 sentence summary. Cover:
  - Total years of relevant experience
  - The 3-5 most distinctive skills or technical areas
  - One notable employer, project, or accomplishment (if any)
  - Any evident gap (career switch, employment gap, junior level for the YOE)

STRICT RULES - these are non-negotiable:
  - Do NOT mention the candidate's name, pronouns (he/she/they), location, \
city, country, nationality, gender, age, ethnicity, religion, marital status, \
appearance, or any other protected attribute.
  - Use neutral phrasing: "the candidate has..." or "with N years of...".
  - Do NOT invent skills or experience the profile doesn't show.
  - Keep it to 2-3 sentences, ~50-80 words total.

This summary will be the executive view a hiring manager sees first. \
Be specific, be honest, be brief.
"""


# ============================================================
# Per-candidate summary call
# ============================================================

async def _summarize_one(
    profile: CommonProfile,
    jd_id: str,
    semaphore: asyncio.Semaphore,
    model: str,
) -> tuple[CommonProfile, str]:
    """Generate a summary for one profile. Returns (profile, summary).

    Errors are caught and logged - the candidate's summary stays empty
    rather than tanking the whole batch. Downstream agents treat empty
    summary as "fall back to raw_text," so the system degrades gracefully.
    """
    async with semaphore:
        client = get_async_client()
        t0 = time.time()

        try:
            prompt = _SUMMARY_PROMPT_TEMPLATE.format(
                profile_text=profile.raw_text[:2000],
                years_experience=f"{profile.years_experience:.1f}",
                skills=", ".join(profile.skills[:15]) or "(none listed)",
            )

            response = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content":
                        "You write concise, bias-blind candidate summaries."},
                    {"role": "user", "content": prompt},
                ],
                response_format=_WireSummary,
                temperature=0.2,    # low - we want consistent, factual summaries
                max_tokens=200,
            )

            wire: _WireSummary = response.choices[0].message.parsed
            usage = response.usage
            latency_ms = (time.time() - t0) * 1000

            # Manually record cost - async client doesn't go through the
            # instrumented chat() wrapper, so we mirror its behavior here.
            record_cost(
                jd_id=jd_id,
                agent="profile_summary",
                model=model,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                latency_ms=latency_ms,
            )

            return profile, (wire.summary or "").strip()

        except Exception as e:
            log_event(jd_id, "profile_summary", "summary_error",
                      profile_id=str(profile.id),
                      error=str(e)[:200])
            return profile, ""


# ============================================================
# Public API
# ============================================================

async def _run_async(
    profiles: list[CommonProfile],
    jd_id: str,
    max_concurrency: int = 8,
) -> list[CommonProfile]:
    """Async implementation. Fan out across all profiles in parallel."""
    semaphore = asyncio.Semaphore(max_concurrency)
    # Prefer the cheap model if configured (settings.openai_model_cheap);
    # fall back to the heavy model so this still works on configurations
    # that only define one model setting. Summaries are short and structured
    # so even the heavy model is acceptable cost-wise here.
    model = getattr(settings, "openai_model_cheap", None) or settings.openai_model_heavy

    tasks = [_summarize_one(p, jd_id, semaphore, model) for p in profiles]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Build new list with summary field populated. Pydantic v2's model_copy
    # with update= is the canonical copy-on-write pattern.
    out: list[CommonProfile] = []
    n_success = 0
    for profile, summary in results:
        if summary:
            out.append(profile.model_copy(update={"summary": summary}))
            n_success += 1
        else:
            out.append(profile)  # summary stays as default ""

    log_event(jd_id, "profile_summary", "batch_end",
              n_input=len(profiles),
              n_summarized=n_success,
              n_failed=len(profiles) - n_success)
    return out


def run_profile_summary(
    profiles: list[CommonProfile],
    jd_id: str,
    max_concurrency: int = 8,
) -> list[CommonProfile]:
    """Generate a bias-blind summary for each profile. Sync entrypoint.

    Args:
        profiles: list of deduped CommonProfile objects
        jd_id: JD UUID for observability
        max_concurrency: cap parallel OpenAI calls (default 8)

    Returns:
        A new list of CommonProfile objects with `summary` field populated.
        Original profiles are not mutated.
    """
    if not profiles:
        log_event(jd_id, "profile_summary", "skip", reason="no_profiles")
        return profiles

    log_event(jd_id, "profile_summary", "batch_start", n=len(profiles))

    # asyncio.run manages the event loop for us. Matches the pattern used
    # by the screening agent.
    return asyncio.run(_run_async(profiles, jd_id, max_concurrency))
# app/agents/ranking.py
"""Ranking agent: finalize the shortlist with per-candidate rationale + bias check.

What it adds on top of screening:
  1. A natural-language overall_rationale per top-N candidate
     (Screening only computed numeric scores + per-criterion reasoning)
  2. Ranking-time bias check — runs Guardrails on every rationale to make sure
     no protected attribute leaked in
  3. Truncates to top_n_shortlist (default 10) — the public shortlist

Why a separate rationale call (vs reusing per-criterion reasoning):
  The per-criterion reasonings are atomic ("met must_003 because X"). The
  ranking rationale is the synthesis a recruiter actually wants to read:
  "Strong on must-haves 1,2,3; weak on 4; solid trajectory; recommend interview."

Uses coverage fields from screening to distinguish "low score with evidence"
from "no evidence in profile" — important for explainability.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings
from app.models import ScoredCandidate, ParsedJD, Criterion
from app.agents.guardrails import check_ranking_for_bias
from app.obs.events import log_event
from app.obs.llm_client import chat


# ============================================================
# Wire schema for the LLM
# ============================================================

class _WireRationale(BaseModel):
    rationale: str = Field(
        description="2-4 sentences synthesizing the candidate's fit. "
                    "Reference specific must-have strengths and any gaps. "
                    "Distinguish 'weak with evidence' from 'no signal in profile'. "
                    "Do NOT mention name, gender, age, religion, nationality, "
                    "or any protected attribute. Be concrete and recruiter-grade."
    )


_RATIONALE_PROMPT_TEMPLATE = """\
You are writing a 2-4 sentence overall rationale for ONE candidate that will \
be shown to a recruiter alongside the shortlist.

JOB:
  Title: {jd_title}
  Seniority required: {seniority}
  YOE range: {yoe_min}-{yoe_max}

PER-CRITERION SCORES FOR THIS CANDIDATE:
{criterion_breakdown}

OVERALL NUMERIC RESULT:
  must_have_avg:    {must_avg:.2f}  (evidence on {must_cov_pct}% of must-haves)
  nice_to_have_avg: {nice_avg:.2f}  (evidence on {nice_cov_pct}% of nice-to-haves)
  overall:          {overall:.2f}
  has_must_have_gap: {has_gap}
  red_flags: {red_flags}

NOTE on coverage: when coverage is low (e.g. 0%), the candidate's profile has \
NO mention of those criteria — that is distinct from low scores with evidence. \
Phrase it accordingly: "no nice-to-have signal in profile" rather than \
"weak on nice-to-haves" when nice_cov_pct is 0%.

WRITE the rationale. Rules:
- 2-4 sentences. Concrete, not generic.
- Lead with the strongest must-have evidence.
- If there's a must-have gap, name the specific gap.
- If coverage is 0% on a category, say so explicitly (it's a profile signal, not weakness).
- Do NOT mention name, gender, age, race, religion, nationality, marital status, \
or appearance.
- Do NOT pad ("great candidate overall"). Specifics only.
"""


def _format_criterion_breakdown(
    scored: ScoredCandidate, criteria: list[Criterion],
) -> str:
    """Compact per-criterion summary for the rationale prompt."""
    by_id = {cs.criterion_id: cs for cs in scored.criterion_scores}
    lines: list[str] = []
    for c in criteria:
        cs = by_id.get(c.id)
        if cs is None:
            continue
        tag = "MUST" if c.is_must_have else "nice"
        ev_marker = "" if cs.has_evidence else "  [no evidence in profile]"
        lines.append(
            f"  [{tag}] {c.text}: score={cs.score:.2f}, conf={cs.confidence:.2f}{ev_marker}"
            f"\n         reasoning: {cs.reasoning[:160]}"
        )
    return "\n".join(lines)


def _generate_rationale(
    scored: ScoredCandidate,
    parsed: ParsedJD,
    jd_title: str,
    jd_id: str,
) -> str:
    """One LLM call per top candidate. Returns the rationale string."""
    prompt = _RATIONALE_PROMPT_TEMPLATE.format(
        jd_title=jd_title,
        seniority=parsed.seniority,
        yoe_min=parsed.yoe_min,
        yoe_max=parsed.yoe_max or "any",
        criterion_breakdown=_format_criterion_breakdown(scored, parsed.criteria),
        must_avg=scored.must_have_score,
        nice_avg=scored.nice_to_have_score,
        must_cov_pct=int(scored.must_have_coverage * 100),
        nice_cov_pct=int(scored.nice_to_have_coverage * 100),
        overall=scored.overall_score,
        has_gap=scored.has_must_have_gap,
        red_flags=", ".join(scored.red_flags) if scored.red_flags else "none",
    )

    wire: _WireRationale = chat(
        messages=[
            {"role": "system", "content": "You write concise, evidence-based, "
                                          "bias-free ranking rationales for recruiters."},
            {"role": "user", "content": prompt},
        ],
        model=settings.openai_model_heavy,
        jd_id=jd_id,
        agent="ranking",
        response_format=_WireRationale,
        temperature=0.2,
        max_tokens=300,
    )
    return wire.rationale


# ============================================================
# Public API
# ============================================================

def run_ranking(
    scored: list[ScoredCandidate],
    parsed: ParsedJD,
    jd_title: str,
    top_n: int | None = None,
) -> list[ScoredCandidate]:
    """Finalize the shortlist:
      1. Take the top_n scored candidates (default: settings.top_n_shortlist)
      2. Generate an overall_rationale for each via LLM
      3. Re-run the ranking-time bias check on each rationale
      4. Return the enriched, ranked list

    The input is already sorted desc by overall_score (from screening).
    """
    jd_id = str(parsed.jd_id)
    top_n = top_n or settings.top_n_shortlist
    shortlist = scored[:top_n]

    log_event(jd_id, "ranking_agent", "start",
              n_input=len(scored), n_shortlist=len(shortlist))

    enriched: list[ScoredCandidate] = []
    bias_flags = 0
    for i, c in enumerate(shortlist, 1):
        rationale = _generate_rationale(c, parsed, jd_title, jd_id)

        # Guardrails re-check: did a protected attribute leak in?
        passed = check_ranking_for_bias(rationale, jd_id=jd_id)
        if not passed:
            bias_flags += 1
            # Regenerate once — if still bad, fall back to a neutral template
            rationale = _generate_rationale(c, parsed, jd_title, jd_id)
            if not check_ranking_for_bias(rationale, jd_id=jd_id):
                log_event(jd_id, "ranking_agent", "rationale_replaced_after_bias",
                          candidate=c.candidate_name)
                rationale = (
                    f"Overall score {c.overall_score:.2f} "
                    f"(must-have {c.must_have_score:.2f} on "
                    f"{int(c.must_have_coverage * 100)}% coverage, "
                    f"nice-to-have {c.nice_to_have_score:.2f} on "
                    f"{int(c.nice_to_have_coverage * 100)}% coverage). "
                    + (f"Red flags: {'; '.join(c.red_flags)}. "
                       if c.red_flags else "")
                    + "See per-criterion scores for evidence."
                )

        # Stash the rationale on the candidate
        enriched.append(c.model_copy(update={"overall_rationale": rationale}))

        log_event(jd_id, "ranking_agent", "ranked",
                  position=i, candidate=c.candidate_name,
                  overall=c.overall_score, has_gap=c.has_must_have_gap)

    log_event(jd_id, "ranking_agent", "end",
              n_shortlist=len(enriched),
              bias_flags_during_run=bias_flags)
    return enriched
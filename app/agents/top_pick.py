# app/agents/top_pick.py
"""Top-Pick agent: one head-to-head LLM call recommending the winner.

Why a separate agent (not just 'highest overall_score wins'):
  Numeric scores capture per-criterion match. They don't capture:
    - Trajectory (junior trending up vs senior plateaued)
    - Composition (perfect on hard must-haves vs even across all)
    - Risk (one weak must-have vs one missing nice-to-have)
    - Coverage gaps (verified weakness vs unknown signal)
  A direct comparison prompt gets the LLM to reason about these qualitative
  trade-offs — the kind of judgment a senior recruiter brings.

Cost note:
  This is the ONE place we may want the better model. ~1 call per JD.
  Even if gpt-4o costs 17x more than gpt-4o-mini, one call is rounding error
  in the overall JD cost, and the output is the headline deliverable.
"""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.config import settings
from app.models import ScoredCandidate, ParsedJD, TopPickRecommendation
from app.obs.events import log_event
from app.obs.llm_client import chat


# ============================================================
# Wire schema
# ============================================================

class _WireTopPick(BaseModel):
    recommended_candidate_id: str = Field(
        description="The UUID (as string) of the recommended candidate. "
                    "MUST be one of the candidate_ids in the input.")
    justification: str = Field(
        description="3-5 sentences explaining WHY this candidate over the others. "
                    "Compare specifically: what does the chosen candidate have that "
                    "the runner-up does not? Reference must-have scores and evidence. "
                    "Do NOT mention name, gender, age, religion, nationality, etc."
    )
    key_tradeoff_vs_runner_up: str = Field(
        description="1 sentence: what you give up by picking the recommended "
                    "candidate over the runner-up. Be honest."
    )
    runner_up_candidate_id: str = Field(
        description="UUID of the second-best candidate (the runner-up you "
                    "compared against)."
    )


# ============================================================
# Prompt
# ============================================================

_TOP_PICK_PROMPT_TEMPLATE = """\
You are a senior recruiter making a final hiring recommendation.

JOB:
  Title: {jd_title}
  Seniority: {seniority}, YOE: {yoe_min}-{yoe_max}

THE TOP {n_candidates} CANDIDATES (already ranked by overall score):

{candidates_block}

Your task: recommend ONE candidate to move forward with for this JD.

Consider:
- Must-have coverage (gaps are expensive)
- Risk vs upside (a steady senior vs a high-ceiling mid-level)
- Trajectory (where their experience is heading)
- Strength of evidence (vague matches vs concrete production work)
- Coverage % distinguishes "verified weakness" from "no signal in profile"

OUTPUT:
- recommended_candidate_id: the UUID of your pick
- justification: 3-5 sentences. Compare to the runner-up specifically.
- key_tradeoff_vs_runner_up: 1 honest sentence on what you give up
- runner_up_candidate_id: the UUID of the second-best candidate

DO NOT mention name, gender, age, religion, race, nationality, marital status, \
or any protected attribute. Use candidate_id when referring to candidates.
"""


def _format_candidates_block(shortlist: list[ScoredCandidate]) -> str:
    """Compact per-candidate summary for the top-pick prompt."""
    blocks: list[str] = []
    for i, c in enumerate(shortlist, 1):
        # Show top criterion scores sorted by score desc (emphasize strengths)
        sorted_scores = sorted(
            c.criterion_scores,
            key=lambda cs: cs.score,
            reverse=True,
        )
        score_lines = []
        for cs in sorted_scores[:8]:  # cap at 8 to keep prompt short
            ev = cs.evidence[:100] if cs.evidence else "—"
            ev_marker = "" if cs.has_evidence else "  [NO EVIDENCE]"
            score_lines.append(
                f"      - {cs.criterion_id} score={cs.score:.2f}{ev_marker}: {cs.reasoning[:120]}"
                f"\n        evidence: {ev!r}"
            )
        red_flags_line = (
            f"   RED FLAGS: {'; '.join(c.red_flags)}\n" if c.red_flags else ""
        )
        blocks.append(
            f"#{i}  candidate_id={c.profile_id}\n"
            f"   overall={c.overall_score:.3f}  "
            f"must={c.must_have_score:.3f} (cov {int(c.must_have_coverage*100)}%)  "
            f"nice={c.nice_to_have_score:.3f} (cov {int(c.nice_to_have_coverage*100)}%)\n"
            f"{red_flags_line}"
            f"   per-criterion (top 8 by score):\n" + "\n".join(score_lines)
        )
    return "\n\n".join(blocks)


# ============================================================
# Public API
# ============================================================

def run_top_pick(
    shortlist: list[ScoredCandidate],
    parsed: ParsedJD,
    jd_title: str,
    compare_top_n: int = 3,
) -> TopPickRecommendation | None:
    """Pick one candidate from the top-N. Returns None if shortlist is empty."""
    jd_id = str(parsed.jd_id)
    if not shortlist:
        log_event(jd_id, "top_pick_agent", "no_candidates")
        return None

    top_n = shortlist[:compare_top_n]
    log_event(jd_id, "top_pick_agent", "start",
              n_compared=len(top_n),
              candidates=[str(c.profile_id) for c in top_n])

    prompt = _TOP_PICK_PROMPT_TEMPLATE.format(
        jd_title=jd_title,
        seniority=parsed.seniority,
        yoe_min=parsed.yoe_min,
        yoe_max=parsed.yoe_max or "any",
        n_candidates=len(top_n),
        candidates_block=_format_candidates_block(top_n),
    )

    wire: _WireTopPick = chat(
        messages=[
            {"role": "system", "content": "You are a senior recruiter. "
                                          "Recommend one candidate with concrete "
                                          "evidence-based justification."},
            {"role": "user", "content": prompt},
        ],
        model=settings.openai_model_heavy,
        jd_id=jd_id,
        agent="top_pick",
        response_format=_WireTopPick,
        temperature=0.2,
        max_tokens=600,
    )

    # Validate UUIDs and ensure recommended is actually in the input
    valid_ids = {str(c.profile_id) for c in top_n}
    if wire.recommended_candidate_id not in valid_ids:
        log_event(jd_id, "top_pick_agent", "invalid_recommended_id",
                  got=wire.recommended_candidate_id, valid=list(valid_ids))
        recommended_id = str(top_n[0].profile_id)
        runner_up_id = str(top_n[1].profile_id) if len(top_n) > 1 else recommended_id
    else:
        recommended_id = wire.recommended_candidate_id
        runner_up_id = wire.runner_up_candidate_id if wire.runner_up_candidate_id in valid_ids else (
            str(top_n[1].profile_id) if len(top_n) > 1 else recommended_id
        )

    # Look up the candidate name for the recommendation object
    by_id = {str(c.profile_id): c for c in top_n}
    candidate_name = by_id[recommended_id].candidate_name

    result = TopPickRecommendation(
        recommended_candidate_id=UUID(recommended_id),
        candidate_name=candidate_name,
        justification=wire.justification,
        key_tradeoff_vs_runner_up=wire.key_tradeoff_vs_runner_up,
        runner_up_id=UUID(runner_up_id),
    )

    log_event(jd_id, "top_pick_agent", "end",
              recommended=candidate_name,
              recommended_id=recommended_id,
              runner_up_id=runner_up_id)
    return result
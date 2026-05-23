# app/agents/screening.py
"""Screening agent: per-criterion candidate scoring with verbatim evidence.

This is the most important agent in the system. Each candidate is scored
against EACH criterion individually — not one opaque score. Each score
carries:
  - a numeric value 0.0-1.0
  - a VERBATIM quote from the profile as evidence (or "No evidence found")
  - 1-2 sentences of reasoning
  - a confidence level
  - a has_evidence flag (distinguishes "scored zero with evidence" from
    "no evidence to evaluate against")

Why this design:
  - Auditability — reviewers can click into any score and see the quote
  - Defensible — "Why was this candidate ranked 4th?" has a real answer
  - Robustness — evidence requirement reduces hallucination
  - Explainability — coverage % distinguishes a profile gap from a low score
  - Aligns with the spec: "per-criterion basis (not a single opaque score),
    with reasoning for each score"

Performance:
  - Per criterion: RAG retrieves ~10 candidates -> ~10 LLM calls
  - For ~9 criteria: ~90 LLM calls per JD
  - Run in parallel via asyncio for speed (~30s on gpt-4o-mini vs ~3min serial)
"""
from __future__ import annotations

import asyncio
import time
from uuid import UUID

from pydantic import BaseModel, Field

from app.config import settings
from app.models import ParsedJD, Criterion, CommonProfile, CriterionScore, ScoredCandidate
from app.rag import HybridIndex, retrieve_for_criterion
from app.obs.events import log_event
from app.obs.llm_client import get_async_client
from app.obs.cost import record_cost


# ============================================================
# Wire-format schema (what the LLM returns per (candidate, criterion))
# ============================================================

class _WireCriterionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0,
        description="0.0 = criterion not met. 1.0 = explicit, strong match.")
    evidence: str = Field(
        description="A VERBATIM quote from the candidate's profile that supports "
                    "the score. Must be copy-paste from the profile, not paraphrased. "
                    "If no evidence exists in the profile, set this to 'No evidence found.'")
    reasoning: str = Field(
        description="1-2 sentences explaining the score given the evidence. "
                    "Concrete, not generic.")
    confidence: float = Field(ge=0.0, le=1.0,
        description="How confident you are in this score. Lower if the profile "
                    "is ambiguous, vague, or lacks signal.")


# ============================================================
# Prompt — strict, focuses on evidence
# ============================================================

_SCORING_PROMPT_TEMPLATE = """\
You are an objective, fair, evidence-based candidate evaluator.

Your task: score how well ONE candidate meets ONE criterion. You will output a \
score from 0.0 to 1.0, a verbatim quote from the profile as evidence, brief \
reasoning, and a confidence level.

SCORING SCALE:
  1.0  — Strong, explicit match. Profile shows clear evidence with specifics \
(years, scale, named tools, named outcomes).
  0.7  — Solid match. Evidence is present but lacks specifics, OR is implied \
strongly by adjacent work.
  0.4  — Partial / tangential match. Some related experience but not a direct hit.
  0.2  — Weak / inferred. The criterion is only loosely supported.
  0.0  — No evidence found, or contradicted by the profile.

EVIDENCE RULES (CRITICAL):
- The `evidence` field MUST be a verbatim, copy-paste quote from the profile below.
- Do NOT paraphrase. Do NOT summarize. Copy exact characters.
- If there is no quote in the profile that supports a non-zero score, the \
evidence field MUST read exactly: "No evidence found."
- If evidence is "No evidence found.", the score MUST be ≤ 0.2.

DO NOT consider any of the following when scoring:
- Candidate name, gender, age, marital status, religion, nationality
- Where they went to school as a status signal (only as relevant credentials)
- Anything that is not directly tied to the criterion

REASONING RULES:
- 1-2 sentences max.
- Reference the evidence concretely (e.g., "5 years at X company doing Y").
- Do not pad with generic statements ("strong candidate", "great fit").

CRITERION TO EVALUATE:
{criterion_text}
(Category: {criterion_category}; This is a {must_or_nice})

CANDIDATE PROFILE:
{profile_text}
"""


def _format_must_nice(c: Criterion) -> str:
    return "MUST-HAVE" if c.is_must_have else "nice-to-have"


# ============================================================
# Single-pair scoring via async OpenAI call
# ============================================================

async def _score_one_pair(
    candidate: CommonProfile,
    criterion: Criterion,
    jd_id: str,
    semaphore: asyncio.Semaphore,
) -> CriterionScore:
    """Score one (candidate, criterion) pair. Async so the agent can fan out."""
    async with semaphore:
        prompt = _SCORING_PROMPT_TEMPLATE.format(
            criterion_text=criterion.text,
            criterion_category=criterion.category,
            must_or_nice=_format_must_nice(criterion),
            profile_text=candidate.raw_text[:3000],   # cap to keep tokens reasonable
        )

        log_event(jd_id, "screening", "score_pair_start",
                  candidate=candidate.full_name, criterion_id=criterion.id)

        t0 = time.time()
        try:
            response = await get_async_client().beta.chat.completions.parse(
                model=settings.openai_model_heavy,
                messages=[
                    {"role": "system", "content": "You are a fair, evidence-based "
                                                   "candidate evaluator. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format=_WireCriterionScore,
                temperature=0.1,
                max_tokens=400,
            )
            wire = response.choices[0].message.parsed
            usage = response.usage
            latency_ms = (time.time() - t0) * 1000

            # Cost tracking
            record_cost(
                jd_id=jd_id, agent="screening",
                model=settings.openai_model_heavy,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                latency_ms=latency_ms,
            )

            # Verbatim-evidence sanity: if the LLM said it has evidence,
            # verify the quote actually appears in the profile (case-insensitive).
            verified_evidence = wire.evidence
            if wire.evidence and wire.evidence != "No evidence found.":
                if wire.evidence.lower().strip() not in candidate.raw_text.lower():
                    # The model hallucinated a quote. Force a no-evidence result.
                    log_event(jd_id, "screening", "evidence_not_verbatim",
                              candidate=candidate.full_name,
                              criterion_id=criterion.id,
                              claimed_evidence=wire.evidence[:120])
                    verified_evidence = (
                        f"[NON-VERBATIM EVIDENCE REJECTED] Original claim: "
                        f"{wire.evidence[:120]}"
                    )
                    # Penalize the score for unverifiable evidence
                    wire.score = min(wire.score, 0.2)
                    wire.confidence = min(wire.confidence, 0.3)

            # has_evidence: True when we have a real verbatim quote
            # False when "No evidence found." or rejected non-verbatim
            has_evidence = (
                verified_evidence != "No evidence found."
                and not verified_evidence.startswith("[NON-VERBATIM")
            )

            log_event(jd_id, "screening", "score_pair_end",
                      candidate=candidate.full_name,
                      criterion_id=criterion.id,
                      score=round(wire.score, 2),
                      confidence=round(wire.confidence, 2),
                      has_evidence=has_evidence,
                      latency_ms=round(latency_ms, 1))

            return CriterionScore(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                score=wire.score,
                evidence=verified_evidence,
                reasoning=wire.reasoning,
                confidence=wire.confidence,
                has_evidence=has_evidence,
            )

        except Exception as e:
            log_event(jd_id, "screening", "score_pair_error",
                      candidate=candidate.full_name,
                      criterion_id=criterion.id,
                      error=str(e))
            # Return a neutral, low-confidence score so one bad call doesn't kill the run.
            # has_evidence=False because the scoring itself failed — we can't claim
            # to have looked at the profile.
            return CriterionScore(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                score=0.0,
                evidence="No evidence found.",
                reasoning=f"Scoring failed: {e!s}",
                confidence=0.0,
                has_evidence=False,
            )


# ============================================================
# Aggregation: per-candidate score from per-criterion scores
# ============================================================

def _aggregate_candidate(
    candidate: CommonProfile,
    criterion_scores: list[CriterionScore],
    criteria_index: dict[str, Criterion],
) -> ScoredCandidate:
    """Roll per-criterion scores into a single ScoredCandidate.

    Two distinct measures emerge:
      - score:    0.0 means "criterion not met" (a real assessment)
      - coverage: fraction of criteria where evidence was actually found.
                  Low coverage tells the recruiter "this candidate has gaps in
                  what we can verify" — distinct from "this candidate scored low".
    """
    must_scores: list[float] = []
    nice_scores: list[float] = []
    must_with_evidence = 0
    nice_with_evidence = 0
    must_total = 0
    nice_total = 0
    red_flags: list[str] = []
    has_gap = False

    for cs in criterion_scores:
        criterion = criteria_index[cs.criterion_id]
        if criterion.is_must_have:
            must_scores.append(cs.score)
            must_total += 1
            if cs.has_evidence:
                must_with_evidence += 1
            if cs.score < settings.must_have_penalty_threshold:
                red_flags.append(
                    f"Weak on must-have '{criterion.text}' (score {cs.score:.2f})"
                )
                has_gap = True
        else:
            nice_scores.append(cs.score)
            nice_total += 1
            if cs.has_evidence:
                nice_with_evidence += 1

    must_avg = sum(must_scores) / len(must_scores) if must_scores else 0.0
    nice_avg = sum(nice_scores) / len(nice_scores) if nice_scores else 0.0

    if has_gap:
        must_avg *= settings.must_have_penalty_multiplier

    overall = 0.75 * must_avg + 0.25 * nice_avg

    must_coverage = must_with_evidence / must_total if must_total else 1.0
    nice_coverage = nice_with_evidence / nice_total if nice_total else 1.0

    return ScoredCandidate(
        profile_id=candidate.id,
        candidate_name=candidate.full_name,
        criterion_scores=criterion_scores,
        must_have_score=round(must_avg, 4),
        nice_to_have_score=round(nice_avg, 4),
        must_have_coverage=round(must_coverage, 4),
        nice_to_have_coverage=round(nice_coverage, 4),
        overall_score=round(overall, 4),
        red_flags=red_flags,
        has_must_have_gap=has_gap,
    )


# ============================================================
# Retrieval step: pick which candidates to score for each criterion
# ============================================================

def _candidates_to_score(
    parsed: ParsedJD,
    index: HybridIndex,
    top_k_per_criterion: int,
    jd_id: str,
) -> set[UUID]:
    """Union across criteria: every candidate that ranks top-k for ANY criterion."""
    selected: set[UUID] = set()
    for criterion in parsed.criteria:
        results = retrieve_for_criterion(
            index=index,
            criterion_text=criterion.text,
            top_k_retrieve=settings.top_k_retrieval,
            top_k_final=top_k_per_criterion,
            yoe_min=parsed.yoe_min,
            yoe_max=parsed.yoe_max,
            location=(parsed.location_constraint.city
                      if not parsed.location_constraint.remote_ok else None),
            jd_id=jd_id,
        )
        for r in results:
            selected.add(UUID(r.candidate_id))

    log_event(jd_id, "screening", "candidate_union_built",
              n_criteria=len(parsed.criteria),
              n_unique_candidates=len(selected))
    return selected


# ============================================================
# Public API
# ============================================================

async def _score_all_async(
    candidates: list[CommonProfile],
    criteria: list[Criterion],
    jd_id: str,
    max_concurrency: int = 8,
) -> dict[UUID, list[CriterionScore]]:
    """Score every (candidate, criterion) pair concurrently."""
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = []
    indexer: list[tuple[UUID, str]] = []
    for cand in candidates:
        for crit in criteria:
            tasks.append(_score_one_pair(cand, crit, jd_id, semaphore))
            indexer.append((cand.id, crit.id))

    results = await asyncio.gather(*tasks)

    bucket: dict[UUID, dict[str, CriterionScore]] = {c.id: {} for c in candidates}
    for (cand_id, crit_id), score in zip(indexer, results):
        bucket[cand_id][crit_id] = score

    ordered: dict[UUID, list[CriterionScore]] = {}
    for cand in candidates:
        ordered[cand.id] = [bucket[cand.id][c.id] for c in criteria]
    return ordered


def run_screening(
    parsed: ParsedJD,
    index: HybridIndex,
    top_k_per_criterion: int = 10,
    max_concurrency: int = 8,
) -> list[ScoredCandidate]:
    """Score candidates against every criterion. Returns a list of ScoredCandidate."""
    jd_id = str(parsed.jd_id)
    log_event(jd_id, "screening_agent", "start",
              n_criteria=len(parsed.criteria),
              top_k_per_criterion=top_k_per_criterion)

    # Step 1: select candidates via RAG (union across criteria)
    candidate_ids = _candidates_to_score(
        parsed, index, top_k_per_criterion, jd_id=jd_id,
    )
    candidates_to_score = [
        p for p in index.all_profiles() if p.id in candidate_ids
    ]

    if not candidates_to_score:
        log_event(jd_id, "screening_agent", "no_candidates")
        return []

    log_event(jd_id, "screening_agent", "candidates_selected",
              n_candidates=len(candidates_to_score),
              n_criteria=len(parsed.criteria),
              total_llm_calls=len(candidates_to_score) * len(parsed.criteria))

    # Step 2: parallel score all (candidate, criterion) pairs
    scores_by_candidate = asyncio.run(_score_all_async(
        candidates=candidates_to_score,
        criteria=parsed.criteria,
        jd_id=jd_id,
        max_concurrency=max_concurrency,
    ))

    # Step 3: aggregate per candidate
    criteria_index = {c.id: c for c in parsed.criteria}
    scored: list[ScoredCandidate] = []
    for cand in candidates_to_score:
        sc = _aggregate_candidate(
            candidate=cand,
            criterion_scores=scores_by_candidate[cand.id],
            criteria_index=criteria_index,
        )
        scored.append(sc)

    # Order by overall score, descending (the Ranking agent will refine this)
    scored.sort(key=lambda s: s.overall_score, reverse=True)

    log_event(jd_id, "screening_agent", "end",
              n_scored=len(scored),
              top_score=scored[0].overall_score if scored else 0.0,
              n_with_must_gap=sum(1 for s in scored if s.has_must_have_gap))

    return scored
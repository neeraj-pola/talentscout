# tests/test_stage7b.py
"""Stage 7B verification — full pipeline through Outreach.

Reuses Stage 7A's sourcing + screening (those have been verified). Adds:
  - run_ranking      → adds overall_rationale, bias-checked
  - run_top_pick     → recommends one candidate from top-3
  - run_outreach     → drafts subject + inmail + email for the top pick

Cost on gpt-4o-mini: ~$0.05 total.
"""
import time
from datetime import date

from app.models import JD
from app.storage.db import init_db
from app.storage.jd_repo import create_jd
from app.tools.sources import LinkedInMockSource
from app.agents import (
    parse_jd, run_sourcing, run_screening,
    run_ranking, run_top_pick, run_outreach_for_top_n,
)
from app.obs.cost import get_cost_summary


def _make_test_jd() -> JD:
    return JD(
        title="Senior ML Engineer (LLM / RAG)",
        description=(
            "We're looking for a senior ML engineer to build and deploy LLM-powered "
            "production systems. You will own the design and operation of a RAG "
            "pipeline serving 1000+ daily queries, including vector indexing, "
            "hybrid retrieval, and prompt orchestration with LangChain. "
            "You will collaborate with platform engineers to ship containerized "
            "services on AWS or Azure with proper observability and CI/CD."
        ),
        must_have_skills=["Python", "LLMs", "RAG", "LangChain", "AWS"],
        nice_to_have_skills=["Kubernetes", "Time series", "Azure"],
        min_years_experience=4,
        max_years_experience=12,
        location="Hyderabad, India",
        remote_ok=True,
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )


def main():
    init_db()
    print("=" * 60)
    print("STAGE 7B VERIFICATION (Ranking + Top-Pick + Outreach)")
    print("=" * 60)

    if not LinkedInMockSource().health_check():
        print("\n✗ Start mock server first: ./scripts/run_mock_server.sh")
        return

    # ----------------------------------------------------------------
    # 0. Prep: full pipeline up to scored candidates (Stage 7A)
    # ----------------------------------------------------------------
    print("\n--- Setup: parse + source + screen (cached/fast on gpt-4o-mini) ---")
    jd = _make_test_jd()
    create_jd(jd)
    parsed = parse_jd(jd)
    src = run_sourcing(parsed)
    print(f"   {src.n_after_dedup} unique profiles, {src.n_merges} merges")

    scored = run_screening(
        parsed=parsed,
        index=src.index,
        top_k_per_criterion=6,
        max_concurrency=8,
    )
    print(f"   {len(scored)} candidates scored, top_score={scored[0].overall_score:.3f}")
    assert len(scored) > 0

    # Show that coverage data from Stage 7A is flowing through
    top = scored[0]
    print(f"   Top candidate coverage: "
          f"must {int(top.must_have_coverage*100)}%, "
          f"nice {int(top.nice_to_have_coverage*100)}%")

    # ----------------------------------------------------------------
    # 1. Ranking agent
    # ----------------------------------------------------------------
    print(f"\n--- Step 1: Ranking Agent ---")
    t0 = time.time()
    shortlist = run_ranking(
        scored=scored,
        parsed=parsed,
        jd_title=jd.title,
        top_n=5,    # keep test cheap
    )
    print(f"   ✓ Ranking complete in {time.time() - t0:.1f}s")
    print(f"   Shortlist of {len(shortlist)} candidates with rationales")
    for i, c in enumerate(shortlist[:3], 1):
        flag = " ⚠" if c.has_must_have_gap else ""
        print(f"\n   #{i}  {c.candidate_name}{flag}  (overall={c.overall_score:.3f})")
        print(f"       coverage: must {int(c.must_have_coverage*100)}%, "
              f"nice {int(c.nice_to_have_coverage*100)}%")
        print(f"       rationale: {c.overall_rationale}")

    # Sanity: every candidate has a non-empty rationale
    for c in shortlist:
        assert c.overall_rationale, f"{c.candidate_name} missing rationale"
        assert len(c.overall_rationale) > 30, "Rationale should be substantive"
    print(f"\n   ✓ All {len(shortlist)} candidates have non-empty rationales")

    # ----------------------------------------------------------------
    # 2. Top-Pick agent
    # ----------------------------------------------------------------
    print(f"\n--- Step 2: Top-Pick Agent ---")
    t0 = time.time()
    top_pick = run_top_pick(
        shortlist=shortlist,
        parsed=parsed,
        jd_title=jd.title,
        compare_top_n=3,
    )
    print(f"   ✓ Top-Pick complete in {time.time() - t0:.1f}s")
    assert top_pick is not None
    print(f"\n   RECOMMENDED: {top_pick.candidate_name}")
    print(f"   candidate_id: {top_pick.recommended_candidate_id}")
    print(f"\n   Justification:")
    print(f"   {top_pick.justification}")
    print(f"\n   Key trade-off vs runner-up: {top_pick.key_tradeoff_vs_runner_up}")

    # Sanity: recommended must be in the shortlist
    shortlist_ids = {c.profile_id for c in shortlist}
    assert top_pick.recommended_candidate_id in shortlist_ids, \
        "Recommended candidate must be from the shortlist"
    print(f"\n   ✓ Recommended candidate is in the shortlist")

    # ----------------------------------------------------------------
    # 3. Outreach agent — for the top pick
    # ----------------------------------------------------------------
    print(f"\n--- Step 3: Outreach Agent ---")

    # Build candidate_id -> raw_text map from the sourcing result
    profiles_by_id = {p.id: p.raw_text for p in src.profiles}

    t0 = time.time()
    # Reorder shortlist so the top pick is first
    recommended = next(
        c for c in shortlist if c.profile_id == top_pick.recommended_candidate_id
    )
    others = [c for c in shortlist if c.profile_id != top_pick.recommended_candidate_id]
    outreach_order = [recommended] + others

    drafts = run_outreach_for_top_n(
        shortlist=outreach_order,
        profiles_by_id=profiles_by_id,
        parsed=parsed,
        jd_title=jd.title,
        n=1,    # just the top pick
    )
    print(f"   ✓ Outreach complete in {time.time() - t0:.1f}s")
    assert len(drafts) == 1
    d = drafts[0]
    print(f"\n   Subject: {d.subject}")
    print(f"\n   LinkedIn InMail ({len(d.linkedin_inmail)} chars):")
    print("   " + d.linkedin_inmail.replace("\n", "\n   "))
    print(f"\n   Email body ({len(d.email_body)} chars):")
    print("   " + d.email_body.replace("\n", "\n   "))
    print(f"\n   Personalization hooks ({len(d.personalization_hooks)}):")
    for h in d.personalization_hooks:
        print(f"     - {h}")

    # Sanity
    assert d.candidate_id == recommended.profile_id
    assert len(d.linkedin_inmail) <= 1800, "InMail should respect ~1500 char limit"
    assert len(d.personalization_hooks) >= 2, "Should have ≥2 hooks"
    assert "rockstar" not in d.email_body.lower(), "Banned cliché slipped in"
    assert "ninja" not in d.email_body.lower(), "Banned cliché slipped in"
    print(f"\n   ✓ Outreach draft passes quality checks")

    # ----------------------------------------------------------------
    # 4. Final cost summary for the whole pipeline
    # ----------------------------------------------------------------
    summary = get_cost_summary(str(jd.id))
    print(f"\n--- Step 4: Full pipeline cost ---")
    print(f"   Total LLM calls: {summary['total_calls']}")
    print(f"   Total tokens:    {summary['total_tokens_in']} in / "
          f"{summary['total_tokens_out']} out")
    print(f"   Total cost:      ${summary['total_usd']:.4f}")
    print(f"   By agent:")
    for agent, stats in sorted(summary["by_agent"].items()):
        print(f"     {agent:25s}: {stats['calls']:4d} calls, ${stats['usd']:.4f}")

    # Cleanup
    src.index.cleanup()

    print("\n" + "=" * 60)
    print("ALL STAGE 7B CHECKS PASSED — full pipeline working end-to-end")
    print("=" * 60)


if __name__ == "__main__":
    main()
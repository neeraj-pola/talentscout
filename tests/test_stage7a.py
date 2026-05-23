# tests/test_stage7a.py
"""Stage 7A verification — Sourcing + Screening agents end-to-end.

Cost warning: this test makes ~50-100 LLM calls on gpt-4o (~$0.30-$0.80).
For cheaper iteration, lower top_k_per_criterion or set the heavy model
to gpt-4o-mini in your .env during development.
"""
import time
from datetime import date

from app.models import JD
from app.storage.db import init_db
from app.storage.jd_repo import create_jd
from app.tools.sources import LinkedInMockSource
from app.agents import parse_jd, run_sourcing, run_screening
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
    print("STAGE 7A VERIFICATION (Sourcing + Screening)")
    print("=" * 60)

    if not LinkedInMockSource().health_check():
        print("\n✗ Start mock server first: ./scripts/run_mock_server.sh")
        return

    # ----------------------------------------------------------------
    # 0. Prep: JD + parse
    # ----------------------------------------------------------------
    jd = _make_test_jd()
    create_jd(jd)
    print(f"\n--- Step 0: Parse the JD ---")
    parsed = parse_jd(jd)
    print(f"   seniority={parsed.seniority}  yoe=[{parsed.yoe_min},{parsed.yoe_max}]")
    print(f"   {len(parsed.criteria)} criteria, "
          f"{sum(1 for c in parsed.criteria if c.is_must_have)} must-have")
    print(f"   queries: {parsed.derived_search_queries}")

    # ----------------------------------------------------------------
    # 1. Sourcing Agent
    # ----------------------------------------------------------------
    print(f"\n--- Step 1: Sourcing Agent ---")
    t0 = time.time()
    src = run_sourcing(parsed)
    t_sourcing = time.time() - t0
    print(f"   ✓ Sourcing complete in {t_sourcing:.1f}s")
    print(f"   raw_counts: {src.raw_counts}")
    print(f"   normalized:  {src.n_normalized}")
    print(f"   after dedup: {src.n_after_dedup} (merges: {src.n_merges})")
    if src.merge_audit:
        m = src.merge_audit[0]
        print(f"   sample merge: {m['merged_into_name']} from "
              f"{m['sources']} -> 1 record")

    assert src.n_normalized > 0, "Sourcing should pull >0 profiles"
    assert src.n_after_dedup <= src.n_normalized, "Dedup must not add records"
    assert src.index.collection.count() == src.n_after_dedup, \
        "Index size must match deduped count"

    # ----------------------------------------------------------------
    # 2. Screening Agent
    # ----------------------------------------------------------------
    # Use a tighter top_k for the test to keep cost reasonable
    print(f"\n--- Step 2: Screening Agent ---")
    print(f"   This will make ~{min(src.n_after_dedup, 15) * len(parsed.criteria)} LLM calls...")
    t0 = time.time()
    scored = run_screening(
        parsed=parsed,
        index=src.index,
        top_k_per_criterion=6,  # 6 candidates per criterion to limit cost in tests
        max_concurrency=8,
    )
    t_screening = time.time() - t0
    print(f"   ✓ Screening complete in {t_screening:.1f}s")
    print(f"   Scored {len(scored)} candidates against {len(parsed.criteria)} criteria")

    # ----------------------------------------------------------------
    # 3. Inspect the top-3 results
    # ----------------------------------------------------------------
    print(f"\n--- Step 3: Top 3 candidates after screening ---")
    for i, c in enumerate(scored[:3], 1):
        flag = " ⚠ MUST-HAVE GAP" if c.has_must_have_gap else ""
        print(f"\n   #{i}  {c.candidate_name}{flag}")
        print(f"       overall={c.overall_score:.3f}  must={c.must_have_score:.3f}  "
              f"nice={c.nice_to_have_score:.3f}")
        for cs in c.criterion_scores[:3]:  # first 3 criteria only
            tag = "MUST" if cs.criterion_id.startswith("must") else "nice"
            ev = cs.evidence
            ev_preview = (ev[:90] + "…") if len(ev) > 90 else ev
            print(f"     [{tag}] {cs.criterion_id}: score={cs.score:.2f} "
                  f"conf={cs.confidence:.2f}")
            print(f"           reasoning: {cs.reasoning[:100]}")
            print(f"           evidence:  {ev_preview!r}")

    # ----------------------------------------------------------------
    # 4. Sanity assertions
    # ----------------------------------------------------------------
    print(f"\n--- Step 4: Quality checks ---")

    # Scored candidates is a non-empty list
    assert len(scored) > 0, "Should have at least some scored candidates"

    # Every candidate has scores for every criterion
    for c in scored:
        assert len(c.criterion_scores) == len(parsed.criteria), \
            f"Candidate {c.candidate_name} has {len(c.criterion_scores)} scores, " \
            f"expected {len(parsed.criteria)}"

    # Scores are sorted desc by overall
    overalls = [c.overall_score for c in scored]
    assert overalls == sorted(overalls, reverse=True), "Should be sorted by overall desc"

    # At least some scores carry verbatim evidence (i.e., were not all "No evidence found")
    has_real_evidence = 0
    for c in scored[:5]:  # top 5
        for cs in c.criterion_scores:
            if cs.evidence != "No evidence found." and not cs.evidence.startswith("[NON-VERBATIM"):
                has_real_evidence += 1
    print(f"   {has_real_evidence} criterion-scores have verbatim evidence (across top-5)")
    assert has_real_evidence >= 5, \
        "Expected at least 5 verbatim-evidence scores across the top-5 candidates"

    # Scores within [0, 1]
    for c in scored:
        for cs in c.criterion_scores:
            assert 0.0 <= cs.score <= 1.0
            assert 0.0 <= cs.confidence <= 1.0

    # Red flags surface correctly: any score < threshold on a must-have -> in red_flags
    for c in scored:
        must_lows = [
            cs for cs in c.criterion_scores
            if cs.criterion_id.startswith("must") and cs.score < 0.3
        ]
        if must_lows:
            assert c.has_must_have_gap is True
            assert len(c.red_flags) >= 1, \
                f"{c.candidate_name} has a low must-have but no red flag recorded"

    print(f"   ✓ All quality checks passed")

    # ----------------------------------------------------------------
    # 5. Cost breakdown
    # ----------------------------------------------------------------
    summary = get_cost_summary(str(jd.id))
    print(f"\n--- Step 5: Cost summary for this JD ---")
    print(f"   Total calls:   {summary['total_calls']}")
    print(f"   Total tokens:  {summary['total_tokens_in']} in / "
          f"{summary['total_tokens_out']} out")
    print(f"   Total cost:    ${summary['total_usd']:.4f}")
    print(f"   By agent:")
    for agent, stats in summary["by_agent"].items():
        print(f"     {agent:20s}: {stats['calls']:3d} calls, "
              f"${stats['usd']:.4f}")

    # Cleanup
    src.index.cleanup()

    print("\n" + "=" * 60)
    print("ALL STAGE 7A CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
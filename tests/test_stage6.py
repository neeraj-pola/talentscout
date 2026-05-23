# tests/test_stage6.py
"""Stage 6 verification — Guardrails + JD Intake agents."""
from datetime import date

from app.models import JD
from app.storage.db import init_db
from app.storage.jd_repo import create_jd
from app.agents import screen_jd, parse_jd, check_ranking_for_bias
from app.obs.cost import get_cost_summary


def _make_clean_jd() -> JD:
    return JD(
        title="Senior ML Engineer (LLM / RAG)",
        description=(
            "We're looking for a senior ML engineer to build and deploy LLM-powered "
            "production systems. You will own the design and operation of a RAG "
            "pipeline serving 1000+ daily queries, including vector indexing, "
            "hybrid retrieval, and prompt orchestration with LangChain. "
            "You will collaborate with platform engineers to ship containerized "
            "services on AWS or Azure with proper observability and CI/CD. "
            "Experience with time-series anomaly detection is a plus."
        ),
        must_have_skills=["Python", "LLMs", "RAG", "LangChain", "AWS"],
        nice_to_have_skills=["Kubernetes", "Time series", "Azure"],
        min_years_experience=5,
        max_years_experience=10,
        location="Hyderabad, India",
        remote_ok=True,
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )


def _make_discriminatory_jd() -> JD:
    return JD(
        title="Young Energetic ML Engineer",
        description=(
            "We are a young, energetic startup looking for a male engineer aged "
            "22-28 to join our team. Hindu candidates preferred. Native English "
            "speaker required. Must be good-looking and presentable for client meetings."
        ),
        must_have_skills=["Python", "ML"],
        nice_to_have_skills=[],
        min_years_experience=1,
        location="Bangalore, India",
        remote_ok=False,
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )


def _make_subtle_jd() -> JD:
    """No explicit slurs — but coded language ('digital natives', 'recent grads only')."""
    return JD(
        title="Digital Marketing Specialist",
        description=(
            "Looking for digital natives who can hit the ground running. "
            "Recent graduates only. Must fit into our fast-paced startup vibe. "
            "She should be comfortable with social media platforms."
        ),
        must_have_skills=["Social Media", "Content"],
        nice_to_have_skills=[],
        min_years_experience=0,
        location="Remote",
        remote_ok=True,
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )


def main():
    init_db()
    print("=" * 60)
    print("STAGE 6 VERIFICATION (Guardrails + JD Intake)")
    print("=" * 60)

    # ============================================================
    # 1. Guardrails — clean JD should PASS
    # ============================================================
    clean = _make_clean_jd()
    create_jd(clean)
    print(f"\n--- Test 1: Guardrails on a CLEAN JD ---")
    print(f"   Title: {clean.title!r}")
    verdict = screen_jd(clean)
    print(f"   is_discriminatory={verdict.is_discriminatory}  severity={verdict.severity}")
    print(f"   reasons: {verdict.reasons}")
    assert verdict.is_discriminatory is False, \
        f"Clean JD should NOT be flagged. Got reasons: {verdict.reasons}"
    print(f"   ✓ Clean JD passed guardrails")

    # ============================================================
    # 2. Guardrails — explicit discriminatory JD should be REJECTED
    # ============================================================
    bad = _make_discriminatory_jd()
    create_jd(bad)
    print(f"\n--- Test 2: Guardrails on an EXPLICITLY DISCRIMINATORY JD ---")
    print(f"   Title: {bad.title!r}")
    verdict = screen_jd(bad)
    print(f"   is_discriminatory={verdict.is_discriminatory}  severity={verdict.severity}")
    print(f"   reasons ({len(verdict.reasons)}):")
    for r in verdict.reasons:
        print(f"     - {r}")
    print(f"   flagged_phrases: {verdict.flagged_phrases}")
    assert verdict.is_discriminatory is True, "Should be flagged"
    assert verdict.severity in ("medium", "high"), \
        f"Should be medium/high severity, got {verdict.severity}"
    print(f"   ✓ Discriminatory JD correctly rejected")

    # ============================================================
    # 3. Guardrails — subtle / coded language
    # ============================================================
    subtle = _make_subtle_jd()
    create_jd(subtle)
    print(f"\n--- Test 3: Guardrails on SUBTLE / CODED LANGUAGE ---")
    print(f"   Title: {subtle.title!r}")
    verdict = screen_jd(subtle)
    print(f"   is_discriminatory={verdict.is_discriminatory}  severity={verdict.severity}")
    print(f"   reasons ({len(verdict.reasons)}):")
    for r in verdict.reasons:
        print(f"     - {r}")
    assert verdict.is_discriminatory is True, \
        "Subtle/coded discrimination should be caught (LLM layer)"
    print(f"   ✓ Subtle discrimination caught by LLM classifier")

    # ============================================================
    # 4. Ranking-time bias check
    # ============================================================
    print(f"\n--- Test 4: Ranking-time bias check ---")
    clean_rationale = (
        "Top candidate due to 8 years of Python production experience, "
        "deep LLM and RAG background, and a strong AWS deployment record."
    )
    bad_rationale = (
        "Top candidate because she is a young, energetic recent graduate "
        "who fits the startup culture."
    )
    assert check_ranking_for_bias(clean_rationale) is True
    assert check_ranking_for_bias(bad_rationale) is False
    print(f"   ✓ Clean rationale passed")
    print(f"   ✓ Biased rationale (gendered + age) caught")

    # ============================================================
    # 5. JD Intake on the clean JD
    # ============================================================
    print(f"\n--- Test 5: JD Intake on the CLEAN JD ---")
    parsed = parse_jd(clean)
    print(f"   seniority: {parsed.seniority}")
    print(f"   yoe: [{parsed.yoe_min}, {parsed.yoe_max}]")
    print(f"   location: city={parsed.location_constraint.city} "
          f"remote_ok={parsed.location_constraint.remote_ok}")
    print(f"   criteria ({len(parsed.criteria)} total, "
          f"{sum(1 for c in parsed.criteria if c.is_must_have)} must, "
          f"{sum(1 for c in parsed.criteria if not c.is_must_have)} nice):")
    for c in parsed.criteria:
        tag = "MUST" if c.is_must_have else "nice"
        print(f"     [{tag}] {c.id} ({c.category:11s}, w={c.weight}): {c.text}")

    print(f"   search queries ({len(parsed.derived_search_queries)}):")
    for q in parsed.derived_search_queries:
        print(f"     - {q!r}")

    # Sanity checks
    assert parsed.seniority in ("mid", "senior", "staff"), \
        f"5-10 YOE should map to senior-ish, got {parsed.seniority}"
    assert len(parsed.criteria) >= 4, "Should produce at least 4 criteria"
    must_count = sum(1 for c in parsed.criteria if c.is_must_have)
    assert must_count >= 3, f"Should have >=3 must-haves, got {must_count}"
    assert 3 <= len(parsed.derived_search_queries) <= 5
    assert parsed.jd_id == clean.id
    # Every criterion should have the right weight per type
    for c in parsed.criteria:
        if c.is_must_have:
            assert c.weight == 1.0
        else:
            assert c.weight == 0.4
    print(f"   ✓ ParsedJD structure looks correct")

    # ============================================================
    # 6. Cost summary
    # ============================================================
    print(f"\n--- Test 6: Cost tracking ---")
    for jd in (clean, bad, subtle):
        s = get_cost_summary(str(jd.id))
        print(f"   {jd.title[:35]:35s} | {s['total_calls']} calls | "
              f"${s['total_usd']:.5f} | tokens={s['total_tokens_in']}/{s['total_tokens_out']}")
    # All three should have at least 1 LLM call (guardrail), clean JD has 2 (intake too)
    clean_summary = get_cost_summary(str(clean.id))
    assert clean_summary["total_calls"] >= 2, \
        "Clean JD should have >=2 LLM calls (guardrail + intake)"
    print(f"   ✓ Cost tracking working per JD")

    print("\n" + "=" * 60)
    print("ALL STAGE 6 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
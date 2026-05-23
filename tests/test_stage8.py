# tests/test_stage8.py
"""Stage 8 verification — LangGraph orchestrator end-to-end.

Tests:
  1. Clean JD -> full pipeline -> completed with top pick + outreach
  2. Discriminatory JD -> stops at guardrails -> no work done
  3. Checkpoint exists in graph_state.db after run
"""
import os
from datetime import date

from app.models import JD
from app.storage.db import init_db
from app.storage.jd_repo import create_jd, get_jd
from app.tools.sources import LinkedInMockSource
from app.orchestrator import run_pipeline, get_checkpoint
from app.obs.cost import get_cost_summary


def _make_clean_jd() -> JD:
    return JD(
        title="Senior ML Engineer (LLM / RAG)",
        description=(
            "Senior ML engineer to design and run a production RAG pipeline "
            "with LangChain on AWS. 5+ years of Python ML production experience."
        ),
        must_have_skills=["Python", "LLMs", "RAG", "LangChain", "AWS"],
        nice_to_have_skills=["Kubernetes", "Azure"],
        min_years_experience=4,
        max_years_experience=10,
        location="Hyderabad, India",
        remote_ok=True,
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )


def _make_bad_jd() -> JD:
    return JD(
        title="Young Engineer",
        description=(
            "Looking for a young, energetic male engineer aged 22-28. "
            "Hindu candidates preferred."
        ),
        must_have_skills=["Python"],
        nice_to_have_skills=[],
        min_years_experience=1,
        location="Bangalore",
        remote_ok=False,
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )


def main():
    init_db()

    # Clean any prior checkpoint to make the run reproducible
    if os.path.exists("graph_state.db"):
        os.remove("graph_state.db")

    print("=" * 60)
    print("STAGE 8 VERIFICATION (LangGraph orchestrator)")
    print("=" * 60)

    if not LinkedInMockSource().health_check():
        print("\n✗ Start mock server first: ./scripts/run_mock_server.sh")
        return

    # ============================================================
    # 1. Clean JD — full pipeline runs end to end
    # ============================================================
    print("\n--- Test 1: clean JD through the full graph ---")
    clean = _make_clean_jd()
    create_jd(clean)
    print(f"   JD: {clean.title!r}  ({clean.id})")

    final = run_pipeline(clean)

    print(f"\n   Final status: {final.get('status')}")
    print(f"   Guardrails verdict: discriminatory="
          f"{final['guardrail_verdict']['is_discriminatory']}")
    print(f"   Parsed criteria: {len(final['parsed_jd']['criteria'])}")
    print(f"   Sourcing: {final['sourcing_result']}")
    print(f"   Shortlist size: {len(final.get('shortlist', []))}")

    if final.get("top_pick"):
        tp = final["top_pick"]
        print(f"   Top pick: {tp['candidate_name']}")
        print(f"   Justification: {tp['justification'][:200]}…")

    if final.get("outreach_drafts"):
        d = final["outreach_drafts"][0]
        print(f"   Outreach subject: {d['subject']!r}")
        print(f"   Outreach hooks: {len(d['personalization_hooks'])}")

    # Hard assertions
    assert final["status"] == "completed", \
        f"Expected status=completed, got {final.get('status')}"
    assert final["guardrail_verdict"]["is_discriminatory"] is False
    assert final["parsed_jd"] is not None
    assert len(final["shortlist"]) > 0
    assert final["top_pick"] is not None
    assert len(final["outreach_drafts"]) >= 1

    # The JDRow in SQLite should reflect status updates
    stored = get_jd(clean.id)
    assert stored.status.value in ("shortlisted", "closed", "completed"), \
        f"JD status in DB should reflect pipeline progress, got {stored.status}"
    print(f"   ✓ JD status in DB: {stored.status.value}")

    # ============================================================
    # 2. Checkpoint persisted
    # ============================================================
    print("\n--- Test 2: checkpoint persisted in graph_state.db ---")
    snap = get_checkpoint(str(clean.id))
    assert snap is not None, "Checkpoint should exist after run"
    assert snap.get("status") == "completed"
    assert snap.get("top_pick") is not None
    print(f"   ✓ Checkpoint has {len(snap)} fields, status={snap['status']}")

    # ============================================================
    # 3. Discriminatory JD halts at guardrails
    # ============================================================
    print("\n--- Test 3: discriminatory JD halts at guardrails ---")
    bad = _make_bad_jd()
    create_jd(bad)
    print(f"   JD: {bad.title!r}")

    bad_final = run_pipeline(bad)

    print(f"   Final status: {bad_final.get('status')}")
    print(f"   Halt reason:  {bad_final.get('halt_reason')}")

    assert bad_final["status"] == "rejected_guardrail"
    assert bad_final["guardrail_verdict"]["is_discriminatory"] is True
    # No downstream work should have happened
    assert bad_final.get("parsed_jd") is None, "JD intake should NOT have run"
    assert not bad_final.get("scored_candidates", []), "Screening should NOT have run"
    assert bad_final.get("top_pick") is None
    assert not bad_final.get("outreach_drafts", [])
    print(f"   ✓ No downstream nodes ran (guardrails halted the graph)")

    # ============================================================
    # 4. Cost summary
    # ============================================================
    print("\n--- Test 4: cost tracking across both runs ---")
    for jd in (clean, bad):
        s = get_cost_summary(str(jd.id))
        print(f"   {jd.title[:35]:35s} | {s['total_calls']:3d} calls | "
              f"${s['total_usd']:.4f}")
    # Clean JD = full pipeline calls; bad JD = just one guardrails call
    clean_summary = get_cost_summary(str(clean.id))
    bad_summary = get_cost_summary(str(bad.id))
    assert clean_summary["total_calls"] > bad_summary["total_calls"], \
        "Clean JD should have more LLM calls than the rejected one"
    print(f"   ✓ Rejected JD made far fewer LLM calls (halted early)")

    print("\n" + "=" * 60)
    print("ALL STAGE 8 CHECKS PASSED — graph orchestrator wired end-to-end")
    print("=" * 60)


if __name__ == "__main__":
    main()
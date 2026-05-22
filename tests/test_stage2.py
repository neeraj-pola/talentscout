# tests/test_stage2.py
"""Stage 2 verification — run with: python -m tests.test_stage2"""
import os
from datetime import date
from uuid import uuid4

from app.models import JD, JDStatus, AuditRecord
from app.storage.db import init_db
from app.storage.jd_repo import create_jd, get_jd, list_jds, update_jd_status, close_jd
from app.storage.audit_repo import create_audit, list_audits
from app.obs.events import log_event, get_events_for_jd
from app.obs.cost import record_cost, get_cost_summary


def main():
    # Clean slate
    for f in ["talentscout.db", "events.jsonl"]:
        if os.path.exists(f):
            os.remove(f)

    print("=" * 60)
    print("STAGE 2 VERIFICATION")
    print("=" * 60)

    # 1. Initialize DB
    init_db()
    print("✓ Database initialized")

    # 2. Create a JD
    jd = JD(
        title="Senior Python Engineer",
        description="Build production ML systems.",
        must_have_skills=["python", "aws", "ml"],
        nice_to_have_skills=["kubernetes"],
        min_years_experience=5,
        location="Remote",
        employment_type="full_time",
        target_hiring_date=date(2026, 8, 1),
    )
    create_jd(jd)
    print(f"✓ Created JD: {jd.id}")

    # 3. Round-trip JD
    retrieved = get_jd(jd.id)
    assert retrieved is not None
    assert retrieved.title == "Senior Python Engineer"
    assert retrieved.must_have_skills == ["python", "aws", "ml"]
    print(f"✓ Retrieved JD: {retrieved.title}")

    # 4. List JDs
    all_jds = list_jds()
    assert len(all_jds) == 1
    print(f"✓ Listed JDs: {len(all_jds)} found")

    # 5. Update status
    update_jd_status(jd.id, JDStatus.SOURCING)
    assert get_jd(jd.id).status == JDStatus.SOURCING
    print("✓ Updated JD status")

    # 6. Log events
    log_event(str(jd.id), "test", "node_start", input_summary={"foo": "bar"})
    log_event(str(jd.id), "test", "node_end", output_summary={"baz": "qux"})
    events = get_events_for_jd(str(jd.id))
    assert len(events) == 2
    print(f"✓ Logged events: {len(events)} in memory, also in events.jsonl")

    # 7. Record cost
    record_cost(str(jd.id), "test_agent", "gpt-4o-mini", 1000, 500, 250.0)
    record_cost(str(jd.id), "test_agent", "gpt-4o",      2000, 1000, 800.0)
    summary = get_cost_summary(str(jd.id))
    print(f"✓ Cost summary: ${summary['total_usd']:.4f} across {summary['total_calls']} calls")
    print(f"  By model: {summary['by_model']}")

    # 8. Close JD + audit
    candidate_id = uuid4()
    close_jd(jd.id, candidate_id, "neeraj@example.com")
    closed = get_jd(jd.id)
    assert closed.status == JDStatus.CLOSED
    assert closed.closed_by == "neeraj@example.com"
    print(f"✓ Closed JD")

    audit = AuditRecord(
        jd_id=jd.id,
        candidate_id=candidate_id,
        closed_by="neeraj@example.com",
        closed_at="2026-05-22T15:00:00",
        justification="Best fit on must-haves.",
        final_ranking_snapshot=[candidate_id],
        total_cost_usd=summary["total_usd"],
        total_tokens=summary["total_tokens_in"] + summary["total_tokens_out"],
        total_llm_calls=summary["total_calls"],
    )
    create_audit(audit)
    audits = list_audits()
    assert len(audits) == 1
    print(f"✓ Created audit record: {audits[0]['justification']}")

    print("=" * 60)
    print("ALL STAGE 2 CHECKS PASSED")
    print("=" * 60)
    print("\nArtifacts created:")
    print("  - talentscout.db  (SQLite, contains jds + audit_log + llm_costs)")
    print("  - events.jsonl    (append-only event log)")
    print("\nInspect with:")
    print("  sqlite3 talentscout.db '.tables'")
    print("  cat events.jsonl | head -5")


if __name__ == "__main__":
    main()
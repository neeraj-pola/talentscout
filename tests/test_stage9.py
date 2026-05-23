# tests/test_stage9a.py
"""Stage 9A verification — hit the FastAPI endpoints via TestClient.

No subprocess needed — FastAPI's TestClient calls the ASGI app directly
in-process. The mock_sources_api server DOES need to be running.
"""
import os

from fastapi.testclient import TestClient

from app.api.server import app
from app.tools.sources import LinkedInMockSource


def main():
    # Clean slate for the test
    for f in ("graph_state.db",):
        if os.path.exists(f):
            os.remove(f)

    print("=" * 60)
    print("STAGE 9A VERIFICATION (FastAPI endpoints)")
    print("=" * 60)

    if not LinkedInMockSource().health_check():
        print("\n✗ Start mock server first: ./scripts/run_mock_server.sh")
        return

    client = TestClient(app)

    # ----------------------------------------------------------------
    # 1. Health endpoint
    # ----------------------------------------------------------------
    print("\n--- Test 1: GET /health ---")
    r = client.get("/health")
    assert r.status_code == 200, r.text
    data = r.json()
    print(f"   status={data['status']}, mock_reachable={data['mock_server_reachable']}")
    assert data["status"] == "ok"
    assert data["mock_server_reachable"] is True

    # ----------------------------------------------------------------
    # 2. List sample JDs (UI helper)
    # ----------------------------------------------------------------
    print("\n--- Test 2: GET /demo/sample-jds ---")
    r = client.get("/demo/sample-jds")
    assert r.status_code == 200
    samples = r.json()
    print(f"   Got {len(samples)} sample JDs:")
    for s in samples:
        print(f"     - {s['label']}")
    assert len(samples) >= 3

    # ----------------------------------------------------------------
    # 3. Create & run a clean JD through the pipeline
    # ----------------------------------------------------------------
    print("\n--- Test 3: POST /jds with a clean JD (runs full pipeline) ---")
    clean_payload = samples[0]["payload"]
    print(f"   Submitting: {clean_payload['title']!r}")
    print(f"   (this will take ~60-120s while the pipeline runs)")
    r = client.post("/jds", json=clean_payload)
    assert r.status_code == 201, r.text
    result = r.json()
    print(f"   ✓ Pipeline completed: status={result['status']}")
    print(f"   jd_id={result['jd_id']}")
    print(f"   shortlist size: {len(result['shortlist'])}")
    if result.get("top_pick"):
        print(f"   top pick: {result['top_pick']['candidate_name']}")

    assert result["status"] == "completed"
    assert len(result["shortlist"]) > 0
    assert result["top_pick"] is not None

    clean_jd_id = result["jd_id"]
    top_pick_id = result["top_pick"]["recommended_candidate_id"]

    # ----------------------------------------------------------------
    # 4. POST a discriminatory JD — should halt at guardrails
    # ----------------------------------------------------------------
    print("\n--- Test 4: POST /jds with a discriminatory JD ---")
    bad_payload = samples[1]["payload"]
    r = client.post("/jds", json=bad_payload)
    assert r.status_code == 201, r.text
    bad_result = r.json()
    print(f"   status={bad_result['status']}")
    print(f"   halt_reason: {bad_result.get('halt_reason')}")
    assert bad_result["status"] == "rejected_guardrail"
    assert bad_result.get("top_pick") is None

    # ----------------------------------------------------------------
    # 5. List JDs
    # ----------------------------------------------------------------
    print("\n--- Test 5: GET /jds ---")
    r = client.get("/jds")
    assert r.status_code == 200
    all_jds = r.json()
    print(f"   {len(all_jds)} JDs in database:")
    for jd in all_jds[:5]:
        print(f"     - {jd['title'][:40]:40s} [{jd['status']}]")
    assert len(all_jds) >= 2

    # ----------------------------------------------------------------
    # 6. JD detail (the heavy endpoint)
    # ----------------------------------------------------------------
    print("\n--- Test 6: GET /jds/{id} (detail) ---")
    r = client.get(f"/jds/{clean_jd_id}")
    assert r.status_code == 200
    detail = r.json()
    print(f"   status={detail['status']}")
    print(f"   parsed_jd: {len(detail['parsed_jd']['criteria'])} criteria")
    print(f"   shortlist: {len(detail['shortlist'])} candidates")
    print(f"   merge_audit: {len(detail['merge_audit'])} merge groups")
    print(f"   cost: {detail['cost_summary']['total_calls']} calls, "
          f"${detail['cost_summary']['total_usd']:.4f}")
    print(f"   events: {len(detail['events'])} logged events")
    assert detail["top_pick"] is not None

    # ----------------------------------------------------------------
    # 7. Cost endpoint
    # ----------------------------------------------------------------
    print("\n--- Test 7: GET /jds/{id}/cost ---")
    r = client.get(f"/jds/{clean_jd_id}/cost")
    assert r.status_code == 200
    cost = r.json()
    print(f"   total_calls={cost['total_calls']}, "
          f"total_usd=${cost['total_usd']:.4f}")
    print(f"   by_agent: {list(cost['by_agent'].keys())}")
    assert cost["total_calls"] > 0

    # ----------------------------------------------------------------
    # 8. Events endpoint
    # ----------------------------------------------------------------
    print("\n--- Test 8: GET /jds/{id}/events ---")
    r = client.get(f"/jds/{clean_jd_id}/events?limit=5")
    assert r.status_code == 200
    ev = r.json()
    print(f"   last {ev['count']} events (sample):")
    for e in ev["events"][-3:]:
        print(f"     {e['ts'][:19]}  {e['agent']:20s}  {e['event']}")

    # ----------------------------------------------------------------
    # 9. Close JD
    # ----------------------------------------------------------------
    print("\n--- Test 9: POST /jds/{id}/close ---")
    close_payload = {
        "closed_by": "neeraj@example.com",
        "candidate_id": top_pick_id,
    }
    r = client.post(f"/jds/{clean_jd_id}/close", json=close_payload)
    assert r.status_code == 200, r.text
    close_result = r.json()
    print(f"   ✓ Closed: {close_result['jd_id']} by {close_result['closed_by']}")
    assert close_result["audit_recorded"] is True

    # Closing twice should fail with 409
    r2 = client.post(f"/jds/{clean_jd_id}/close", json=close_payload)
    assert r2.status_code == 409
    print(f"   ✓ Idempotent: second close attempt returned 409 Conflict")

    # ----------------------------------------------------------------
    # 10. Audit list
    # ----------------------------------------------------------------
    print("\n--- Test 10: GET /audits ---")
    r = client.get("/audits")
    assert r.status_code == 200
    audits = r.json()
    print(f"   {len(audits)} audit record(s):")
    for a in audits[:3]:
        print(f"     - jd={a['jd_id'][:8]}... by {a['closed_by']} "
              f"({a['total_llm_calls']} calls, ${a['total_cost_usd']:.4f})")
    assert len(audits) >= 1

    print("\n" + "=" * 60)
    print("ALL STAGE 9A CHECKS PASSED — REST API working end-to-end")
    print("=" * 60)


if __name__ == "__main__":
    main()
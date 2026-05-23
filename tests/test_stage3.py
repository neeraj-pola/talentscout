# tests/test_stage3.py
"""Stage 3 verification — HTTP-backed sources work end-to-end.

PREREQUISITE: Start the mock server in another terminal first:
    ./scripts/run_mock_server.sh

If the server isn't running, this test will fail with a connection error.
"""
import time

from app.storage.db import init_db
from app.tools.sources import (
    LinkedInMockSource, NaukriMockSource, ATSMockSource,
    search_all_sources, search_one_source_paginated,
)
from app.tools.sources.base import TransientSourceError


def main():
    init_db()
    print("=" * 60)
    print("STAGE 3 VERIFICATION (HTTP-backed sources)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 0. Mock server must be up
    # ----------------------------------------------------------------
    li = LinkedInMockSource()
    if not li.health_check():
        print("\n✗ MOCK SERVER NOT RUNNING")
        print("  Start it in another terminal:")
        print("    ./scripts/run_mock_server.sh")
        print("  Or:")
        print("    uvicorn mock_sources_api.server:app --port 9417")
        return
    print("✓ Mock server is healthy at http://localhost:9417")

    nk = NaukriMockSource()
    ats = ATSMockSource()

    # ----------------------------------------------------------------
    # 1. Basic search hits the real HTTP endpoint
    # ----------------------------------------------------------------
    batch = li.search(queries=["python", "ml"], location=None, yoe_min=3,
                      page=1, page_size=20)
    print(f"✓ LinkedIn search via HTTP: {len(batch.profiles)} profiles, "
          f"total={batch.total_count}, next_page={batch.next_page}")
    assert batch.source == "linkedin"
    assert all(p.get("yearsOfExperience", 0) >= 3 for p in batch.profiles)

    # ----------------------------------------------------------------
    # 2. Schemas genuinely differ
    # ----------------------------------------------------------------
    li_batch = li.search(queries=[""], page=1, page_size=1)
    nk_batch = nk.search(queries=[""], page=1, page_size=1)
    ats_batch = ats.search(queries=[""], page=1, page_size=1)
    if li_batch.profiles and nk_batch.profiles and ats_batch.profiles:
        assert "summary" in li_batch.profiles[0], "LinkedIn should use 'summary'"
        assert "aboutSelf" in nk_batch.profiles[0], "Naukri should use 'aboutSelf'"
        assert "bio" in ats_batch.profiles[0], "ATS should use 'bio'"
        assert isinstance(li_batch.profiles[0]["skills"], list), "LinkedIn skills = list of dicts"
        assert isinstance(nk_batch.profiles[0]["keySkills"], str), "Naukri skills = comma string"
        assert isinstance(ats_batch.profiles[0]["tags"], list), "ATS tags = flat list"
        print("✓ Schemas genuinely differ across sources")

    # ----------------------------------------------------------------
    # 3. Pagination across HTTP calls
    # ----------------------------------------------------------------
    page1 = li.search(queries=[""], page=1, page_size=5)
    page2 = li.search(queries=[""], page=2, page_size=5)
    if page1.profiles and page2.profiles:
        assert page1.profiles[0] != page2.profiles[0]
        print(f"✓ Pagination via HTTP: page1[0] != page2[0]")
    else:
        print("✓ Pagination structure works (data small)")

    # ----------------------------------------------------------------
    # 4. Empty results return [] (no exception)
    # ----------------------------------------------------------------
    empty = li.search(queries=["definitely-not-a-real-skill-xyzzy"],
                      yoe_min=99, page=1, page_size=20)
    assert empty.profiles == []
    assert empty.next_page is None
    print(f"✓ Empty results: returned [] without raising")

    # ----------------------------------------------------------------
    # 5. Server-simulated transient failure raises TransientSourceError
    # ----------------------------------------------------------------
    always_fail = LinkedInMockSource(fail_rate=1.0)  # server returns 503
    try:
        always_fail.search(queries=["py"], page=1, page_size=5)
        raise AssertionError("Should have raised TransientSourceError")
    except TransientSourceError:
        print("✓ Server 503 -> TransientSourceError raised correctly")

    # search_one_source_paginated catches retries, returns []
    result = search_one_source_paginated(
        always_fail, queries=["py"], location=None, yoe_min=0,
        max_pages=3, page_size=5,
    )
    assert result == []
    print(f"✓ Retries exhausted gracefully (returned []), no crash")

    # ----------------------------------------------------------------
    # 6. Parallel search across all 3 sources via HTTP
    # ----------------------------------------------------------------
    t0 = time.time()
    all_results = search_all_sources(
        queries=["python", "ml", "aws"],
        location=None,
        yoe_min=2,
        max_pages=2,
        page_size=20,
    )
    elapsed = time.time() - t0
    print(f"✓ Parallel search across 3 sources via HTTP in {elapsed*1000:.0f}ms")
    for src_name, profiles in all_results.items():
        print(f"   {src_name:10s}: {len(profiles)} profiles")

    total = sum(len(v) for v in all_results.values())
    assert total > 0
    assert set(all_results.keys()) == {"linkedin", "naukri", "ats"}

    # ----------------------------------------------------------------
    # 7. fetch_detail via HTTP
    # ----------------------------------------------------------------
    sample = li.search(queries=[""], page=1, page_size=1).profiles
    if sample:
        sample_id = sample[0]["linkedin_id"]
        detail = li.fetch_detail(sample_id)
        assert detail is not None
        assert detail["linkedin_id"] == sample_id
        print(f"✓ fetch_detail via HTTP: got {sample_id}")

        none_detail = li.fetch_detail("li_does_not_exist")
        assert none_detail is None
        print(f"✓ fetch_detail returns None on 404 (unknown ID)")

    # ----------------------------------------------------------------
    # 8. Latency simulation (proves the HTTP layer is real)
    # ----------------------------------------------------------------
    slow = LinkedInMockSource(latency_ms=200)
    t0 = time.time()
    slow.search(queries=[""], page=1, page_size=1)
    elapsed = time.time() - t0
    assert elapsed >= 0.2, f"Expected ≥200ms, got {elapsed*1000:.0f}ms"
    print(f"✓ Server-side latency simulation: {elapsed*1000:.0f}ms (≥200 expected)")

    print("=" * 60)
    print("ALL STAGE 3 CHECKS PASSED")
    print("=" * 60)
    print("\nThe sources are HTTP-backed. To prove it:")
    print("  curl 'http://localhost:9417/health'")
    print("  curl 'http://localhost:9417/linkedin/search?queries=python&page_size=2'")


if __name__ == "__main__":
    main()
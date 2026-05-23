# tests/test_stage4.py
"""Stage 4 verification — normalization + deduplication.

Requires: mock server running on port 9417 (./scripts/run_mock_server.sh).
"""
from collections import Counter

from app.storage.db import init_db
from app.tools.sources import search_all_sources, LinkedInMockSource
from app.normalize import normalize_batch, canonicalize_skill
from app.dedup import deduplicate


def main():
    init_db()
    print("=" * 60)
    print("STAGE 4 VERIFICATION (normalization + dedup)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 0. Mock server must be up
    # ----------------------------------------------------------------
    if not LinkedInMockSource().health_check():
        print("\n✗ MOCK SERVER NOT RUNNING. Start it first:")
        print("    ./scripts/run_mock_server.sh")
        return

    # ----------------------------------------------------------------
    # 1. Skill canonicalization sanity
    # ----------------------------------------------------------------
    assert canonicalize_skill("Python") == "python"
    assert canonicalize_skill("python3") == "python"
    assert canonicalize_skill("Py") == "python"
    assert canonicalize_skill("K8s") == "kubernetes"
    assert canonicalize_skill("LLMs") == "llms"
    assert canonicalize_skill("Large Language Models") == "llms"
    assert canonicalize_skill("UnknownSkillXyz") == "unknownskillxyz"  # pass-through
    print("✓ Skill canonicalization: Python/python3/Py -> 'python' etc.")

    # ----------------------------------------------------------------
    # 2. Pull ALL profiles via the HTTP sources
    # ----------------------------------------------------------------
    # Use empty queries + low page_size to get everything available
    raw_by_source = search_all_sources(
        queries=[],   # empty match-all
        location=None,
        yoe_min=0,
        max_pages=5,
        page_size=20,
    )
    raw_total = sum(len(v) for v in raw_by_source.values())
    print(f"\n✓ Pulled raw profiles from all sources:")
    for src, profs in raw_by_source.items():
        print(f"   {src:10s}: {len(profs)} raw")
    print(f"   {'TOTAL':10s}: {raw_total} raw rows")

    # ----------------------------------------------------------------
    # 3. Normalize all profiles
    # ----------------------------------------------------------------
    normalized = normalize_batch(raw_by_source)
    assert len(normalized) == raw_total, "Every raw profile should normalize"
    print(f"\n✓ Normalized {len(normalized)} profiles to CommonProfile schema")

    # Spot-check schema consistency
    sample = normalized[0]
    assert hasattr(sample, "full_name") and sample.full_name
    assert hasattr(sample, "skills") and isinstance(sample.skills, list)
    assert hasattr(sample, "raw_text") and len(sample.raw_text) > 50
    assert sample.source in ("linkedin", "naukri", "ats")
    print(f"   Sample: {sample.full_name} | {sample.source} | "
          f"{len(sample.skills)} skills | {len(sample.experiences)} jobs")

    # Skills should be canonical (lowercase, no 'python3'/'Py' variants)
    all_skills = [s for p in normalized for s in p.skills]
    bad = [s for s in all_skills if s in {"Python", "python3", "Py", "K8s", "LLMs"}]
    assert not bad, f"Found uncanonicalized skills: {bad}"
    print(f"   ✓ All {len(set(all_skills))} unique skills are canonical")

    source_counts = Counter(p.source for p in normalized)
    print(f"   By source: {dict(source_counts)}")

    # ----------------------------------------------------------------
    # 4. Deduplicate
    # ----------------------------------------------------------------
    deduped, audit = deduplicate(normalized)
    n_merges = len(audit)
    n_collapsed = len(normalized) - len(deduped)
    print(f"\n✓ Deduplication complete:")
    print(f"   {len(normalized)} normalized -> {len(deduped)} unique "
          f"({n_collapsed} records collapsed across {n_merges} merge groups)")

    # Sanity: every deduped profile is still a CommonProfile
    assert all(hasattr(p, "full_name") for p in deduped)

    # ----------------------------------------------------------------
    # 5. Show 3 example merges
    # ----------------------------------------------------------------
    if audit:
        print(f"\n   Example merges (first 3):")
        for m in audit[:3]:
            print(f"     • {m['merged_into_name']}: "
                  f"merged {m['n_records']} records from {m['sources']}")
            if m['reasons']:
                print(f"       reason: {m['reasons'][0]}")

    # ----------------------------------------------------------------
    # 6. Ground-truth check using _canonical_id we planted in the seed
    # ----------------------------------------------------------------
    # Every profile from the seed has a hidden `_canonical_id` in metadata.
    # Two profiles with the same `_canonical_id` are the same real person.
    # After dedup, each cluster should ideally collapse to one record.
    canonical_by_source_pair: dict[str, set[str]] = {}
    for p in normalized:
        cid = p.metadata.get("_canonical_id")
        if cid:
            canonical_by_source_pair.setdefault(cid, set()).add(p.source_id)

    true_dupes = {cid: srcs for cid, srcs in canonical_by_source_pair.items() if len(srcs) > 1}
    print(f"\n   Ground truth: {len(true_dupes)} canonical people exist in 2+ sources")

    # How many of the true-dupe canonical IDs ended up in ONE merged record?
    found_clusters = 0
    for cid, src_ids in true_dupes.items():
        # find which deduped record contains these source_ids
        matches = [
            p for p in deduped
            if p.source_id in src_ids
            or any(sid in src_ids for sid in p.metadata.get("merged_source_ids", []))
        ]
        # All source_ids of this canonical person should appear in ONE deduped record
        unique_records = {p.id for p in matches}
        if len(unique_records) == 1 and len(matches) == 1:
            found_clusters += 1

    recall = found_clusters / max(1, len(true_dupes))
    print(f"   Dedup recall on ground truth: "
          f"{found_clusters}/{len(true_dupes)} clusters correctly merged "
          f"({recall*100:.0f}%)")
    # We aim for >= 70% recall — some name+employer collisions are genuinely ambiguous
    assert recall >= 0.7, f"Dedup recall too low: {recall:.2f}"

    print("\n" + "=" * 60)
    print("ALL STAGE 4 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
# app/dedup/dedup.py
"""Deterministic deduplication of CommonProfiles across sources.

Approach: blocking + pairwise similarity. Blocking is the standard trick to
avoid O(n²) — group records that *could* be duplicates into small buckets,
then do expensive comparison only within each bucket.

Blocking key: first letter of name + first word of location.
Similarity:   Jaro-Winkler on full name + employer/education overlap.

Merge rule:   pick the profile with the most complete data as base,
              union skills, keep all source_ids in merged_from + metadata.
"""
from collections import defaultdict
from uuid import UUID

import jellyfish

from app.models import CommonProfile
from app.obs.events import log_event


NAME_SIM_THRESHOLD = 0.92      # Jaro-Winkler — names that pass
COMPANY_OVERLAP_REQUIRED = 1    # ≥1 shared employer to confirm same person
INSTITUTION_OVERLAP_REQUIRED = 1  # OR ≥1 shared school


# ============================================================
# 1. Blocking
# ============================================================

def _block_key(p: CommonProfile) -> str:
    """Group candidates that could be duplicates.

    Key: first letter of first name (lowercase) + first word of location.
    Examples:
      ("Aarav Sharma", "Bangalore, India") -> "a|bangalore,"
      ("Aarav Sharma", "Bangalore")        -> "a|bangalore"
    """
    name = p.full_name.strip().lower()
    first_initial = name[0] if name else "?"
    loc_first = p.location.strip().lower().split()[0] if p.location else "?"
    return f"{first_initial}|{loc_first}"


def _build_blocks(profiles: list[CommonProfile]) -> dict[str, list[CommonProfile]]:
    blocks: dict[str, list[CommonProfile]] = defaultdict(list)
    for p in profiles:
        blocks[_block_key(p)].append(p)
    return blocks


# ============================================================
# 2. Pairwise similarity within a block
# ============================================================

def _name_similarity(a: str, b: str) -> float:
    """Jaro-Winkler. 1.0 = identical, 0.0 = completely different."""
    if not a or not b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(a.lower().strip(), b.lower().strip())


def _shared_companies(a: CommonProfile, b: CommonProfile) -> set[str]:
    ca = {e.company.lower().strip() for e in a.experiences if e.company}
    cb = {e.company.lower().strip() for e in b.experiences if e.company}
    return ca & cb


def _shared_institutions(a: CommonProfile, b: CommonProfile) -> set[str]:
    ia = {e.institution.lower().strip() for e in a.education if e.institution}
    ib = {e.institution.lower().strip() for e in b.education if e.institution}
    return ia & ib


def _is_duplicate(a: CommonProfile, b: CommonProfile) -> tuple[bool, str]:
    """Decide if two profiles are the same person. Returns (yes_no, reason)."""
    # Sanity: never merge two profiles from the SAME source — they're already distinct
    if a.source == b.source:
        return False, "same source"

    name_sim = _name_similarity(a.full_name, b.full_name)
    if name_sim < NAME_SIM_THRESHOLD:
        return False, f"name similarity {name_sim:.2f} below {NAME_SIM_THRESHOLD}"

    companies = _shared_companies(a, b)
    institutions = _shared_institutions(a, b)

    if len(companies) >= COMPANY_OVERLAP_REQUIRED:
        return True, f"name_sim={name_sim:.2f}, shared employer(s): {sorted(companies)}"
    if len(institutions) >= INSTITUTION_OVERLAP_REQUIRED:
        return True, f"name_sim={name_sim:.2f}, shared school(s): {sorted(institutions)}"

    # High name match but no employer/school confirmation — leave as separate
    # (avoids merging two real different people with similar names)
    return False, f"name_sim={name_sim:.2f} OK but no employer/school overlap"


# ============================================================
# 3. Merge
# ============================================================

def _completeness_score(p: CommonProfile) -> int:
    """Rough 'how complete is this record' score. Higher = better base for merge."""
    score = 0
    score += 10 if p.contact_email else 0
    score += len(p.skills)
    score += len(p.experiences) * 3
    score += len(p.education) * 2
    score += len(p.raw_text) // 100
    return score


def _merge_pair(base: CommonProfile, other: CommonProfile) -> CommonProfile:
    """Merge `other` into `base`, returning a new CommonProfile.
    The most complete record is the base; the other contributes missing fields."""
    merged_skills = list({*base.skills, *other.skills})

    merged = base.model_copy(update={
        "skills": sorted(merged_skills),
        "contact_email": base.contact_email or other.contact_email,
        "headline": base.headline or other.headline,
        "merged_from": [*base.merged_from, *other.merged_from, other.id],
        "metadata": {
            **other.metadata,
            **base.metadata,
            "merged_source_ids": (
                base.metadata.get("merged_source_ids", [base.source_id])
                + [other.source_id]
            ),
            "merged_sources": list({
                *base.metadata.get("merged_sources", [base.source]),
                other.source,
            }),
        },
    })
    return merged


# ============================================================
# 4. Public API
# ============================================================

def deduplicate(
    profiles: list[CommonProfile],
    jd_id: str | None = None,
) -> tuple[list[CommonProfile], list[dict]]:
    """Deduplicate a list of CommonProfiles.

    Returns:
        (deduped_profiles, merge_audit_records)
        merge_audit_records = list of {"merged_into": id, "reason": "...", "from": [ids]}
    """
    log_event(jd_id, "dedup", "start", n_input=len(profiles))

    blocks = _build_blocks(profiles)
    log_event(jd_id, "dedup", "blocks_built",
              n_blocks=len(blocks),
              max_block_size=max((len(v) for v in blocks.values()), default=0))

    # Union-Find structure: profile.id -> "root" id of its cluster
    parent: dict[UUID, UUID] = {p.id: p.id for p in profiles}
    merge_reasons: dict[tuple[UUID, UUID], str] = {}

    def find(x: UUID) -> UUID:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: UUID, b: UUID, reason: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            merge_reasons[(ra, rb)] = reason

    # Compare within each block
    pair_comparisons = 0
    for key, members in blocks.items():
        if len(members) <= 1:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_comparisons += 1
                a, b = members[i], members[j]
                is_dup, reason = _is_duplicate(a, b)
                if is_dup:
                    union(a.id, b.id, reason)

    log_event(jd_id, "dedup", "comparisons_done",
              pair_comparisons=pair_comparisons)

    # Build clusters
    clusters: dict[UUID, list[CommonProfile]] = defaultdict(list)
    by_id = {p.id: p for p in profiles}
    for pid in parent:
        clusters[find(pid)].append(by_id[pid])

    # Merge each cluster — base = most complete record
    deduped: list[CommonProfile] = []
    audit: list[dict] = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            deduped.append(cluster[0])
            continue
        cluster_sorted = sorted(cluster, key=_completeness_score, reverse=True)
        base = cluster_sorted[0]
        merged = base
        for other in cluster_sorted[1:]:
            merged = _merge_pair(merged, other)
        deduped.append(merged)
        audit.append({
            "merged_into": str(merged.id),
            "merged_into_name": merged.full_name,
            "n_records": len(cluster),
            "sources": sorted({p.source for p in cluster}),
            "source_ids": [p.source_id for p in cluster],
            "reasons": list({
                merge_reasons.get((find(c.id), c.id), "merged via transitive cluster")
                for c in cluster_sorted[1:]
                if (find(c.id), c.id) in merge_reasons
            }),
        })

    log_event(jd_id, "dedup", "end",
              n_input=len(profiles), n_output=len(deduped),
              n_merges=len(audit),
              n_collapsed_records=len(profiles) - len(deduped))

    return deduped, audit
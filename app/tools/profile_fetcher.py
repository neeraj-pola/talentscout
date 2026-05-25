# app/tools/profile_fetcher.py
"""Profile detail fetcher tool.

Wraps each source's `fetch_detail(source_id)` with consistent retries,
empty-result handling, and observability. Agents call `fetch_profile_detail`
rather than reaching into source clients directly, so the tool boundary is
uniform across all 6 spec-required operations.

Public API:
    fetch_profile_detail(source, source_id, jd_id=None) -> dict | None
    fetch_profile_details_bulk(refs, jd_id=None) -> dict[str, dict]

Behaviors:
  - Retries on TransientSourceError (network, 5xx, 429) up to 3 attempts
    with exponential backoff
  - Returns None on 404 (profile no longer exists at source) — agents
    handle "missing profile" cleanly rather than crashing
  - Permanent errors (auth, malformed source_id) surface as raised
    PermanentSourceError — these indicate config bugs, not data gaps
"""
from __future__ import annotations

from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, RetryError,
)

from app.tools.sources import SOURCES
from app.tools.sources.base import (
    HTTPProfileSource, TransientSourceError, PermanentSourceError,
)
from app.obs.events import log_event


# Source name → instance lookup. SOURCES is initialized at module import time
# in app.tools.sources.__init__, so this stays in sync with whatever sources
# are registered globally.
_SOURCE_BY_NAME: dict[str, HTTPProfileSource] = {s.source_name: s for s in SOURCES}


@retry(
    retry=retry_if_exception_type(TransientSourceError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=5),
    reraise=True,
)
def _fetch_with_retries(source: HTTPProfileSource, source_id: str) -> dict | None:
    """Inner retry-wrapped call. Returns None for 404, raises on permanent errors."""
    return source.fetch_detail(source_id)


def fetch_profile_detail(
    source: str,
    source_id: str,
    jd_id: str | None = None,
) -> dict | None:
    """Fetch the full profile record from one source by its source-side ID.

    Args:
        source:     "linkedin" | "naukri" | "ats"
        source_id:  The source-side ID for this candidate
        jd_id:      Optional JD UUID for event correlation

    Returns:
        Raw source-shaped dict, or None if the source returned 404
        (profile not found / deleted at source).

    Raises:
        ValueError:              Unknown source name
        TransientSourceError:    Repeated network failures (after retries)
        PermanentSourceError:    Auth, malformed ID, 4xx (other than 404)
    """
    if source not in _SOURCE_BY_NAME:
        raise ValueError(
            f"Unknown source '{source}'. Known: {list(_SOURCE_BY_NAME.keys())}"
        )

    src = _SOURCE_BY_NAME[source]
    log_event(jd_id, f"tool.profile_fetcher", "fetch_start",
              source=source, source_id=source_id)

    try:
        detail = _fetch_with_retries(src, source_id)
    except (TransientSourceError, RetryError) as e:
        log_event(jd_id, "tool.profile_fetcher", "give_up_after_retries",
                  source=source, source_id=source_id, error=str(e))
        return None
    except PermanentSourceError as e:
        log_event(jd_id, "tool.profile_fetcher", "permanent_error",
                  source=source, source_id=source_id, error=str(e))
        raise

    log_event(jd_id, "tool.profile_fetcher", "fetch_end",
              source=source, source_id=source_id,
              found=detail is not None,
              fields=list(detail.keys()) if detail else [])
    return detail


def fetch_profile_details_bulk(
    refs: list[tuple[str, str]],
    jd_id: str | None = None,
) -> dict[str, dict | None]:
    """Fetch multiple profiles by (source, source_id) pairs.

    Each fetch is independent — a transient failure on one doesn't abort
    the others. Results are returned keyed by `source:source_id`.

    Args:
        refs:   List of (source_name, source_id) tuples
        jd_id:  Optional JD UUID for event correlation

    Returns:
        Dict mapping "source:source_id" -> profile dict (or None if missing).
    """
    log_event(jd_id, "tool.profile_fetcher", "bulk_fetch_start", n=len(refs))
    results: dict[str, dict | None] = {}
    for source, source_id in refs:
        key = f"{source}:{source_id}"
        try:
            results[key] = fetch_profile_detail(source, source_id, jd_id=jd_id)
        except PermanentSourceError:
            # Permanent errors are config bugs; log + record None, don't crash
            results[key] = None
    found = sum(1 for v in results.values() if v is not None)
    log_event(jd_id, "tool.profile_fetcher", "bulk_fetch_end",
              n_requested=len(refs), n_found=found)
    return results
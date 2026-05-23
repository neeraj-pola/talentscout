# app/tools/sources/__init__.py
"""Parallel multi-source search with per-source retries.

Public API:
    SOURCES                         — default list of three sources
    search_all_sources(...)         — parallel call across sources
    search_one_source_paginated(...) — page through one source
"""
import concurrent.futures

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError,
)

from app.tools.sources.base import (
    HTTPProfileSource, TransientSourceError, PermanentSourceError,
)
from app.tools.sources.linkedin import LinkedInMockSource
from app.tools.sources.naukri import NaukriMockSource
from app.tools.sources.ats import ATSMockSource
from app.models import RawProfileBatch
from app.obs.events import log_event


SOURCES: list[HTTPProfileSource] = [
    LinkedInMockSource(),
    NaukriMockSource(),
    ATSMockSource(),
]


@retry(
    retry=retry_if_exception_type(TransientSourceError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=5),
    reraise=True,
)
def _search_with_retries(source: HTTPProfileSource, **kwargs) -> RawProfileBatch:
    """Retry only on transient errors (network, 5xx, 429)."""
    return source.search(**kwargs)


def search_one_source_paginated(
    source: HTTPProfileSource,
    queries: list[str],
    location: str | None,
    yoe_min: int,
    max_pages: int = 3,
    page_size: int = 20,
    jd_id: str | None = None,
) -> list[dict]:
    """Page through a single source. Robust to per-page transient failures."""
    all_profiles: list[dict] = []
    page = 1
    while page <= max_pages:
        try:
            batch = _search_with_retries(
                source,
                queries=queries, location=location, yoe_min=yoe_min,
                page=page, page_size=page_size,
            )
        except (TransientSourceError, RetryError) as e:
            log_event(jd_id, f"tool.{source.source_name}", "give_up_after_retries",
                      error=str(e), page=page)
            break
        except PermanentSourceError as e:
            log_event(jd_id, f"tool.{source.source_name}", "permanent_error",
                      error=str(e), page=page)
            break

        all_profiles.extend(batch.profiles)
        log_event(jd_id, f"tool.{source.source_name}", "page_complete",
                  page=page, page_size=len(batch.profiles),
                  total_so_far=len(all_profiles))

        if batch.next_page is None:
            break
        page = batch.next_page

    return all_profiles


def search_all_sources(
    queries: list[str],
    location: str | None = None,
    yoe_min: int = 0,
    max_pages: int = 3,
    page_size: int = 20,
    sources: list[HTTPProfileSource] | None = None,
    jd_id: str | None = None,
) -> dict[str, list[dict]]:
    """Query all sources in parallel. Per-source failures don't block others."""
    sources = sources or SOURCES
    log_event(jd_id, "sourcing", "parallel_search_start",
              sources=[s.source_name for s in sources], queries=queries,
              location=location, yoe_min=yoe_min)

    results: dict[str, list[dict]] = {s.source_name: [] for s in sources}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as ex:
        future_to_src = {
            ex.submit(
                search_one_source_paginated,
                s, queries, location, yoe_min, max_pages, page_size, jd_id,
            ): s for s in sources
        }
        for fut in concurrent.futures.as_completed(future_to_src):
            src = future_to_src[fut]
            try:
                results[src.source_name] = fut.result()
            except Exception as e:
                log_event(jd_id, f"tool.{src.source_name}", "source_failed_completely",
                          error=str(e))
                results[src.source_name] = []

    log_event(jd_id, "sourcing", "parallel_search_end",
              counts={k: len(v) for k, v in results.items()},
              total=sum(len(v) for v in results.values()))
    return results
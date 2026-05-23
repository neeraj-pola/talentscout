# app/tools/sources/base.py
"""Abstract base for profile sources. All sources hit a REAL HTTP endpoint.

This is the key architectural choice: real or mock, the source class issues
HTTP GETs against some base URL. In production, swap the base URL to a real
LinkedIn Talent Solutions endpoint with no other code changes.
"""
from abc import ABC, abstractmethod
from typing import Literal

import httpx

from app.config import settings
from app.models import RawProfileBatch


class TransientSourceError(Exception):
    """Retryable error — network blip, 5xx, rate limit."""


class PermanentSourceError(Exception):
    """Non-retryable error — 4xx (except 429), bad query, auth failure."""


class HTTPProfileSource(ABC):
    """Base class for any HTTP-backed source.

    Concrete sources (LinkedIn, Naukri, ATS) just declare their URL path
    prefix and ID field name; the search/fetch logic is shared here.
    """

    source_name: Literal["linkedin", "naukri", "ats"]
    path_prefix: str  # e.g. "/linkedin"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
        fail_rate: float = 0.0,      # passed to server to simulate failures
        latency_ms: int = 0,          # passed to server to simulate latency
    ):
        self.base_url = (base_url or settings.mock_sources_base_url).rstrip("/")
        self.timeout = timeout
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _get(self, path: str, params: dict) -> dict:
        """Issue a GET and translate HTTP errors into our exception types."""
        # Merge simulation knobs into params
        params = {**params, "fail_rate": self.fail_rate, "latency_ms": self.latency_ms}
        try:
            r = self._client.get(path, params=params)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            raise TransientSourceError(f"{self.source_name} network error: {e}") from e

        if r.status_code in (429, 502, 503, 504):
            raise TransientSourceError(f"{self.source_name} returned {r.status_code}")
        if r.status_code == 404:
            return {}  # caller decides what to do
        if r.status_code >= 400:
            raise PermanentSourceError(f"{self.source_name} returned {r.status_code}: {r.text}")
        return r.json()

    def search(
        self,
        queries: list[str],
        location: str | None = None,
        yoe_min: int = 0,
        page: int = 1,
        page_size: int = 20,
    ) -> RawProfileBatch:
        params: dict = {
            "queries": queries,
            "yoe_min": yoe_min,
            "page": page,
            "page_size": page_size,
        }
        if location:
            params["location"] = location

        data = self._get(f"{self.path_prefix}/search", params)
        return RawProfileBatch(
            source=self.source_name,
            profiles=data.get("profiles", []),
            next_page=data.get("next_page"),
            total_count=data.get("total_count"),
        )

    @abstractmethod
    def fetch_detail(self, source_id: str) -> dict | None:
        ...

    def _fetch_by_path(self, source_id: str) -> dict | None:
        data = self._get(f"{self.path_prefix}/profile/{source_id}", params={})
        return data if data else None

    def health_check(self) -> bool:
        """Used by tests / startup probes."""
        try:
            r = self._client.get("/health", params={})
            return r.status_code == 200
        except Exception:
            return False
# ui/api_client.py
"""Thin HTTP client for the TalentScout API. Streamlit pages use this
instead of calling requests directly so we keep the URL config in one place."""
from __future__ import annotations

import os
from typing import Any

import requests

API_BASE_URL = os.getenv("TALENTSCOUT_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 180.0  # pipeline runs are slow


class APIError(Exception):
    pass


def _req(method: str, path: str, **kwargs) -> Any:
    url = f"{API_BASE_URL}{path}"
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    try:
        r = requests.request(method, url, **kwargs)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot reach API at {API_BASE_URL}. Is the API server running?")
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise APIError(f"API {r.status_code}: {detail}")
    return r.json()


def health() -> dict:
    return _req("GET", "/health")


def list_jds() -> list[dict]:
    return _req("GET", "/jds")


def create_jd(payload: dict) -> dict:
    return _req("POST", "/jds", json=payload)


def get_jd_detail(jd_id: str) -> dict:
    return _req("GET", f"/jds/{jd_id}")


def close_jd(jd_id: str, closed_by: str, candidate_id: str) -> dict:
    return _req(
        "POST", f"/jds/{jd_id}/close",
        json={"closed_by": closed_by, "candidate_id": candidate_id},
    )


def refine_jd(jd_id: str, message: str) -> dict:
    """Submit a refinement message. Returns one assistant turn + refined shortlist.

    Long-running turns (find_similar) can take 30-60s, so we lean on the default
    180s timeout. If the timeout fires, the user should resend.
    """
    return _req("POST", f"/jds/{jd_id}/refine", json={"message": message})


def list_audits() -> list[dict]:
    return _req("GET", "/audits")


def get_sample_jds() -> list[dict]:
    return _req("GET", "/demo/sample-jds")


def get_cost(jd_id: str) -> dict:
    return _req("GET", f"/jds/{jd_id}/cost")
# mock_sources_api/server.py
"""Mock external sources HTTP server.

Simulates three external recruitment APIs (LinkedIn, Naukri, ATS) backed by
the seed JSON files. Real HTTP boundary so the ProfileSource clients exercise
the network stack, retry logic, and timeout handling end-to-end.

Run:
    uvicorn mock_sources_api.server:app --port 9417

Endpoints:
    GET  /health
    GET  /linkedin/search?queries=&location=&yoe_min=&page=&page_size=&fail_rate=
    GET  /linkedin/profile/{linkedin_id}
    GET  /naukri/search?...
    GET  /naukri/profile/{naukri_id}
    GET  /ats/search?...
    GET  /ats/profile/{ats_id}
"""
import json
import random
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

DATA_DIR = Path(__file__).parent.parent / "data"

app = FastAPI(
    title="TalentScout Mock Sources API",
    description="HTTP-backed mocks for LinkedIn, Naukri, and internal ATS. "
                "Same shape as real recruitment APIs would have.",
    version="1.0.0",
)


def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / f"{name}_profiles.json").read_text())


_DATA = {
    "linkedin": _load("linkedin"),
    "naukri": _load("naukri"),
    "ats": _load("ats"),
}


def _maybe_fail(fail_rate: float, source: str):
    """Simulate transient API failures via a 503 response."""
    if random.random() < fail_rate:
        raise HTTPException(status_code=503, detail=f"{source} upstream temporarily unavailable")


def _maybe_latency(latency_ms: int):
    """Simulate API latency to make the demo feel real."""
    if latency_ms > 0:
        time.sleep(latency_ms / 1000.0)


def _parse_total_exp(s: str) -> float:
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return 0.0


def _filter_linkedin(rows: list[dict], queries: list[str], location: str | None, yoe_min: int) -> list[dict]:
    out = []
    for p in rows:
        if yoe_min and p.get("yearsOfExperience", 0) < yoe_min:
            continue
        if location and location.lower() != "remote":
            loc = p.get("location", {}).get("name", "").lower()
            if location.lower() not in loc and "remote" not in loc:
                continue
        blob = (
            p.get("headline", "") + " " + p.get("summary", "") + " " +
            " ".join(s["name"] for s in p.get("skills", []))
        ).lower()
        if not queries or any(any(t.lower() in blob for t in q.split()) for q in queries):
            out.append(p)
    return out


def _filter_naukri(rows: list[dict], queries: list[str], location: str | None, yoe_min: int) -> list[dict]:
    out = []
    for p in rows:
        yrs = _parse_total_exp(p.get("totalExp", "0 years"))
        if yoe_min and yrs < yoe_min:
            continue
        if location and location.lower() != "remote":
            loc = p.get("currentLocation", "").lower()
            if location.lower() not in loc and "remote" not in loc:
                continue
        blob = (
            p.get("currentDesignation", "") + " " +
            p.get("aboutSelf", "") + " " +
            p.get("keySkills", "")
        ).lower()
        if not queries or any(any(t.lower() in blob for t in q.split()) for q in queries):
            out.append(p)
    return out


def _filter_ats(rows: list[dict], queries: list[str], location: str | None, yoe_min: int) -> list[dict]:
    out = []
    for p in rows:
        if yoe_min and p.get("tenure_years", 0) < yoe_min:
            continue
        if location and location.lower() != "remote":
            loc = p.get("city", "").lower()
            if location.lower() not in loc and "remote" not in loc:
                continue
        blob = (
            p.get("role", "") + " " + p.get("bio", "") + " " +
            " ".join(p.get("tags", []))
        ).lower()
        if not queries or any(any(t.lower() in blob for t in q.split()) for q in queries):
            out.append(p)
    return out


FILTERS = {
    "linkedin": _filter_linkedin,
    "naukri":   _filter_naukri,
    "ats":      _filter_ats,
}
ID_FIELDS = {
    "linkedin": "linkedin_id",
    "naukri":   "naukri_id",
    "ats":      "ats_id",
}


def _paginate(results: list[dict], page: int, page_size: int) -> tuple[list[dict], int | None]:
    start = (page - 1) * page_size
    end = start + page_size
    page_results = results[start:end]
    next_page = page + 1 if end < len(results) else None
    return page_results, next_page


def _search_handler(
    source: Literal["linkedin", "naukri", "ats"],
    queries: list[str], location: str | None, yoe_min: int,
    page: int, page_size: int, fail_rate: float, latency_ms: int,
):
    _maybe_fail(fail_rate, source)
    _maybe_latency(latency_ms)
    results = FILTERS[source](_DATA[source], queries, location, yoe_min)
    page_results, next_page = _paginate(results, page, page_size)
    return {
        "source": source,
        "profiles": page_results,
        "next_page": next_page,
        "total_count": len(results),
        "page": page,
        "page_size": page_size,
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "sources": {k: len(v) for k, v in _DATA.items()},
    }


@app.get("/linkedin/search")
def linkedin_search(
    queries: list[str] = Query(default=[]),
    location: str | None = None,
    yoe_min: int = 0,
    page: int = 1,
    page_size: int = 20,
    fail_rate: float = 0.0,
    latency_ms: int = 0,
):
    return _search_handler("linkedin", queries, location, yoe_min,
                           page, page_size, fail_rate, latency_ms)


@app.get("/linkedin/profile/{linkedin_id}")
def linkedin_profile(linkedin_id: str, fail_rate: float = 0.0, latency_ms: int = 0):
    _maybe_fail(fail_rate, "linkedin")
    _maybe_latency(latency_ms)
    for p in _DATA["linkedin"]:
        if p["linkedin_id"] == linkedin_id:
            return p
    raise HTTPException(status_code=404, detail="not found")


@app.get("/naukri/search")
def naukri_search(
    queries: list[str] = Query(default=[]),
    location: str | None = None,
    yoe_min: int = 0,
    page: int = 1,
    page_size: int = 20,
    fail_rate: float = 0.0,
    latency_ms: int = 0,
):
    return _search_handler("naukri", queries, location, yoe_min,
                           page, page_size, fail_rate, latency_ms)


@app.get("/naukri/profile/{naukri_id}")
def naukri_profile(naukri_id: str, fail_rate: float = 0.0, latency_ms: int = 0):
    _maybe_fail(fail_rate, "naukri")
    _maybe_latency(latency_ms)
    for p in _DATA["naukri"]:
        if p["naukri_id"] == naukri_id:
            return p
    raise HTTPException(status_code=404, detail="not found")


@app.get("/ats/search")
def ats_search(
    queries: list[str] = Query(default=[]),
    location: str | None = None,
    yoe_min: int = 0,
    page: int = 1,
    page_size: int = 20,
    fail_rate: float = 0.0,
    latency_ms: int = 0,
):
    return _search_handler("ats", queries, location, yoe_min,
                           page, page_size, fail_rate, latency_ms)


@app.get("/ats/profile/{ats_id}")
def ats_profile(ats_id: str, fail_rate: float = 0.0, latency_ms: int = 0):
    _maybe_fail(fail_rate, "ats")
    _maybe_latency(latency_ms)
    for p in _DATA["ats"]:
        if p["ats_id"] == ats_id:
            return p
    raise HTTPException(status_code=404, detail="not found")
# app/api/server.py
"""TalentScout REST API.

Thin layer over the orchestrator and storage. All business logic lives in
app/agents/ and app/orchestrator/. The API translates HTTP <-> domain objects.

Endpoints:
  GET  /health                — service health + mock-server reachability
  GET  /jds                   — list all JDs (lightweight)
  POST /jds                   — create AND immediately run a JD through the pipeline
  GET  /jds/{id}              — full detail for one JD (shortlist, top pick, events, cost)
  POST /jds/{id}/close        — close a JD with a chosen candidate, write audit record
  GET  /audits                — list all audit records
  GET  /jds/{id}/cost         — cost summary for one JD
  GET  /jds/{id}/events       — event log for one JD (for live UI tail)
"""
from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import ensure_db_initialized
from app.api.schemas import (
    CreateJDRequest, CloseJDRequest,
    PipelineRunResponse, JDDetailResponse, JDSummary,
    AuditRecordResponse, HealthResponse,RefineRequest, RefineResponse,
)
from app.models import JD, AuditRecord
from app.orchestrator import run_pipeline, get_checkpoint
from app.storage.jd_repo import (
    create_jd, get_jd, list_jds, close_jd,
)
from app.storage.audit_repo import create_audit, list_audits
from app.storage.db import get_session, JDRow
from app.obs.cost import get_cost_summary
from app.obs.events import get_events_for_jd
from app.tools.sources import LinkedInMockSource
from app.agents.refinement import run_refinement


# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="TalentScout API",
    description="AI agent for end-to-end recruitment funnel automation.",
    version="1.0.0",
)

# CORS — wide open for local dev. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    ensure_db_initialized()


# ============================================================
# Health
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    mock_up = False
    try:
        mock_up = LinkedInMockSource().health_check()
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        mock_server_reachable=mock_up,
        db_initialized=True,
    )


# ============================================================
# JDs
# ============================================================

@app.get("/jds", response_model=list[JDSummary])
def list_all_jds():
    """List all JDs (lightweight summary)."""
    jds = list_jds()
    return [
        JDSummary(
            id=jd.id,
            title=jd.title,
            location=jd.location,
            status=jd.status.value,
            created_at=jd.created_at.isoformat(),
            target_hiring_date=jd.target_hiring_date.isoformat(),
            closed_at=jd.closed_at.isoformat() if jd.closed_at else None,
            closed_by=jd.closed_by,
        )
        for jd in jds
    ]


@app.post("/jds", response_model=PipelineRunResponse, status_code=status.HTTP_201_CREATED)
def create_and_run_jd(req: CreateJDRequest):
    """Create a JD and run it through the full pipeline synchronously.

    Note: This blocks for ~60-120 seconds while the pipeline runs.
    For production, this would be a 202 Accepted + background job pattern.
    For a take-home demo, synchronous is simpler and the user sees the result.
    """
    # Build the domain JD
    jd = JD(
        title=req.title,
        description=req.description,
        must_have_skills=req.must_have_skills,
        nice_to_have_skills=req.nice_to_have_skills,
        min_years_experience=req.min_years_experience,
        max_years_experience=req.max_years_experience,
        location=req.location,
        remote_ok=req.remote_ok,
        employment_type=req.employment_type,
        target_hiring_date=req.target_hiring_date,
    )

    # Persist the JD first so it exists in the DB even if the pipeline crashes
    create_jd(jd)

    # Run the orchestrator
    try:
        final_state = run_pipeline(jd)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {e!s}",
        )

    return PipelineRunResponse(
        jd_id=jd.id,
        status=final_state.get("status", "unknown"),
        halt_reason=final_state.get("halt_reason"),
        guardrail_verdict=final_state.get("guardrail_verdict"),
        parsed_jd=final_state.get("parsed_jd"),
        sourcing_result=final_state.get("sourcing_result"),
        shortlist=final_state.get("shortlist", []),
        top_pick=final_state.get("top_pick"),
        outreach_drafts=final_state.get("outreach_drafts", []),
    )


@app.get("/jds/{jd_id}", response_model=JDDetailResponse)
def get_jd_detail(jd_id: UUID):
    """Return everything needed to render the JD detail page."""
    jd = get_jd(jd_id)
    if jd is None:
        raise HTTPException(status_code=404, detail=f"JD {jd_id} not found")

    # Read the cached pipeline output from the DB
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        parsed_jd = json.loads(row.parsed_jd_json) if row and row.parsed_jd_json else None
        shortlist = json.loads(row.shortlist_json) if row and row.shortlist_json else []
        top_pick = json.loads(row.top_pick_json) if row and row.top_pick_json else None
        outreach_drafts = json.loads(row.outreach_json) if row and row.outreach_json else []
        # New fields added for sourcing, profiles, guardrails, refinement
        profiles = json.loads(row.profiles_json) if row and row.profiles_json else []
        sourcing_summary_db = json.loads(row.sourcing_json) if row and row.sourcing_json else None
        merge_audit_db = json.loads(row.merge_audit_json) if row and row.merge_audit_json else []
        guardrail_verdict = json.loads(row.guardrail_verdict_json) if row and row.guardrail_verdict_json else None

    # Load refinement state (conversation history + filter stack + cost)
    from app.storage.jd_repo import load_refinement_state
    refinement_state = load_refinement_state(jd_id) or {
        "conversation_history": [],
        "filter_stack": [],
        "total_refinement_cost_usd": 0.0,
    }

    # Compute refined_shortlist by applying the persisted filter_stack to
    # the full shortlist. The UI displays this so what the user sees matches
    # the backend's filter state.
    filter_stack = refinement_state.get("filter_stack") or []
    if filter_stack and shortlist:
        try:
            from app.agents.refinement import _apply_filter_stack
            profiles_by_id = {p["id"]: p for p in profiles}
            refined_shortlist = _apply_filter_stack(shortlist, profiles_by_id, filter_stack)
        except Exception:
            refined_shortlist = shortlist
    else:
        refined_shortlist = shortlist
    refinement_state["refined_shortlist"] = refined_shortlist

    # Pull supplementary info from the checkpoint (fallback for older JDs that
    # don't have sourcing/merge data in the new columns)
    snapshot = get_checkpoint(str(jd_id)) or {}

    return JDDetailResponse(
        jd=jd.model_dump(mode="json"),
        status=jd.status.value,
        parsed_jd=parsed_jd,
        shortlist=shortlist,
        top_pick=top_pick,
        outreach_drafts=outreach_drafts,
        merge_audit=merge_audit_db or snapshot.get("merge_audit", []),
        sourcing_summary=sourcing_summary_db or snapshot.get("sourcing_result"),
        cost_summary=get_cost_summary(str(jd_id)),
        events=get_events_for_jd(str(jd_id)),
        guardrail_verdict=guardrail_verdict,
        refinement_state=refinement_state,
        profiles=profiles,
    )


@app.post("/jds/{jd_id}/close", status_code=status.HTTP_200_OK)
def close_jd_endpoint(jd_id: UUID, req: CloseJDRequest):
    """Close the JD with a chosen candidate. Writes an AuditRecord."""
    jd = get_jd(jd_id)
    if jd is None:
        raise HTTPException(status_code=404, detail=f"JD {jd_id} not found")

    if jd.status.value == "closed":
        raise HTTPException(
            status_code=409,
            detail=f"JD {jd_id} is already closed",
        )

    # Load shortlist + top pick to extract justification + ranking snapshot
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        shortlist = json.loads(row.shortlist_json) if row and row.shortlist_json else []
        top_pick = json.loads(row.top_pick_json) if row and row.top_pick_json else None

    # Build the justification: use the top-pick justification if the chosen
    # candidate IS the top pick, otherwise note the override
    if top_pick and UUID(top_pick["recommended_candidate_id"]) == req.candidate_id:
        justification = top_pick.get("justification", "Closed with recommended top pick.")
    else:
        justification = "Closed with manually selected candidate (overriding system top pick)."

    ranking_snapshot = [
        UUID(c["profile_id"]) for c in shortlist
    ]

    # Cost data for the audit record
    cost = get_cost_summary(str(jd_id))

    audit = AuditRecord(
        jd_id=jd_id,
        candidate_id=req.candidate_id,
        closed_by=req.closed_by,
        closed_at=datetime.utcnow().isoformat(),
        justification=justification,
        final_ranking_snapshot=ranking_snapshot,
        total_cost_usd=cost["total_usd"],
        total_tokens=cost["total_tokens_in"] + cost["total_tokens_out"],
        total_llm_calls=cost["total_calls"],
    )

    # Atomic: close JD + write audit
    close_jd(jd_id, req.candidate_id, req.closed_by)
    create_audit(audit)

    return {
        "ok": True,
        "jd_id": str(jd_id),
        "candidate_id": str(req.candidate_id),
        "closed_by": req.closed_by,
        "closed_at": audit.closed_at,
        "audit_recorded": True,
    }


# ============================================================
# Cost and observability endpoints
# ============================================================

@app.get("/jds/{jd_id}/cost")
def get_cost(jd_id: UUID):
    return get_cost_summary(str(jd_id))


@app.get("/jds/{jd_id}/events")
def get_events(jd_id: UUID, limit: int = 100):
    events = get_events_for_jd(str(jd_id))
    return {"jd_id": str(jd_id), "count": len(events), "events": events[-limit:]}


@app.get("/audits", response_model=list[AuditRecordResponse])
def list_all_audits():
    return [AuditRecordResponse(**a) for a in list_audits()]

@app.post("/jds/{jd_id}/refine", response_model=RefineResponse)
def refine_jd_endpoint(jd_id: UUID, req: RefineRequest):
    """Apply one natural-language refinement turn against a shortlisted JD.

    The refinement agent uses OpenAI tool calling to classify the recruiter's
    intent into one of 11 tools (filter by YOE/location/skill, explain,
    compare, regenerate outreach, find similar, get details, clarify), then
    dispatches the matching handler. Conversation history and filter stack
    persist across turns via refinement_state on the JD row.
    """
    # Verify JD exists and is in a state that supports refinement
    jd = get_jd(jd_id)
    if jd is None:
        raise HTTPException(status_code=404, detail=f"JD {jd_id} not found")

    if jd.status.value == "closed":
        raise HTTPException(
            status_code=409,
            detail="Cannot refine a closed JD. Reopen or create a new one.",
        )

    if jd.status.value not in ("shortlisted",):
        raise HTTPException(
            status_code=409,
            detail=(
                f"JD status is '{jd.status.value}'. Refinement is only available "
                "after the pipeline has produced a shortlist."
            ),
        )

    # Run the refinement turn
    result = run_refinement(str(jd_id), req.message)

    return RefineResponse(
        assistant_message=result.assistant_message,
        intent=result.intent,
        parameters=result.parameters,
        active_filters=result.active_filters,
        refined_shortlist=result.refined_shortlist,
        metadata=result.metadata,
        cost_usd=result.cost_usd,
        n_llm_calls=result.n_llm_calls,
        turn_latency_ms=result.turn_latency_ms,
    )

# ============================================================
# Demo / catalog endpoints (handy for the UI)
# ============================================================

@app.get("/demo/sample-jds")
def sample_jds():
    """Sample JDs the UI can prefill into the intake form."""
    return [
        {
            "label": "Senior ML Engineer (clean)",
            "payload": {
                "title": "Senior ML Engineer (LLM / RAG)",
                "description": (
                    "Senior ML engineer to build and deploy LLM-powered production "
                    "systems. Own the design and operation of a RAG pipeline serving "
                    "1000+ daily queries with LangChain on AWS. CI/CD and observability."
                ),
                "must_have_skills": ["Python", "LLMs", "RAG", "LangChain", "AWS"],
                "nice_to_have_skills": ["Kubernetes", "Azure", "Time series"],
                "min_years_experience": 5,
                "max_years_experience": 12,
                "location": "Hyderabad, India",
                "remote_ok": True,
                "employment_type": "full_time",
                "target_hiring_date": "2026-08-01",
            },
        },
        {
            "label": "Discriminatory JD (will be rejected)",
            "payload": {
                "title": "Young Energetic Engineer",
                "description": (
                    "Looking for a young, energetic male engineer aged 22-28. "
                    "Hindu candidates preferred. Native English speaker required."
                ),
                "must_have_skills": ["Python"],
                "nice_to_have_skills": [],
                "min_years_experience": 1,
                "max_years_experience": 3,
                "location": "Bangalore, India",
                "remote_ok": False,
                "employment_type": "full_time",
                "target_hiring_date": "2026-08-01",
            },
        },
        {
            "label": "Niche-skill JD (showcases hybrid retrieval)",
            "payload": {
                "title": "ML Engineer — Time Series Forecasting",
                "description": (
                    "ML engineer with hands-on time-series experience: ARIMA, "
                    "Prophet, anomaly detection. Python + production ML."
                ),
                "must_have_skills": ["Python", "Time series", "Machine Learning"],
                "nice_to_have_skills": ["Prophet", "LSTM"],
                "min_years_experience": 3,
                "max_years_experience": 8,
                "location": "Remote",
                "remote_ok": True,
                "employment_type": "full_time",
                "target_hiring_date": "2026-09-01",
            },
        },
    ]
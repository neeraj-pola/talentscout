# app/storage/jd_repo.py
"""Thin repository for JDs. Converts between Pydantic <-> SQLAlchemy rows."""
import json
from datetime import datetime
from uuid import UUID

from app.models import JD, JDStatus, ParsedJD, ScoredCandidate, TopPickRecommendation, OutreachDraft
from app.storage.db import JDRow, get_session


def _jd_to_row(jd: JD) -> JDRow:
    return JDRow(
        id=str(jd.id),
        title=jd.title,
        description=jd.description,
        must_have_skills=json.dumps(jd.must_have_skills),
        nice_to_have_skills=json.dumps(jd.nice_to_have_skills),
        min_years_experience=jd.min_years_experience,
        max_years_experience=jd.max_years_experience,
        location=jd.location,
        remote_ok=jd.remote_ok,
        employment_type=jd.employment_type,
        target_hiring_date=jd.target_hiring_date.isoformat(),
        status=jd.status.value,
        created_at=jd.created_at,
        closed_at=jd.closed_at,
        closed_by=jd.closed_by,
        closed_with_candidate_id=str(jd.closed_with_candidate_id) if jd.closed_with_candidate_id else None,
    )


def _row_to_jd(row: JDRow) -> JD:
    from datetime import date as _date
    return JD(
        id=UUID(row.id),
        title=row.title,
        description=row.description,
        must_have_skills=json.loads(row.must_have_skills),
        nice_to_have_skills=json.loads(row.nice_to_have_skills or "[]"),
        min_years_experience=row.min_years_experience,
        max_years_experience=row.max_years_experience,
        location=row.location,
        remote_ok=bool(row.remote_ok),
        employment_type=row.employment_type,
        target_hiring_date=_date.fromisoformat(row.target_hiring_date),
        status=JDStatus(row.status),
        created_at=row.created_at,
        closed_at=row.closed_at,
        closed_by=row.closed_by,
        closed_with_candidate_id=UUID(row.closed_with_candidate_id) if row.closed_with_candidate_id else None,
    )


def create_jd(jd: JD) -> JD:
    with get_session() as s:
        s.add(_jd_to_row(jd))
    return jd


def get_jd(jd_id: UUID | str) -> JD | None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        return _row_to_jd(row) if row else None


def list_jds() -> list[JD]:
    with get_session() as s:
        rows = s.query(JDRow).order_by(JDRow.created_at.desc()).all()
        return [_row_to_jd(r) for r in rows]


def update_jd_status(jd_id: UUID | str, status: JDStatus) -> None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row:
            row.status = status.value


def save_parsed_jd(jd_id: UUID | str, parsed: ParsedJD) -> None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row:
            row.parsed_jd_json = parsed.model_dump_json()


def save_shortlist(jd_id: UUID | str, shortlist: list[ScoredCandidate]) -> None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row:
            row.shortlist_json = json.dumps([c.model_dump(mode="json") for c in shortlist])


def save_top_pick(jd_id: UUID | str, top_pick: TopPickRecommendation) -> None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row:
            row.top_pick_json = top_pick.model_dump_json()


def save_outreach(jd_id: UUID | str, drafts: list[OutreachDraft]) -> None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row:
            row.outreach_json = json.dumps([d.model_dump(mode="json") for d in drafts])


def close_jd(jd_id: UUID | str, candidate_id: UUID, closed_by: str) -> None:
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row:
            row.status = JDStatus.CLOSED.value
            row.closed_at = datetime.utcnow()
            row.closed_by = closed_by
            row.closed_with_candidate_id = str(candidate_id)

# ============================================================
# Sourcing, profiles, guardrails, refinement state
# ============================================================
# Added to support refinement (needs profiles + refinement_state) and the
# bias firewall (profile_summary writes back to profiles_json).

def save_sourcing(
    jd_id: UUID | str,
    sourcing_summary: dict,
    merge_audit: list[dict],
) -> None:
    """Persist the sourcing agent's summary + the dedup merge audit trail."""
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None:
            return
        row.sourcing_json = json.dumps(sourcing_summary)
        row.merge_audit_json = json.dumps(merge_audit)
        s.commit()


def save_profiles(jd_id: UUID | str, profiles: list[dict]) -> None:
    """Persist the deduped candidate profiles (with bias-blind summaries)."""
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None:
            return
        row.profiles_json = json.dumps(profiles)
        s.commit()


def load_profiles(jd_id: UUID | str) -> list[dict]:
    """Read the deduped profiles back. Returns [] if never saved."""
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None or not row.profiles_json:
            return []
        try:
            return json.loads(row.profiles_json)
        except json.JSONDecodeError:
            return []


def save_guardrail_verdict(jd_id: UUID | str, verdict) -> None:
    """Persist the guardrail agent's verdict (reasons + flagged phrases)."""
    # verdict can be a pydantic model or a dict — handle both
    if hasattr(verdict, "model_dump"):
        data = verdict.model_dump(mode="json")
    elif hasattr(verdict, "dict"):
        data = verdict.dict()
    else:
        data = dict(verdict)
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None:
            return
        row.guardrail_verdict_json = json.dumps(data)
        s.commit()


def load_refinement_state(jd_id: UUID | str) -> dict:
    """Read refinement state (conversation history + filter stack + total cost).

    Returns a fresh empty state if the JD has never been refined.
    """
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None or not row.refinement_state_json:
            return {
                "conversation_history": [],
                "filter_stack": [],
                "total_refinement_cost_usd": 0.0,
            }
        try:
            state = json.loads(row.refinement_state_json)
        except json.JSONDecodeError:
            return {
                "conversation_history": [],
                "filter_stack": [],
                "total_refinement_cost_usd": 0.0,
            }
        # Defensive defaults for older state shapes
        state.setdefault("conversation_history", [])
        state.setdefault("filter_stack", [])
        state.setdefault("total_refinement_cost_usd", 0.0)
        return state


def save_refinement_state(
    jd_id: UUID | str,
    conversation_history: list[dict],
    filter_stack: list[dict],
    total_refinement_cost_usd: float,
) -> None:
    """Persist one refinement turn's updated state."""
    state = {
        "conversation_history": conversation_history,
        "filter_stack": filter_stack,
        "total_refinement_cost_usd": float(total_refinement_cost_usd),
    }
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None:
            return
        row.refinement_state_json = json.dumps(state)
        s.commit()
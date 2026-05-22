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
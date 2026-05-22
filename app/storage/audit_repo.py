# app/storage/audit_repo.py
import json
from uuid import UUID

from app.models import AuditRecord
from app.storage.db import AuditRow, get_session


def create_audit(record: AuditRecord) -> None:
    with get_session() as s:
        s.add(AuditRow(
            jd_id=str(record.jd_id),
            candidate_id=str(record.candidate_id),
            closed_by=record.closed_by,
            justification=record.justification,
            final_ranking_snapshot=json.dumps([str(x) for x in record.final_ranking_snapshot]),
            total_cost_usd=record.total_cost_usd,
            total_tokens=record.total_tokens,
            total_llm_calls=record.total_llm_calls,
        ))


def list_audits() -> list[dict]:
    with get_session() as s:
        rows = s.query(AuditRow).order_by(AuditRow.closed_at.desc()).all()
        return [
            {
                "jd_id": r.jd_id,
                "candidate_id": r.candidate_id,
                "closed_by": r.closed_by,
                "closed_at": r.closed_at.isoformat() if r.closed_at else None,
                "justification": r.justification,
                "final_ranking_snapshot": json.loads(r.final_ranking_snapshot),
                "total_cost_usd": r.total_cost_usd,
                "total_tokens": r.total_tokens,
                "total_llm_calls": r.total_llm_calls,
            }
            for r in rows
        ]
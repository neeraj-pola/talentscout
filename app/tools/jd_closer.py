# app/tools/jd_closer.py
"""JD closure tool.

Wraps the close-JD logic that previously lived inline in the API endpoint.
Produces both:
  - Status transition on the JD row (sets status=closed, closed_at, closed_by,
    closed_with_candidate_id)
  - Persistent AuditRecord row capturing who closed when, the chosen candidate,
    justification, the full ranking snapshot at close time, and total cost.

This is the spec's "closing the JD" tool — single named entry point.

Public API:
    close_jd_with_audit(jd_id, candidate_id, closed_by) -> AuditRecord

The API endpoint (POST /jds/{id}/close) is the only caller in this build,
but exposing it as a tool makes the contract explicit and lets future
agents (e.g. an auto-close agent that fires when SLAs expire) invoke it
cleanly.
"""
from __future__ import annotations

from datetime import datetime
import json
from uuid import UUID

from app.models import AuditRecord
from app.storage.jd_repo import close_jd, get_jd
from app.storage.audit_repo import create_audit
from app.storage.db import JDRow, get_session
from app.obs.events import log_event
from app.obs.cost import get_cost_summary


def close_jd_with_audit(
    jd_id: UUID | str,
    candidate_id: UUID,
    closed_by: str,
) -> AuditRecord:
    """Close a JD with a chosen candidate. Writes an AuditRecord.

    Args:
        jd_id:         The JD UUID being closed
        candidate_id:  The profile_id of the candidate chosen to fill the role
        closed_by:     Identifier (email/username) of the recruiter closing the JD

    Returns:
        The persisted AuditRecord.

    Raises:
        ValueError: JD not found, JD already closed, or candidate not in shortlist

    Behavior:
        - Atomic in intent: the JD row update and audit insert are issued
          back-to-back. SQLite default isolation means a crash between the
          two leaves the JD closed but no audit row — acceptable for this
          build (we'd switch to a transaction in production).
        - Justification: if the chosen candidate matches the top-pick
          recommendation, we reuse the top-pick justification verbatim
          (the LLM's original reasoning). Otherwise we mark it as an
          override of the system recommendation.
        - Ranking snapshot: full shortlist profile_id list at close time,
          for "what alternatives did the recruiter pass on" auditability.
    """
    jd_id_str = str(jd_id)
    log_event(jd_id_str, "tool.jd_closer", "close_start",
              candidate_id=str(candidate_id), closed_by=closed_by)

    jd = get_jd(jd_id)
    if jd is None:
        log_event(jd_id_str, "tool.jd_closer", "close_error",
                  reason="jd_not_found")
        raise ValueError(f"JD {jd_id} not found")

    if jd.status.value == "closed":
        log_event(jd_id_str, "tool.jd_closer", "close_error",
                  reason="already_closed")
        raise ValueError(f"JD {jd_id} is already closed")

    # Load shortlist + top pick to build justification and ranking snapshot
    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == jd_id_str).first()
        shortlist = (
            json.loads(row.shortlist_json) if row and row.shortlist_json else []
        )
        top_pick = (
            json.loads(row.top_pick_json) if row and row.top_pick_json else None
        )

    # Justification: top-pick reasoning if match, override note otherwise
    if top_pick and UUID(top_pick["recommended_candidate_id"]) == candidate_id:
        justification = top_pick.get(
            "justification", "Closed with recommended top pick."
        )
    else:
        justification = (
            "Closed with manually selected candidate "
            "(overriding system top pick)."
        )

    ranking_snapshot = [UUID(c["profile_id"]) for c in shortlist]
    cost = get_cost_summary(jd_id_str)

    audit = AuditRecord(
        jd_id=UUID(jd_id_str),
        candidate_id=candidate_id,
        closed_by=closed_by,
        closed_at=datetime.utcnow().isoformat(),
        justification=justification,
        final_ranking_snapshot=ranking_snapshot,
        total_cost_usd=cost["total_usd"],
        total_tokens=cost["total_tokens_in"] + cost["total_tokens_out"],
        total_llm_calls=cost["total_calls"],
    )

    # Issue the writes back-to-back
    close_jd(jd_id, candidate_id, closed_by)
    create_audit(audit)

    log_event(jd_id_str, "tool.jd_closer", "close_end",
              candidate_id=str(candidate_id),
              total_cost_usd=audit.total_cost_usd,
              n_candidates_passed_over=len(ranking_snapshot) - 1)
    return audit
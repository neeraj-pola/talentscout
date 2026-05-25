# app/obs/timeline.py
"""Compute per-node timing + cost for a JD's pipeline run.

Reads the events log + cost rows and produces a list of NodeStat rows
that the UI flowchart consumes.

Why a separate module: keeps the UI dumb. The UI just renders what this
returns; all event-parsing logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal

from app.obs.events import get_events_for_jd
from app.storage.db import get_session, CostRow


# The eight orchestrator nodes, in pipeline order. profile_summary runs
# after sourcing/dedup and before screening — it generates a bias-blind
# 2-3 sentence summary per candidate that downstream agents can reference
# without re-reading raw_text and without risk of name/location leakage.
NODE_ORDER = [
    "guardrails",
    "jd_intake",
    "sourcing",
    "profile_summary",
    "screening",
    "ranking",
    "top_pick",
    "outreach",
]


@dataclass
class NodeStat:
    name: str
    status: Literal["completed", "skipped", "rejected", "pending", "running"]
    duration_s: float | None       # None if didn't run
    cost_usd: float
    n_llm_calls: int
    started_at: str | None         # ISO timestamp
    ended_at: str | None


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None


def compute_node_stats(jd_id: str) -> list[NodeStat]:
    """Return a NodeStat per orchestrator node, in pipeline order.

    A node's status is:
      - completed: graph emitted node_start AND node_end
      - rejected: only for guardrails — JD was flagged discriminatory
      - skipped:  pipeline halted before reaching this node
      - pending:  pipeline never ran (no events for this JD)
    """
    events = get_events_for_jd(jd_id)
    if not events:
        return [
            NodeStat(name=n, status="pending", duration_s=None, cost_usd=0.0,
                     n_llm_calls=0, started_at=None, ended_at=None)
            for n in NODE_ORDER
        ]

    # Build per-node start/end timestamps from graph events
    starts: dict[str, str] = {}
    ends: dict[str, str] = {}
    rejected_at_guardrails = False

    for e in events:
        if e.get("agent") != "graph":
            continue
        node = e.get("node")
        if not node:
            continue
        if e.get("event") == "node_start":
            starts.setdefault(node, e.get("ts", ""))
        elif e.get("event") == "node_end":
            ends[node] = e.get("ts", "")
            if node == "guardrails" and e.get("is_discriminatory"):
                rejected_at_guardrails = True

    # Per-node cost + call count from the llm_costs table
    cost_by_agent: dict[str, tuple[float, int]] = {}
    with get_session() as s:
        rows = (
            s.query(CostRow)
            .filter(CostRow.jd_id == jd_id)
            .all()
        )
        for r in rows:
            usd, n = cost_by_agent.get(r.agent, (0.0, 0))
            cost_by_agent[r.agent] = (usd + r.usd, n + 1)

    # Map the orchestrator node name to the agent name used in cost logs.
    # Most line up directly; a few sub-agents within a node also bill, so
    # we aggregate everything that belongs under each node.
    node_to_agent_aliases = {
        "guardrails":       ["guardrails"],
        "jd_intake":        ["jd_intake"],
        "sourcing":         ["rag.indexer", "rag.retriever"],
        "profile_summary":  ["profile_summary"],
        "screening":        ["screening", "rag.retriever"],
        "ranking":          ["ranking"],
        "top_pick":         ["top_pick"],
        "outreach":         ["outreach"],
    }
    # `rag.retriever` is used by both sourcing (during indexing) and screening
    # (per-criterion). For visual clarity we attribute it to screening only.
    node_to_agent_aliases["sourcing"] = ["rag.indexer"]

    stats: list[NodeStat] = []
    for node in NODE_ORDER:
        start = starts.get(node)
        end = ends.get(node)
        duration: float | None = None
        if start and end:
            t0 = _parse_ts(start)
            t1 = _parse_ts(end)
            if t0 and t1:
                duration = max(0.0, (t1 - t0).total_seconds())

        # Aggregate cost + call count
        usd_total = 0.0
        n_calls_total = 0
        for ag in node_to_agent_aliases.get(node, []):
            u, n = cost_by_agent.get(ag, (0.0, 0))
            usd_total += u
            n_calls_total += n

        # Status logic
        if node == "guardrails":
            if start and end:
                status = "rejected" if rejected_at_guardrails else "completed"
            else:
                status = "pending"
        else:
            if rejected_at_guardrails:
                status = "skipped"
            elif start and end:
                status = "completed"
            elif start and not end:
                status = "running"
            else:
                status = "pending"

        stats.append(NodeStat(
            name=node,
            status=status,
            duration_s=duration,
            cost_usd=round(usd_total, 6),
            n_llm_calls=n_calls_total,
            started_at=start,
            ended_at=end,
        ))

    return stats


def stats_to_dicts(stats: list[NodeStat]) -> list[dict]:
    """JSON-friendly for the API if we want to expose this endpoint later."""
    return [asdict(s) for s in stats]
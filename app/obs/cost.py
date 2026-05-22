# app/obs/cost.py
"""Per-LLM-call cost tracking. Writes to SQLite, exposes per-JD totals."""
from app.storage.db import CostRow, get_session

# OpenAI pricing as of 2025-10 (USD per 1M tokens). Update if needed.
PRICING = {
    "gpt-4o":           {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":      {"in": 0.15,  "out": 0.60},
    "gpt-4o-2024-08-06":{"in": 2.50,  "out": 10.00},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
}


def calc_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate USD cost for one LLM call."""
    p = PRICING.get(model)
    if not p:
        # Unknown model — fall back to gpt-4o-mini pricing rather than 0
        p = PRICING["gpt-4o-mini"]
    return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]


def record_cost(
    jd_id: str,
    agent: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
) -> float:
    """Record one LLM call's cost. Returns the USD cost."""
    usd = calc_cost_usd(model, tokens_in, tokens_out)
    with get_session() as s:
        s.add(CostRow(
            jd_id=jd_id, agent=agent, model=model,
            tokens_in=tokens_in, tokens_out=tokens_out,
            usd=usd, latency_ms=latency_ms,
        ))
    return usd


def get_cost_summary(jd_id: str) -> dict:
    """Return aggregate cost for a JD."""
    with get_session() as s:
        rows = s.query(CostRow).filter(CostRow.jd_id == jd_id).all()
        return {
            "total_usd": sum(r.usd for r in rows),
            "total_tokens_in": sum(r.tokens_in for r in rows),
            "total_tokens_out": sum(r.tokens_out for r in rows),
            "total_calls": len(rows),
            "by_agent": _group_by(rows, lambda r: r.agent),
            "by_model": _group_by(rows, lambda r: r.model),
        }


def _group_by(rows, keyfn):
    out = {}
    for r in rows:
        k = keyfn(r)
        if k not in out:
            out[k] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0}
        out[k]["calls"] += 1
        out[k]["tokens_in"] += r.tokens_in
        out[k]["tokens_out"] += r.tokens_out
        out[k]["usd"] += r.usd
    return out
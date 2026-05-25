# app/orchestrator/graph.py
"""LangGraph assembly + public entrypoint.

The graph is:

    START
      │
      ▼
  guardrails ──(rejected)──► END
      │
      ▼
  jd_intake
      │
      ▼
  sourcing
      │
      ▼
  screening
      │
      ▼
  ranking
      │
      ▼
  top_pick
      │
      ▼
  outreach ──► END

Linear with one conditional branch after guardrails. We chose linear over
parallel-everywhere because each agent depends on the previous one's output —
and the spec's parallelism requirement (Tech Req 4) is satisfied by the
SOURCING agent's internal ThreadPoolExecutor and the SCREENING agent's
asyncio fan-out.

State is checkpointed to SQLite via SqliteSaver. This means:
  - A crash doesn't lose work (resume from last checkpoint)
  - Conversational refinement is possible (re-enter at jd_intake with a
    modified JD; downstream nodes re-run)
  - The Activity Log in the UI can replay events from the checkpoint
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END
import warnings

# LangGraph emits a future-deprecation warning we don't need to act on yet.
warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
    category=Warning,
)

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings
from app.models import JD
from app.obs.events import log_event
from app.orchestrator.state import TalentScoutState, empty_state
from app.orchestrator.nodes import (
    node_guardrails, route_after_guardrails,
    node_jd_intake, node_sourcing,
    node_profile_summary,
    node_screening,
    node_ranking, node_top_pick, node_outreach,
    cleanup_caches,
)


# ============================================================
# Graph construction (module-level — compiled once, reused)
# ============================================================

def _build_graph():
    """Construct the StateGraph. Pure construction — no I/O.

    Node names are prefixed with `n_` to avoid collisions with state field
    names (LangGraph rejects a node whose name matches a state key, since
    that would create ambiguous edges).
    """
    g = StateGraph(TalentScoutState)

    g.add_node("n_guardrails", node_guardrails)
    g.add_node("n_jd_intake", node_jd_intake)
    g.add_node("n_sourcing", node_sourcing)
    g.add_node("n_profile_summary", node_profile_summary)
    g.add_node("n_screening", node_screening)
    g.add_node("n_ranking", node_ranking)
    g.add_node("n_top_pick", node_top_pick)
    g.add_node("n_outreach", node_outreach)

    g.set_entry_point("n_guardrails")

    # Conditional after guardrails — halt if discriminatory
    g.add_conditional_edges(
        "n_guardrails",
        route_after_guardrails,
        {
            "continue": "n_jd_intake",
            "halt": END,
        },
    )

    # Linear from here
    g.add_edge("n_jd_intake", "n_sourcing")
    g.add_edge("n_sourcing", "n_profile_summary")
    g.add_edge("n_profile_summary", "n_screening")
    g.add_edge("n_screening", "n_ranking")
    g.add_edge("n_ranking", "n_top_pick")
    g.add_edge("n_top_pick", "n_outreach")
    g.add_edge("n_outreach", END)

    return g

# Build once at import time
_GRAPH_DEF = _build_graph()


def _compile_graph(checkpoint_db_path: str = "graph_state.db"):
    """Compile the graph with a SQLite checkpointer.

    Uses a context-managed connection (LangGraph >=0.2 expects this pattern).
    Each call returns a fresh compiled graph bound to a new SqliteSaver —
    fine for our usage (one pipeline run at a time).
    """
    Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    saver_ctx = SqliteSaver.from_conn_string(checkpoint_db_path)
    saver = saver_ctx.__enter__()
    compiled = _GRAPH_DEF.compile(checkpointer=saver)
    # Stash the context so the caller can close it cleanly
    compiled._saver_ctx = saver_ctx  # type: ignore[attr-defined]
    return compiled


# ============================================================
# Public API
# ============================================================

def run_pipeline(jd: JD, checkpoint_db_path: str = "graph_state.db") -> dict:
    """Run the full pipeline for a JD. Returns the final state.

    Synchronous. Internal agents that use asyncio (screening) handle their
    own event loop.
    """
    jd_id = str(jd.id)
    log_event(jd_id, "graph", "pipeline_start", jd_title=jd.title)

    initial = empty_state(jd_id=jd_id, jd_dict=jd.model_dump(mode="json"))
    config = {"configurable": {"thread_id": jd_id}}

    compiled = _compile_graph(checkpoint_db_path)

    try:
        final_state = compiled.invoke(initial, config=config)
        log_event(jd_id, "graph", "pipeline_end",
                  status=final_state.get("status"),
                  halt_reason=final_state.get("halt_reason"))
        return final_state
    except Exception as e:
        log_event(jd_id, "graph", "pipeline_error", error=str(e))
        raise
    finally:
        cleanup_caches(jd_id)
        # Close the sqlite checkpointer connection cleanly
        try:
            compiled._saver_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
        except Exception:
            pass


def resume_pipeline(jd_id: str, checkpoint_db_path: str = "graph_state.db") -> dict:
    """Resume a previously-checkpointed pipeline run for the same jd_id."""
    config = {"configurable": {"thread_id": jd_id}}
    compiled = _compile_graph(checkpoint_db_path)
    try:
        log_event(jd_id, "graph", "pipeline_resume")
        final_state = compiled.invoke(None, config=config)
        return final_state
    finally:
        cleanup_caches(jd_id)
        try:
            compiled._saver_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
        except Exception:
            pass


def get_checkpoint(jd_id: str, checkpoint_db_path: str = "graph_state.db") -> dict | None:
    """Read the latest checkpoint state for a JD (for the UI / activity log)."""
    config = {"configurable": {"thread_id": jd_id}}
    compiled = _compile_graph(checkpoint_db_path)
    try:
        snapshot = compiled.get_state(config)
        return snapshot.values if snapshot else None
    finally:
        try:
            compiled._saver_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
        except Exception:
            pass
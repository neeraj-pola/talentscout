# app/obs/events.py
"""Structured event logging. Every agent/tool/LLM call emits an event.

Events go to two places:
1. events.jsonl — append-only file (durable, replayable)
2. In-memory deque per JD — for live UI display
"""
import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

EVENTS_FILE = Path("events.jsonl")
_lock = threading.Lock()
_memory_log: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

logger = logging.getLogger("talentscout")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def log_event(
    jd_id: str | None,
    agent: str,
    event: str,
    **fields: Any,
) -> dict:
    """Emit a structured event.

    Args:
        jd_id: JD this event belongs to (None for system events)
        agent: agent or tool name (e.g. "screening", "tool.search_linkedin")
        event: event type (e.g. "node_start", "node_end", "llm_call", "tool_call", "error")
        **fields: any additional structured fields
    """
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "jd_id": jd_id,
        "agent": agent,
        "event": event,
        **fields,
    }
    with _lock:
        with EVENTS_FILE.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        if jd_id:
            _memory_log[jd_id].append(record)
    logger.info(f"[{agent}] {event} jd={jd_id} {fields}")
    return record


def get_events_for_jd(jd_id: str) -> list[dict]:
    """Return all in-memory events for a JD (for live UI display)."""
    return list(_memory_log.get(jd_id, []))


def clear_events_for_jd(jd_id: str) -> None:
    _memory_log.pop(jd_id, None)
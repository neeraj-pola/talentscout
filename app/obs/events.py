# app/obs/events.py
"""Structured event logging. Every agent/tool/LLM call emits an event.

Events go to two places:
1. events.jsonl — append-only file (durable, replayable)
2. In-memory deque per JD — for fast live UI display during one process lifetime
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
    """Return all events for a JD.

    Reads from the in-memory ring buffer first (fast, live during a single
    process's lifetime). If that's empty for this JD — typically because
    the caller is a different process than the one that recorded the
    events (e.g. UI process vs API process, or after an API restart) —
    falls back to scanning events.jsonl on disk.

    Why this matters: compute_node_stats() in app/obs/timeline.py reads
    these events to compute per-node status. If the caller's in-memory
    log is empty for the JD, the function would return all "pending"
    statuses even though the pipeline completed successfully and the
    events are on disk.
    """
    in_memory = list(_memory_log.get(jd_id, []))
    if in_memory:
        return in_memory

    # Disk fallback. O(N) over events.jsonl; fine for take-home scale.
    # If the file ever gets large enough to make this slow, replace with
    # a JD-indexed cache (re-built per-process on first access).
    if not EVENTS_FILE.exists():
        return []
    matches = []
    try:
        with EVENTS_FILE.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("jd_id") == jd_id:
                    matches.append(record)
    except OSError:
        return []
    return matches


def clear_events_for_jd(jd_id: str) -> None:
    _memory_log.pop(jd_id, None)
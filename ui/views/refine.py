# ui/views/refine.py
"""Refine tab — natural-language chat against a JD's shortlist.

Layout:
    Top bar:       Active filter pills · refined count · total refinement cost
    Left column:
        - Empty state suggestions (only when there's no history)
        - Fixed-height SCROLLABLE chat container (~480px) with soft beige bg
        - Chat input below the scroll area
    Right column:  Live filtered-shortlist preview

State management:
    All conversation state is persisted server-side in the JD row's
    refinement_state_json column. Each user message hits POST /jds/{id}/refine
    which loads, processes, persists, and returns. The UI just refetches
    GET /jds/{id} after each turn to render the current state.

Rendering strategy:
    The chat is rendered as our own <div class="refine-chat-row {role}">
    markup inside an st.container(height=N), instead of using
    st.chat_message(). This gives us full control over alignment and styling
    without fighting Streamlit's internal DOM.

CSS scoping:
    A marker div `.refine-chat-wrapper` is emitted directly BEFORE the
    st.container. The container becomes its adjacent sibling. CSS rules use
    `.refine-chat-wrapper + div ...` to target ONLY that one container —
    never ancestor wrappers Streamlit emits around the page.
"""
from __future__ import annotations

import html as html_lib

import streamlit as st

from ui import api_client
from ui.styles import eyebrow, score_bar, cost_chip


# ============================================================
# Constants
# ============================================================

SUGGESTIONS = [
    "Show only candidates with 5+ years",
    "Why did candidate #1 rank first?",
    "Compare #1 and #2",
    "What skills does candidate #1 have?",
    "Show me more candidates like #1",
    "Remove the red-flagged ones",
    "Rewrite the outreach for #1 more casually",
    "Show all candidates again",
]

CHAT_SCROLL_HEIGHT_PX = 480  # ~7-8 short turns visible at once

# CSS — fully under our control. No reliance on Streamlit's chat_message DOM.
CHAT_CSS = """
<style>
/* The 480px scroll container — soft beige background, only this one.
   Targeted via adjacent-sibling so it never cascades upward.
   `data-test-scroll-behavior="normal"` is what Streamlit puts on bounded
   containers created with st.container(height=N), so it uniquely identifies
   the scroll surface. */
.refine-chat-wrapper + div div[data-testid="stVerticalBlockBorderWrapper"][data-test-scroll-behavior="normal"] {
  background: #fbf6e6 !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: var(--radius-md) !important;
}

/* The chat surface. Lives inside the st.container(height=N) above; just
   a vertical flex stack of rows. */
.refine-chat-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

/* Each turn is its own row so we can left/right-align the bubble within it */
.refine-chat-row {
  display: flex;
  width: 100%;
}
.refine-chat-row.user      { justify-content: flex-end; }
.refine-chat-row.assistant { justify-content: flex-start; }

/* Bubble — asymmetric border-radius gives the speech-bubble feel */
.refine-chat-bubble {
  padding: 0.6rem 0.9rem;
  font-size: 0.92rem;
  line-height: 1.5;
  word-wrap: break-word;
  max-width: 78%;
}
.refine-chat-bubble.user {
  background: var(--accent-yellow-soft);
  border: 1px solid var(--accent-yellow);
  border-radius: 18px 18px 4px 18px;
  color: var(--text-primary);
}
.refine-chat-bubble.assistant {
  background: var(--surface-white);
  border: 1px solid var(--border-soft);
  border-radius: 18px 18px 18px 4px;
  color: var(--text-primary);
  max-width: 88%;
}

/* Tiny metadata strip beneath assistant bubbles (intent + cost pills) */
.refine-chat-meta {
  display: flex;
  gap: 0.4rem;
  margin-top: 0.4rem;
  flex-wrap: wrap;
}
.refine-chat-meta .pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--text-muted);
  background: #f5f3ec;
  padding: 2px 8px;
  border-radius: 999px;
}
</style>
<script>
// Auto-scroll the chat list to the bottom on each render
(function() {
  setTimeout(function() {
    var end = window.parent.document.querySelector('[data-chat-end="1"]');
    if (end) end.scrollIntoView({behavior: 'instant', block: 'end'});
  }, 50);
})();
</script>
"""


# ============================================================
# Public entry point — called from detail.py via the new Refine tab
# ============================================================

def render(jd_id: str, detail: dict):
    """Render the Refine tab for a JD."""
    status = detail.get("status", "draft")
    if status in ("rejected_guardrail", "draft"):
        st.info(
            "Refinement is only available after the pipeline has produced a shortlist. "
            f"This JD's status is `{status}`."
        )
        return

    shortlist = detail.get("shortlist") or []
    if not shortlist:
        st.info("No shortlist available to refine.")
        return

    # Inject chat styling once per render
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    refinement_state = detail.get("refinement_state") or {}
    history = refinement_state.get("conversation_history") or []
    filter_stack = refinement_state.get("filter_stack") or []
    total_cost = refinement_state.get("total_refinement_cost_usd", 0.0) or 0.0
    # The API computes refined_shortlist by applying the persisted filter_stack
    # to the full shortlist. We display this filtered view so what the user
    # sees matches what the backend's filter state actually contains. Falls
    # back to the full shortlist when no filters are active.
    refined_shortlist = refinement_state.get("refined_shortlist") or shortlist

    _render_topbar(filter_stack, history, total_cost)

    left, right = st.columns([1.35, 1.0], gap="medium")

    with left:
        _render_chat_panel(jd_id, history)

    with right:
        _render_filtered_shortlist(refined_shortlist, filter_stack, n_total=len(shortlist))


# ============================================================
# Top bar
# ============================================================

def _render_topbar(filter_stack: list[dict], history: list[dict], total_cost: float):
    """Active-filter pills + total cost chip + turn counter."""
    st.markdown(eyebrow("REFINE · NATURAL-LANGUAGE REFINEMENT"), unsafe_allow_html=True)

    if filter_stack:
        pills_html = '<div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.6rem;">'
        for f in filter_stack:
            label = _format_filter(f)
            pills_html += (
                '<span style="background:var(--accent-yellow-soft); color:var(--text-primary); '
                'padding:4px 12px; border-radius:999px; border:1px solid var(--accent-yellow); '
                'font-family:JetBrains Mono, monospace; font-size:0.75rem; font-weight:500;">'
                f'{label}'
                '</span>'
            )
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#9a978f; font-family:JetBrains Mono, monospace; '
            'font-size:0.78rem; margin-bottom:0.6rem;">No active filters · '
            'showing full shortlist</div>',
            unsafe_allow_html=True,
        )

    n_turns = len([h for h in history if h.get("role") == "user"])
    cost_row = (
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'margin-bottom:1rem;">'
        f'<div>'
        f'<span class="cost-chip">'
        f'<strong>${total_cost:.4f}</strong> · {n_turns} turn{"s" if n_turns != 1 else ""}'
        f'</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(cost_row, unsafe_allow_html=True)


def _format_filter(f: dict) -> str:
    """Human-readable label for one filter in the stack."""
    ftype = f.get("type", "?")
    val = f.get("value")
    if ftype == "location":
        return f"location: {val}"
    if ftype == "yoe_min":
        return f"YOE ≥ {val}"
    if ftype == "yoe_max":
        return f"YOE ≤ {val}"
    if ftype == "skill":
        return f"skill: {val}"
    if ftype == "exclude_flagged":
        return "no red flags"
    return f"{ftype}: {val}"


# ============================================================
# Chat panel — renders entirely as our own HTML divs
# ============================================================

def _render_chat_panel(jd_id: str, history: list[dict]):
    """Render the chat panel: empty-state suggestions OR scrollable history.

    Each turn is a plain <div class="refine-chat-row {role}"> with a child
    bubble. The wrapping st.container(height=...) gives us the fixed-height
    scroll area; the .refine-chat-wrapper marker right before it lets our
    CSS target ONLY that container for the beige background.
    """
    if not history:
        # Empty state — suggestion grid above the input
        st.markdown(
            '<div style="background:var(--bg-beige-soft); padding:1.2rem 1.4rem; '
            'border-radius:14px; margin-bottom:1rem;">'
            '<div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">'
            'Ask anything about this shortlist'
            '</div>'
            '<div style="color:#6b6963; font-size:0.85rem; margin-bottom:0.8rem;">'
            'Filter, explain, compare, find similar candidates, or rewrite outreach — '
            'all in natural language. State persists across page reloads.'
            '</div>'
            '<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
            'letter-spacing:0.05em; color:#9a978f; text-transform:uppercase;">'
            'TRY ONE OF THESE'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    _send_message(jd_id, suggestion)
    else:
        # Marker div for CSS scoping — the next sibling is the scroll container
        st.markdown('<div class="refine-chat-wrapper"></div>', unsafe_allow_html=True)
        # Build the entire chat as one big HTML string, then render inside
        # an st.container with fixed height for scrolling.
        chat_box = st.container(height=CHAT_SCROLL_HEIGHT_PX, border=True)
        with chat_box:
            chat_html = ['<div class="refine-chat-list">']

            for turn in history:
                role = turn.get("role", "?")
                content = turn.get("content", "")
                # Escape HTML in content to prevent injection
                safe = html_lib.escape(content).replace("\n", "<br/>")

                if role == "user":
                    chat_html.append(
                        f'<div class="refine-chat-row user">'
                        f'<div class="refine-chat-bubble user">{safe}</div>'
                        f'</div>'
                    )
                elif role == "assistant":
                    # Optional metadata strip (intent + cost pills)
                    meta = ""
                    intent = turn.get("intent")
                    cost = turn.get("cost_usd", 0.0)
                    if intent and intent not in ("empty", "error"):
                        meta = (
                            f'<div class="refine-chat-meta">'
                            f'<span class="pill">intent: {html_lib.escape(str(intent))}</span>'
                            f'<span class="pill">${cost:.4f}</span>'
                            f'</div>'
                        )

                    chat_html.append(
                        f'<div class="refine-chat-row assistant">'
                        f'<div class="refine-chat-bubble assistant">'
                        f'{safe}'
                        f'{meta}'
                        f'</div>'
                        f'</div>'
                    )

            # End marker for auto-scroll
            chat_html.append('<div data-chat-end="1"></div>')
            chat_html.append('</div>')

            st.markdown("".join(chat_html), unsafe_allow_html=True)

    # Chat input always at the bottom, outside the scroll container
    user_msg = st.chat_input("Ask anything about this shortlist…")
    if user_msg:
        _send_message(jd_id, user_msg)


def _send_message(jd_id: str, message: str):
    """Submit a message, refresh the page so the new history renders."""
    with st.spinner("Thinking…"):
        try:
            api_client.refine_jd(jd_id, message)
        except api_client.APIError as e:
            st.error(f"Refinement failed: {e}")
            return
    st.rerun()


# ============================================================
# Filtered shortlist (right pane)
# ============================================================

def _render_filtered_shortlist(
    shortlist: list[dict],
    filter_stack: list[dict],
    n_total: int | None = None,
):
    """Render a compact view of candidates.

    `shortlist` is already filtered by the API based on the persisted
    filter_stack. `n_total` is the unfiltered count (passed separately so
    we can show 'N of M shown' in the header).
    """
    n_shown = len(shortlist)
    n_total = n_total if n_total is not None else n_shown
    n_active = sum(1 for _ in filter_stack)

    if n_active > 0:
        st.markdown(
            f'<div style="display:flex; align-items:baseline; justify-content:space-between; '
            f'margin-bottom:0.7rem;">'
            f'<span style="font-family:JetBrains Mono, monospace; font-size:0.75rem; '
            f'letter-spacing:0.08em; color:#9a978f; text-transform:uppercase;">'
            f'CURRENT SHORTLIST'
            f'</span>'
            f'<span style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
            f'color:#9a978f;">'
            f'{n_shown} of {n_total} · {n_active} filter{"s" if n_active != 1 else ""} active'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-family:JetBrains Mono, monospace; font-size:0.75rem; '
            f'letter-spacing:0.08em; color:#9a978f; text-transform:uppercase; '
            f'margin-bottom:0.7rem;">'
            f'CURRENT SHORTLIST · {n_total} CANDIDATES'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not shortlist:
        st.info("No candidates match the current filters.")
        return

    for i, c in enumerate(shortlist[:10], 1):
        gap_html = ""
        if c.get("has_must_have_gap"):
            gap_html = (
                ' <span style="color:#dc2626; font-family:JetBrains Mono, monospace; '
                'font-size:0.7rem;">⚠</span>'
            )

        st.markdown(
            f'<div style="background:#ffffff; border:1px solid #ece8dc; border-radius:10px; '
            f'padding:0.6rem 0.85rem; margin-bottom:0.4rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div>'
            f'<span style="font-family:JetBrains Mono, monospace; color:#9a978f; '
            f'font-size:0.72rem;">#{i:02d}</span>'
            f' &nbsp; <strong style="font-size:0.95rem;">{c.get("candidate_name", "?")}</strong>'
            f'{gap_html}'
            f'</div>'
            f'<div>{score_bar(c.get("overall_score", 0.0))}</div>'
            f'</div>'
            f'<div style="font-family:JetBrains Mono, monospace; font-size:0.7rem; '
            f'color:#6b6963; margin-top:0.3rem;">'
            f'must {c.get("must_have_coverage", 0)*100:.0f}% · '
            f'nice {c.get("nice_to_have_coverage", 0)*100:.0f}%'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
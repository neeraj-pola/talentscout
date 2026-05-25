# ui/views/pipeline.py
"""Pipeline view — unified flowchart + cost breakdown + activity log."""
from __future__ import annotations

import math

import streamlit as st

from app.obs.timeline import compute_node_stats, NodeStat
from ui import api_client
from ui.styles import eyebrow


# ============================================================
# Layout constants
# ============================================================

NODE_W = 175
NODE_H = 105
GAP_X = 35
GAP_Y = 50

NODE_POSITIONS = {
    "guardrails":       (0, 0),
    "jd_intake":        (1, 0),
    "sourcing":         (2, 0),
    "profile_summary":  (3, 0),
    "screening":        (4, 0),
    "ranking":          (4, 1),
    "top_pick":         (3, 1),
    "outreach":         (2, 1),
}

NODE_DISPLAY = {
    "guardrails":       "GUARDRAILS",
    "jd_intake":        "JD INTAKE",
    "sourcing":         "SOURCING",
    "profile_summary":  "PROFILE SUMMARY",
    "screening":        "SCREENING",
    "ranking":          "RANKING",
    "top_pick":         "TOP PICK",
    "outreach":         "OUTREACH",
}

EDGES = [
    ("guardrails", "jd_intake"),
    ("jd_intake", "sourcing"),
    ("sourcing", "profile_summary"),
    ("profile_summary", "screening"),
    ("screening", "ranking"),
    ("ranking", "top_pick"),
    ("top_pick", "outreach"),
]

NODE_ORDER = [
    "guardrails", "jd_intake", "sourcing", "profile_summary",
    "screening", "ranking", "top_pick", "outreach",
]

COLORS = {
    "card_bg":       "#ffffff",
    "card_bg_done":  "#fff9e0",
    "card_bg_skip":  "#f5f3ec",
    "card_bg_fail":  "#fde2e2",
    "border_soft":   "#ece8dc",
    "border_done":   "#fbd35a",
    "border_fail":   "#dc2626",
    "border_skip":   "#d6d3c8",
    "text":          "#0a0a0a",
    "text_muted":    "#6b6963",
    "text_dim":      "#9a978f",
    "arrow":         "#c9c5b8",
    "arrow_done":    "#e0a800",
    "end_bg":        "#0a0a0a",
    "end_text":      "#fafaf7",
}


# ============================================================
# Per-node warm pastel palette — applied when node is completed.
# Skipped / rejected / pending fall back to the COLORS dict above.
# ============================================================

NODE_PALETTE = {
    "guardrails":       {"bg": "#fef3c7", "border": "#facc15"},   # butter yellow
    "jd_intake":        {"bg": "#fde4cf", "border": "#fb923c"},   # soft peach
    "sourcing":         {"bg": "#fcd6c8", "border": "#f87171"},   # blush
    "profile_summary":  {"bg": "#fef5d8", "border": "#e0a800"},   # honey (bias-blind)
    "screening":        {"bg": "#fde2e2", "border": "#dc2626"},   # warm red (most expensive!)
    "ranking":          {"bg": "#e6e2cc", "border": "#a3a073"},   # sage
    "top_pick":         {"bg": "#dde7d3", "border": "#84a866"},   # mint
    "outreach":         {"bg": "#e2dcef", "border": "#a78bfa"},   # lavender
}


# ============================================================
# Agent descriptions for cost-breakdown hover tooltips
# ============================================================

AGENT_DESCRIPTIONS = {
    "guardrails":
        "Screens the JD for discriminatory or biased language. "
        "Two-layer defense: regex patterns catch obvious cases, then an LLM "
        "classifier catches subtle/coded language. Tokens = JD text + classifier prompt.",
    "jd_intake":
        "Parses the raw JD into structured criteria (must-haves, nice-to-haves, "
        "YOE, location, seniority) using one LLM call with strict JSON schema. "
        "High token count = full JD + extraction schema in the prompt.",
    "screening":
        "Scores each (candidate × criterion) pair individually with verbatim "
        "evidence quotes. The heaviest agent — runs ~10 candidates × N criteria "
        "in async fan-out. Tokens scale with profile text × criteria count.",
    "profile_summary":
        "Generates a bias-blind 2-3 sentence summary for each candidate after "
        "dedup, before screening. Deliberately omits name, location, and "
        "protected attributes so downstream agents (ranking, outreach) can "
        "reference the summary without bias risk. Async fan-out across all "
        "deduped profiles, cheap gpt-4o-mini model.",
    "ranking":
        "Generates a 2-4 sentence rationale per shortlisted candidate, then "
        "re-runs guardrails on the rationale to catch any bias leakage. "
        "One LLM call per shortlisted candidate.",
    "top_pick":
        "Single head-to-head LLM call comparing the top 3 shortlisted candidates. "
        "Returns a justification and the key trade-off vs runner-up. "
        "Highest leverage call in the pipeline — small cost, headline output.",
    "outreach":
        "Drafts subject + LinkedIn InMail + email body + personalization hooks "
        "for the top 3 non-flagged candidates. One LLM call per candidate.",
    "rag.indexer":
        "Embeds candidate profiles into the vector index using OpenAI "
        "text-embedding-3-small. Token count = sum of all profile texts.",
    "rag.retriever":
        "Embeds each criterion query for semantic search. Runs once per criterion "
        "during retrieval. Cheap because queries are short.",
}


def _agent_description(agent: str) -> str:
    return AGENT_DESCRIPTIONS.get(agent, f"LLM-driven {agent} step in the pipeline.")


# ============================================================
# Helpers
# ============================================================

def _status_glyph(status: str) -> str:
    return {
        "completed": "✓",
        "rejected":  "✗",
        "skipped":   "◷",
        "running":   "●",
        "pending":   "⋯",
    }.get(status, "⋯")


def _node_colors(status: str, node_name: str | None = None) -> tuple[str, str]:
    if status == "rejected":
        return COLORS["card_bg_fail"], COLORS["border_fail"]
    if status == "skipped" or status == "pending":
        return COLORS["card_bg_skip"], COLORS["border_skip"]
    if status == "completed" and node_name in NODE_PALETTE:
        palette = NODE_PALETTE[node_name]
        return palette["bg"], palette["border"]
    if status == "completed":
        return COLORS["card_bg_done"], COLORS["border_done"]
    return COLORS["card_bg"], COLORS["border_soft"]


def _fmt_duration(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 1:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


def _fmt_cost(usd: float) -> str:
    """Always show dollars with 4 decimals. No 'm' suffix."""
    if usd == 0:
        return "$0"
    return f"${usd:.4f}"


# ============================================================
# SVG primitives — inline polygon arrowheads (Streamlit-safe)
# ============================================================

def _arrowhead(x: int, y: int, angle_deg: float, color: str) -> str:
    """Inline polygon arrowhead at (x, y) pointing in `angle_deg` direction.
    angle_deg: 0 = right, 90 = down, 180 = left, 270 = up.
    """
    size = 7
    ang = math.radians(angle_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    pts = [(0, 0), (-size, -size * 0.55), (-size, size * 0.55)]
    rotated = [
        (x + p[0] * cos_a - p[1] * sin_a, y + p[0] * sin_a + p[1] * cos_a)
        for p in pts
    ]
    pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in rotated)
    return f'<polygon points="{pts_str}" fill="{color}"/>'


def _arrow_h(x1: int, y1: int, x2: int, y2: int, active: bool, dashed: bool = False) -> str:
    """Horizontal arrow from (x1, y1) to (x2, y2)."""
    color = COLORS["arrow_done"] if active else COLORS["arrow"]
    width = "2" if active else "1.5"
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    angle = 0 if x2 > x1 else 180
    line_end_x = x2 - 6 if x2 > x1 else x2 + 6
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{line_end_x}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dash}/>'
        + _arrowhead(x2, y2, angle, color)
    )


def _arrow_v(x1: int, y1: int, x2: int, y2: int, active: bool, dashed: bool = False) -> str:
    """Vertical arrow."""
    color = COLORS["arrow_done"] if active else COLORS["arrow"]
    width = "2" if active else "1.5"
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    angle = 90 if y2 > y1 else 270
    line_end_y = y2 - 6 if y2 > y1 else y2 + 6
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{line_end_y}" '
        f'stroke="{color}" stroke-width="{width}"{dash}/>'
        + _arrowhead(x2, y2, angle, color)
    )


def _corner_arrow(x1: int, y1: int, x2: int, y2: int, active: bool) -> str:
    """L-shaped path: horizontal from (x1,y1) to (x2,y1), then vertical to (x2,y2)."""
    color = COLORS["arrow_done"] if active else COLORS["arrow"]
    width = "2" if active else "1.5"
    angle = 90 if y2 > y1 else 270
    line_end_y = y2 - 6 if y2 > y1 else y2 + 6
    return (
        f'<path d="M {x1} {y1} L {x2} {y1} L {x2} {line_end_y}" '
        f'fill="none" stroke="{color}" stroke-width="{width}"/>'
        + _arrowhead(x2, y2, angle, color)
    )


def _node_svg(x: int, y: int, node: NodeStat) -> str:
    bg, border = _node_colors(node.status, node.name)
    glyph = _status_glyph(node.status)
    display = NODE_DISPLAY.get(node.name, node.name.upper())
    muted = COLORS["text_muted"]
    dim = COLORS["text_dim"]
    text_color = COLORS["text"]
    opacity = "0.55" if node.status in ("skipped", "pending") else "1"

    return f'''
    <g opacity="{opacity}">
      <rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="10" ry="10"
            fill="{bg}" stroke="{border}" stroke-width="1.5"/>
      <text x="{x + NODE_W - 14}" y="{y + 22}" text-anchor="end"
            font-family="JetBrains Mono, monospace" font-size="14" font-weight="600"
            fill="{border if node.status in ('completed', 'rejected') else dim}">
        {glyph}
      </text>
      <text x="{x + 14}" y="{y + 24}" font-family="JetBrains Mono, monospace"
            font-size="11" font-weight="600" letter-spacing="0.1em"
            fill="{muted}">{display}</text>
      <text x="{x + 14}" y="{y + 56}" font-family="Inter, sans-serif"
            font-size="22" font-weight="800" fill="{text_color}">
        {_fmt_duration(node.duration_s)}
      </text>
      <text x="{x + 14}" y="{y + 78}" font-family="JetBrains Mono, monospace"
            font-size="10.5" fill="{muted}">
        {_fmt_cost(node.cost_usd)} · {node.n_llm_calls} call{"s" if node.n_llm_calls != 1 else ""}
      </text>
      <text x="{x + 14}" y="{y + 95}" font-family="JetBrains Mono, monospace"
            font-size="9.5" letter-spacing="0.08em" fill="{dim}">
        {node.status.upper()}
      </text>
    </g>
    '''


# ============================================================
# Renderers
# ============================================================

def _render_flowchart(stats: list[NodeStat]) -> None:
    by_name: dict[str, NodeStat] = {s.name: s for s in stats}

    # ----- summary chip strip (right-justified) -----
    total_duration = sum((s.duration_s or 0) for s in stats)
    total_cost = sum(s.cost_usd for s in stats)
    total_calls = sum(s.n_llm_calls for s in stats)
    n_completed = sum(1 for s in stats if s.status == "completed")
    n_rejected = sum(1 for s in stats if s.status == "rejected")
    n_skipped = sum(1 for s in stats if s.status == "skipped")

    rejected_pill = (
        f'<span class="stat-pill" style="background:#fde2e2; color:#991b1b;">'
        f'{n_rejected} rejected</span>' if n_rejected else ""
    )
    skipped_pill = (
        f'<span class="stat-pill" style="background:#f5f3ec; color:#6b6963;">'
        f'{n_skipped} skipped</span>' if n_skipped else ""
    )
    summary = (
        f'<div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1rem; justify-content:flex-end;">'
        f'  <span class="stat-pill">{n_completed} completed</span>'
        f'  {rejected_pill}'
        f'  {skipped_pill}'
        f'  <span class="stat-pill">{_fmt_duration(total_duration)} total</span>'
        f'  <span class="stat-pill">{_fmt_cost(total_cost)} · {total_calls} LLM calls</span>'
        f'</div>'
    )
    st.markdown(summary, unsafe_allow_html=True)

    # ----- compute pixel positions -----
    pad = 20
    # Dynamically size SVG to fit however many columns the layout uses.
    # max(col) + 1 = number of columns; same for rows. This way adding a node
    # to NODE_POSITIONS just works without touching width/height calculations.
    n_cols = max(col for col, _ in NODE_POSITIONS.values()) + 1
    n_rows = max(row for _, row in NODE_POSITIONS.values()) + 1
    width = pad * 2 + n_cols * NODE_W + (n_cols - 1) * GAP_X
    height = pad * 2 + n_rows * NODE_H + (n_rows - 1) * GAP_Y + 30

    def pos(node_name: str) -> tuple[int, int]:
        col, row = NODE_POSITIONS[node_name]
        x = pad + col * (NODE_W + GAP_X)
        y = pad + row * (NODE_H + GAP_Y)
        return x, y

    # ----- arrows -----
    arrows_svg = ""
    for src, dst in EDGES:
        if src not in by_name or dst not in by_name:
            continue
        src_stat = by_name[src]
        dst_stat = by_name[dst]
        active = (
            src_stat.status == "completed"
            and dst_stat.status in ("completed", "rejected")
        )
        sx, sy = pos(src)
        dx, dy = pos(dst)
        src_col, src_row = NODE_POSITIONS[src]
        dst_col, dst_row = NODE_POSITIONS[dst]

        if src_row == dst_row:
            if dst_col > src_col:
                arrows_svg += _arrow_h(
                    sx + NODE_W, sy + NODE_H // 2,
                    dx, dy + NODE_H // 2,
                    active,
                )
            else:
                arrows_svg += _arrow_h(
                    sx, sy + NODE_H // 2,
                    dx + NODE_W, dy + NODE_H // 2,
                    active,
                )
        else:
            arrows_svg += _corner_arrow(
                sx + NODE_W // 2, sy + NODE_H,
                dx + NODE_W // 2, dy,
                active,
            )

    # ----- nodes -----
    nodes_svg = ""
    for node_name in NODE_ORDER:
        if node_name not in by_name:
            continue
        x, y = pos(node_name)
        nodes_svg += _node_svg(x, y, by_name[node_name])

    svg = f'''
    <div class="pipeline-flowchart">
    <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"
         style="width:100%; height:auto; max-width:{width}px; display:block; margin:0 auto;">
      {arrows_svg}
      {nodes_svg}
    </svg>
    </div>
    '''
    st.markdown(svg, unsafe_allow_html=True)

    # ----- legend -----
    st.markdown(
        '<div style="display:flex; gap:1.5rem; justify-content:center; margin-top:1rem;'
        ' font-family:JetBrains Mono, monospace; font-size:0.75rem; color:#6b6963;">'
        '<span><span style="color:#15803d;">✓</span> completed</span>'
        '<span><span style="color:#dc2626;">✗</span> rejected</span>'
        '<span><span style="color:#9a978f;">◷</span> skipped</span>'
        '<span><span style="color:#9a978f;">⋯</span> pending</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_cost_breakdown(cost: dict) -> None:
    st.markdown(eyebrow("COST BREAKDOWN"), unsafe_allow_html=True)
    by_agent = cost.get("by_agent", {})
    if not by_agent:
        st.caption("No LLM cost data for this JD yet.")
        return
    for agent, stats in sorted(by_agent.items(), key=lambda x: x[1]["usd"], reverse=True):
        description = _agent_description(agent)
        # Tooltip via standard `title` attribute — works in every browser
        st.markdown(
            f'<div class="card cost-card" title="{description}" '
            f'style="padding:0.7rem 1rem; margin-bottom:0.5rem; cursor:help;">'
            f'<div style="display:flex; justify-content:space-between;">'
            f'<span style="font-family:JetBrains Mono, monospace; font-size:0.85rem;">{agent}</span>'
            f'<span style="font-family:JetBrains Mono, monospace; font-size:0.85rem;">'
            f'<strong>${stats["usd"]:.4f}</strong></span></div>'
            f'<div style="color:#9a978f; font-size:0.75rem; font-family:JetBrains Mono, monospace;">'
            f'{stats["calls"]} calls · {stats["tokens_in"]}/{stats["tokens_out"]} tokens'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def _render_activity_log(events: list) -> None:
    """Render the activity log with run-length compression.

    Without compression, screening's ~200+ score_pair_start/score_pair_end
    events fill the window and push earlier agents (guardrails, jd_intake,
    sourcing, profile_summary) out of view. We collapse consecutive
    identical (agent, event) pairs into a single row with a count, so the
    full pipeline is visible from start to finish.

    Compression is purely cosmetic — the underlying events list is
    unchanged; we just render runs as one line.
    """
    if not events:
        st.markdown(
            '<div class="activity-log">(no events recorded — try a fresh JD run)</div>',
            unsafe_allow_html=True,
        )
        return

    # Run-length encode consecutive identical (agent, event) pairs.
    # Each entry: (first_ts, agent, event, count).
    compressed: list[tuple[str, str, str, int]] = []
    for e in events:
        ts = e.get("ts", "")[11:19]
        agent = e.get("agent", "")[:22]
        evt = e.get("event", "")
        if compressed and compressed[-1][1] == agent and compressed[-1][2] == evt:
            # Same as previous — bump the count, keep the first timestamp
            prev_ts, prev_agent, prev_evt, prev_n = compressed[-1]
            compressed[-1] = (prev_ts, prev_agent, prev_evt, prev_n + 1)
        else:
            compressed.append((ts, agent, evt, 1))

    # Show the last 150 compressed rows — way more pipeline coverage than
    # raw last-100 since each row may represent many underlying events.
    rows = []
    for ts, agent, evt, n in compressed[-150:]:
        count_suffix = (
            f' <span style="color:#9a978f; font-size:0.78em;'
            f' margin-left:0.3em;">×{n}</span>'
        ) if n > 1 else ""
        rows.append(
            f'<span class="log-ts">{ts}</span> '
            f'<span class="log-ag">{agent:22s}</span> '
            f'<span class="log-evt">{evt}</span>'
            f'{count_suffix}'
        )

    st.markdown(
        '<div class="activity-log">' + "<br/>".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# Rejection view — replaces the SVG flowchart for rejected JDs
# ============================================================

def _render_rejection_view(jd_id: str, guardrails_stat: NodeStat, detail: dict, events: list) -> None:
    """Simpler rejection-specific layout.

    For rejected JDs we skip the SVG flowchart (most nodes would be empty
    anyway) and present the rejection reasoning as the headline content.
    This actually tells a better story for the recruiter: "here's exactly
    why we halted the pipeline before spending money on retrieval."
    """
    st.markdown(eyebrow("PIPELINE FLOWCHART"), unsafe_allow_html=True)

    # Top stat strip — same layout as the normal pipeline view
    summary = (
        '<div style="display:flex; gap:0.5rem; flex-wrap:wrap; '
        'margin-bottom:1rem; justify-content:flex-end;">'
        '<span class="stat-pill" style="background:#fde2e2; color:#991b1b;">'
        '1 rejected</span>'
        '<span class="stat-pill" style="background:#f5f3ec; color:#6b6963;">'
        '6 skipped</span>'
        f'<span class="stat-pill">{_fmt_duration(guardrails_stat.duration_s)} total</span>'
        f'<span class="stat-pill">{_fmt_cost(guardrails_stat.cost_usd)} · '
        f'{guardrails_stat.n_llm_calls} LLM calls</span>'
        '</div>'
    )
    st.markdown(summary, unsafe_allow_html=True)

    # Pull guardrail verdict if the API surfaced it
    guardrail_verdict = detail.get("guardrail_verdict") or {}
    reasons = guardrail_verdict.get("reasons", [])
    flagged_phrases = guardrail_verdict.get("flagged_phrases", [])

    # The rejection card
    st.markdown(
        '<div class="card" style="border-left:4px solid #dc2626; padding:1.5rem 1.75rem;">'
        '<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">'
        '<span style="font-family:JetBrains Mono, monospace; font-size:1.3rem; '
        'color:#dc2626;">✗</span>'
        '<span style="font-family:JetBrains Mono, monospace; font-size:0.78rem; '
        'letter-spacing:0.1em; text-transform:uppercase; color:#991b1b;">'
        'GUARDRAILS REJECTED</span>'
        '</div>'
        '<div style="font-size:1.4rem; font-weight:800; margin-bottom:0.8rem;">'
        'Pipeline halted before sourcing.'
        '</div>'
        '<p style="color:#0a0a0a; line-height:1.6;">'
        'The JD failed the two-layer fairness check (regex + LLM classifier). '
        'No candidate data was retrieved or scored. '
        f'Total cost: <strong>{_fmt_cost(guardrails_stat.cost_usd)}</strong>, '
        f'runtime: <strong>{_fmt_duration(guardrails_stat.duration_s)}</strong>.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Reasons (from the classifier)
    if reasons:
        st.markdown(eyebrow("WHY IT WAS REJECTED"), unsafe_allow_html=True)
        reasons_html = (
            '<div class="card" style="padding:1rem 1.25rem;">'
            '<ul style="margin:0; padding-left:1.2rem; line-height:1.7; color:#0a0a0a;">'
            + "".join(f"<li>{r}</li>" for r in reasons)
            + '</ul>'
            '</div>'
        )
        st.markdown(reasons_html, unsafe_allow_html=True)

    # Flagged phrases (the specific coded language)
    if flagged_phrases:
        st.markdown(eyebrow("FLAGGED PHRASES"), unsafe_allow_html=True)
        phrases_html = (
            '<div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:1rem;">'
            + "".join(
                f'<span style="background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; '
                f'padding:4px 10px; border-radius:999px; font-family:JetBrains Mono, monospace; '
                f'font-size:0.78rem;">"{p}"</span>'
                for p in flagged_phrases
            )
            + '</div>'
        )
        st.markdown(phrases_html, unsafe_allow_html=True)

    # Activity log + cost breakdown still useful
    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_log, col_cost = st.columns([1.6, 1])
    with col_log:
        st.markdown(eyebrow("ACTIVITY LOG"), unsafe_allow_html=True)
        _render_activity_log(events)
    with col_cost:
        try:
            cost = api_client.get_cost(jd_id)
        except api_client.APIError:
            cost = {"by_agent": {}}
        _render_cost_breakdown(cost)


# ============================================================
# Public entry point
# ============================================================

def render(jd_id: str):
    """Render the unified pipeline view."""
    stats = compute_node_stats(jd_id)

    try:
        cost = api_client.get_cost(jd_id)
    except api_client.APIError:
        cost = {"by_agent": {}}

    try:
        detail = api_client.get_jd_detail(jd_id)
        events = detail.get("events", [])
    except api_client.APIError:
        detail = {}
        events = []

    # If guardrails rejected this JD, render the simpler rejection view
    # instead of the SVG flowchart. Tells a better story ("here's exactly
    # why we halted") and sidesteps a Streamlit-side rendering quirk with
    # the flowchart when most nodes are skipped.
    by_name = {s.name: s for s in stats}
    guardrails_stat = by_name.get("guardrails")
    if guardrails_stat and guardrails_stat.status == "rejected":
        _render_rejection_view(jd_id, guardrails_stat, detail, events)
        return

    st.markdown(eyebrow("PIPELINE FLOWCHART"), unsafe_allow_html=True)
    _render_flowchart(stats)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    col_log, col_cost = st.columns([1.6, 1])
    with col_log:
        st.markdown(eyebrow("ACTIVITY LOG"), unsafe_allow_html=True)
        _render_activity_log(events)
    with col_cost:
        _render_cost_breakdown(cost)
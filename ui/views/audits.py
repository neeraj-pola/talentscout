# ui/pages/audits.py
"""Page 4 — closed-JD audit log."""
from __future__ import annotations

import streamlit as st

from ui import api_client
from ui.styles import eyebrow, cost_chip


def render():
    st.markdown(eyebrow("04 / AUDIT LOG"), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Closed <span class="highlight">JDs</span>.</h1>',
        unsafe_allow_html=True,
    )

    try:
        audits = api_client.list_audits()
    except api_client.APIError as e:
        st.error(str(e))
        return

    if not audits:
        st.info("No closed JDs yet. Close one from the JD Detail page.")
        return

    st.markdown(
        f'<p style="color:#6b6963;">{len(audits)} closure record{"s" if len(audits) != 1 else ""}.</p>',
        unsafe_allow_html=True,
    )

    for a in audits:
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">'
            f'<div style="flex:1;">'
            f'<div style="font-family:JetBrains Mono, monospace; font-size:0.75rem; color:#9a978f; '
            f'text-transform:uppercase; letter-spacing:0.05em;">'
            f'JD {a["jd_id"][:8]}…  ·  closed {a.get("closed_at", "")[:19] if a.get("closed_at") else "—"}'
            f'</div>'
            f'<div style="margin-top:0.4rem;"><strong>{a["closed_by"]}</strong> chose '
            f'<code>{a["candidate_id"][:8]}…</code></div>'
            f'<div style="color:#6b6963; font-size:0.88rem; margin-top:0.5rem;">{a["justification"]}</div>'
            f'</div>'
            f'<div>{cost_chip(a["total_cost_usd"], a["total_llm_calls"])}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
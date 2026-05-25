# ui/pages/dashboard.py
"""Page 2 — list of all JDs."""
from __future__ import annotations

import streamlit as st

from ui import api_client
from ui.styles import eyebrow, status_pill


def render():
    st.markdown(eyebrow("02 / DASHBOARD"), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">JD <span class="highlight">pipeline</span>.</h1>',
        unsafe_allow_html=True,
    )

    try:
        jds = api_client.list_jds()
    except api_client.APIError as e:
        st.error(str(e))
        return

    if not jds:
        st.info("No JDs yet. Submit one from the JD Intake page.")
        return

    st.markdown(
        f'<p style="color:#6b6963;">{len(jds)} JD{"s" if len(jds) != 1 else ""} in the system.</p>',
        unsafe_allow_html=True,
    )

    for jd in jds:
        card_html = f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
                <div style="flex:1;">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#9a978f; text-transform:uppercase; letter-spacing:0.05em;">
                        {jd["id"][:8]}…  ·  {jd["location"]}
                    </div>
                    <div style="font-size:1.2rem; font-weight:700; margin:0.3rem 0;">{jd["title"]}</div>
                    <div style="color:#6b6963; font-size:0.85rem; font-family:'JetBrains Mono', monospace;">
                        Created {jd["created_at"][:10]}  ·  Target {jd["target_hiring_date"]}
                    </div>
                </div>
                <div>{status_pill(jd["status"])}</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        if st.button(f"Open →", key=f"open_{jd['id']}"):
            st.session_state["selected_jd_id"] = jd["id"]
            st.session_state["page"] = "JD Detail"
            st.rerun()
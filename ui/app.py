# ui/app.py
"""TalentScout Streamlit UI — entry point and page router."""

# ui/app.py
import sys
from pathlib import Path

# Make the project root importable so `from ui.styles import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import streamlit as st

from ui.styles import inject
from ui import api_client
from ui.views import intake, dashboard, detail, audits

# ============================================================
# Page config — must be the first Streamlit call
# ============================================================
st.set_page_config(
    page_title="TalentScout",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1.5rem;">'
        '<div style="width:36px; height:36px; border-radius:50%; background:#fbd35a;'
        ' display:flex; align-items:center; justify-content:center; font-weight:900;">T</div>'
        '<div style="font-family:Inter; font-weight:900; font-size:1.4rem;">TalentScout.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Health indicator
    try:
        h = api_client.health()
        api_ok = True
        mock_ok = h.get("mock_server_reachable", False)
    except Exception:
        api_ok = False
        mock_ok = False

    api_color = "#22c55e" if api_ok else "#dc2626"
    mock_color = "#22c55e" if mock_ok else "#dc2626"
    st.markdown(
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.75rem; '
        f'color:#6b6963; margin-bottom:1rem;">'
        f'<span style="color:{api_color}">●</span> API '
        f'<span style="color:{mock_color}; margin-left:0.7rem;">●</span> mock sources'
        f'</div>',
        unsafe_allow_html=True,
    )

    pages = ["JD Intake", "Dashboard", "JD Detail", "Audit Log"]
    if "page" not in st.session_state:
        st.session_state["page"] = "JD Intake"

    choice = st.radio(
        "Navigate",
        pages,
        index=pages.index(st.session_state["page"]) if st.session_state["page"] in pages else 0,
        key="page_radio",
    )
    if choice != st.session_state["page"]:
        st.session_state["page"] = choice
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
        'color:#9a978f; line-height:1.6;">'
        'TalentScout POC<br/>'
        'Multi-agent recruiting<br/>'
        '+ RAG + LangGraph'
        '</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# Router
# ============================================================
page = st.session_state.get("page", "JD Intake")
if page == "JD Intake":
    intake.render()
elif page == "Dashboard":
    dashboard.render()
elif page == "JD Detail":
    detail.render()
elif page == "Audit Log":
    audits.render()
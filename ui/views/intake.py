# ui/pages/intake.py
"""Page 1 — JD intake form."""
from __future__ import annotations

from datetime import date

import streamlit as st

from ui import api_client
from ui.styles import eyebrow


def render():
    st.markdown(eyebrow("01 / JD INTAKE"), unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Submit a <span class="highlight">new role</span>.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b6963; font-size:1.05rem; max-width:680px;">'
        "Describe the role. Guardrails will screen for fairness, then the pipeline "
        "will source, dedup, score, rank, and recommend a top candidate — usually "
        "in under two minutes."
        "</p>",
        unsafe_allow_html=True,
    )

    # Sample JD selector
    st.markdown("&nbsp;")
    with st.expander("Need a starting point? Pick a sample JD"):
        try:
            samples = api_client.get_sample_jds()
            cols = st.columns(len(samples))
            for col, s in zip(cols, samples):
                with col:
                    if st.button(s["label"], key=f"sample_{s['label']}"):
                        st.session_state["intake_prefill"] = s["payload"]
                        st.rerun()
        except api_client.APIError as e:
            st.error(str(e))

    prefill = st.session_state.get("intake_prefill", {})

    # ----- form -----
    with st.form("jd_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            title = st.text_input(
                "Role title",
                value=prefill.get("title", ""),
                placeholder="e.g. Senior ML Engineer (LLM / RAG)",
            )
        with col2:
            employment_type = st.selectbox(
                "Employment type",
                ["full_time", "contract", "intern"],
                index=["full_time", "contract", "intern"].index(
                    prefill.get("employment_type", "full_time")
                ),
            )

        description = st.text_area(
            "Description",
            value=prefill.get("description", ""),
            placeholder="What will this person own and ship? Be specific about scale, stack, and impact.",
            height=180,
        )

        col3, col4 = st.columns(2)
        with col3:
            must_have = st.text_input(
                "Must-have skills (comma-separated)",
                value=", ".join(prefill.get("must_have_skills", [])),
                placeholder="Python, LLMs, RAG, LangChain, AWS",
            )
        with col4:
            nice_have = st.text_input(
                "Nice-to-have skills (comma-separated)",
                value=", ".join(prefill.get("nice_to_have_skills", [])),
                placeholder="Kubernetes, Time series, Azure",
            )

        col5, col6, col7 = st.columns(3)
        with col5:
            min_yoe = st.number_input(
                "Min YOE",
                min_value=0, max_value=40, step=1,
                value=int(prefill.get("min_years_experience", 3)),
            )
        with col6:
            max_yoe = st.number_input(
                "Max YOE (0 = no max)",
                min_value=0, max_value=40, step=1,
                value=int(prefill.get("max_years_experience") or 0),
            )
        with col7:
            target_date = st.date_input(
                "Target hiring date",
                value=date.fromisoformat(prefill["target_hiring_date"])
                if prefill.get("target_hiring_date") else date.today(),
            )

        col8, col9 = st.columns([2, 1])
        with col8:
            location = st.text_input(
                "Location",
                value=prefill.get("location", ""),
                placeholder="Hyderabad, India",
            )
        with col9:
            remote_ok = st.checkbox(
                "Remote OK",
                value=bool(prefill.get("remote_ok", False)),
            )

        submitted = st.form_submit_button("Submit and run pipeline →")

    if submitted:
        if not title or not description or not must_have:
            st.error("Title, description, and at least one must-have skill are required.")
            return

        payload = {
            "title": title,
            "description": description,
            "must_have_skills": [s.strip() for s in must_have.split(",") if s.strip()],
            "nice_to_have_skills": [s.strip() for s in nice_have.split(",") if s.strip()],
            "min_years_experience": int(min_yoe),
            "max_years_experience": int(max_yoe) if max_yoe > 0 else None,
            "location": location or "Remote",
            "remote_ok": remote_ok,
            "employment_type": employment_type,
            "target_hiring_date": target_date.isoformat(),
        }

        with st.spinner("Pipeline running — guardrails, parsing, sourcing, screening, ranking… (60-120s)"):
            try:
                result = api_client.create_jd(payload)
            except api_client.APIError as e:
                st.error(str(e))
                return

        st.session_state.pop("intake_prefill", None)
        st.success(f"Pipeline finished — status: **{result['status']}**")
        st.session_state["selected_jd_id"] = result["jd_id"]
        st.session_state["page"] = "JD Detail"
        st.rerun()
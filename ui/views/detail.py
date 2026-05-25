# ui/views/detail.py
"""Single JD detail page — full pipeline output with all tabs."""
from __future__ import annotations

import streamlit as st

from ui import api_client
from ui.styles import (
    eyebrow, stat_pill, status_pill, tag,
    score_bar, evidence_quote, cost_chip,
)
from ui.views import pipeline, refine


# ============================================================
# Helpers
# ============================================================

def _has_must_have_gap(candidate: dict, parsed_jd: dict | None = None) -> bool:
    """Pull the backend's authoritative gap flag.

    The ranking agent sets this on ScoredCandidate based on its own threshold
    logic (typically: any must-have scored 0.0, or must coverage below a floor).
    We trust that decision rather than re-deriving client-side.
    """
    return bool(candidate.get("has_must_have_gap"))


def _synthesize_rationale(candidate: dict) -> str:
    """Fallback rationale builder when backend doesn't provide overall_rationale.

    Pulls top must-have hits and gaps to produce a 1-2 sentence summary.
    Only used if `overall_rationale` is missing — which shouldn't happen
    on fresh JDs but protects against older data.
    """
    crits = candidate.get("criterion_scores", [])
    must_hits = [cs for cs in crits
                 if cs.get("criterion_id", "").startswith("must_") and cs.get("score", 0) >= 0.5]
    must_gaps = [cs for cs in crits
                 if cs.get("criterion_id", "").startswith("must_") and cs.get("score", 0) < 0.3]

    parts = []
    if must_hits:
        names = ", ".join(cs["criterion_text"] for cs in must_hits[:3])
        parts.append(f"Demonstrates {names}.")
    else:
        parts.append("Weak coverage of required skills.")
    if must_gaps:
        names = ", ".join(cs["criterion_text"] for cs in must_gaps[:2])
        parts.append(f"Gaps: no evidence of {names}.")
    return " ".join(parts)


# ============================================================
# Page entry point
# ============================================================

def render():
    jd_id = st.session_state.get("selected_jd_id")
    if not jd_id:
        st.warning("No JD selected. Go to Dashboard and pick one.")
        return

    try:
        detail = api_client.get_jd_detail(jd_id)
    except api_client.APIError as e:
        st.error(f"Failed to load JD: {e}")
        return

    jd = detail["jd"]
    parsed_jd = detail.get("parsed_jd")
    shortlist = detail.get("shortlist", [])
    top_pick = detail.get("top_pick")
    outreach_drafts = detail.get("outreach_drafts", [])
    sourcing = detail.get("sourcing_summary")
    status = detail.get("status", "draft")
    # Full deduped profiles keyed by id — used to surface bias-blind summaries
    # produced by the profile_summary agent on each shortlist card. Empty for
    # JDs run before the profile_summary node existed; the UI handles that
    # gracefully by simply omitting the summary block.
    profiles_by_id = {p["id"]: p for p in detail.get("profiles", [])}

    # ---- Header ----
    _render_header(jd, status, detail.get("cost_summary", {}))

    # ---- Tabs ----
    # Refine sits between Outreach and Sourcing — it's a candidate-interaction
    # surface (chat-driven exploration of the shortlist), not a back-office view.
    tabs = st.tabs([
        "Overview", "Shortlist", "Top Pick", "Outreach",
        "Refine", "Sourcing", "Pipeline",
    ])

    with tabs[0]:
        _render_overview(jd, parsed_jd)

    with tabs[1]:
        _render_shortlist(shortlist, parsed_jd, profiles_by_id)

    with tabs[2]:
        if top_pick:
            _render_top_pick(top_pick, shortlist, jd_id, status)
        else:
            st.info("No top-pick recommendation yet.")

    with tabs[3]:
        _render_outreach_tab(outreach_drafts, shortlist, top_pick, parsed_jd)

    with tabs[4]:
        refine.render(jd_id, detail)

    with tabs[5]:
        _render_sourcing(sourcing, detail.get("merge_audit", []))

    with tabs[6]:
        pipeline.render(jd_id)


# ============================================================
# Header
# ============================================================

def _render_header(jd: dict, status: str, cost_summary: dict):
    jd_id_short = jd["id"][:8].upper()
    st.markdown(eyebrow(f"JD · {jd_id_short}"), unsafe_allow_html=True)
    st.markdown(f"# {jd['title']}")

    location = jd.get("location") or ("Remote" if jd.get("remote_ok") else "—")
    employment = jd.get("employment_type", "full_time").replace("_", " ")
    min_yoe = jd.get("min_yoe", 0)

    pills = [
        status_pill(status),
        stat_pill(location),
        stat_pill(employment),
        stat_pill(f"{min_yoe}+ years"),
    ]
    if cost_summary:
        pills.append(cost_chip(
            cost_summary.get("total_usd", 0.0),
            cost_summary.get("total_calls", 0),
        ))
    st.markdown(
        '<div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1.2rem;">'
        + "".join(pills)
        + '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Overview tab
# ============================================================

def _render_overview(jd: dict, parsed_jd: dict | None):
    st.markdown(eyebrow("JOB DESCRIPTION"), unsafe_allow_html=True)
    st.markdown(
        f'<div class="card"><p style="white-space:pre-wrap;">{jd["description"]}</p></div>',
        unsafe_allow_html=True,
    )

    if not parsed_jd:
        st.info("JD has not been parsed yet.")
        return

    criteria = parsed_jd.get("criteria", [])
    must_haves = [c for c in criteria if c.get("is_must_have")]
    nice_to_haves = [c for c in criteria if not c.get("is_must_have")]

    # ---- Must-have section ----
    st.markdown(eyebrow(f"MUST-HAVE CRITERIA · {len(must_haves)}"), unsafe_allow_html=True)
    if must_haves:
        cards_html = '<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:0.7rem; margin-bottom:1.5rem;">'
        for c in must_haves:
            category = c.get("category", "skill")
            weight = c.get("weight", 1.0)
            cards_html += (
                f'<div style="background:#fff8e1; border:1px solid #fbd35a; border-radius:10px; '
                f'padding:0.7rem 0.9rem;">'
                f'<div style="font-family:JetBrains Mono, monospace; font-size:0.65rem; '
                f'letter-spacing:0.08em; text-transform:uppercase; color:#92400e; margin-bottom:0.3rem;">'
                f'{category} · weight {weight:.1f}</div>'
                f'<div style="font-weight:600; font-size:0.92rem; color:#0a0a0a;">'
                f'{c.get("text", "")}</div>'
                f'</div>'
            )
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.caption("No must-have criteria parsed.")

    # ---- Nice-to-have section ----
    st.markdown(eyebrow(f"NICE-TO-HAVE CRITERIA · {len(nice_to_haves)}"), unsafe_allow_html=True)
    if nice_to_haves:
        cards_html = '<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:0.7rem;">'
        for c in nice_to_haves:
            category = c.get("category", "skill")
            weight = c.get("weight", 0.4)
            cards_html += (
                f'<div style="background:#ffffff; border:1px solid #ece8dc; border-radius:10px; '
                f'padding:0.7rem 0.9rem;">'
                f'<div style="font-family:JetBrains Mono, monospace; font-size:0.65rem; '
                f'letter-spacing:0.08em; text-transform:uppercase; color:#6b6963; margin-bottom:0.3rem;">'
                f'{category} · weight {weight:.1f}</div>'
                f'<div style="font-weight:500; font-size:0.92rem; color:#0a0a0a;">'
                f'{c.get("text", "")}</div>'
                f'</div>'
            )
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.caption("No nice-to-have criteria parsed.")
# ============================================================
# Shortlist tab
# ============================================================

def _render_shortlist(shortlist: list, parsed_jd: dict | None = None,
                      profiles_by_id: dict | None = None):
    if not shortlist:
        st.info("No shortlist generated yet.")
        return

    profiles_by_id = profiles_by_id or {}

    st.markdown(eyebrow(f"SHORTLIST · {len(shortlist)} CANDIDATES"), unsafe_allow_html=True)

    for i, c in enumerate(shortlist, 1):
        has_gap = _has_must_have_gap(c, parsed_jd)
        gap_flag = ""
        if has_gap:
            gap_flag = (
                ' <span style="color:#dc2626; font-family:JetBrains Mono, monospace;'
                ' font-size:0.75rem;">⚠ must-have gap</span>'
            )

        # Prefer the backend's rationale; fall back to client synthesis if missing
        summary = c.get("overall_rationale") or _synthesize_rationale(c)

        # Bias-blind professional summary from the profile_summary agent.
        # This is the JD-agnostic "executive view" — describes the candidate
        # in general terms (YOE, skills, notable employers, gaps). Distinct
        # from `overall_rationale` above which is JD-specific reasoning.
        # Render it as its own labeled block above the JD-fit rationale so
        # the reader sees "who this person is" before "why they fit here."
        profile = profiles_by_id.get(c.get("profile_id"), {})
        profile_summary_text = (profile.get("summary") or "").strip()
        profile_summary_html = ""
        if profile_summary_text:
            profile_summary_html = (
                '<div style="margin-top:0.8rem; padding:0.65rem 0.85rem; '
                'background:#f5ecd0; border-left:3px solid #fbd35a; border-radius:6px;">'
                '<div style="font-family:JetBrains Mono, monospace; font-size:0.7rem; '
                'letter-spacing:0.06em; text-transform:uppercase; color:#6b6963; '
                'margin-bottom:0.3rem;">PROFILE SUMMARY</div>'
                f'<div style="color:#0a0a0a; font-size:0.88rem; line-height:1.45;">'
                f'{profile_summary_text}</div>'
                '</div>'
            )

        # Red flags from the backend (if any)
        red_flags = c.get("red_flags", [])

        red_flags_html = ""
        if red_flags:
            red_flags_html = (
                '<div style="margin-top:0.6rem; padding:0.5rem 0.8rem; background:#fee2e2; '
                'border-left:3px solid #dc2626; border-radius:6px; font-size:0.82rem; color:#991b1b;">'
                '<span style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
                'letter-spacing:0.05em; text-transform:uppercase;">RED FLAGS</span><br/>'
                + "<br/>".join(red_flags)
                + '</div>'
            )

        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
            f'<div>'
            f'<div style="font-family:JetBrains Mono, monospace; color:#9a978f; font-size:0.78rem;">'
            f'#{i:02d}</div>'
            f'<div style="font-size:1.2rem; font-weight:700;">{c["candidate_name"]}{gap_flag}</div>'
            f'<div style="color:#6b6963; font-size:0.85rem;">{c.get("title", "")}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-family:JetBrains Mono, monospace; color:#9a978f; font-size:0.72rem;">OVERALL</div>'
            f'{score_bar(c["overall_score"])}'
            f'<div style="margin-top:0.4rem; font-family:JetBrains Mono, monospace; font-size:0.72rem; color:#6b6963;">'
            f'must {c["must_have_coverage"]*100:.0f}% · nice {c["nice_to_have_coverage"]*100:.0f}%'
            f'</div>'
            f'</div>'
            f'</div>'
            f'{profile_summary_html}'
            f'<div style="margin-top:0.8rem; color:#0a0a0a; font-size:0.9rem;">'
            f'<div style="font-family:JetBrains Mono, monospace; font-size:0.7rem; '
            f'letter-spacing:0.06em; text-transform:uppercase; color:#6b6963; '
            f'margin-bottom:0.3rem;">FIT FOR THIS ROLE</div>'
            f'{summary}'
            f'</div>'
            f'{red_flags_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"Per-criterion scores for {c['candidate_name']}"):
            for crit in c.get("criterion_scores", []):
                cov = " (no profile signal)" if not crit.get("has_evidence", True) else ""
                st.markdown(
                    f'<div style="margin-bottom:0.6rem;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="font-weight:500;">{crit["criterion_text"]}</span>'
                    f'<span>{score_bar(crit["score"])}{cov}</span>'
                    f'</div>'
                    f'<div style="color:#6b6963; font-size:0.85rem; margin-top:0.2rem; font-style:italic;">'
                    f'{crit.get("reasoning", "")}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    evidence_quote(crit.get("evidence", ""), crit.get("has_evidence", True)),
                    unsafe_allow_html=True,
                )


# ============================================================
# Top Pick tab — with UUID-to-name substitution
# ============================================================

def _render_top_pick(top_pick: dict, shortlist: list, jd_id: str, status: str):
    """Render the recommended candidate card.

    The LLM is prompted to refer to candidates by UUID (not name) to avoid
    name-based bias (gender, ethnicity inference). We substitute names back
    at render time — bias protection stays in the prompt, readability in the UI.
    """
    id_to_name = {c["profile_id"]: c["candidate_name"] for c in shortlist}

    def substitute_uuids(text: str) -> str:
        if not text:
            return text
        for uid, name in id_to_name.items():
            text = text.replace(uid, name)
            text = text.replace(uid[:8], name)
        return text

    name = top_pick.get("candidate_name", "Unknown")
    justification_clean = substitute_uuids(top_pick.get("justification", ""))
    tradeoff_clean = substitute_uuids(top_pick.get("key_tradeoff_vs_runner_up", ""))

    st.markdown(
        f'<div class="top-pick-card">'
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.78rem; '
        f'letter-spacing:0.1em; text-transform:uppercase; color:#92400e;">'
        f'★ RECOMMENDED CANDIDATE</div>'
        f'<div style="font-size:2.4rem; font-weight:900; margin: 0.4rem 0 0.8rem 0; letter-spacing:-0.02em;">'
        f'<span class="highlight">{name}</span></div>'
        f'<div style="color:#0a0a0a; font-size:0.98rem; line-height:1.6; margin-bottom:1rem;">'
        f'{justification_clean}</div>'
        f'<div style="background: rgba(255,255,255,0.5); padding: 0.8rem 1rem; border-radius:10px;">'
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
        f'letter-spacing:0.08em; text-transform:uppercase; color:#92400e; margin-bottom:0.3rem;">'
        f'KEY TRADE-OFF VS RUNNER-UP</div>'
        f'<div style="font-size:0.9rem; color:#0a0a0a;">{tradeoff_clean}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if status == "closed":
        st.info("This JD has already been closed.")
        return

    st.markdown(eyebrow("CLOSE THIS JD"), unsafe_allow_html=True)
    closed_by = st.text_input("Your email / username", key="close_who",
                              placeholder="neeraj@example.com")
    candidate_options = {c["profile_id"]: c["candidate_name"] for c in shortlist}
    default_idx = (
        list(candidate_options.keys()).index(top_pick["recommended_candidate_id"])
        if top_pick["recommended_candidate_id"] in candidate_options else 0
    )
    chosen_id = st.selectbox(
        "Close with candidate",
        options=list(candidate_options.keys()),
        format_func=lambda x: f"{candidate_options[x]}"
                              + (" (recommended)" if x == top_pick["recommended_candidate_id"] else ""),
        index=default_idx,
    )
    if st.button("Approve & close JD"):
        if not closed_by.strip():
            st.error("Please enter your email/username.")
        else:
            try:
                api_client.close_jd(jd_id, closed_by.strip(), chosen_id)
                st.success("JD closed and audit record written.")
                st.rerun()
            except api_client.APIError as e:
                st.error(str(e))


# ============================================================
# Outreach tab — trust the backend's eligibility decision
# ============================================================

def _render_outreach_tab(outreach_drafts: list, shortlist: list, top_pick: dict | None, parsed_jd: dict | None = None):
    """Display outreach drafts via a candidate dropdown.

    Trust the backend's eligibility decision: show all candidates the
    pipeline drafted outreach for. The backend already filtered out
    candidates with must-have gaps before generating drafts.
    """
    if not outreach_drafts:
        st.info("No outreach drafts generated yet.")
        return

    drafts_by_id = {d["candidate_id"]: d for d in outreach_drafts}

    recommended_id = top_pick["recommended_candidate_id"] if top_pick else None
    candidates_in_drafts = []
    for cand_id in drafts_by_id:
        sl_entry = next((c for c in shortlist if c["profile_id"] == cand_id), None)
        candidates_in_drafts.append({
            "id": cand_id,
            "name": sl_entry["candidate_name"] if sl_entry else "Unknown",
            "score": sl_entry["overall_score"] if sl_entry else 0.0,
        })

    candidates_in_drafts.sort(key=lambda c: (
        0 if c["id"] == recommended_id else 1,
        -c["score"],
    ))

    options = [c["id"] for c in candidates_in_drafts]

    def fmt(uid: str) -> str:
        cand = next(c for c in candidates_in_drafts if c["id"] == uid)
        star = " ★" if uid == recommended_id else ""
        return f"{cand['name']}{star}  —  {cand['score']:.2f}"

    chosen_id = st.selectbox(
        "View outreach for",
        options=options,
        format_func=fmt,
        index=0,
        key="outreach_pick",
    )
    _render_outreach(drafts_by_id[chosen_id], shortlist)


def _substitute_name_placeholders(text: str, candidate_name: str) -> str:
    """Substitute the {candidate_name} placeholder with the real name.

    The outreach agent uses {candidate_name} as a placeholder during drafting
    so the LLM never sees the candidate's actual name (keeps the draft path
    bias-blind — the LLM can't be influenced by name-based associations).
    The UI substitutes here at render time.

    Also blanks the rogue [Your Name] template the LLM sometimes emits when
    trying to write a recruiter signature. We sign off as the team handle
    ("TalentScout Recruiting") rather than impersonate a specific recruiter,
    so that bracket-style placeholder always represents a model hallucination
    rather than a deliberate intent.
    """
    out = text.replace("{candidate_name}", candidate_name)
    # Drop "[Your Name]" and the immediately surrounding glue text like
    # "My name is [Your Name], and " so the result still reads cleanly.
    out = out.replace("My name is [Your Name], and ", "")
    out = out.replace("[Your Name]", "TalentScout Recruiting")
    return out


def _render_outreach(draft: dict, shortlist: list):
    """Render a single outreach draft (subject, InMail, email, hooks)."""
    cand_name = next(
        (c["candidate_name"] for c in shortlist if c["profile_id"] == draft["candidate_id"]),
        "Candidate",
    )

    st.markdown(eyebrow(f"OUTREACH · {cand_name.upper()}"), unsafe_allow_html=True)

    subject = _substitute_name_placeholders(draft.get("subject", ""), cand_name)
    inmail = _substitute_name_placeholders(draft.get("linkedin_inmail", ""), cand_name)
    email = _substitute_name_placeholders(draft.get("email_body", ""), cand_name)
    hooks = draft.get("personalization_hooks", [])

    inmail_html = inmail.replace("\n", "<br/>")
    email_html = email.replace("\n", "<br/>")

    hooks_html = ""
    if hooks:
        hooks_html = (
            '<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
            'letter-spacing:0.08em; text-transform:uppercase; color:#6b6963; '
            'margin-top:1rem;">PERSONALIZATION HOOKS</div>'
            '<ul style="margin-top:0.4rem; color:#6b6963; font-size:0.88rem;">'
            + "".join(f"<li>{h}</li>" for h in hooks)
            + "</ul>"
        )

    st.markdown(
        f'<div class="card">'
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
        f'letter-spacing:0.08em; color:#9a978f;">SUBJECT</div>'
        f'<div style="font-weight:600; font-size:1.05rem; margin-bottom:1rem;">'
        f'{subject}</div>'
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
        f'letter-spacing:0.08em; color:#9a978f;">LINKEDIN INMAIL ({len(inmail)} CHARS)</div>'
        f'<div style="margin: 0.4rem 0 1rem 0; line-height:1.6;">{inmail_html}</div>'
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.72rem; '
        f'letter-spacing:0.08em; color:#9a978f;">EMAIL BODY ({len(email)} CHARS)</div>'
        f'<div style="margin-top:0.4rem; line-height:1.6;">{email_html}</div>'
        f'{hooks_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Sourcing tab
# ============================================================

def _render_sourcing(sourcing: dict | None, merge_audit: list):
    if not sourcing:
        st.info("Sourcing has not run yet.")
        return

    st.markdown(eyebrow("SOURCING SUMMARY"), unsafe_allow_html=True)

    # ---- Source-by-source breakdown ----
    col1, col2, col3, col4 = st.columns(4)
    raw = sourcing.get("raw_counts", {})
    with col1:
        st.metric("LinkedIn", raw.get("linkedin", 0))
    with col2:
        st.metric("Naukri", raw.get("naukri", 0))
    with col3:
        st.metric("ATS", raw.get("ats", 0))
    with col4:
        st.metric("After dedup", sourcing.get("n_after_dedup", 0))

    # ---- Funnel narrative ----
    total_raw = sum(raw.values())
    n_normalized = sourcing.get("n_normalized", total_raw)
    n_after_dedup = sourcing.get("n_after_dedup", 0)
    n_merges = sourcing.get("n_merges", 0)

    st.markdown(
        f'<div class="card-beige" style="margin-top:1rem;">'
        f'<div style="font-family:JetBrains Mono, monospace; font-size:0.78rem; color:#6b6963; '
        f'line-height:1.7;">'
        f'<strong style="color:#0a0a0a;">{total_raw}</strong> raw profiles across 3 sources → '
        f'<strong style="color:#0a0a0a;">{n_normalized}</strong> normalized → '
        f'<strong style="color:#0a0a0a;">{n_after_dedup}</strong> unique '
        f'({n_merges} duplicate{"s" if n_merges != 1 else ""} merged).'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ---- Per-source counts as visual tags ----
    st.markdown(eyebrow("RAW COUNTS PER SOURCE"), unsafe_allow_html=True)
    source_pills = '<div style="display:flex; gap:0.5rem; margin-bottom:1.5rem;">'
    palette = {
        "linkedin": ("#dde7f5", "#1d4ed8"),
        "naukri":   ("#fde2e2", "#dc2626"),
        "ats":      ("#dde7d3", "#15803d"),
    }
    for src, count in raw.items():
        bg, border = palette.get(src.lower(), ("#f5f3ec", "#9a978f"))
        source_pills += (
            f'<div style="background:{bg}; border:1px solid {border}; border-radius:10px; '
            f'padding:0.5rem 0.9rem; flex:1; text-align:center;">'
            f'<div style="font-family:JetBrains Mono, monospace; font-size:0.7rem; '
            f'letter-spacing:0.05em; color:{border};">{src.upper()}</div>'
            f'<div style="font-size:1.4rem; font-weight:800; color:#0a0a0a; margin-top:0.2rem;">'
            f'{count}</div>'
            f'<div style="font-size:0.72rem; color:#6b6963;">raw profiles</div>'
            f'</div>'
        )
    source_pills += '</div>'
    st.markdown(source_pills, unsafe_allow_html=True)

    # ---- Merge audit ----
    if merge_audit:
        st.markdown(eyebrow(f"MERGE AUDIT · {len(merge_audit)} ENTRIES"), unsafe_allow_html=True)
        st.caption("Profiles that the dedup agent identified as the same person across sources.")

        with st.expander("View all merge decisions", expanded=False):
            for m in merge_audit[:50]:
                merged_name = m.get("merged_into_name", "?")
                merged_id_short = m.get("merged_into", "")[:8]
                sources = m.get("sources", [])
                source_ids = m.get("source_ids", [])
                n_records = m.get("n_records", len(source_ids))
                reasons = m.get("reasons", [])

                # Source badges
                source_badges = ""
                source_palette = {
                    "linkedin": ("#dde7f5", "#1d4ed8"),
                    "naukri":   ("#fde2e2", "#dc2626"),
                    "ats":      ("#dde7d3", "#15803d"),
                }
                for s, sid in zip(sources, source_ids):
                    bg, border = source_palette.get(s.lower(), ("#f5f3ec", "#9a978f"))
                    source_badges += (
                        f'<span style="background:{bg}; color:{border}; padding:2px 8px; '
                        f'border-radius:4px; font-family:JetBrains Mono, monospace; font-size:0.7rem; '
                        f'margin-right:0.3rem;">{s}:{sid}</span>'
                    )

                reason_str = ", ".join(reasons) if reasons else "name + content similarity"

                st.markdown(
                    f'<div style="padding:0.7rem 0; border-bottom:1px solid #ece8dc;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:baseline;">'
                    f'<div>'
                    f'<strong style="font-size:0.95rem;">{merged_name}</strong> '
                    f'<span style="font-family:JetBrains Mono, monospace; font-size:0.72rem; color:#9a978f;">'
                    f'({merged_id_short}…)</span>'
                    f'</div>'
                    f'<span style="font-family:JetBrains Mono, monospace; font-size:0.72rem; color:#6b6963;">'
                    f'{n_records} record{"s" if n_records != 1 else ""}</span>'
                    f'</div>'
                    f'<div style="margin-top:0.4rem;">{source_badges}</div>'
                    f'<div style="margin-top:0.3rem; font-size:0.78rem; color:#6b6963; font-style:italic;">'
                    f'matched by: {reason_str}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No merges occurred — every profile was unique across sources.")
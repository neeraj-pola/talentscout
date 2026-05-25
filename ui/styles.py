# ui/styles.py
"""Portfolio-inspired theme injected into every page.

Pulled from neerajkumarpola.dev: cream backgrounds, warm beige accent
sections, golden-yellow CTAs and highlights, JetBrains Mono for numbered
section headers and tag pills, massive bold display headings, white
cards with soft shadows.
"""

CSS = """
<style>
/* -----------------------------------------------------------
   Fonts — match the portfolio
   ----------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* -----------------------------------------------------------
   Design tokens
   ----------------------------------------------------------- */
:root {
  --bg-cream:        #fafaf7;
  --bg-beige:        #f5ecd0;
  --bg-beige-soft:   #faf3e0;
  --surface-white:   #ffffff;

  --text-primary:    #0a0a0a;
  --text-secondary:  #6b6963;
  --text-muted:      #9a978f;

  --accent-yellow:        #fbd35a;
  --accent-yellow-soft:   #fef3c7;
  --accent-yellow-hover:  #f5c63f;
  --accent-orange:        #d97706;

  --color-success:   #15803d;
  --color-warn:      #b45309;
  --color-danger:    #b91c1c;

  --border-soft:     #ece8dc;
  --shadow-card:     0 1px 2px rgba(10, 10, 10, 0.04),
                     0 8px 24px rgba(10, 10, 10, 0.06);
  --shadow-card-hover: 0 2px 4px rgba(10, 10, 10, 0.06),
                       0 14px 36px rgba(10, 10, 10, 0.10);

  --radius-sm:  6px;
  --radius-md:  14px;
  --radius-lg:  22px;
  --radius-pill: 999px;
}

/* -----------------------------------------------------------
   Streamlit root overrides
   ----------------------------------------------------------- */
.stApp {
  background: var(--bg-cream);
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  color: var(--text-primary);
}

.block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 4rem !important;
  max-width: 1200px;
}

/* Hide Streamlit's default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface-white);
  border-right: 1px solid var(--border-soft);
}
section[data-testid="stSidebar"] .stRadio label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: var(--text-secondary);
  letter-spacing: 0.03em;
}
section[data-testid="stSidebar"] h2 {
  font-family: 'Inter', sans-serif;
  font-weight: 800;
}

/* -----------------------------------------------------------
   Typography
   ----------------------------------------------------------- */
h1, h2, h3, h4 {
  font-family: 'Inter', sans-serif;
  letter-spacing: -0.02em;
  color: var(--text-primary);
}
h1 { font-weight: 900; font-size: 3rem; line-height: 1.05; }
h2 { font-weight: 800; font-size: 2.25rem; line-height: 1.1; }
h3 { font-weight: 700; font-size: 1.35rem; line-height: 1.2; }

p, .stMarkdown { color: var(--text-primary); line-height: 1.6; }

/* -----------------------------------------------------------
   Custom helper classes
   ----------------------------------------------------------- */

/* Section eyebrow — "01 / JD INTAKE" mono header */
.section-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--text-secondary);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.section-eyebrow::before {
  content: "";
  display: inline-block;
  width: 28px;
  height: 3px;
  background: var(--accent-yellow);
  border-radius: 2px;
}

/* Yellow highlighter behind text (like your "Pola" highlight) */
.highlight {
  background: linear-gradient(
    180deg,
    transparent 60%,
    var(--accent-yellow) 60%,
    var(--accent-yellow) 92%,
    transparent 92%
  );
  padding: 0 0.18em;
}

/* Hero-style header */
.hero-title {
  font-size: 3.4rem;
  font-weight: 900;
  letter-spacing: -0.025em;
  line-height: 1.02;
  margin: 0.4rem 0 1.2rem 0;
}

/* Stat pill — "98% Accuracy" / "12x Speedup" style */
.stat-pill {
  display: inline-block;
  background: var(--accent-yellow-soft);
  color: var(--text-primary);
  padding: 4px 12px;
  border-radius: var(--radius-pill);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 500;
  margin-right: 6px;
}

/* Status pills */
.status-pill {
  display: inline-block;
  padding: 4px 12px;
  border-radius: var(--radius-pill);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.status-running     { background: #fef3c7; color: #92400e; }
.status-completed   { background: #dcfce7; color: #166534; }
.status-shortlisted { background: #dcfce7; color: #166534; }
.status-closed      { background: #e5e5e5; color: #525252; }
.status-rejected    { background: #fee2e2; color: #991b1b; }
.status-draft       { background: #f3f4f6; color: #4b5563; }
.status-sourcing,
.status-screening,
.status-parsed      { background: #fef3c7; color: #92400e; }

/* Tag — "Python · LLMs · RAG" mono pills */
.tag {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-secondary);
  padding: 2px 8px;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-pill);
  margin: 2px 4px 2px 0;
  background: var(--surface-white);
}
.tag-must {
  background: var(--accent-yellow-soft);
  color: var(--text-primary);
  border-color: var(--accent-yellow);
  font-weight: 600;
}

/* Cards */
.card {
  background: var(--surface-white);
  border-radius: var(--radius-md);
  padding: 1.5rem 1.75rem;
  box-shadow: var(--shadow-card);
  margin-bottom: 1rem;
  transition: box-shadow 0.18s ease;
}
.card:hover { box-shadow: var(--shadow-card-hover); }

.card-beige {
  background: var(--bg-beige-soft);
  border-radius: var(--radius-md);
  padding: 1.5rem 1.75rem;
  margin-bottom: 1rem;
}

/* Special top-pick card (highlighted) */
.top-pick-card {
  background: linear-gradient(135deg, #fff8e1 0%, #fde68a 100%);
  border-radius: var(--radius-lg);
  padding: 2rem 2.25rem;
  box-shadow: 0 10px 40px rgba(251, 211, 90, 0.35);
  margin-bottom: 1.5rem;
}

/* Score bar — horizontal pill */
.score-bar {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
}
.score-bar-track {
  width: 90px;
  height: 6px;
  background: #ece8dc;
  border-radius: 3px;
  overflow: hidden;
}
.score-bar-fill {
  height: 100%;
  border-radius: 3px;
}

/* Evidence quote box */
.evidence-quote {
  border-left: 3px solid var(--accent-yellow);
  padding: 0.4rem 0.85rem;
  font-family: 'Inter', sans-serif;
  font-size: 0.88rem;
  color: var(--text-secondary);
  background: var(--bg-beige-soft);
  border-radius: 0 6px 6px 0;
  margin: 0.4rem 0;
  font-style: italic;
}
.evidence-empty { color: var(--text-muted); font-style: italic; }

/* Cost chip */
.cost-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--surface-white);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-pill);
  padding: 4px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--text-secondary);
}
.cost-chip strong { color: var(--text-primary); font-weight: 600; }

/* Activity log — terminal-ish */
.activity-log {
  background: #1a1a1a;
  border-radius: var(--radius-md);
  padding: 1rem 1.25rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  line-height: 1.55;
  color: #e5e5e5;
  max-height: 320px;
  overflow-y: auto;
}
.activity-log .log-ts   { color: #6b7280; }
.activity-log .log-ag   { color: #fbd35a; }
.activity-log .log-evt  { color: #ffffff; }
.activity-log .log-info { color: #9ca3af; }

/* -----------------------------------------------------------
   Buttons — including st.form_submit_button which uses
   a different selector (.stFormSubmitButton)
   ----------------------------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button,
button[data-testid="stFormSubmitButton"],
div[data-testid="stFormSubmitButton"] button {
  background: var(--accent-yellow) !important;
  color: var(--text-primary) !important;
  border: none !important;
  border-radius: var(--radius-pill) !important;
  padding: 0.55rem 1.4rem !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  box-shadow: 0 1px 3px rgba(10, 10, 10, 0.08) !important;
  transition: all 0.18s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
  background: var(--accent-yellow-hover) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(251, 211, 90, 0.4) !important;
}

.stButton > button:disabled,
.stFormSubmitButton > button:disabled,
div[data-testid="stFormSubmitButton"] button:disabled {
  background: #e5e5e5 !important;
  color: var(--text-muted) !important;
  cursor: not-allowed;
}

/* Catch any leftover Streamlit primary button styling */
button[kind="primary"], button[kind="primaryFormSubmit"] {
  background: var(--accent-yellow) !important;
  color: var(--text-primary) !important;
  border: none !important;
}

/* Secondary button modifier */
button[kind="secondary"] {
  background: var(--surface-white) !important;
  border: 1px solid var(--border-soft) !important;
  color: var(--text-primary) !important;
}

/* -----------------------------------------------------------
   Form widgets
   ----------------------------------------------------------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"], .stDateInput input {
  background: var(--surface-white) !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border-soft) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}

/* Kill any default invalid/required browser outline on inputs */
.stTextInput input:not(:focus),
.stTextArea textarea:not(:focus),
.stNumberInput input:not(:focus),
.stDateInput input:not(:focus) {
  outline: none !important;
  box-shadow: none !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stDateInput input:focus {
  border-color: var(--accent-yellow) !important;
  box-shadow: 0 0 0 3px rgba(251, 211, 90, 0.2) !important;
  outline: none !important;
}

label, .stRadio label p {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}

/* Expander */
details {
  background: var(--surface-white);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-soft);
  margin-bottom: 0.5rem;
}
details summary {
  padding: 0.7rem 1rem;
  cursor: pointer;
  font-weight: 500;
}

/* Metric */
[data-testid="stMetric"] {
  background: var(--surface-white);
  padding: 1rem 1.2rem;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-card);
}

/* Metric value/label colors — fix the disappearing-into-bg issue */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] * {
  color: var(--text-primary) !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] * {
  color: var(--text-secondary) !important;
}

/* Selectbox — force white surface with dark text */
.stSelectbox div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] input {
  background: var(--surface-white) !important;
  color: var(--text-primary) !important;
}
.stSelectbox div[data-baseweb="select"] [data-baseweb="tag"] {
  background: var(--accent-yellow-soft) !important;
}

/* Selectbox open-state dropdown menu */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
  background: var(--surface-white) !important;
  color: var(--text-primary) !important;
}
div[data-baseweb="popover"] li:hover {
  background: var(--accent-yellow-soft) !important;
}

/* Number input — the +/- steppers */
.stNumberInput input,
.stNumberInput div[data-baseweb="input"] {
  background: var(--surface-white) !important;
  color: var(--text-primary) !important;
}
.stNumberInput button {
  background: var(--surface-white) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-soft) !important;
}
.stNumberInput button:hover {
  background: var(--accent-yellow-soft) !important;
}

/* Date input */
.stDateInput input {
  background: var(--surface-white) !important;
  color: var(--text-primary) !important;
}

/* Checkbox label color */
.stCheckbox label, .stCheckbox label p {
  color: var(--text-primary) !important;
}

/* Highlight class on a metric — clip cleanly */
[data-testid="stMetricValue"] .highlight {
  background: transparent;
  padding: 0;
}

/* Code blocks */
code {
  background: var(--bg-beige-soft) !important;
  color: var(--text-primary) !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.85em !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 1px solid var(--border-soft);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 6px 6px 0 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  letter-spacing: 0.04em;
  padding: 8px 16px;
  color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
  background: var(--accent-yellow-soft) !important;
  color: var(--text-primary) !important;
  font-weight: 600;
}

/* Divider */
hr { border-color: var(--border-soft) !important; }

/* Pipeline flowchart container */
.pipeline-flowchart {
  background: var(--surface-white);
  border-radius: var(--radius-md);
  padding: 1.5rem 1rem;
  box-shadow: var(--shadow-card);
  margin-bottom: 1rem;
  overflow-x: auto;
}
/* -----------------------------------------------------------
   Chat (st.chat_message + st.chat_input) — match portfolio palette
   ----------------------------------------------------------- */

/* User chat bubble — warm beige */
div[data-testid="stChatMessage"][data-testid*="user"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: var(--bg-beige) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.8rem 1.1rem !important;
}

/* Assistant chat bubble — soft cream */
div[data-testid="stChatMessage"][data-testid*="assistant"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: var(--surface-white) !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.8rem 1.1rem !important;
}

/* Fallback for any chat message wrapper (Streamlit's class names shift) */
div[data-testid="stChatMessage"] {
  background: var(--surface-white);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-md);
  padding: 0.8rem 1.1rem !important;
  margin-bottom: 0.6rem !important;
}

/* Chat message text — force readable color */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
  color: var(--text-primary) !important;
}

/* Chat avatar — small monochrome circle that matches theme */
div[data-testid="stChatMessage"] img[data-testid="chatAvatarIcon-user"],
div[data-testid="stChatMessage"] img[data-testid="chatAvatarIcon-assistant"],
div[data-testid="stChatMessage"] > div:first-child {
  background: var(--accent-yellow-soft) !important;
  border-radius: 50% !important;
}

/* Chat input — sticky bottom bar, cream surface with yellow focus */
div[data-testid="stChatInput"],
div[data-testid="stChatInputContainer"] {
  background: transparent !important;
  border: none !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInputContainer"] textarea,
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInputContainer"] > div {
  background: var(--surface-white) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: var(--radius-md) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.95rem !important;
  caret-color: var(--accent-orange) !important;
}

div[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInputContainer"] textarea::placeholder {
  color: var(--text-muted) !important;
}

div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stChatInputContainer"] textarea:focus {
  border-color: var(--accent-yellow) !important;
  box-shadow: 0 0 0 3px rgba(251, 211, 90, 0.2) !important;
  outline: none !important;
}

/* Chat input submit button — golden yellow to match the rest */
div[data-testid="stChatInput"] button,
div[data-testid="stChatInputContainer"] button {
  background: var(--accent-yellow) !important;
  color: var(--text-primary) !important;
  border-radius: 50% !important;
  border: none !important;
}

div[data-testid="stChatInput"] button:hover,
div[data-testid="stChatInputContainer"] button:hover {
  background: var(--accent-yellow-hover) !important;
}

/* Kill any red focus ring Streamlit defaults to */
div[data-testid="stChatInput"] *:focus-visible,
div[data-testid="stChatInputContainer"] *:focus-visible {
  outline: none !important;
}
</style>
"""


def inject() -> None:
    """Call once per page (Streamlit re-renders) to inject the theme."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# Tiny helper builders — keeps page code clean
# ============================================================

def eyebrow(text: str) -> str:
    """01 / EDUCATION style mono header."""
    return f'<div class="section-eyebrow">{text}</div>'


def status_pill(status: str) -> str:
    cls = f"status-{status.lower().replace('_', '-')}"
    return f'<span class="status-pill {cls}">{status.replace("_", " ")}</span>'


def stat_pill(text: str) -> str:
    return f'<span class="stat-pill">{text}</span>'


def tag(text: str, must: bool = False) -> str:
    cls = "tag tag-must" if must else "tag"
    return f'<span class="{cls}">{text}</span>'


def cost_chip(usd: float, calls: int) -> str:
    return (
        f'<span class="cost-chip">'
        f'<strong>${usd:.4f}</strong> · {calls} calls'
        f'</span>'
    )


def score_bar(score: float) -> str:
    """Horizontal bar pill colored by score."""
    pct = max(0, min(100, int(score * 100)))
    if score >= 0.7:
        color = "#22c55e"
    elif score >= 0.4:
        color = "#fbd35a"
    elif score >= 0.2:
        color = "#f97316"
    else:
        color = "#dc2626"
    return (
        f'<span class="score-bar">'
        f'<span class="score-bar-track">'
        f'<span class="score-bar-fill" style="width:{pct}%; background:{color};"></span>'
        f'</span>'
        f'<span>{score:.2f}</span>'
        f'</span>'
    )


def evidence_quote(text: str, has_evidence: bool = True) -> str:
    if not text or text == "No evidence found." or not has_evidence:
        return f'<div class="evidence-quote evidence-empty">No verbatim evidence in profile.</div>'
    safe = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    return f'<div class="evidence-quote">"{safe}"</div>'
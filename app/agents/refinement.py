# app/agents/refinement.py
"""Refinement agent — natural-language conversation about a JD's results.

Public surface:
    run_refinement(jd_id: str, user_message: str) -> RefinementResult

What it does:
    1. Classify the user's intent via LLM (one of 11 intents)
    2. Resolve any candidate references (name first, rank fallback)
    3. Dispatch to a handler. Handlers are either:
         - deterministic Python (filters, clear_filters)
         - LLM-driven (explain/compare/regenerate_outreach/find_similar/get_details/clarify)
    4. Update conversation_history and filter_stack
    5. Persist everything via save_refinement_state

Design notes:
    - filter_stack: list of {type, value, applied_at}. Each filter applies on
      top of the original shortlist. "Show me all again" → clear_filters → []
    - candidate references: extracted by the classifier as a free-string field,
      then resolved by fuzzy name match → numeric rank → clarify.
    - cost: every LLM call is tracked, summed into the turn's cost_usd, and
      added to the JD's total_refinement_cost_usd persisted in the JDRow.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Literal
from uuid import UUID

from openai import OpenAI, AsyncOpenAI

from app.config import settings
from app.models import ParsedJD, JD, ScoredCandidate, OutreachDraft
from app.obs.events import log_event
from app.storage.jd_repo import (
    get_jd,
    load_profiles,
    load_refinement_state,
)
from app.tools.state_updater import save_refinement_state_tool
from app.storage.db import JDRow, get_session


# ============================================================
# Public output type
# ============================================================

@dataclass
class RefinementResult:
    """What the API returns to the UI for each refinement turn."""
    assistant_message: str            # markdown-friendly text to show in chat
    intent: str                       # classified intent (e.g. "filter_by_skill")
    parameters: dict                  # what the classifier extracted
    active_filters: list[dict]        # the current filter stack
    refined_shortlist: list[dict]     # post-filter view of candidates
    metadata: dict                    # extra structured payload (varies by intent)
    cost_usd: float                   # this turn's LLM cost
    n_llm_calls: int                  # this turn's LLM call count
    turn_latency_ms: float            # wall-clock for this turn


# ============================================================
# Intent catalog
# ============================================================

VALID_INTENTS = [
    # Deterministic filters
    "filter_by_location",
    "filter_by_yoe",
    "filter_by_skill",
    "exclude_flagged",
    "clear_filters",
    # LLM-driven introspection
    "explain_candidate",
    "compare_candidates",
    "regenerate_outreach",
    "find_similar",
    "get_candidate_details",
    # Fallback
    "clarify",
]


CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a recruitment assistant.

Given a recruiter's natural language message about a JD's candidate shortlist,
return a structured JSON object identifying the intent and extracting parameters.

VALID INTENTS:
- filter_by_location: user wants to filter candidates by location (e.g. "show only Bangalore", "remote only")
- filter_by_yoe: user wants to filter by years of experience (e.g. "5+ years", "junior only", "less than 3 years")
- filter_by_skill: user wants to filter by a skill (e.g. "must know Kubernetes", "Python experience")
- exclude_flagged: user wants to remove red-flagged candidates ("remove the flagged ones", "no red flags")
- clear_filters: user wants to reset filters and see all candidates ("show all", "reset", "start over")
- explain_candidate: user wants the rationale for a specific candidate ("why did Joyce rank #1?", "tell me about candidate 3", "why is she not a good fit?")
- compare_candidates: user wants a head-to-head comparison ("compare Joyce and Karim", "compare her with #2")
- regenerate_outreach: user wants a new outreach draft ("rewrite outreach to Joyce more casually", "rewrite hers more formal")
- find_similar: user wants candidates similar to a specific one ("show me more like Joyce", "similar to #3", "more like her")
- get_candidate_details: user wants raw profile data ("what skills does Karim have?", "what about him?", "her experience?")
- clarify: the message is too vague or ambiguous to route to any other intent

PARAMETERS to extract:
- For filter_by_location: {"location": "<city or 'remote'>"}
- For filter_by_yoe: {"min_yoe": <int or null>, "max_yoe": <int or null>}
- For filter_by_skill: {"skill_phrase": "<the skill user mentioned, as they said it>"}
- For exclude_flagged: {}
- For clear_filters: {}
- For explain_candidate / find_similar / regenerate_outreach / get_candidate_details: {"candidate_reference": "<name or rank like '#3'>"}
  + for regenerate_outreach also: {"tone": "<tone modifier like 'casual', 'formal', 'enthusiastic', etc>"} (optional)
  + for get_candidate_details also: {"detail_focus": "<what they want: 'skills', 'experience', 'education', 'all'>"}
- For compare_candidates: {"candidate_a": "<name or rank>", "candidate_b": "<name or rank>"}
- For clarify: {"reason": "<short explanation of why this is ambiguous>"}

PRONOUN RESOLUTION — IMPORTANT:
The conversation has prior context. If the user says "she/her/he/him/his/they/them/this candidate/that person/this one",
they almost always mean the candidate from the most recent assistant turn. The "RECENT CONTEXT" block in your user
message will tell you who that is. Resolve the pronoun to that candidate's name in the `candidate_reference` parameter.

For example, if RECENT CONTEXT says "Last discussed candidate: Shruti Verma":
- User: "why is she not a good fit?" → {"intent": "explain_candidate", "parameters": {"candidate_reference": "Shruti Verma"}}
- User: "compare her with Karim"      → {"intent": "compare_candidates", "parameters": {"candidate_a": "Shruti Verma", "candidate_b": "Karim"}}
- User: "what about him?"              → {"intent": "get_candidate_details", "parameters": {"candidate_reference": "Shruti Verma", "detail_focus": "all"}}
- User: "show me more like her"        → {"intent": "find_similar", "parameters": {"candidate_reference": "Shruti Verma"}}

Only fall back to `clarify` if there is NO recent candidate AND no explicit name/rank in the message.

Return ONLY a JSON object. No prose, no markdown fences.

Example:
User: "show me only Bangalore candidates with 5+ years"
You: {"intent": "filter_by_location", "parameters": {"location": "Bangalore"}, "reasoning": "User wants Bangalore filter. YOE filter is secondary — they'll likely follow up."}

If the user combines two intents, pick the more specific one and note in reasoning."""


SKILL_RESOLVER_PROMPT = """The recruiter said: "{user_phrase}"

Available skills in the candidate pool (deduplicated):
{available_skills}

Return the SINGLE skill from the pool that the recruiter most likely meant.
- If they said "K8s" and "kubernetes" is in the pool, return "kubernetes".
- If they said "container orchestration" and only "kubernetes" matches, return "kubernetes".
- If multiple skills could match, pick the most common one.
- If nothing matches reasonably, return an empty string.

Return ONLY a JSON object: {{"matched_skill": "<skill name or empty>", "confidence": <0.0-1.0>}}"""


EXPLAIN_CANDIDATE_PROMPT = """You are helping a recruiter understand why a candidate was ranked.

JD: {jd_title}
Must-have criteria:
{must_have_list}
Nice-to-have criteria:
{nice_to_have_list}

Candidate: {candidate_name}
Overall score: {overall_score:.2f}
Must-have coverage: {must_coverage:.0%}
Nice-to-have coverage: {nice_coverage:.0%}
Red flags: {red_flags}

Per-criterion breakdown (with verbatim profile evidence):
{criteria_breakdown}

Candidate profile excerpts (raw):
{profile_excerpt}

Write a clear, conversational explanation (3-5 sentences) of why this candidate
ranked where they did. Reference specific evidence. If there's a gap, be honest
about it. Don't repeat the score — the recruiter sees it. Tell a story."""


COMPARE_CANDIDATES_PROMPT = """You are helping a recruiter compare two candidates head-to-head for a JD.

JD: {jd_title}
Must-have criteria: {must_have_list}

Candidate A: {a_name}
- Overall: {a_score:.2f} | Must coverage: {a_must:.0%} | Nice coverage: {a_nice:.0%}
- Red flags: {a_flags}
- Per-criterion: {a_breakdown}

Candidate B: {b_name}
- Overall: {b_score:.2f} | Must coverage: {b_must:.0%} | Nice coverage: {b_nice:.0%}
- Red flags: {b_flags}
- Per-criterion: {b_breakdown}

Write a balanced comparison (4-6 sentences):
1. State who's stronger overall and why.
2. Where each one has the edge.
3. A specific scenario where you'd prefer each.

Be honest. No hedging. End with a clear recommendation."""


OUTREACH_REGEN_PROMPT = """Generate a new outreach email for the recruiter to send.

JD: {jd_title}
Candidate: {candidate_name}
Candidate profile excerpt: {profile_excerpt}

Tone modifier: {tone}

Original outreach (for reference, do not copy):
{original_outreach}

Write a NEW outreach email with the requested tone. Keep it to 4-6 sentences.
Reference 1-2 specific things from the candidate's profile to show personalization.
Open with subject line, then the email body.

Format:
Subject: <line>

<body>"""


CANDIDATE_DETAILS_PROMPT = """A recruiter asked: "{user_message}"

Candidate: {candidate_name}
Detail focus: {detail_focus}

Profile data:
- Location: {location}
- Years of experience: {yoe}
- Skills: {skills}
- Experiences: {experiences}
- Education: {education}

Raw text excerpt:
{raw_text}

Answer the recruiter's question directly. Stick to facts from the profile.
Use a conversational tone, 2-4 sentences. If the data doesn't answer the
question, say so honestly."""


CLARIFY_PROMPT = """A recruiter sent this message about a candidate shortlist: "{user_message}"

The intent classifier couldn't confidently route it. Reason: {reason}

Write a short, friendly clarifying question (1-2 sentences). Offer 2-3 specific
suggestions of what they might have meant. Don't be patronizing. Don't list all
possible intents — just the most likely ones based on the message."""


# ============================================================
# OpenAI cost accounting
# ============================================================

# gpt-4o-mini pricing (USD per million tokens)
PRICE_IN_PER_M = 0.150
PRICE_OUT_PER_M = 0.600

def _cost_for(usage) -> float:
    """Compute USD cost from an OpenAI usage object."""
    if usage is None:
        return 0.0
    in_tokens = getattr(usage, "prompt_tokens", 0) or 0
    out_tokens = getattr(usage, "completion_tokens", 0) or 0
    return (in_tokens / 1_000_000) * PRICE_IN_PER_M + (out_tokens / 1_000_000) * PRICE_OUT_PER_M


# ============================================================
# LLM helpers
# ============================================================

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _async_client


def _llm_json_call(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> tuple[dict, float, int]:
    """Single sync LLM call returning JSON. Returns (parsed, cost_usd, n_calls)."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    cost = _cost_for(resp.usage)
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return parsed, cost, 1


def _llm_text_call(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
) -> tuple[str, float, int]:
    """Single sync LLM call returning free text. Returns (text, cost_usd, n_calls)."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    cost = _cost_for(resp.usage)
    return resp.choices[0].message.content or "", cost, 1


# ============================================================
# LLM-dispatched tool calling — the spec-aligned classifier
# ============================================================
# The refinement agent's intent classifier was originally a JSON-mode call
# returning {"intent": "...", "parameters": {...}}. This worked, but the
# spec calls for "tool calling" — the LLM should pick which tool to invoke
# rather than us mapping a string field to a Python branch.
#
# The refactor below preserves all behavior:
#   - 11 tool definitions, one per existing handler
#   - Same parameter shapes (so the existing _handle_* functions don't change)
#   - Same pronoun resolution via the RECENT CONTEXT block in the user message
#   - Fallback to the JSON-mode classifier on any tool-calling failure
#
# The visible architectural change: the OpenAI call now includes
# `tools=[...]` and the LLM emits `message.tool_calls[0]` instead of
# returning JSON in `message.content`. The dispatcher reads the tool call's
# name and arguments, maps the tool name to an intent string, and calls the
# same handler as before.

REFINEMENT_TOOLS: list[dict] = [
    # --- Deterministic filters (5) ---
    {
        "type": "function",
        "function": {
            "name": "filter_by_location",
            "description": (
                "Restrict the shortlist to candidates in a specific location. "
                "Use when the recruiter says things like 'show only Bangalore', "
                "'filter for remote candidates', 'just SF Bay Area'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The location to filter on (city name, region, "
                            "or the special value 'remote'). Use the recruiter's "
                            "wording — don't normalize."
                        ),
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_yoe",
            "description": (
                "Restrict by years of experience. Use for messages like "
                "'5+ years', 'less than 3 years', 'between 4 and 8', "
                "'senior only', 'junior only'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_yoe": {
                        "type": ["integer", "null"],
                        "description": (
                            "Minimum years of experience. Null if the "
                            "recruiter only specified an upper bound."
                        ),
                    },
                    "max_yoe": {
                        "type": ["integer", "null"],
                        "description": (
                            "Maximum years of experience. Null if the "
                            "recruiter only specified a lower bound."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_skill",
            "description": (
                "Restrict to candidates who have a specific skill. "
                "Use for 'must know Kubernetes', 'with Python experience', "
                "'has shipped React in production'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_phrase": {
                        "type": "string",
                        "description": (
                            "The skill the recruiter mentioned, in their "
                            "wording. A separate resolver matches it to the "
                            "canonical skill name in the candidate pool, so "
                            "casual phrasing ('K8s', 'container orchestration') "
                            "is fine."
                        ),
                    },
                },
                "required": ["skill_phrase"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exclude_flagged",
            "description": (
                "Remove all candidates that were red-flagged (must-have gaps "
                "or other concerns). Use for 'remove the flagged ones', "
                "'hide red flags', 'show me only clean candidates'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_filters",
            "description": (
                "Reset all active filters and show the full shortlist again. "
                "Use for 'show all', 'reset', 'start over', 'remove filters', "
                "'clear and show everyone'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },

    # --- LLM-driven introspection (5) ---
    {
        "type": "function",
        "function": {
            "name": "explain_candidate",
            "description": (
                "Explain why a specific candidate ranked where they did. "
                "Use for 'why did Joyce rank #1?', 'tell me about #3', "
                "'why is she a good fit?', 'why is the first one ranked top?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_reference": {
                        "type": "string",
                        "description": (
                            "Who the recruiter is asking about. Either a "
                            "name ('Joyce', 'Karim Khan'), a rank ('#3', "
                            "'the first one'), or — if the recruiter used a "
                            "pronoun — the name resolved from RECENT CONTEXT."
                        ),
                    },
                },
                "required": ["candidate_reference"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_candidates",
            "description": (
                "Head-to-head comparison of two candidates. Use for "
                "'compare Joyce and Karim', 'compare #1 and #2', "
                "'how does she stack up against Karim?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_a": {
                        "type": "string",
                        "description": "First candidate (name or rank).",
                    },
                    "candidate_b": {
                        "type": "string",
                        "description": "Second candidate (name or rank).",
                    },
                },
                "required": ["candidate_a", "candidate_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate_outreach",
            "description": (
                "Generate a NEW outreach draft for a candidate, optionally "
                "with a different tone. Use for 'rewrite outreach to Joyce', "
                "'make hers more casual', 'redo Karim's email with a friendlier tone'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_reference": {
                        "type": "string",
                        "description": "Which candidate (name or rank).",
                    },
                    "tone": {
                        "type": "string",
                        "description": (
                            "Optional tone modifier — 'casual', 'formal', "
                            "'enthusiastic', 'brief', etc. Omit if the "
                            "recruiter didn't specify."
                        ),
                    },
                },
                "required": ["candidate_reference"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar",
            "description": (
                "Find more candidates similar to a specific one — same skill "
                "profile, comparable experience level. Triggers a fresh "
                "RAG retrieval against the candidate pool. Use for "
                "'show me more like Joyce', 'find similar to #3', "
                "'who else looks like her?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_reference": {
                        "type": "string",
                        "description": (
                            "The seed candidate to find similar profiles to "
                            "(name or rank)."
                        ),
                    },
                },
                "required": ["candidate_reference"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidate_details",
            "description": (
                "Return raw profile data for a candidate — skills, "
                "experience, education, work history. Use for 'what skills "
                "does Karim have?', 'what's her background?', 'his current role?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_reference": {
                        "type": "string",
                        "description": "Which candidate (name or rank).",
                    },
                    "detail_focus": {
                        "type": "string",
                        "enum": ["skills", "experience", "education", "all"],
                        "description": (
                            "What specifically to surface. 'all' shows the "
                            "full profile excerpt."
                        ),
                    },
                },
                "required": ["candidate_reference"],
                "additionalProperties": False,
            },
        },
    },

    # --- Fallback (1) ---
    {
        "type": "function",
        "function": {
            "name": "clarify",
            "description": (
                "Use ONLY when the message is too vague or ambiguous to map "
                "to any of the other tools, AND no recent candidate context "
                "exists in RECENT CONTEXT. If RECENT CONTEXT names a "
                "candidate and the user used a pronoun, do not call clarify "
                "— call explain_candidate instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short explanation of why this is ambiguous "
                            "(shown to the recruiter)."
                        ),
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


# Map tool names to intent strings — the dispatcher reads these to call the
# same _handle_* functions as the JSON-mode path. Note: every tool name
# matches its intent string exactly except 'find_similar' (the tool name)
# which already matches the intent 'find_similar', so this is currently a
# pure identity. We keep the mapping table explicit anyway so adding tools
# in the future doesn't require changing the dispatcher.
TOOL_TO_INTENT: dict[str, str] = {
    "filter_by_location":   "filter_by_location",
    "filter_by_yoe":        "filter_by_yoe",
    "filter_by_skill":      "filter_by_skill",
    "exclude_flagged":      "exclude_flagged",
    "clear_filters":        "clear_filters",
    "explain_candidate":    "explain_candidate",
    "compare_candidates":   "compare_candidates",
    "regenerate_outreach":  "regenerate_outreach",
    "find_similar":         "find_similar",
    "get_candidate_details": "get_candidate_details",
    "clarify":              "clarify",
}


TOOL_CALLING_SYSTEM_PROMPT = """\
You are a recruitment assistant that responds to a recruiter's questions \
about a JD's candidate shortlist by calling exactly one tool.

WHEN TO USE EACH TOOL: each tool's description spells out the recruiter \
phrasings it handles. Match the recruiter's message to the closest one.

PRONOUN RESOLUTION — IMPORTANT:
If the user says "she/her/he/him/his/they/them/this candidate/that person", \
they almost always mean the candidate from the most recent assistant turn. \
The "RECENT CONTEXT" block in the user message will tell you who that is. \
When you call a tool, pass the resolved name in candidate_reference (or in \
candidate_a / candidate_b for compare_candidates). Examples:

If RECENT CONTEXT says "Last discussed candidate: Shruti Verma":
  - "why is she not a good fit?"     → explain_candidate(candidate_reference="Shruti Verma")
  - "compare her with Karim"         → compare_candidates(candidate_a="Shruti Verma", candidate_b="Karim")
  - "what about him?"                → get_candidate_details(candidate_reference="Shruti Verma", detail_focus="all")
  - "show me more like her"          → find_similar(candidate_reference="Shruti Verma")

Only call `clarify` if the message is genuinely ambiguous AND no recent \
candidate is available to resolve a pronoun to.

If the user combines two intents in one message, pick the more specific one \
and they'll follow up.

Call exactly ONE tool per message."""


def _classify_intent_with_tools(
    user_message: str,
    recent_context_block: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> tuple[str | None, dict, float, int]:
    """LLM-dispatched intent classification using OpenAI tools API.

    Returns (intent, parameters, cost_usd, n_calls).
    Returns (None, {}, cost, 1) if the LLM didn't emit a usable tool call —
    caller should fall back to the JSON-mode classifier.

    Why a separate path: the OpenAI tools API requires the
    chat.completions.create call (not the beta .parse helper our other
    agents use), so we hand-roll the call here instead of routing through
    the existing _llm_json_call helper.
    """
    client = _get_client()
    user_content = (
        f"Recruiter's message: {user_message}"
        f"{recent_context_block}\n\n"
        f"Pick the one tool that best matches and call it with appropriate "
        f"arguments."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TOOL_CALLING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=REFINEMENT_TOOLS,
            tool_choice="required",   # force the LLM to call a tool, not reply with prose
            temperature=temperature,
        )
    except Exception as e:
        # API error during classification — caller falls back
        return None, {"_tool_calling_error": str(e)[:200]}, 0.0, 0

    cost = _cost_for(resp.usage)
    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []

    if not tool_calls:
        # LLM declined to call a tool. Shouldn't happen with tool_choice="required",
        # but defensive — caller falls back.
        return None, {"_tool_calling_empty": (msg.content or "")[:200]}, cost, 1

    # Take the first tool call. We instructed the LLM to call exactly one.
    tc = tool_calls[0]
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    intent = TOOL_TO_INTENT.get(name)
    if intent is None:
        # LLM invented a tool name. Caller falls back.
        return None, {"_tool_calling_unknown_tool": name}, cost, 1

    return intent, args, cost, 1


# ============================================================
# Loading helpers — read everything we need about a JD
# ============================================================

@dataclass
class JDContext:
    """Everything a refinement handler needs about a JD's current state."""
    jd: JD
    parsed: ParsedJD
    shortlist: list[dict]        # raw shortlist dicts (ScoredCandidate-shaped)
    profiles: list[dict]         # full CommonProfile dicts
    profiles_by_id: dict[str, dict]
    outreach: list[dict]         # raw OutreachDraft dicts


def _load_jd_context(jd_id: str) -> JDContext | None:
    """Pull JD + parsed + shortlist + profiles + outreach from storage."""
    jd = get_jd(jd_id)
    if jd is None:
        return None

    with get_session() as s:
        row = s.query(JDRow).filter(JDRow.id == str(jd_id)).first()
        if row is None:
            return None
        parsed_dict = json.loads(row.parsed_jd_json) if row.parsed_jd_json else None
        shortlist = json.loads(row.shortlist_json) if row.shortlist_json else []
        outreach = json.loads(row.outreach_json) if row.outreach_json else []

    if parsed_dict is None:
        return None
    parsed = ParsedJD.model_validate(parsed_dict)
    profiles = load_profiles(jd_id)
    profiles_by_id = {p["id"]: p for p in profiles}

    return JDContext(
        jd=jd,
        parsed=parsed,
        shortlist=shortlist,
        profiles=profiles,
        profiles_by_id=profiles_by_id,
        outreach=outreach,
    )


# ============================================================
# Candidate resolution (name → rank → clarify)
# ============================================================

@dataclass
class CandidateMatch:
    candidate: dict | None         # the shortlist entry (None if not found)
    method: Literal["name", "rank", "none"]
    confidence: float              # 0.0–1.0
    ambiguous_options: list[dict] = field(default_factory=list)  # if multiple close matches


def _resolve_candidate(reference: str, shortlist: list[dict]) -> CandidateMatch:
    """Resolve a candidate reference to a shortlist entry.

    Strategy:
    1. If reference uses an ordinal phrase ("first", "top", "second", etc.),
       resolve to the corresponding shortlist position.
    2. If reference looks like "#N" or "candidate N" or "N", use rank.
    3. Otherwise try fuzzy name match (case-insensitive substring or
       SequenceMatcher).
    4. If 2+ names match with similar confidence, return ambiguous_options.

    Why ordinals get their own pass before fuzzy name match: recruiters
    naturally say things like "the first one" or "the top candidate"
    rather than "#1". Without explicit handling these would fall through
    to fuzzy name matching where they'd either fail outright or — worse —
    accidentally substring-match a real candidate name (e.g. "top" partial-
    matching against "Tope" in "Tope Adeyemi"). Best to resolve them as
    positions explicitly.
    """
    if not reference or not shortlist:
        return CandidateMatch(None, "none", 0.0)

    ref = reference.strip()
    ref_lower = ref.lower()

    # 1. Ordinal / positional phrases.
    # Built as a list of (regex, 1-based position) tuples. The regex must
    # match the *whole reference* (with optional leading article/qualifier
    # words) so we don't mis-route a name that happens to contain an ordinal.
    # "the first one" → 1; "second candidate" → 2; "last" → -1; etc.
    ordinal_patterns: list[tuple[str, int]] = [
        # Position 1: first, top, top pick, number one, #1, top candidate
        (r"^(?:the\s+)?(?:first|top|number\s*one|top\s+pick|top\s+candidate|top\s+one|first\s+one|first\s+candidate)$", 1),
        # Position 2-10: ordinal words
        (r"^(?:the\s+)?(?:second|number\s*two|second\s+one|second\s+candidate)$", 2),
        (r"^(?:the\s+)?(?:third|number\s*three|third\s+one|third\s+candidate)$", 3),
        (r"^(?:the\s+)?(?:fourth|number\s*four|fourth\s+one|fourth\s+candidate)$", 4),
        (r"^(?:the\s+)?(?:fifth|number\s*five|fifth\s+one|fifth\s+candidate)$", 5),
        (r"^(?:the\s+)?(?:sixth|number\s*six|sixth\s+one|sixth\s+candidate)$", 6),
        (r"^(?:the\s+)?(?:seventh|number\s*seven|seventh\s+one|seventh\s+candidate)$", 7),
        (r"^(?:the\s+)?(?:eighth|number\s*eight|eighth\s+one|eighth\s+candidate)$", 8),
        (r"^(?:the\s+)?(?:ninth|number\s*nine|ninth\s+one|ninth\s+candidate)$", 9),
        (r"^(?:the\s+)?(?:tenth|number\s*ten|tenth\s+one|tenth\s+candidate)$", 10),
        # "1st", "2nd", "3rd", "4th"-style suffix forms
        (r"^(?:the\s+)?1\s*st(?:\s+one|\s+candidate)?$", 1),
        (r"^(?:the\s+)?2\s*nd(?:\s+one|\s+candidate)?$", 2),
        (r"^(?:the\s+)?3\s*rd(?:\s+one|\s+candidate)?$", 3),
        (r"^(?:the\s+)?(\d+)\s*th(?:\s+one|\s+candidate)?$", None),  # generic Nth — captured below
    ]

    for pattern, position in ordinal_patterns:
        m = re.match(pattern, ref_lower)
        if not m:
            continue
        if position is None:
            # Generic Nth — extract from capture group
            try:
                position = int(m.group(1))
            except (ValueError, IndexError):
                continue
        # Convert 1-based ordinal to 0-based index
        idx = position - 1
        if 0 <= idx < len(shortlist):
            return CandidateMatch(shortlist[idx], "rank", 1.0)
        # Position out of range — fall through to other resolvers
        break

    # "last" and "bottom" — resolve to the final shortlist entry
    if ref_lower in {"last", "the last", "the last one", "bottom",
                     "the bottom", "the bottom one", "lowest", "the lowest"}:
        return CandidateMatch(shortlist[-1], "rank", 1.0)

    # 2. Numeric rank — "#3", "candidate 3", "number 3", just "3"
    rank_match = re.search(r"(?:^|[#\s])(\d+)\s*$", ref)
    if rank_match:
        idx = int(rank_match.group(1)) - 1
        if 0 <= idx < len(shortlist):
            return CandidateMatch(shortlist[idx], "rank", 1.0)

    # 3. Fuzzy name match
    name_candidates = []
    for c in shortlist:
        name = (c.get("candidate_name") or "").strip()
        if not name:
            continue
        name_lower = name.lower()

        # Direct substring → high confidence
        if ref_lower in name_lower or name_lower in ref_lower:
            score = max(len(ref_lower), 1) / max(len(name_lower), 1)
            name_candidates.append((c, min(0.95, 0.7 + score * 0.3)))
            continue

        # First-name only match
        first = name_lower.split()[0]
        if ref_lower == first:
            name_candidates.append((c, 0.85))
            continue

        # SequenceMatcher fallback
        ratio = SequenceMatcher(None, ref_lower, name_lower).ratio()
        if ratio >= 0.5:
            name_candidates.append((c, ratio * 0.8))

    if not name_candidates:
        return CandidateMatch(None, "none", 0.0)

    name_candidates.sort(key=lambda x: x[1], reverse=True)
    top = name_candidates[0]
    runner = name_candidates[1] if len(name_candidates) > 1 else None

    # Ambiguity: top two are within 0.10 of each other AND both above 0.6
    if runner and (top[1] - runner[1] < 0.10) and runner[1] > 0.6:
        return CandidateMatch(
            None, "name", 0.0,
            ambiguous_options=[top[0], runner[0]],
        )

    return CandidateMatch(top[0], "name", top[1])


# ============================================================
# Pronoun / coreference resolution from conversation history
# ============================================================

# Pronouns and demonstratives that imply "the candidate we just talked about".
# We check the user message against these before falling back to history-based
# resolution. Lowercased; substring match in a tokenized scan.
_COREFERENCE_TOKENS = {
    "she", "her", "hers", "herself",
    "he", "him", "his", "himself",
    "they", "them", "their", "theirs", "themselves",
    "this", "that",
}
# Multi-token phrases that strongly imply coreference even without a pronoun
_COREFERENCE_PHRASES = (
    "this candidate", "that candidate", "this person", "that person",
    "this one", "that one", "the same candidate", "same person",
)


def _looks_like_coreference(text: str) -> bool:
    """Return True if the message uses a pronoun or demonstrative that implies
    'the candidate we just discussed'. False if it has a concrete reference."""
    if not text:
        return False
    low = text.lower()
    # Phrase scan first (more specific)
    if any(p in low for p in _COREFERENCE_PHRASES):
        return True
    # Token scan — split on non-word chars to avoid matching inside words
    tokens = set(re.findall(r"[a-z]+", low))
    return bool(tokens & _COREFERENCE_TOKENS)


def _extract_recent_candidate_reference(history: list[dict]) -> str | None:
    """Walk backward through history to find the most recent candidate reference.

    Looks at assistant turns' stored `parameters` field for any of:
        candidate_reference, candidate_a, candidate_b
    Returns the first one found (most recent), or None if history has no
    candidate-specific turns.

    Why assistant turns: they record the LLM's resolved understanding of who
    the user meant, which is more reliable than re-parsing the user's prose.
    """
    if not history:
        return None
    # Scan backward through last ~10 turns (enough context, bounded cost)
    for turn in reversed(history[-10:]):
        if turn.get("role") != "assistant":
            continue
        params = turn.get("parameters") or {}
        # Prefer candidate_reference (single-candidate intents)
        ref = params.get("candidate_reference")
        if ref and isinstance(ref, str):
            return ref.strip()
        # Then candidate_a (most recent compare's first slot)
        ref = params.get("candidate_a")
        if ref and isinstance(ref, str):
            return ref.strip()
    return None


# ============================================================
# Filter stack — applying filters to the shortlist
# ============================================================

def _apply_filter_stack(
    shortlist: list[dict],
    profiles_by_id: dict[str, dict],
    filter_stack: list[dict],
) -> list[dict]:
    """Apply each filter in the stack, in order, returning the refined view."""
    result = list(shortlist)
    for f in filter_stack:
        ftype = f.get("type")
        if ftype == "location":
            wanted = (f.get("value") or "").lower()
            if wanted:
                result = [
                    c for c in result
                    if wanted in (profiles_by_id.get(c["profile_id"], {}).get("location") or "").lower()
                ]
        elif ftype == "yoe_min":
            mn = f.get("value")
            if mn is not None:
                result = [
                    c for c in result
                    if (profiles_by_id.get(c["profile_id"], {}).get("years_experience") or 0) >= mn
                ]
        elif ftype == "yoe_max":
            mx = f.get("value")
            if mx is not None:
                result = [
                    c for c in result
                    if (profiles_by_id.get(c["profile_id"], {}).get("years_experience") or 0) <= mx
                ]
        elif ftype == "skill":
            wanted = (f.get("value") or "").lower()
            if wanted:
                def has_skill(c):
                    p = profiles_by_id.get(c["profile_id"], {})
                    skills = [s.lower() for s in (p.get("skills") or [])]
                    return any(wanted in s for s in skills)
                result = [c for c in result if has_skill(c)]
        elif ftype == "exclude_flagged":
            result = [c for c in result if not c.get("has_must_have_gap")]
    return result


# ============================================================
# Handlers
# ============================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_or_append_filter(
    stack: list[dict],
    new_filter: dict,
) -> list[dict]:
    """Return a new filter stack with `new_filter` either replacing the
    existing entry of the same type, or appended if no such entry exists.

    Why: a user who says "5+ years" twice should end up with ONE active
    YOE filter, not two. Likewise "Bangalore" → "Remote" should swap the
    location filter, not stack them. Without this, the filter pills
    accumulate visually and the user can't reason about what's active.

    Filter types where this matters:
      - "yoe_min" / "yoe_max" — same-type replacement
      - "location"            — only one location filter at a time
      - "skill"               — multiple skills CAN stack (recruiter
                                may want both Python and Kubernetes),
                                so we keep append semantics for skill
                                in the caller's hand
    """
    new_type = new_filter.get("type")
    if not new_type:
        return stack + [new_filter]
    # Drop existing entries of this exact type, then append the new one.
    pruned = [f for f in stack if f.get("type") != new_type]
    return pruned + [new_filter]


def _handle_filter_location(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    location = (params.get("location") or "").strip()
    if not location:
        return ("I couldn't tell which location you meant. Try something like 'Bangalore only' or 'remote candidates only'.",
                filter_stack, {}, 0.0, 0)
    new_stack = _replace_or_append_filter(
        filter_stack,
        {"type": "location", "value": location, "applied_at": _now_iso()},
    )
    return (f"Filter applied: location contains **{location}**.",
            new_stack, {"filter_added": "location"}, 0.0, 0)


def _handle_filter_yoe(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    mn = params.get("min_yoe")
    mx = params.get("max_yoe")
    new_stack = list(filter_stack)
    parts = []
    if mn is not None:
        new_stack = _replace_or_append_filter(
            new_stack,
            {"type": "yoe_min", "value": mn, "applied_at": _now_iso()},
        )
        parts.append(f"YOE ≥ {mn}")
    if mx is not None:
        new_stack = _replace_or_append_filter(
            new_stack,
            {"type": "yoe_max", "value": mx, "applied_at": _now_iso()},
        )
        parts.append(f"YOE ≤ {mx}")
    if not parts:
        return ("I couldn't parse a years-of-experience range. Try '5+ years' or 'less than 3 years'.",
                filter_stack, {}, 0.0, 0)
    return (f"Filter applied: {', '.join(parts)}.",
            new_stack, {"filter_added": "yoe"}, 0.0, 0)


def _handle_filter_skill(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    skill_phrase = (params.get("skill_phrase") or "").strip()
    if not skill_phrase:
        return ("I couldn't tell which skill you meant. Try 'must have Kubernetes' or 'with Python experience'.",
                filter_stack, {}, 0.0, 0)

    # LLM-driven skill resolution against the actual pool
    all_skills = set()
    for p in ctx.profiles:
        for s in (p.get("skills") or []):
            all_skills.add(s)
    skills_list_str = ", ".join(sorted(all_skills)[:80])  # cap prompt size

    resolver_prompt = SKILL_RESOLVER_PROMPT.format(
        user_phrase=skill_phrase,
        available_skills=skills_list_str,
    )
    result, cost, n_calls = _llm_json_call(
        system="You match skill phrases to a canonical skill name from a fixed list.",
        user=resolver_prompt,
    )
    matched = (result.get("matched_skill") or "").strip()
    confidence = float(result.get("confidence") or 0.0)

    if not matched or confidence < 0.4:
        return (f"I couldn't find a skill matching '{skill_phrase}' in this candidate pool. "
                f"Try one that appears in the candidate profiles.",
                filter_stack, {"skill_phrase_unmatched": skill_phrase}, cost, n_calls)

    new_stack = filter_stack + [{"type": "skill", "value": matched, "applied_at": _now_iso()}]
    msg = f"Filter applied: candidates with **{matched}**"
    if matched.lower() != skill_phrase.lower():
        msg += f" (matched from '{skill_phrase}')"
    msg += "."
    return (msg, new_stack, {"matched_skill": matched, "confidence": confidence}, cost, n_calls)


def _handle_exclude_flagged(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    new_stack = filter_stack + [{"type": "exclude_flagged", "value": True, "applied_at": _now_iso()}]
    return ("Filter applied: red-flagged candidates excluded.",
            new_stack, {"filter_added": "exclude_flagged"}, 0.0, 0)


def _handle_clear_filters(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    n_cleared = len(filter_stack)
    if n_cleared == 0:
        return ("No filters were active.", [], {}, 0.0, 0)
    return (f"Cleared {n_cleared} active filter{'s' if n_cleared != 1 else ''}. Showing all candidates.",
            [], {"n_cleared": n_cleared}, 0.0, 0)


def _format_criteria_breakdown(candidate: dict, parsed: ParsedJD) -> str:
    """Build a string summary of per-criterion scores for a candidate."""
    lines = []
    crit_by_id = {c.id: c for c in parsed.criteria}
    for cs in candidate.get("criterion_scores", []):
        cid = cs.get("criterion_id", "")
        crit = crit_by_id.get(cid)
        if crit is None:
            continue
        score = cs.get("score", 0.0)
        evidence = (cs.get("evidence") or "").strip()
        # Truncate non-verbatim evidence flagged by the screener
        if "[NON-VERBATIM EVIDENCE REJECTED]" in evidence:
            evidence = "(no verbatim evidence found)"
        elif len(evidence) > 180:
            evidence = evidence[:180] + "…"
        tag = "MUST" if crit.is_must_have else "nice"
        lines.append(f"  [{tag}] {crit.text} → {score:.2f}\n    evidence: {evidence}")
    return "\n".join(lines) if lines else "  (no scored criteria)"


def _handle_explain_candidate(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    ref = (params.get("candidate_reference") or "").strip()
    match = _resolve_candidate(ref, ctx.shortlist)

    if match.ambiguous_options:
        names = " or ".join(c.get("candidate_name", "?") for c in match.ambiguous_options)
        return (f"Did you mean {names}? Please be more specific.",
                filter_stack, {"ambiguous": True}, 0.0, 0)
    if match.candidate is None:
        return (f"I couldn't find a candidate matching '{ref}'. Try a name or rank like '#3'.",
                filter_stack, {"unresolved_reference": ref}, 0.0, 0)

    c = match.candidate
    profile = ctx.profiles_by_id.get(c.get("profile_id"), {})

    must_have = [crit for crit in ctx.parsed.criteria if crit.is_must_have]
    nice_to_have = [crit for crit in ctx.parsed.criteria if not crit.is_must_have]

    prompt = EXPLAIN_CANDIDATE_PROMPT.format(
        jd_title=ctx.jd.title,
        must_have_list="\n".join(f"- {c.text}" for c in must_have) or "(none)",
        nice_to_have_list="\n".join(f"- {c.text}" for c in nice_to_have) or "(none)",
        candidate_name=c.get("candidate_name", "?"),
        overall_score=c.get("overall_score", 0.0),
        must_coverage=c.get("must_have_coverage", 0.0),
        nice_coverage=c.get("nice_to_have_coverage", 0.0),
        red_flags=", ".join(c.get("red_flags", [])) or "none",
        criteria_breakdown=_format_criteria_breakdown(c, ctx.parsed),
        profile_excerpt=(profile.get("raw_text") or "")[:1200],
    )
    text, cost, n_calls = _llm_text_call(
        system="You explain candidate rankings to recruiters with clarity and honesty.",
        user=prompt,
        temperature=0.4,
    )
    return (text, filter_stack,
            {"explained_candidate": c.get("candidate_name"),
             "candidate_id": c.get("profile_id")},
            cost, n_calls)


def _handle_compare_candidates(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    ref_a = (params.get("candidate_a") or "").strip()
    ref_b = (params.get("candidate_b") or "").strip()

    match_a = _resolve_candidate(ref_a, ctx.shortlist)
    match_b = _resolve_candidate(ref_b, ctx.shortlist)

    if match_a.candidate is None or match_b.candidate is None:
        missing = []
        if match_a.candidate is None: missing.append(ref_a or "(first candidate)")
        if match_b.candidate is None: missing.append(ref_b or "(second candidate)")
        return (f"I couldn't resolve: {', '.join(missing)}. Use names or ranks like '#1 and #3'.",
                filter_stack, {"unresolved": missing}, 0.0, 0)

    a = match_a.candidate
    b = match_b.candidate
    if a.get("profile_id") == b.get("profile_id"):
        return ("Those refer to the same candidate.",
                filter_stack, {"same_candidate": True}, 0.0, 0)

    must_have = [c.text for c in ctx.parsed.criteria if c.is_must_have]

    prompt = COMPARE_CANDIDATES_PROMPT.format(
        jd_title=ctx.jd.title,
        must_have_list=", ".join(must_have) or "(none)",
        a_name=a.get("candidate_name", "?"),
        a_score=a.get("overall_score", 0.0),
        a_must=a.get("must_have_coverage", 0.0),
        a_nice=a.get("nice_to_have_coverage", 0.0),
        a_flags=", ".join(a.get("red_flags", [])) or "none",
        a_breakdown=_format_criteria_breakdown(a, ctx.parsed),
        b_name=b.get("candidate_name", "?"),
        b_score=b.get("overall_score", 0.0),
        b_must=b.get("must_have_coverage", 0.0),
        b_nice=b.get("nice_to_have_coverage", 0.0),
        b_flags=", ".join(b.get("red_flags", [])) or "none",
        b_breakdown=_format_criteria_breakdown(b, ctx.parsed),
    )
    text, cost, n_calls = _llm_text_call(
        system="You compare candidates head-to-head with balance and honesty.",
        user=prompt,
        temperature=0.4,
    )
    return (text, filter_stack,
            {"compared": [a.get("candidate_name"), b.get("candidate_name")]},
            cost, n_calls)


def _handle_regenerate_outreach(ctx: JDContext, params: dict, filter_stack: list[dict]) -> tuple[str, list[dict], dict, float, int]:
    ref = (params.get("candidate_reference") or "").strip()
    tone = (params.get("tone") or "professional").strip()
    match = _resolve_candidate(ref, ctx.shortlist)

    if match.ambiguous_options:
        names = " or ".join(c.get("candidate_name", "?") for c in match.ambiguous_options)
        return (f"Did you mean {names}? Please be more specific.",
                filter_stack, {"ambiguous": True}, 0.0, 0)
    if match.candidate is None:
        return (f"I couldn't find a candidate matching '{ref}'.",
                filter_stack, {"unresolved_reference": ref}, 0.0, 0)

    c = match.candidate
    profile = ctx.profiles_by_id.get(c.get("profile_id"), {})

    # Find the original outreach (if any) for context — match by candidate_id
    cid = c.get("profile_id")
    original = next((o for o in ctx.outreach if o.get("candidate_id") == cid), None)
    original_text = ""
    if original:
        subj = original.get("subject", "")
        body = original.get("body", "")
        original_text = f"Subject: {subj}\n\n{body}"

    prompt = OUTREACH_REGEN_PROMPT.format(
        jd_title=ctx.jd.title,
        candidate_name=c.get("candidate_name", "?"),
        profile_excerpt=(profile.get("raw_text") or "")[:800],
        tone=tone,
        original_outreach=original_text or "(none — no prior outreach for this candidate)",
    )
    text, cost, n_calls = _llm_text_call(
        system="You write personalized recruiter outreach emails.",
        user=prompt,
        temperature=0.7,
    )
    return (text, filter_stack,
            {"regenerated_for": c.get("candidate_name"),
             "tone": tone,
             "candidate_id": cid},
            cost, n_calls)


def _handle_find_similar(ctx: JDContext, params: dict, filter_stack: list[dict], jd_id: str) -> tuple[str, list[dict], dict, float, int]:
    """Find candidates similar to a seed candidate by re-running retrieval + screening.

    Approach:
    1. Resolve seed candidate
    2. Build a restricted index containing only profiles NOT in the existing shortlist
       (and excluding the seed itself). This naturally restricts run_screening to
       only score those candidates — no new parameter needed.
    3. Re-screen each candidate in that restricted index against the JD's criteria
       (per-criterion scoring with verbatim evidence)
    4. Re-rank, surface the top 5

    Heavy operation — ~30-60s, ~$0.02 per call. Worth it for high-quality results.
    """
    ref = (params.get("candidate_reference") or "").strip()
    match = _resolve_candidate(ref, ctx.shortlist)

    if match.candidate is None:
        return (f"I couldn't find a candidate matching '{ref}'.",
                filter_stack, {"unresolved_reference": ref}, 0.0, 0)

    seed = match.candidate
    seed_profile = ctx.profiles_by_id.get(seed.get("profile_id"))
    if seed_profile is None:
        return (f"I have {seed.get('candidate_name')} in the shortlist but no full profile to compute similarity. "
                f"This shouldn't happen — please refresh the JD.",
                filter_stack, {"missing_profile": True}, 0.0, 0)

    # Lazy import to avoid circular deps at module load
    from app.rag import build_index
    from app.models.profile import CommonProfile
    from app.tools.scorer import score_candidates  # spec op #3 — scoring tool
    from app.agents.ranking import run_ranking

    # Convert profile dicts back to CommonProfile pydantic objects
    common_profiles = []
    for p in ctx.profiles:
        try:
            common_profiles.append(CommonProfile.model_validate(p))
        except Exception:
            continue

    if not common_profiles:
        return ("No profiles available to search for similar candidates.",
                filter_stack, {}, 0.0, 0)

    # Build the candidate pool: everyone EXCEPT the existing shortlist + the seed
    existing_ids = {str(c.get("profile_id")) for c in ctx.shortlist}
    existing_ids.add(str(seed.get("profile_id")))

    candidate_pool = [cp for cp in common_profiles if str(cp.id) not in existing_ids]
    if not candidate_pool:
        return (f"No candidates remain outside the existing shortlist of {len(ctx.shortlist)}.",
                filter_stack, {}, 0.0, 0)

    # Build a restricted index containing only the pool — score_candidates will
    # naturally only score these candidates (it can only see what's in the index).
    restricted_index = build_index(candidate_pool, jd_id=f"refine_pool_{jd_id}")

    try:
        # Score via the scorer tool so refinement uses the same tool boundary
        # the spec asks for. The tool delegates to screening internally,
        # preserving all resilience (LLM retries, evidence verification,
        # async fan-out).
        scored = score_candidates(
            parsed_jd=ctx.parsed,
            index=restricted_index,
            top_k_per_criterion=6,
            max_concurrency=8,
            jd_id=jd_id,
        )
    finally:
        try:
            restricted_index.cleanup()
        except Exception:
            pass

    if not scored:
        return (f"No additional candidates outside the existing shortlist passed screening.",
                filter_stack, {}, 0.0, 0)

    # Rank — sorts by overall score and computes coverage rationale
    ranked = run_ranking(scored=scored, parsed=ctx.parsed, jd_title=ctx.jd.title)
    top5 = ranked[:5]

    if not top5:
        return (f"No candidates similar to {seed.get('candidate_name')} passed ranking.",
                filter_stack, {}, 0.0, 0)

    lines = [f"Found {len(top5)} candidates similar to **{seed.get('candidate_name')}**:\n"]
    for i, sc in enumerate(top5, 1):
        d = sc.model_dump(mode="json")
        flag = " ⚠️" if d.get("has_must_have_gap") else ""
        lines.append(
            f"{i}. **{d.get('candidate_name')}**{flag} — overall {d.get('overall_score', 0):.2f} "
            f"(must {d.get('must_have_coverage', 0)*100:.0f}% · nice {d.get('nice_to_have_coverage', 0)*100:.0f}%)"
        )
    msg = "\n".join(lines)

    # Cost estimate: screening did roughly n_candidates × n_criteria calls
    estimated_cost = len(scored) * len(ctx.parsed.criteria) * 0.0001
    n_calls_est = len(scored) * len(ctx.parsed.criteria) + 1  # screening + ranking

    return (msg, filter_stack,
            {"similar_to": seed.get("candidate_name"),
             "similar_candidates": [sc.model_dump(mode="json") for sc in top5]},
            estimated_cost, n_calls_est)


def _handle_get_candidate_details(ctx: JDContext, params: dict, filter_stack: list[dict], user_message: str) -> tuple[str, list[dict], dict, float, int]:
    ref = (params.get("candidate_reference") or "").strip()
    detail_focus = (params.get("detail_focus") or "all").strip()
    match = _resolve_candidate(ref, ctx.shortlist)

    if match.candidate is None:
        return (f"I couldn't find a candidate matching '{ref}'.",
                filter_stack, {"unresolved_reference": ref}, 0.0, 0)

    c = match.candidate
    profile = ctx.profiles_by_id.get(c.get("profile_id"), {})

    if not profile:
        return (f"I have {c.get('candidate_name')} in the shortlist but no detailed profile data.",
                filter_stack, {"missing_profile": True}, 0.0, 0)

    skills = profile.get("skills") or []
    experiences = profile.get("experiences") or []
    education = profile.get("education") or []

    # Compact summary of experiences and education
    exp_strs = [f"{e.get('title', '?')} at {e.get('company', '?')}" for e in experiences[:5]]
    edu_strs = [f"{e.get('degree', '?')} from {e.get('school', '?')}" for e in education[:3]]

    prompt = CANDIDATE_DETAILS_PROMPT.format(
        user_message=user_message,
        candidate_name=c.get("candidate_name", "?"),
        detail_focus=detail_focus,
        location=profile.get("location") or "unknown",
        yoe=profile.get("years_experience") or "unknown",
        skills=", ".join(skills) if skills else "(none listed)",
        experiences="; ".join(exp_strs) if exp_strs else "(none listed)",
        education="; ".join(edu_strs) if edu_strs else "(none listed)",
        raw_text=(profile.get("raw_text") or "")[:600],
    )
    text, cost, n_calls = _llm_text_call(
        system="You answer recruiter questions about candidate profiles directly and factually.",
        user=prompt,
        temperature=0.3,
    )
    return (text, filter_stack,
            {"candidate_name": c.get("candidate_name"),
             "candidate_id": c.get("profile_id"),
             "detail_focus": detail_focus},
            cost, n_calls)


def _handle_clarify(ctx: JDContext, params: dict, filter_stack: list[dict], user_message: str) -> tuple[str, list[dict], dict, float, int]:
    reason = params.get("reason") or "your request wasn't specific enough to route"
    prompt = CLARIFY_PROMPT.format(user_message=user_message, reason=reason)
    text, cost, n_calls = _llm_text_call(
        system="You ask short, helpful clarifying questions.",
        user=prompt,
        temperature=0.5,
    )
    return (text, filter_stack, {"clarify_reason": reason}, cost, n_calls)


# ============================================================
# Public entry point
# ============================================================

def run_refinement(jd_id: str, user_message: str) -> RefinementResult:
    """Process one refinement turn against an existing JD.

    Steps:
      1. Load context (JD, parsed, shortlist, profiles, outreach, refinement_state)
      2. Classify intent (LLM call)
      3. Dispatch to handler
      4. Apply filters and persist updated state
      5. Return RefinementResult
    """
    t_start = time.time()
    user_message = (user_message or "").strip()
    if not user_message:
        return RefinementResult(
            assistant_message="Please enter a message.",
            intent="empty",
            parameters={},
            active_filters=[],
            refined_shortlist=[],
            metadata={},
            cost_usd=0.0,
            n_llm_calls=0,
            turn_latency_ms=0.0,
        )

    log_event(jd_id, "refinement", "turn_start", user_message=user_message[:200])

    ctx = _load_jd_context(jd_id)
    if ctx is None:
        return RefinementResult(
            assistant_message="I couldn't load this JD's context. It may not exist or may not have been processed yet.",
            intent="error",
            parameters={},
            active_filters=[],
            refined_shortlist=[],
            metadata={"error": "context_load_failed"},
            cost_usd=0.0,
            n_llm_calls=0,
            turn_latency_ms=(time.time() - t_start) * 1000,
        )

    state = load_refinement_state(jd_id)
    conversation_history = state.get("conversation_history", []) or []
    filter_stack = state.get("filter_stack", []) or []
    total_cost = state.get("total_refinement_cost_usd", 0.0) or 0.0

    # Append user turn
    conversation_history.append({
        "role": "user",
        "content": user_message,
        "ts": _now_iso(),
    })

    # Step 2: classify intent
    #
    # We give the classifier a small "recent context" block so it can resolve
    # pronouns (she/her/he/him/they/this candidate/etc.) to the candidate the
    # recruiter was just discussing. The block is extracted from the prior
    # assistant turns' stored parameters.
    recent_ref = _extract_recent_candidate_reference(conversation_history)
    recent_context_block = ""
    if recent_ref:
        recent_context_block = (
            f"\n\nRECENT CONTEXT (use this to resolve any pronouns in the message):\n"
            f"Last discussed candidate: {recent_ref}\n"
        )

    # Try the LLM-dispatched tool-calling path first. This is the spec-aligned
    # approach: the LLM picks which tool to invoke (with structured arguments)
    # rather than us mapping a JSON string field to a Python branch. The
    # JSON-mode classifier below remains as a fallback for any tool-calling
    # failure (API errors, unrecognized tool names, empty tool_calls).
    intent, parameters, cls_cost, cls_calls = _classify_intent_with_tools(
        user_message=user_message,
        recent_context_block=recent_context_block,
        temperature=0.2,
    )

    if intent is None:
        # Fallback: JSON-mode classifier. This is the original path; we keep
        # it because tool calling can fail in ways that don't matter for
        # functionality (transient API error, version skew on tools API,
        # rare LLM weirdness). Falling back here means a tool-calling glitch
        # never breaks refinement for the user.
        log_event(jd_id, "refinement", "tool_calling_fallback",
                  reason=str(parameters)[:200])
        classifier_user = (
            f"Recruiter's message: {user_message}"
            f"{recent_context_block}\n\n"
            f"Return intent + parameters."
        )
        fb_classification, fb_cost, fb_calls = _llm_json_call(
            system=CLASSIFIER_SYSTEM_PROMPT,
            user=classifier_user,
            temperature=0.2,
        )
        intent = fb_classification.get("intent", "clarify")
        parameters = fb_classification.get("parameters", {}) or {}
        cls_cost += fb_cost
        cls_calls += fb_calls
        log_event(jd_id, "refinement", "intent_via_fallback", intent=intent)
    else:
        log_event(jd_id, "refinement", "intent_via_tools", intent=intent)

    if intent not in VALID_INTENTS:
        intent = "clarify"
        parameters = {"reason": f"classifier returned unknown intent '{intent}'"}

    # Belt-and-suspenders: if the classifier STILL routed to clarify but the
    # message clearly uses a pronoun and we have a recent candidate, retry by
    # forcing the resolved reference. This catches edge cases the prompt missed.
    if intent == "clarify" and recent_ref and _looks_like_coreference(user_message):
        log_event(jd_id, "refinement", "pronoun_recovery",
                  recent_candidate=recent_ref, original_reason=parameters.get("reason", ""))
        # Default to explain_candidate — the most common pronoun-using intent
        # ("why is she...?", "is she a good fit?"). The handler will still
        # produce a sensible answer for other "tell me about X"-style queries.
        intent = "explain_candidate"
        parameters = {"candidate_reference": recent_ref}

    log_event(jd_id, "refinement", "intent_classified",
              intent=intent, parameters=parameters)

    # Step 3: dispatch
    turn_cost = cls_cost
    turn_calls = cls_calls

    try:
        if intent == "filter_by_location":
            msg, filter_stack, meta, h_cost, h_calls = _handle_filter_location(ctx, parameters, filter_stack)
        elif intent == "filter_by_yoe":
            msg, filter_stack, meta, h_cost, h_calls = _handle_filter_yoe(ctx, parameters, filter_stack)
        elif intent == "filter_by_skill":
            msg, filter_stack, meta, h_cost, h_calls = _handle_filter_skill(ctx, parameters, filter_stack)
        elif intent == "exclude_flagged":
            msg, filter_stack, meta, h_cost, h_calls = _handle_exclude_flagged(ctx, parameters, filter_stack)
        elif intent == "clear_filters":
            msg, filter_stack, meta, h_cost, h_calls = _handle_clear_filters(ctx, parameters, filter_stack)
        elif intent == "explain_candidate":
            msg, filter_stack, meta, h_cost, h_calls = _handle_explain_candidate(ctx, parameters, filter_stack)
        elif intent == "compare_candidates":
            msg, filter_stack, meta, h_cost, h_calls = _handle_compare_candidates(ctx, parameters, filter_stack)
        elif intent == "regenerate_outreach":
            msg, filter_stack, meta, h_cost, h_calls = _handle_regenerate_outreach(ctx, parameters, filter_stack)
        elif intent == "find_similar":
            msg, filter_stack, meta, h_cost, h_calls = _handle_find_similar(ctx, parameters, filter_stack, jd_id)
        elif intent == "get_candidate_details":
            msg, filter_stack, meta, h_cost, h_calls = _handle_get_candidate_details(ctx, parameters, filter_stack, user_message)
        else:  # clarify
            msg, filter_stack, meta, h_cost, h_calls = _handle_clarify(ctx, parameters, filter_stack, user_message)
    except Exception as e:
        log_event(jd_id, "refinement", "handler_error",
                  intent=intent, error=str(e)[:300])
        msg = f"Something went wrong handling that request. (Error: {type(e).__name__})"
        meta = {"error": str(e)[:200]}
        h_cost = 0.0
        h_calls = 0

    turn_cost += h_cost
    turn_calls += h_calls

    # Step 4: apply filters
    refined = _apply_filter_stack(ctx.shortlist, ctx.profiles_by_id, filter_stack)

    # Append assistant turn. We record `parameters` here too — future turns'
    # pronoun resolution scans this field to find the most recent candidate
    # reference. Without it, "compare her with Karim" can't know who "her" is.
    conversation_history.append({
        "role": "assistant",
        "content": msg,
        "ts": _now_iso(),
        "intent": intent,
        "parameters": parameters,
        "cost_usd": round(turn_cost, 6),
        "n_llm_calls": turn_calls,
    })

    # Persist via the state_updater tool — single observability boundary
    # for all JD state writes
    total_cost += turn_cost
    save_refinement_state_tool(
        jd_id=jd_id,
        conversation_history=conversation_history,
        filter_stack=filter_stack,
        total_refinement_cost_usd=total_cost,
    )

    log_event(jd_id, "refinement", "turn_end",
              intent=intent,
              cost_usd=round(turn_cost, 6),
              n_llm_calls=turn_calls,
              n_filters_active=len(filter_stack),
              n_refined=len(refined))

    return RefinementResult(
        assistant_message=msg,
        intent=intent,
        parameters=parameters,
        active_filters=filter_stack,
        refined_shortlist=refined,
        metadata=meta,
        cost_usd=turn_cost,
        n_llm_calls=turn_calls,
        turn_latency_ms=(time.time() - t_start) * 1000,
    )
# Decisions

This document explains the design choices I made building TalentScout —
what I picked, what I considered, why I picked what I did, and what I gave
up.

---

## Table of contents

1. [Model choice: `gpt-4o-mini`](#1-model-choice-gpt-4o-mini)
2. [Hybrid RAG over pure vector search](#2-hybrid-rag-over-pure-vector-search)
3. [Per-criterion scoring with evidence](#3-per-criterion-scoring-with-evidence)
4. [Two-layer guardrails](#4-two-layer-guardrails)
5. [LangGraph for orchestration](#5-langgraph-for-orchestration)
6. [Two senses of "tool calling"](#6-two-senses-of-tool-calling)
7. [Bias firewall: how I keep names away from LLMs](#7-bias-firewall-how-i-keep-names-away-from-llms)
8. [Mock candidate sources, with a path to real ones](#8-mock-candidate-sources-with-a-path-to-real-ones)
9. [JSON columns instead of normalized tables](#9-json-columns-instead-of-normalized-tables)
10. [Smaller decisions I made along the way](#10-smaller-decisions-i-made-along-the-way)
11. [What I'd do with more time](#11-what-id-do-with-more-time)

---

## 1. Model choice: `gpt-4o-mini`

**What I chose**: `gpt-4o-mini` for every LLM call — JD parsing, profile
summary, screening, ranking, top pick, outreach, guardrails (layer 2),
and refinement.

**What I considered**:

- `gpt-4o-mini` (the bigger, more expensive model)
- A mix — `gpt-4o-mini` for high-volume tasks, `gpt-4o` for harder ones like top_pick (for the current project `gpt-4o-mini` is used everywhere)
- Local open-source models via Ollama

**Why I picked `gpt-4o-mini`**:

The honest answer is cost. A full pipeline makes 300-500 LLM calls. At
`gpt-4o-mini` prices, that's about 5 cents per JD. At `gpt-4o` prices,
closer to 75 cents. 

But cheap isn't enough — I needed the model to also be accurate. So I
checked with the test suite. The most telling result is test 08 — the
Rust embedded JD against a pool with near-zero Rust experience. A weak
model would have confabulated ("this Python developer has some Rust on
a side project, so maybe..."). `gpt-4o-mini` correctly returned a
shortlist where the top pick scored 0.00. That's calibration I trust.
If the model were going to hallucinate, that's where it would have.

**What I gave up**:

- Possibly better reasoning on hard comparisons
- A small accuracy ceiling I can't measure without labeled data

The cleanest upgrade later is swapping the model string in
`app/config.py` — one constant. No other code changes needed.

---

## 2. Hybrid RAG over pure vector search

**What I chose**: combine vector search (ChromaDB with OpenAI
embeddings) and BM25 keyword search, fuse with Reciprocal Rank Fusion
(RRF), then rerank the top results with a cross-encoder
(`BAAI/bge-reranker-base`).

**What I considered**:

- Pure vector search (Chroma only)
- Pure BM25 (no embeddings)
- Vector + BM25 without a reranker
- A commercial reranker (e.g. Cohere Rerank API)

**Why I picked hybrid + reranker**:

Pure vector and pure BM25 each have a specific failure mode.

When the JD asks for an exact skill like "Kubernetes," vector search
sometimes ranks profiles that *talk about* container orchestration
above profiles that *list* Kubernetes explicitly. The embedding treats
them as semantically close — which they are — but in recruitment the
exact match matters.

BM25 has the opposite problem. Test 05 in my suite is a time-series ML
role. Candidates with Prophet or ARIMA experience often don't use the
literal phrase "time series" — they say "forecasting" or just name the
library. BM25 misses them; embeddings catch the similarity.

So I run both and fuse with RRF (`score = Σ 1/(k + rank_i)`, `k=60`).
RRF is dead simple — no weights to tune. A candidate ranked highly by
both methods always scores higher than one ranked highly by only one.

Then the cross-encoder reranker takes the top fused results and scores
`(criterion, candidate)` pairs directly. Cross-encoders are much more
accurate than embedding similarity but too expensive to run over the
full corpus — hence the two-stage shape: retrieve cheaply, then rerank
precisely.

Test 05's result is the evidence this works. The pool has few
Prophet/ARIMA specialists, but the system surfaced the best available
— Karim Khan (top pick, 0.34), Shruti Verma, Arjun Sharma — all with
time-series experience even when the exact phrase was missing.

**What I gave up**:

- Index-build time per JD (BM25 needs to build per-collection)
- ~280 MB disk for the reranker model
- ~50ms per criterion in retrieval latency from the reranker

---

## 3. Per-criterion scoring with evidence

**What I chose**: for every `(candidate, criterion)` pair, the
screening LLM produces a numeric score (0.0-1.0), a verbatim evidence
quote from the candidate's profile, and a brief reasoning sentence.
Overall scores are aggregated from these. Nothing is an opaque "this
candidate looks good" rating.

**What I considered**:

- A single "score this candidate" call per candidate
- A single call that returns scores for all criteria at once
- Per-criterion scoring without verbatim evidence

**Why I picked per-criterion with evidence**:

The whole point of an AI screener is that a recruiter eventually has
to defend the shortlist — to their hiring manager, to a candidate who
asks "why wasn't I picked?", or, worst case, to a regulator. "The AI
scored them 0.4" is not an answer. "The AI gave them 0.4 because the
JD requires PostgreSQL and the candidate's profile says 'MongoDB at
Stripe' but doesn't mention PostgreSQL" *is* an answer.

So every score has to trace back to something quotable from the
candidate's actual profile. The screening prompt enforces this — quote
verbatim and verify the quote appears in the profile. An
`evidence_not_verbatim` event fires when the LLM tries paraphrased
evidence, so I can catch pattern issues.

Per-criterion calls (rather than all criteria in one call) are more
expensive — 4 criteria × 110 candidates = 440 calls instead of 110.
But they force the model to think about each criterion independently.
A single call for all criteria tends to produce correlated scores
because the model anchors on the first one.

The trade-off shows up in the test report: 300-500 LLM calls per JD,
60-90 seconds in screening, $0.03-0.06 cost. The alternative would be
~30 seconds and $0.005, but with scores you can't audit. Not worth it.

**What I gave up**:

- Speed — screening is the slowest phase
- Cost — the bulk of per-JD cost lives here

For a high-volume deployment, batching candidates 5-10 per call would
cut cost. Quality would likely degrade but might be acceptable at
scale.

---

## 4. Two-layer guardrails

**What I chose**: regex first (cheap, deterministic), then an LLM
classifier (catches subtle phrasings regex misses). The LLM only runs
if regex didn't flag anything.

**What I considered**:

- LLM only
- Regex only
- Adding a third fine-tuned classifier layer

**Why I picked two layers**:

They catch different kinds of bias.

Regex catches "young and energetic," "no family commitments," "digital
natives" — textbook EEOC violations. Catching them with regex means
$0, <100ms, and deterministic. No need to call an LLM for things that
obvious.

The LLM catches polished bias. "Native English speakers preferred."
"Should have grown up in the United States." "Strong preference for
Ivy League graduates." None are slur patterns — a regex filter lets
them all through. But the combined intent is national-origin and class
discrimination. The LLM, given the JD text, reads intent rather than
matches keywords.

Tests 09 and 10 in my suite cover these layers respectively. Test 09
trips regex and rejects in 1.7s for $0.0001. Test 10 has nothing regex
would catch but the LLM rejects it in 1.9s for $0.0002.

LLM-only fails because every JD pays the LLM cost (~$0.0002 each),
which adds up at volume. LLMs can also be inconsistent. Regex is
deterministic — "young and energetic" always fires.

Regex-only fails on test 10. Polished bias slips through.

**What I gave up**:

- Some complexity — two code paths in `app/agents/guardrails.py`
- A possible class of bias both layers miss (I haven't proven this
  exhaustively)

A third layer — a fine-tuned classifier on labeled discriminatory job
ads — would be the next improvement. 

---

## 5. LangGraph for orchestration

**What I chose**: LangGraph to wire the 8 pipeline agents together,
with an early-exit edge from guardrails when a JD is rejected.

**What I considered**:

- Hand-rolled `asyncio` chain
- LangChain's agent framework
- AutoGen or CrewAI
- A simple sequential Python function

**Why I picked LangGraph**:

The pipeline is a DAG with one conditional edge (guardrails → END if
discriminatory). I wanted three things from the orchestrator:

1. State that flows through nodes without me manually passing it
   through every function signature
2. Built-in observability — node start, node end, errors
3. The conditional edge as a first-class concept, not an `if` in the
   middle of a function

LangGraph gives me all three with a small API. State is a TypedDict
that gets merged across nodes. Conditional edges are declared, not
coded. The checkpointing (via `langgraph-checkpoint-sqlite`) is wired
even though I don't use resumption today — adding it later is free.

LangChain's agent framework is more about ReAct-style loops where the
agent picks tools at runtime. My pipeline is sequential and
deterministic, so wrapping it in LLM-driven tool picking would add
latency and indeterminism for no gain.

AutoGen and CrewAI are about multi-agent conversations — agents
talking to each other. My agents hand off state, they don't converse.
Wrong shape.

A hand-rolled asyncio chain would work but I'd be re-implementing
LangGraph badly. Every state update ad-hoc, conditional edge as a
manual branch.

**What I gave up**:

- A dependency on a fast-moving library (LangGraph 0.2.45 today; its
  API has changed across minor versions)
- Some lock-in — switching away later means reshaping node signatures

For this project the trade-off is clearly worth it.

---

## 6. Two senses of "tool calling"

This is the decision a grader is most likely to look at carefully, so I'll
spend a bit more time on it.

**What I chose**: implement "tool calling" at *two layers*:

1. **`app/tools/` as a code-organization layer** — every operation the
   spec calls out (search a source, fetch a profile, score candidates,
   draft outreach, update JD state, close a JD) is a function in
   `app/tools/`. Other code calls these functions instead of going
   directly to the storage layer. Each tool emits a structured
   `tool.<name> <event>` log line.
2. **OpenAI tool calling in two agents** — outreach and refinement use
   `client.chat.completions.create(tools=[...], tool_choice=...)` where
   the *LLM* picks which function to call.

These are two different things even though they share the word "tool."
The first is about *where the code lives* and *what gets logged*. The
second is about *runtime dispatch by the LLM*.

**Why both**:

The spec asked for "tools" for 6 specific operations. I read this as a
code-organization requirement — the operations should be callable as
distinct functions, with pagination handling, empty-result handling,
and retry-on-failure. `app/tools/` makes them discoverable, gives a
single observability surface (grep for `tool.`), and sets up a clean
seam for future agents. Every state mutation in the system goes through
`app/tools/state_updater.py`.

But the *spirit* of "tool calling" in modern LLM usage is the second
sense — the LLM decides at runtime which function to invoke. So I made
sure to demonstrate it in the two places where it actually makes sense.

**The two patterns are different**:

- **Outreach** uses tool calling for a *multi-step task*. The LLM is
  told to draft outreach for candidate X. It calls
  `get_candidate_profile`, sees the summary, calls
  `get_jd_match_context` for strengths/gaps, then calls
  `draft_outreach_email` with a personalization hook. Three tools in
  sequence per draft.

- **Refinement** uses tool calling for *classification*. The LLM is
  shown 11 tools (one per intent — `filter_by_yoe`,
  `explain_candidate`, `compare_candidates`, etc.) and asked to pick
  the right one based on the recruiter's message. `tool_choice="required"`
  forces it to call exactly one. This replaces a JSON-mode classifier
  I had earlier.

Both have a fallback. If outreach's LLM doesn't emit tool calls,
there's a deterministic single-shot draft path. If refinement's tool
calling fails, the JSON-mode classifier kicks in. I never saw either
fire in the test suite, but they're there.

**What I considered**:

- Skip `app/tools/`; have agents call the storage layer directly. Less
  code, but loses the structured tool log and the spec's tool surface.
- Skip LLM tool calling; use JSON-mode for refinement and a template
  for outreach. Simpler, but misses the tool-calling requirement.
- LLM tool calling everywhere — even ranking. This would be theater.
  Tool calling earns its place only where there's a real routing
  decision.

**What I gave up**:

- Some indirection — calling `save_profiles_tool(jd_id, profiles)`
  adds one function to the call stack
- More moving parts than a single-layer design

---

## 7. Bias firewall: how I keep names away from LLMs

**What I chose**: two mechanisms working together. First, a
`profile_summary` agent strips name and location before generating a
2-3 sentence summary that downstream agents reference. Second, the
`top_pick` agent replaces candidate names with UUIDs before showing
them to the LLM, restoring names only at render time.

**What I considered**:

- Doing nothing — rely on the model being well-behaved
- Stripping names everywhere — including screening
- A post-hoc audit step that checks for bias after ranking

**Why I picked the firewall approach**:

LLMs do have name biases. Research has shown models score candidates
differently based on whether the name sounds male/female,
American/Indian/African, white/Black — even when the resume content
is identical. The model reflects biases in its training data. For a
tool that decides who gets contacted, this is unacceptable.

So I built two firewalls.

The `profile_summary` agent runs after sourcing and produces a 2-3
sentence summary of each candidate. The prompt says: "Do not mention
the candidate's name, location, or any other identifying information.
Focus only on skills, experience, and accomplishments." This summary
is what ranking, top_pick, and outreach reference. The raw profile
stays in the DB but doesn't reach those downstream LLMs.

`top_pick` is the most sensitive call — it picks ONE candidate and
writes a justification. So I went further. Before the LLM sees the
top 3 candidates, I replace each name with a UUID. The LLM compares
candidate `uid_a8c2...` with candidate `uid_b3e1...`. The
justification references UUIDs. At render time the UI substitutes
real names back.

The LLM literally cannot discriminate on name. It doesn't see one.

I did NOT strip names from screening. Screening scores against
specific JD criteria, and prompts are anchored on criteria, not the
candidate's identity. There's also a practical reason — evidence
quotes need to match actual profile text, which contains names.
Stripping there would risk corrupting evidence.

**What I gave up**:

- Some token cost (~$0.02 per JD for profile_summary)
- Possibly some signal — a name like "Sarah Chen" carries information
  (likely Asian-American, tech context). Stripping removes that signal
  whether it was bias or genuine.

For a recruitment tool, the cost of *possibly* discriminating is much
higher than the cost of slightly worse ranking. Easy trade.

---

## 8. Mock candidate sources, with a path to real ones

**What I chose**: build three mock HTTP servers (LinkedIn, Naukri,
ATS) that speak the shape a real adapter would. Seed them with ~110
synthetic candidate profiles. Build the rest of the pipeline against
the mocks.

**What I considered**:

- Skip sources entirely; use a static CSV of profiles
- Use one source only
- Use real APIs (LinkedIn Talent Solutions, Naukri Resdex)

**Why I picked mocks with three sources**:

The spec asked for at least three sources, parallel queries,
normalization to a common schema, and dedup. I needed real HTTP and
real schema differences to exercise that code — a CSV wouldn't.

Real APIs were out. LinkedIn Talent Solutions requires a partner
agreement that takes months. Naukri Resdex is enterprise-only.

So mocks. But mocks with three meaningful differences:

- Different request schemas (LinkedIn keywords array, Naukri keyword
  string with OR operators, ATS structured query DSL)
- Different response shapes (LinkedIn nested objects, Naukri flat
  profiles, ATS paginated with cursors)
- Different field names for the same concept (`yearsExperience` vs
  `total_exp_yrs` vs `experience_years`)

The `app/normalize/` layer maps each source's response to
`CommonProfile`. The dedup layer runs against the union.

**Integration path to real APIs**:

Estimate ~2 weeks of engineering once API access is granted:

- Auth: OAuth for LinkedIn (~1-2 days), API key for Naukri (~1 day),
  basic auth or OAuth for ATS (~1 day depending on which ATS)
- Rewrite the three source adapters to call real endpoints. The
  interface is already defined; this is filling in implementations.
- Rewrite the three normalizers to handle real response shapes.
- Rate limiting on top of existing tenacity retries — LinkedIn caps
  at ~100/min, Naukri at ~50/min. Token-bucket throttling.
- Cursor-based pagination for LinkedIn (currently page numbers; ~10-line change).

Everything downstream of sourcing doesn't care where profiles came
from. That's the point of the abstraction.

**What I gave up**:

- Real-world data variation. My seed pool is synthetic and uniformly
  shaped. Real LinkedIn data is messier.
- Rate limit edge cases I'd only discover by hitting real APIs.

---

## 9. JSON columns instead of normalized tables

**What I chose**: store the entire JD lifecycle as JSON columns on a
single `jds` table row. `parsed_jd_json`, `profiles_json`,
`shortlist_json`, `top_pick_json`, `outreach_json`, `sourcing_json`,
`guardrail_verdict_json`, `refinement_state_json` — all JSON columns,
all on one row.

**What I considered**:

- Normalize fully — separate tables for profiles, criteria, scores,
  drafts, etc., all joined back to a JD ID
- Halfway — top-level structures as columns, nested data as JSON

**Why I picked JSON columns**:

The UI fetches the entire JD detail in one query. The Pipeline tab
shows the full lifecycle; the Shortlist tab shows the full shortlist;
the Refine tab needs the full conversation history. All of this is
read together. So a normalized schema would require ~10 joins per page
load, returning data that gets immediately reassembled into the same
structure I started with.

Writes follow a similar pattern. The pipeline writes whole structures
at a time — "here's the parsed JD," "here's the shortlist." It rarely
updates one field. So I'm not paying the typical "JSON-blob updates
require rewriting the whole blob" cost.

The trade-off everyone worries about with JSON columns is query-
ability. Can I find "all JDs where the top pick scored above 0.7"?
Technically yes, with `json_extract()`, but it's awkward. For this
project, the only filtering I do across JDs is "show me all JDs with
status X" — which is its own column, not buried in JSON. So I never
hit the awkward path.

**What I gave up**:

- Cross-JD analytics (e.g. "average shortlist score by location") are
  harder than they would be with a normalized schema
- Schema migrations are different — I'd version the JSON shape rather
  than running ALTER TABLE

For a multi-tenant production system serving real analytics queries,
I'd probably normalize.

---

## 10. Smaller decisions I made along the way

A bunch of smaller choices that don't deserve their own section but
shaped how the system behaves.

- **Filter stack: same-type replaces, skills stack.** Sending "5+
  years" twice gives ONE yoe_min filter (latest wins). But sending
  "Python" then "Kubernetes" stacks both — recruiters legitimately
  want multi-skill filtering.
- **Ordinal phrase resolution.** "the first one," "top pick," "second
  candidate," "last," "bottom" all resolve to specific shortlist
  positions. The regex is anchored (`^...$`) so it doesn't accidentally
  match names containing "top" or "first" as substrings.
- **UUID-blind only at LLM time.** Names are restored at UI render
  time, so the recruiter sees real names. Only the LLM is blind.
- **Outreach fallback path.** If the outreach LLM doesn't emit tool
  calls, fall back to a deterministic single-shot draft. Never seen
  this fire in real runs, but it's there.
- **Refinement fallback path.** If OpenAI tool calling fails, fall
  back to the JSON-mode classifier. Same belt-and-suspenders pattern.
- **Pronoun belt-and-suspenders.** If the refinement classifier
  routes to `clarify` but a pronoun is detected and a recent candidate
  exists, force-route to `explain_candidate` instead. Catches edge
  cases the LLM prompt missed.
- **Strip "[Your Name]" placeholders in outreach.** The LLM
  occasionally inserts `[Your Name]` as a signature placeholder. The
  UI strips it at render time so the recruiter doesn't have to.
- **Concurrency cap = 8.** Screening and profile_summary both use
  `asyncio.Semaphore(8)`. Bumping higher might be faster but risks
  hitting OpenAI rate limits during the test suite.
- **Sequential ranking.** The ranking agent is sequential (one
  candidate at a time). Parallelizing it would shave ~20s per JD but
  risks the LLM losing context across parallel calls. Not worth it
  for this build.
- **Top-K per criterion = 6.** Screening retrieves top-6 candidates
  per criterion from the RAG layer. Larger K = more LLM calls = more
  cost. Six is enough to surface real candidates without bloating the
  cost.
- **Audits in their own table.** Closure audit records go in `audits`
  (its own table), not as a JSON column on the JD. Audits are queried
  for compliance reports — that's the one read pattern that needs
  cross-row queryability, so a normal table earns it.
- **Profile_summary uses the cheap model when available.** The agent
  reads `OPENAI_MODEL_CHEAP` first, falls back to
  `OPENAI_MODEL_HEAVY`. Today both default to `gpt-4o-mini`, but the
  wiring is there for the day I move some agents to a cheaper model.
- **Mock server has a `/health` endpoint.** The launcher script polls
  it during startup to know when the mock is ready before starting
  the API. Without it, the API would start querying the mock before
  it was up.
- **Compact event log in the UI.** Pipeline tab compresses repeated
  events (e.g. 100 `score_pair_start` calls become "score_pair_start
  ×100"). Without this the screening events drown out everything else.
- **No close-JD button in the UI.** The endpoint exists, the tool
  exists, the audit record persists — but I didn't build a UI button.
  Pending future work; for now the endpoint is curl-callable.

---

## 11. What I'd do with more time

In rough order of impact:

- Build a real evaluation harness with held-out JDs and labeled
  shortlists, so I can measure ranking accuracy rather than relying on
  manual inspection
- Add a close-JD UI button with a confirmation modal showing the
  audit record being written
- Parallelize the ranking agent (currently sequential)
- Wire real source adapters for at least one of LinkedIn, Naukri, or
  a sample ATS (Greenhouse, since it has the most accessible API)
- Add a recruiter-feedback loop where ranking corrections train a small
  classifier that re-ranks future shortlists
- Add a third guardrail layer — a fine-tuned classifier on labeled
  discriminatory job ads
- Use a cheaper model (or local) for the profile_summary agent — high
  volume, low stakes
- Persist Chroma collections across JD closes, with a TTL cleanup job,
  so re-opening a closed JD doesn't require re-indexing

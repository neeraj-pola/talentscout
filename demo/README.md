# Demo JDs

Each file in this folder is a job description you can copy into the
TalentScout Intake form to exercise a specific behavior. Open the UI,
click **New JD**, copy the fields from the markdown file into the
form, and submit.

## Suggested demo order

If you have ~10 minutes and want to see the system at its best, run
these three:

1. **`02_backend_engineer_fintech.md`** — clean happy path, strong
   top pick around 0.77. Best demo of normal operation.
2. **`08_niche_rust_embedded.md`** — calibration test, top pick scores
   0.00 (correct behavior). Shows the system refuses to hallucinate.
3. **`09_coded_age_bias.md`** — guardrail rejection in under 2
   seconds. Shows two-layer bias defense.

## After running JD 05 (Time Series), try the Refine tab

`05_time_series_ml_engineer.md` produces a shortlist that's good for
demoing the natural-language Refine feature. After it completes,
switch to the Refine tab and try:

```
show me only candidates with 5+ years
why is the first one a good fit?
compare #1 and #2
clear all filters and show everyone
```

Each turn dispatches via OpenAI tool calling — 11 tools registered,
one per recruiter intent.

## Full list

| File | What it shows |
|------|---------------|
| `01_senior_ml_engineer.md` | Happy path — baseline ML role |
| `02_backend_engineer_fintech.md` | Domain depth — payments specialism |
| `03_staff_platform_engineer.md` | Senior YOE band (8-15 years) |
| `04_mobile_engineer_ios.md` | Mobile discipline coverage |
| `05_time_series_ml_engineer.md` | Hybrid retrieval + best for Refine demo |
| `06_data_engineer_streaming.md` | Cross-domain (Kafka + Spark + dbt) |
| `07_vague_software_engineer.md` | Underspecified JD resilience |
| `08_niche_rust_embedded.md` | Calibration — must not hallucinate |
| `09_coded_age_bias.md` | Regex-layer guardrail rejection |
| `10_polished_nationality_bias.md` | LLM-layer guardrail rejection |

## Cost expectations

Running any single happy-path JD: about $0.03 – $0.08.
Running the two guardrail-rejection JDs: about $0.0002 total.
Running all 10: about $0.40.

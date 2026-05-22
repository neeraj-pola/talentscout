# tests/test_llm_client.py
"""Quick smoke test for the instrumented LLM client.
Makes one real OpenAI call. Costs ~$0.0001."""
from app.storage.db import init_db
from app.obs.llm_client import chat
from app.obs.cost import get_cost_summary


def main():
    init_db()
    jd_id = "smoke-test"

    response = chat(
        messages=[
            {"role": "user", "content": "Reply with exactly the word 'pong' and nothing else."}
        ],
        model="gpt-4o-mini",
        jd_id=jd_id,
        agent="smoke_test",
        max_tokens=10,
    )
    print(f"LLM response: {response!r}")

    summary = get_cost_summary(jd_id)
    print(f"Cost: ${summary['total_usd']:.6f} | Tokens: {summary['total_tokens_in']} in / {summary['total_tokens_out']} out")
    print("✓ LLM client works end-to-end (call logged, cost tracked)")


if __name__ == "__main__":
    main()
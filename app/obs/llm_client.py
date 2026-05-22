# app/obs/llm_client.py
"""Instrumented OpenAI client. Every LLM call goes through here.

Why: gives us free retries, observability, cost tracking, and a single
swap-point if we ever change provider.
"""
import json
import time
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type,
)
from openai import APIError, RateLimitError, APIConnectionError

from app.config import settings
from app.obs.cost import record_cost
from app.obs.events import log_event

_client = OpenAI(api_key=settings.openai_api_key)
T = TypeVar("T", bound=BaseModel)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def chat(
    messages: list[dict],
    model: str | None = None,
    jd_id: str | None = None,
    agent: str = "unknown",
    response_format: Type[BaseModel] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str | BaseModel:
    """Call OpenAI chat. Logs cost + event. If response_format is a Pydantic
    class, return a parsed instance; otherwise return raw string.
    """
    model = model or settings.openai_model_heavy
    t0 = time.time()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    log_event(jd_id, agent, "llm_call_start", model=model, n_messages=len(messages))

    try:
        if response_format is not None:
            # Structured output via Pydantic class
            response = _client.beta.chat.completions.parse(
                **kwargs, response_format=response_format,
            )
            content = response.choices[0].message.parsed
            raw_text = response.choices[0].message.content or ""
        else:
            response = _client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            raw_text = content or ""

        usage = response.usage
        latency_ms = (time.time() - t0) * 1000
        usd = record_cost(
            jd_id=jd_id or "unknown",
            agent=agent,
            model=model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            latency_ms=latency_ms,
        )
        log_event(
            jd_id, agent, "llm_call_end",
            model=model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            usd=round(usd, 6),
            latency_ms=round(latency_ms, 1),
            output_preview=raw_text[:120],
        )
        return content

    except Exception as e:
        log_event(jd_id, agent, "llm_call_error", error=str(e), model=model)
        raise


def embed(texts: list[str], jd_id: str | None = None, agent: str = "embed") -> list[list[float]]:
    """Get embeddings. Logged and cost-tracked."""
    model = settings.openai_embedding_model
    t0 = time.time()
    log_event(jd_id, agent, "embed_start", model=model, n_texts=len(texts))

    response = _client.embeddings.create(model=model, input=texts)
    latency_ms = (time.time() - t0) * 1000

    total_tokens = response.usage.total_tokens
    usd = record_cost(
        jd_id=jd_id or "unknown",
        agent=agent,
        model=model,
        tokens_in=total_tokens,
        tokens_out=0,
        latency_ms=latency_ms,
    )
    log_event(
        jd_id, agent, "embed_end",
        model=model, tokens_in=total_tokens,
        usd=round(usd, 6), latency_ms=round(latency_ms, 1),
    )
    return [d.embedding for d in response.data]
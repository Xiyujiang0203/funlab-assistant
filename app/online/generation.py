"""DeepSeek generation (from test/6llm)."""
from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.online.context_expand import build_context
from app.online.query_rewrite import get_llm

SYSTEM = (
    "你是 FUNLAB 实验室助手。只根据给定检索片段回答，用简洁中文。"
    "若片段不足以回答，明确说不知道，不要编造。"
)


def generate_answer(query: str, docs: list[dict], *, stream: bool = False) -> str | Iterator[str]:
    llm, model = get_llm()
    context = build_context(docs)
    user = f"检索片段：\n{context}\n\n用户问题：{query}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    if not stream:
        resp = llm.chat.completions.create(model=model, messages=messages, temperature=0.2)
        return (resp.choices[0].message.content or "").strip()

    def _gen() -> Iterator[str]:
        events = llm.chat.completions.create(
            model=model, messages=messages, temperature=0.2, stream=True
        )
        for event in events:
            delta = event.choices[0].delta.content or ""
            if delta:
                yield delta

    return _gen()

"""① Query 改写：把口语/过短问题改写成适合检索的 query，并产出关键词与子问题。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

REWRITE_SYSTEM = """你是检索查询规划器。面向 FUNLAB 实验室文档（管理规范、Seafile/ZeroTier 使用指南）。
用户问题可能过短或表述不清，请补全语境，输出 JSON，不要其它文字：
{
  "rewrite": "完整、适合向量检索的问句，补关键词与领域用语",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "sub_queries": ["可独立检索的子问题1", "子问题2"]
}
要求：
- rewrite：保留原意，写成文档风格完整问句（例：原问「怎么调优？」→「RAG 系统中向量检索准确率低，有哪些优化方法？」）
- keywords：2~6 个检索关键词/专有名词/数值（如 ZeroTier、93afae59636b818c、周工时）
- sub_queries：1~3 个拆开的子问题；若原问题已很简单可只给 1 个
- 全部中文，不要解释"""


@dataclass
class QueryPlan:
    original: str
    rewrite: str
    keywords: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)

    @property
    def keyword_query(self) -> str:
        return " ".join(self.keywords).strip()


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def rewrite_query(llm: OpenAI, model: str, query: str) -> str:
    """只做 query 改写，返回改写后的检索句。"""
    plan = plan_query(llm, model, query)
    return plan.rewrite or query.strip()


def plan_query(llm: OpenAI, model: str, query: str) -> QueryPlan:
    """改写 + 关键词 + 子问题，供多路召回使用。"""
    original = query.strip()
    plan = QueryPlan(original=original, rewrite=original)
    if not original:
        return plan
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": original},
            ],
            temperature=0.2,
        )
        data = _parse_json(resp.choices[0].message.content or "")
    except Exception:
        return plan

    rewrite = str(data.get("rewrite") or "").strip() or original
    keywords = [str(x).strip() for x in (data.get("keywords") or []) if str(x).strip()]
    subs = [str(x).strip() for x in (data.get("sub_queries") or []) if str(x).strip()]
    # 去重且避开与原句/改写完全相同
    seen = {original, rewrite}
    unique_subs = []
    for s in subs:
        if s not in seen:
            unique_subs.append(s)
            seen.add(s)
    return QueryPlan(
        original=original,
        rewrite=rewrite,
        keywords=keywords[:6],
        sub_queries=unique_subs[:3],
    )

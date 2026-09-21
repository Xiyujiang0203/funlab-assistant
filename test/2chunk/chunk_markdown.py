"""Classify cleaned Markdown then chunk; write preview md under test/chunk/output."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_INPUT = ROOT / "test" / "1pdf2markdown" / "1.3ocr"
DEFAULT_OUTPUT = HERE / "output"

DOC_TYPES = ("structure", "prose", "semantic", "fixed")

CLASSIFY_SYSTEM = """你是文档切分策略分类器。只输出一个 JSON 对象，不要其它文字。
根据文档前几页内容，判断最适合的切分策略：
- structure: 有清晰 Markdown 标题/章节结构，应按标题切分
- prose: 普通散文/连续叙述，应用递归字符切分
- semantic: 话题频繁切换或质量要求高，应做语义切分
- fixed: 只需快速验证原型，用固定长度切分

JSON 格式：{"doc_type":"structure|prose|semantic|fixed","reason":"一句话原因"}"""


def preview_text(text: str, max_chars: int = 3500) -> str:
    return text[:max_chars]


def classify_doc_type(client: OpenAI, model: str, text: str) -> tuple[str, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": preview_text(text)},
        ],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        doc_type = str(data.get("doc_type", "prose")).lower().strip()
        reason = str(data.get("reason", "")).strip()
    except json.JSONDecodeError:
        doc_type, reason = "prose", f"分类解析失败，回退 prose: {raw[:120]}"
    if doc_type not in DOC_TYPES:
        reason = f"未知类型 {doc_type}，回退 prose"
        doc_type = "prose"
    return doc_type, reason


def split_structure(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    docs = splitter.split_text(text)
    chunks: list[dict] = []
    overflow = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
    for doc in docs:
        meta = dict(doc.metadata)
        content = doc.page_content.strip()
        if not content:
            continue
        if len(content) <= chunk_size:
            chunks.append({"text": content, "meta": meta})
            continue
        for part in overflow.split_text(content):
            if part.strip():
                chunks.append({"text": part.strip(), "meta": meta})
    return chunks


def split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return [{"text": c, "meta": {}} for c in splitter.split_text(text) if c.strip()]


def split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[""],
    )
    return [{"text": c, "meta": {}} for c in splitter.split_text(text) if c.strip()]


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def split_semantic(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Paragraph-first semantic-ish merge; oversized paragraphs fall back to recursive."""
    paras = _paragraphs(text)
    if not paras:
        return []
    chunks: list[dict] = []
    buf = ""
    for para in paras:
        if len(para) > chunk_size:
            if buf:
                chunks.append({"text": buf, "meta": {"strategy": "semantic"}})
                buf = ""
            chunks.extend(
                {
                    "text": c,
                    "meta": {"strategy": "semantic-overflow"},
                }
                for c in RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n", "。", "！", "？", " ", ""],
                ).split_text(para)
                if c.strip()
            )
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append({"text": buf, "meta": {"strategy": "semantic"}})
            if chunk_overlap > 0 and buf:
                tail = buf[-chunk_overlap:]
                buf = f"{tail}\n\n{para}".strip()
                if len(buf) > chunk_size:
                    buf = para
            else:
                buf = para
    if buf:
        chunks.append({"text": buf, "meta": {"strategy": "semantic"}})
    return chunks


STRATEGIES = {
    "structure": split_structure,
    "prose": split_recursive,
    "semantic": split_semantic,
    "fixed": split_fixed,
}


def write_chunk_preview(
    source: Path,
    doc_type: str,
    reason: str,
    chunks: list[dict],
    out_path: Path,
) -> None:
    lines = [
        f"# Chunk Preview · {source.name}",
        "",
        f"- source: `{source.as_posix()}`",
        f"- doc_type: `{doc_type}`",
        f"- reason: {reason}",
        f"- chunk_count: {len(chunks)}",
        "",
        "---",
        "",
    ]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("meta") or {}
        meta_bits = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "-"
        lines.append(f"## Chunk {i}")
        lines.append("")
        lines.append(f"- chars: {len(chunk['text'])}")
        lines.append(f"- meta: {meta_bits}")
        lines.append("")
        lines.append("```markdown")
        lines.append(chunk["text"])
        lines.append("```")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def process_file(
    path: Path,
    out_dir: Path,
    client: OpenAI,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    force_type: str | None,
) -> Path:
    text = path.read_text(encoding="utf-8")
    if force_type:
        doc_type, reason = force_type, "手动指定"
    else:
        doc_type, reason = classify_doc_type(client, model, text)
    chunks = STRATEGIES[doc_type](text, chunk_size, chunk_overlap)
    out_path = out_dir / f"{path.stem}.chunks.md"
    write_chunk_preview(path, doc_type, reason, chunks, out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify + chunk cleaned markdown")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--force-type", choices=DOC_TYPES, default=None)
    args = parser.parse_args()

    if not args.force_type and not args.api_key:
        print("缺少 API key：传 --api-key 或设置 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2

    src = args.input.resolve()
    files = [src] if src.is_file() else sorted(src.glob("*.md"))
    if not files:
        print(f"未找到 markdown：{src}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=args.api_key or "unused", base_url=args.base_url)
    out_dir = args.output_dir.resolve()
    for md in files:
        out = process_file(
            md,
            out_dir,
            client,
            args.model,
            args.chunk_size,
            args.chunk_overlap,
            args.force_type,
        )
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

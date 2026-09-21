"""Call deepseek-flash to clean Markdown tables and list markers."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
MD_DIR = HERE / "markdown"
CLEANED_DIR = HERE / "cleaned"
SYSTEM = """你是 Markdown 整理助手。只输出整理后的完整 Markdown，不要解释。

规则：
1. 把乱码/异常列表符号（如 、•、●、◆、■、○、以及私用区字符）统一改成 Markdown 无序列表 `- ` 或有序 `1. `。
2. 复杂/破碎/单元格错位的表格，重构为更符合人类阅读的格式：优先规范 Markdown 表；若跨行合并、多级表头过复杂，可改为「小标题 + 列表/键值对」，勿丢失信息。
3. 保留标题层级、段落含义、图片引用路径，不要编造原文没有的内容。
4. 去掉无意义空行噪音，保持可读。"""


def split_chunks(text: str, max_chars: int = 8000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?=\n#{1,6} )", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if buf and len(buf) + len(part) > max_chars:
            chunks.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        chunks.append(buf)
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
            continue
        for i in range(0, len(chunk), max_chars):
            final.append(chunk[i : i + max_chars])
    return final


def fix_image_paths(text: str, md_dir: Path, out_dir: Path) -> str:
    """Rewrite image links so they resolve from cleaned/ to markdown/ assets."""

    def rewrite(path: str) -> str:
        path = path.strip()
        if path.startswith(("http://", "https://", "data:")):
            return path
        raw = path[1:-1] if path.startswith("<") and path.endswith(">") else path
        if raw.startswith("../"):
            return path
        rel = Path(os.path.relpath(md_dir / raw, out_dir)).as_posix()
        return f"<{rel}>" if path.startswith("<") else rel

    def repl_angle(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        return f"![{alt}]({rewrite(f'<{path}>')})"

    def repl_plain(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        return f"![{alt}]({rewrite(path)})"

    text = re.sub(r"!\[([^\]]*)\]\(<([^>]+)>\)", repl_angle, text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)<][^)]*)\)", repl_plain, text)
    return text


def refine_text(client: OpenAI, model: str, text: str) -> str:
    outs: list[str] = []
    chunks = split_chunks(text)
    for i, chunk in enumerate(chunks, 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"这是第 {i}/{len(chunks)} 段，请整理：\n\n{chunk}",
                },
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        content = re.sub(r"^```(?:markdown)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content)
        outs.append(content.strip())
    return "\n\n".join(outs).strip() + "\n"


def iter_markdown_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".md":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.md"))
    raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine Markdown with deepseek-flash")
    parser.add_argument("input", type=Path, nargs="?", default=MD_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    args = parser.parse_args()
    if not args.api_key:
        print("缺少 API key：传 --api-key 或设置 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    files = iter_markdown_files(args.input.resolve())
    if not files:
        print("未找到 markdown 文件", file=sys.stderr)
        return 1

    md_dir = args.input.resolve() if args.input.resolve().is_dir() else args.input.resolve().parent
    for md in files:
        print(f"refine: {md}")
        cleaned = refine_text(client, args.model, md.read_text(encoding="utf-8"))
        cleaned = fix_image_paths(cleaned, md_dir, out_dir)
        out = out_dir / md.name
        out.write_text(cleaned, encoding="utf-8")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Replace markdown images with Chinese captions from Qwen-VL."""
from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_INPUT = ROOT / "test" / "1pdf2markdown" / "1.2cleaned"
DEFAULT_OUTPUT = HERE
IMAGE_ROOT = ROOT / "test" / "1pdf2markdown" / "1.1markdown"

IMG_RE = re.compile(r"!\[([^\]]*)\]\((<[^>]+>|[^)\s]+)\)")
PROMPT = (
    "这是文档中的截图。请用一段简洁中文概括图片内容与关键信息"
    "（界面名称、按钮、输入内容、表格要点等）。不要用列表，不要编造看不清的文字。"
)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_image(md_file: Path, raw: str) -> Path | None:
    path = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
    path = path.strip().strip('"').strip("'")
    candidates = [
        (md_file.parent / path).resolve(),
        (IMAGE_ROOT / Path(path).name).resolve() if "/" not in path.replace("\\", "/") else None,
    ]
    # ../markdown/xxx -> ../1.1markdown/xxx
    fixed = path.replace("../markdown/", "../1.1markdown/").replace("..\\markdown\\", "..\\1.1markdown\\")
    candidates.append((md_file.parent / fixed).resolve())
    # basename under 1.1markdown/*/
    name = Path(path).name
    if name:
        candidates.extend(IMAGE_ROOT.rglob(name))
    for c in candidates:
        if c and c.is_file():
            return c
    return None


def caption_image(path: Path, api_key: str, model: str, base_url: str) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url(path)}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "max_tokens": 400,
            "temperature": 0.2,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"vl failed {resp.status_code}: {resp.text[:300]}")
    text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def process_markdown(
    md_path: Path,
    out_dir: Path,
    *,
    api_key: str,
    model: str,
    base_url: str,
) -> Path:
    text = md_path.read_text(encoding="utf-8")
    cache: dict[str, str] = {}

    def repl(m: re.Match[str]) -> str:
        raw = m.group(2)
        img = resolve_image(md_path, raw)
        if img is None:
            return f"（图片缺失：{raw}）"
        key = str(img.resolve())
        if key not in cache:
            print(f"  caption: {img.name}")
            cache[key] = caption_image(img, api_key, model, base_url)
            time.sleep(0.15)
        return f"（图片说明：{cache[key]}）"

    new_text = IMG_RE.sub(repl, text)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / md_path.name
    out_path.write_text(new_text, encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace images with Qwen-VL Chinese captions")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen3-vl-flash")
    args = parser.parse_args()
    if not args.api_key:
        print("缺少 API key：传 --api-key 或设置 DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    src = args.input.resolve()
    files = [src] if src.is_file() else sorted(src.glob("*.md"))
    if not files:
        print(f"未找到 markdown：{src}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    for md in files:
        print(f"process: {md.name}")
        out = process_markdown(
            md,
            out_dir,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
        )
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

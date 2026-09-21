"""Embed chunk preview markdown via DashScope / SiliconFlow.

默认走阿里云百炼 OpenAI 兼容接口：
  model=text-embedding-v4  （官方标注属于 Qwen3-Embedding 系列）

若要用真正的 Qwen/Qwen3-Embedding-0.6B：
  1) SiliconFlow：--provider siliconflow --api-key <硅基流动key>
  2) 百炼模型单元部署后：--model <部署后的 code>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_INPUT = ROOT / "test" / "2chunk" / "output"
DEFAULT_OUTPUT = HERE / "output"

PROVIDERS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "text-embedding-v4",
        "batch_size": 10,
        "env_key": "DASHSCOPE_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "batch_size": 16,
        "env_key": "SILICONFLOW_API_KEY",
    },
}


def parse_chunks_md(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    chunks: list[dict] = []
    pattern = re.compile(
        r"## Chunk (\d+)\n\n- chars: (\d+)\n- meta: ([^\n]*)\n\n```markdown\n(.*?)```",
        re.S,
    )
    for m in pattern.finditer(text):
        chunks.append(
            {
                "chunk_id": int(m.group(1)),
                "chars": int(m.group(2)),
                "meta": m.group(3).strip(),
                "text": m.group(4).strip(),
            }
        )
    return chunks


def embed_batch(
    texts: list[str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    dimensions: int,
    timeout: int = 120,
) -> list[list[float]]:
    url = f"{base_url.rstrip('/')}/embeddings"
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": dimensions,
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"embedding failed {resp.status_code}: {resp.text[:500]}")
    data = sorted(resp.json()["data"], key=lambda x: x["index"])
    return [row["embedding"] for row in data]


def embed_texts(
    texts: list[str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    dimensions: int,
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(
            embed_batch(
                batch,
                api_key=api_key,
                base_url=base_url,
                model=model,
                dimensions=dimensions,
            )
        )
        if i + batch_size < len(texts):
            time.sleep(0.05)
    return vectors


def process_file(
    path: Path,
    out_dir: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    dimensions: int,
    batch_size: int,
) -> Path:
    chunks = parse_chunks_md(path)
    if not chunks:
        raise RuntimeError(f"no chunks parsed: {path}")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(
        texts,
        api_key=api_key,
        base_url=base_url,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.embeddings.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk, vec in zip(chunks, vectors):
            row = {
                "source": path.name,
                "chunk_id": chunk["chunk_id"],
                "chars": chunk["chars"],
                "meta": chunk["meta"],
                "text": chunk["text"],
                "model": model,
                "dimensions": len(vec),
                "embedding": vec,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = out_dir / f"{path.stem}.embeddings.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "source": path.name,
                "model": model,
                "base_url": base_url,
                "dimensions": dimensions,
                "chunk_count": len(chunks),
                "output": out_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks with Qwen embedding API")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=list(PROVIDERS), default="dashscope")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    conf = PROVIDERS[args.provider]
    api_key = args.api_key or os.getenv(conf["env_key"], "") or os.getenv("SILICONFLOW_API_KEY", "")
    base_url = args.base_url or conf["base_url"]
    model = args.model or conf["model"]
    batch_size = args.batch_size or conf["batch_size"]
    if not api_key:
        print(f"缺少 API key：传 --api-key 或设置 {conf['env_key']}", file=sys.stderr)
        return 2

    src = args.input.resolve()
    files = [src] if src.is_file() else sorted(src.glob("*.chunks.md"))
    if not files:
        print(f"未找到 *.chunks.md：{src}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    for path in files:
        out = process_file(
            path,
            out_dir,
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=args.dimensions,
            batch_size=batch_size,
        )
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

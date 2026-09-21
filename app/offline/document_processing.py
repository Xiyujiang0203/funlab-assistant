"""Offline: load knowledge docs (md/txt/pdf)."""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts)


def load_document(path: Path) -> dict | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".html", ".htm"}:
        text = load_text_file(path)
    elif suffix == ".pdf":
        text = load_pdf_text(path)
    else:
        return None
    text = text.strip()
    if not text:
        return None
    return {"source": path.name, "path": str(path), "text": text}


def load_documents(file_paths: list[Path] | None = None) -> list[dict]:
    if file_paths:
        paths = file_paths
    else:
        root = get_settings().knowledge_path
        root.mkdir(parents=True, exist_ok=True)
        paths = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    docs = []
    for p in paths:
        doc = load_document(p)
        if doc:
            docs.append(doc)
    return docs

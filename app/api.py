"""FastAPI app: chat + RAG admin, backed by test/ pipeline."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.admin_service import (
    build_context_preview,
    extract_uploaded_document,
    generate_preview,
    process_document_text,
    process_query_preview,
    rerank_preview,
    retrieve_preview as admin_retrieve_preview,
    split_text_preview,
    store_text_chunks,
    vectorize_texts_preview,
)
from app.config import PROJECT_ROOT, get_settings
from app.ingest_service import ingest
from app.offline.document_processing import load_document
from app.online.pipeline import run_online_retrieval
from app.rag_state import dump_rag_runtime, get_rag_runtime, update_rag_runtime

app = FastAPI(title="Funlab Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""


class RagConfigRequest(BaseModel):
    rag_enabled: bool
    embed_model: str
    rerank_model: str
    chunk_size: int
    chunk_overlap: int
    retrieve_top_k: int
    rerank_top_n: int


class SplitPreviewRequest(BaseModel):
    text: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class RetrievePreviewRequest(BaseModel):
    query: str
    top_k: int = 5


class DocumentUpdateRequest(BaseModel):
    content: str


class AdminTextRequest(BaseModel):
    text: str


class AdminSplitRequest(BaseModel):
    text: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class AdminVectorizeRequest(BaseModel):
    texts: list[str]


class AdminStoreRequest(BaseModel):
    chunks: list[str]
    source: str = "manual"
    rebuild: bool = False


class AdminQueryRequest(BaseModel):
    query: str


class AdminRetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0


class AdminRerankRequest(BaseModel):
    query: str
    hits: list[dict]


class AdminContextRequest(BaseModel):
    hits: list[dict]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_knowledge_file(path_str: str) -> Path:
    root = get_settings().knowledge_path.resolve()
    target = (root / path_str).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文档不存在")
    return target


def _stream_generator(message: str, thread_id: str):
    yield _sse("start", {"thread_id": thread_id})
    try:
        from app.online.generation import generate_answer

        runtime = get_rag_runtime()
        if runtime.rag_enabled:
            result = run_online_retrieval(message)
            yield _sse(
                "tool_call",
                {
                    "tool": "rag_retrieve",
                    "input": {
                        "query": message,
                        "rewrite": result.plan.rewrite,
                        "paths": len(result.path_queries),
                        "hits": len(result.hits),
                    },
                },
            )
            sources = [
                {"source": h.get("source"), "chunk_ids": h.get("chunk_ids")}
                for h in result.hits
            ]
            yield _sse("tool_result", {"content": json.dumps(sources, ensure_ascii=False)})
            tokens = generate_answer(message, result.hits, stream=True)
        else:
            tokens = generate_answer(message, [], stream=True)
        for token in tokens:  # type: ignore[union-attr]
            yield _sse("token", {"text": token})
    except Exception as e:
        yield _sse("error", {"message": str(e)})
    yield _sse("done", {})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_generator(req.message, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id, "Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/rag/config")
async def rag_config():
    settings = get_settings()
    return {
        **dump_rag_runtime(),
        "embed_model_options": ["text-embedding-v4"],
        "rerank_model_options": ["qwen3-rerank"],
        "knowledge_dir": str(settings.knowledge_path),
    }


@app.put("/api/rag/config")
async def update_rag_config(req: RagConfigRequest):
    update_rag_runtime(
        rag_enabled=req.rag_enabled,
        embed_model=req.embed_model,
        rerank_model=req.rerank_model,
        chunk_size=max(50, req.chunk_size),
        chunk_overlap=max(0, req.chunk_overlap),
        retrieve_top_k=max(1, req.retrieve_top_k),
        rerank_top_n=max(1, req.rerank_top_n),
    )
    return dump_rag_runtime() | {"knowledge_dir": str(get_settings().knowledge_path)}


@app.get("/api/rag/documents")
async def list_rag_documents():
    root = get_settings().knowledge_path
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            items.append(
                {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(root)),
                    "size": file_path.stat().st_size,
                }
            )
    return {"documents": items}


@app.post("/api/rag/documents")
async def upload_rag_document(file: UploadFile = File(...)):
    root = get_settings().knowledge_path
    root.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "upload.txt").name
    target = root / filename
    if target.exists():
        target = root / f"{target.stem}_{uuid.uuid4().hex[:8]}{target.suffix}"
    content = await file.read()
    target.write_bytes(content)
    count = ingest(rebuild=False, file_paths=[target])
    return {"ok": True, "file": target.name, "path": str(target.relative_to(root)), "chunks": count}


@app.get("/api/rag/document")
async def get_rag_document(path: str):
    file_path = _resolve_knowledge_file(path)
    doc = load_document(file_path)
    content = (doc or {}).get("text", "")
    return {"name": file_path.name, "path": path, "content": content}


@app.put("/api/rag/document")
async def update_rag_document(path: str, req: DocumentUpdateRequest):
    file_path = _resolve_knowledge_file(path)
    if file_path.suffix.lower() not in {".md", ".txt", ".html", ".htm"}:
        raise HTTPException(status_code=400, detail="当前仅支持修改 txt/md/html 文档")
    file_path.write_text(req.content, encoding="utf-8")
    return {"ok": True, "path": path}


@app.delete("/api/rag/document")
async def delete_rag_document(path: str):
    file_path = _resolve_knowledge_file(path)
    file_path.unlink()
    return {"ok": True, "path": path}


@app.post("/api/rag/preview-split")
async def preview_split(req: SplitPreviewRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    return split_text_preview(req.text, req.chunk_size, req.chunk_overlap)


@app.post("/api/rag/retrieve-preview")
async def retrieve_preview(req: RetrievePreviewRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    return admin_retrieve_preview(req.query, top_k=max(1, req.top_k))


@app.post("/api/rag/rebuild")
async def rebuild_rag():
    count = ingest(rebuild=True)
    return {"ok": True, "chunks": count}


@app.post("/api/admin/offline/document-process")
async def admin_document_process(req: AdminTextRequest):
    return process_document_text(req.text)


@app.post("/api/admin/offline/document-file")
async def admin_document_file(file: UploadFile = File(...)):
    import asyncio

    filename = file.filename or "upload.md"
    content = await file.read()
    try:
        return await asyncio.to_thread(extract_uploaded_document, filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"文件解析失败: {exc}") from exc


@app.post("/api/admin/offline/split")
async def admin_split(req: AdminSplitRequest):
    return split_text_preview(req.text, req.chunk_size, req.chunk_overlap)


@app.post("/api/admin/offline/vectorize")
async def admin_vectorize(req: AdminVectorizeRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    try:
        return vectorize_texts_preview(req.texts)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"向量化失败: {exc}") from exc


@app.post("/api/admin/offline/store")
async def admin_store(req: AdminStoreRequest):
    if not req.chunks:
        raise HTTPException(status_code=400, detail="chunks cannot be empty")
    try:
        return store_text_chunks(req.chunks, source=req.source, rebuild=req.rebuild)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"写入向量数据库失败: {exc}") from exc


@app.post("/api/admin/online/query-process")
async def admin_query_process(req: AdminQueryRequest):
    return process_query_preview(req.query)


@app.post("/api/admin/online/retrieve")
async def admin_retrieve(req: AdminRetrieveRequest):
    return admin_retrieve_preview(req.query, top_k=max(1, req.top_k), score_threshold=max(0.0, req.score_threshold))


@app.post("/api/admin/online/rerank")
async def admin_rerank(req: AdminRerankRequest):
    return rerank_preview(req.query, req.hits)


@app.post("/api/admin/online/context")
async def admin_context(req: AdminContextRequest):
    return build_context_preview(req.hits)


@app.post("/api/admin/online/generate")
async def admin_generate(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    return generate_preview(req.message, thread_id)


frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(frontend_dir / "index.html"))

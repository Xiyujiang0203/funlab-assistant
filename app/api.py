import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import FunlabAssistant
from app.config import PROJECT_ROOT, get_settings
from app.ingest_service import ingest
from app.loaders import load_file
from app.rag_state import dump_rag_runtime, update_rag_runtime
from app.retriever import KnowledgeRetriever
from app.reranker import rerank_hits
from app.splitter import get_text_splitter
from app.text_clean import clean_rag_text

app = FastAPI(title="Funlab Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_assistant: FunlabAssistant | None = None


def get_assistant() -> FunlabAssistant:
    global _assistant
    if _assistant is None:
        _assistant = FunlabAssistant()
    return _assistant


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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _merge_tool_call_args(pending: dict[str, dict], index_map: dict[int, str], chunk) -> None:
    for tc in getattr(chunk, "tool_call_chunks", None) or []:
        idx = tc.get("index", 0)
        tc_id = tc.get("id")
        if tc_id:
            index_map[idx] = tc_id
        else:
            tc_id = index_map.get(idx, f"idx-{idx}")
        entry = pending.setdefault(tc_id, {"tool": "", "args": ""})
        if tc.get("name"):
            entry["tool"] = tc["name"]
        entry["args"] += tc.get("args") or ""

    for tc in getattr(chunk, "tool_calls", None) or []:
        tc_id = tc.get("id") or tc.get("name") or "default"
        entry = pending.setdefault(str(tc_id), {"tool": "", "args": ""})
        if tc.get("name"):
            entry["tool"] = tc["name"]
        args = tc.get("args")
        if isinstance(args, dict) and args:
            entry["args"] = args
        elif isinstance(args, str) and args and not entry["args"]:
            entry["args"] = args


def _parse_tool_input(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"query": raw.strip()}
    return {}


def _resolve_knowledge_file(path_str: str) -> Path:
    root = get_settings().knowledge_path.resolve()
    target = (root / path_str).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文档不存在")
    return target


def _stream_generator(message: str, thread_id: str):
    assistant = get_assistant()
    config = {"configurable": {"thread_id": thread_id}}
    pending_tool_calls: dict[str, dict] = {}
    tool_call_index_map: dict[int, str] = {}
    emitted_tool_calls: set[str] = set()

    yield _sse("start", {"thread_id": thread_id})

    try:
        for chunk, meta in assistant.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="messages",
        ):
            chunk_type = getattr(chunk, "type", None)

            if chunk_type and str(chunk_type).lower().startswith("ai"):
                _merge_tool_call_args(pending_tool_calls, tool_call_index_map, chunk)

            if chunk_type == "tool":
                tool_name = getattr(chunk, "name", "")
                tool_call_id = getattr(chunk, "tool_call_id", "") or tool_name
                pending = pending_tool_calls.get(str(tool_call_id), {})
                tool_input = _parse_tool_input(pending.get("args"))
                if tool_call_id not in emitted_tool_calls and tool_input:
                    yield _sse(
                        "tool_call",
                        {"tool": tool_name or pending.get("tool", ""), "input": tool_input},
                    )
                    emitted_tool_calls.add(str(tool_call_id))

                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                if content:
                    yield _sse("tool_result", {"content": str(content)})

            elif chunk_type == "tool_result" or (
                hasattr(chunk, "content")
                and hasattr(chunk, "tool_call_id")
                and chunk_type != "ai"
            ):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                yield _sse("tool_result", {"content": str(content)})

            elif chunk_type and str(chunk_type).lower().startswith("ai"):
                content = chunk.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and block.get("text"):
                                yield _sse("token", {"text": block["text"]})
                            elif block.get("type") == "tool_use":
                                tool_input = block.get("input") or {}
                                if tool_input:
                                    yield _sse(
                                        "tool_call",
                                        {
                                            "tool": block.get("name", ""),
                                            "input": tool_input,
                                        },
                                    )
                elif isinstance(content, str) and content:
                    yield _sse("token", {"text": content})

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
        "embed_model_options": [
            "BAAI/bge-m3",
            "Pro/BAAI/bge-m3",
        ],
        "rerank_model_options": [
            "BAAI/bge-reranker-v2-m3",
            "Pro/BAAI/bge-reranker-v2-m3",
        ],
        "knowledge_dir": str(settings.knowledge_path),
    }


@app.put("/api/rag/config")
async def update_rag_config(req: RagConfigRequest):
    runtime = update_rag_runtime(
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
            stat = file_path.stat()
            items.append(
                {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(root)),
                    "size": stat.st_size,
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
    docs = load_file(file_path)
    content = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
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
    cleaned = clean_rag_text(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="文本不能为空")
    splitter = get_text_splitter(req.chunk_size, req.chunk_overlap)
    chunks = splitter.split_text(cleaned)
    return {
        "count": len(chunks),
        "chunks": [{"index": i, "text": text} for i, text in enumerate(chunks)],
    }


@app.post("/api/rag/retrieve-preview")
async def retrieve_preview(req: RetrievePreviewRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    retriever = KnowledgeRetriever()
    hits = retriever.search(req.query, top_k=max(1, req.top_k))
    return {"hits": hits}


@app.post("/api/rag/rebuild")
async def rebuild_rag():
    count = ingest(rebuild=True)
    return {"ok": True, "chunks": count}


frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(frontend_dir / "index.html"))

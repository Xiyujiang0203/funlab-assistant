import json
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import FunlabAssistant
from app.config import PROJECT_ROOT

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


frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(frontend_dir / "index.html"))

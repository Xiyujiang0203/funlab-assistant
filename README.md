# Funlab 智能问答助手

FUNLAB 实验室私有知识库 RAG：`web`（Next.js）经 API Route 代理调用 `app`（FastAPI）。

流水线来自 `test/`：文档入库 → 改写/多路召回/RRF → 重排 → 上下文扩展 → DeepSeek 生成。

---

## 架构：web 如何调用 app

浏览器只访问 Next.js（默认 `http://localhost:3000`）。前端 `fetch("/api/...")` 打到 Next Route，再由服务端转发到 FastAPI `http://127.0.0.1:8001`。

```
浏览器
  │  /api/chat、/api/rag/*、/api/admin/*
  ▼
web/app/api/*          （Next.js 代理）
  │  http://127.0.0.1:8001/...
  ▼
app/api.py             （FastAPI）
  ├── offline/         切分 · 嵌入 · 写入 Qdrant
  └── online/          改写 · 多路召回 · 重排 · 扩窗 · 生成
```

| 前端入口 | Next 代理文件 | 后端接口 |
|----------|---------------|----------|
| 聊天 SSE | `web/app/api/chat/route.ts` | `POST /api/chat` |
| 知识库配置/文档 | `web/app/api/rag/[...path]/route.ts` | `/api/rag/*` |
| Admin 分步调试 | `web/app/api/admin/[...path]/route.ts` | `/api/admin/*` |

对应前端封装：

- 聊天：`web/lib/sse.ts` → `/api/chat`
- RAG 面板：`web/lib/rag.ts` → `/api/rag/...`
- Admin：`web/lib/admin.ts` → `/api/admin/offline|online/...`

两边需同时启动；后端未开时，Admin 会报「后端不可用(8001)」。

---

## 功能

- 入库：`md / txt / html / pdf` → 切分 → DashScope `text-embedding-v4` → 本地 Qdrant
- 检索：Query 改写 + 多路召回（原问/改写/关键词/子问题）+ RRF + `qwen3-rerank` + 前后 chunk 扩窗
- 生成：DeepSeek Flash，SSE 流式（`start` / `tool_call` / `tool_result` / `token` / `done`）
- 页面：聊天（`/`）、RAG 配置、Admin 离线/在线分步（`/admin`）

---

## 目录

```
funlab-assistant/
├── .env / .env.example
├── run_api.py                 # 启动 FastAPI :8001
├── environment.yml            # conda 环境 funlab
├── app/
│   ├── api.py                 # HTTP 入口（web 最终打到这里）
│   ├── config.py / rag_state.py
│   ├── ingest_service.py
│   ├── admin_service.py
│   ├── qdrant_client.py
│   ├── offline/               # 文档 → chunk → embed → Qdrant
│   └── online/                # 改写 → 多路召回 → 重排 → 扩窗 → 生成
├── web/                       # Next.js；api/* 仅做代理
├── test/                      # 流水线实验脚本（1pdf…7valuation、6llm）
└── data/
    ├── knowledge/             # 知识库原文
    └── qdrant/                # Qdrant 本地存储
```

---

## 环境

| 依赖 | 说明 |
|------|------|
| conda `funlab` | 见 `environment.yml` |
| Node.js 18+ | Next.js |
| `DEEPSEEK_API_KEY` | 改写 + 生成 |
| `DASHSCOPE_API_KEY` | 嵌入 + 重排 |

```powershell
conda env create -f environment.yml
conda activate funlab

cd web
npm install

copy .env.example .env
```

`.env` 关键关注：

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-flash
DASHSCOPE_API_KEY=...
EMBED_MODEL=text-embedding-v4
QDRANT_PATH=./data/qdrant
KNOWLEDGE_DIR=./data/knowledge
```

---

## 启动

### 1. 灌库（可选）

把文档放进 `data/knowledge/`，或从测试 OCR 结果灌入：

```powershell
python -c "from app.ingest_service import seed_test_docs; print(seed_test_docs(rebuild=True))"
# 或全量重建
python -c "from app.ingest_service import ingest; print(ingest(rebuild=True))"
```

### 2. 后端

```powershell
conda activate funlab
python run_api.py
```

健康检查：`http://127.0.0.1:8001/health`

### 3. 前端

```powershell
cd web
npm run dev
```

打开：**http://localhost:3000**  
Admin：**http://localhost:3000/admin**

---

## web ↔ app 接口对照

### 聊天

| 浏览器 | 后端 |
|--------|------|
| `POST /api/chat` `{ message, thread_id }` | `POST http://127.0.0.1:8001/api/chat` SSE |

### RAG 面板（`web/lib/rag.ts`）

| 浏览器 | 后端 |
|--------|------|
| `GET/PUT /api/rag/config` | `/api/rag/config` |
| `GET/POST /api/rag/documents` | `/api/rag/documents` |
| `GET/PUT/DELETE /api/rag/document` | `/api/rag/document` |
| `POST /api/rag/preview-split` | `/api/rag/preview-split` |
| `POST /api/rag/retrieve-preview` | `/api/rag/retrieve-preview` |
| `POST /api/rag/rebuild` | `/api/rag/rebuild` |

### Admin（`web/lib/admin.ts`）

| 浏览器 | 后端 | app 逻辑 |
|--------|------|----------|
| `POST /api/admin/offline/document-process` | 同路径 | 文本清洗预览 |
| `POST /api/admin/offline/document-file` | 同路径 | PDF/MD 抽文本 |
| `POST /api/admin/offline/split` | 同路径 | `offline.chunking` |
| `POST /api/admin/offline/vectorize` | 同路径 | DashScope 嵌入 |
| `POST /api/admin/offline/store` | 同路径 | 写入 Qdrant |
| `POST /api/admin/online/query-process` | 同路径 | Query 改写 |
| `POST /api/admin/online/retrieve` | 同路径 | 多路召回 |
| `POST /api/admin/online/rerank` | 同路径 | Qwen 重排 |
| `POST /api/admin/online/context` | 同路径 | 上下文扩窗 |
| `POST /api/admin/online/generate` | 同路径 | DeepSeek 生成 |

---

## 常见问题

**前端报后端不可用(8001)**：先启动 `python run_api.py`。

**嵌入 404 / model_not_found**：`.env` 中 `EMBED_MODEL=text-embedding-v4`，并配置 `DASHSCOPE_API_KEY`。

**检索为空**：确认已灌库；Qdrant 本地目录同一时间只能被一个进程打开（关掉占用中的 `rag_cli`）。

**无流式输出**：确认 `:8001` 在跑，且浏览器走的是 `/api/chat` 代理而非直连错误地址。

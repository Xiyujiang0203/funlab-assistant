# Funlab Intelligent Q&A Assistant

Agent + RAG assistant for the Funlab lab private knowledge base, built on `langchain==1.2.x`.

Place lab documents in `data/knowledge/`, ingest them, then chat in Chinese via the **Next.js UI** (or optional Gradio).

---

## Features

- Multi-format ingestion: `txt / md / html / docx / pdf / pptx / ppt`
- Vector search: Milvus (BGE-M3 embeddings, 1024-dim, COSINE)
- Agent: answers via `search_funlab_knowledge` tool
- Web chat: Next.js frontend + FastAPI SSE streaming; reasoning steps shown above the reply
- Short-term memory: `InMemorySaver` + `SummarizationMiddleware`
- Optional: Gradio UI (`main.py`)
- Optional: MinerU API for complex PDF/PPT (`MINERU_API_TOKEN`)
- Optional: LangSmith tracing

---

## Project Structure

```
funlab-assistant/
├── .env                              # API keys (not committed)
├── .env.example                      # Config template
├── run_api.py                        # FastAPI server (8001)
├── main.py                           # Gradio server (7860, optional)
├── requirements-extra.txt
├── milvus-standalone-docker-compose.yml
├── app/
│   ├── api.py                        # FastAPI + SSE
│   ├── agent.py                      # FunlabAssistant core
│   ├── tools.py                      # search_funlab_knowledge @tool
│   ├── retriever.py                  # Vector retrieval
│   ├── vectorstore.py                # Milvus
│   ├── embeddings.py                 # SiliconFlow BGE-M3
│   ├── loaders.py                    # Document loaders
│   ├── ingest_service.py             # Ingestion logic
│   └── ui.py                         # Gradio UI (optional)
├── web/                              # Next.js frontend
├── scripts/
│   ├── ingest.py                     # Ingest CLI
│   └── verify_e2e.py                 # E2E smoke test
└── data/knowledge/                   # Lab documents (see README)
```

---

## Prerequisites

| Dependency | Notes |
|------------|-------|
| conda env `langchain1.2` | Python backend |
| Node.js 18+ | Next.js frontend |
| Docker | Milvus |
| LLM key | `DEEPSEEK_API_KEY` or `CLOSEAI_API_KEY` |
| Embedding key | `SILICONFLOW_API_KEY` (BGE-M3) |

---

## Setup

```powershell
conda activate langchain1.2
pip install -r requirements-extra.txt

cd web
npm install
```

Copy and fill `.env`:

```powershell
copy .env.example .env
```

---

## Run

### 1. Start Milvus

```powershell
docker compose -f milvus-standalone-docker-compose.yml up -d
```

### 2. Ingest documents

```powershell
python scripts/ingest.py --rebuild
```

### 3. Start backend + frontend

```powershell
# Terminal 1: FastAPI
python run_api.py

# Terminal 2: Next.js
cd web
npm run dev
```

Open: **http://localhost:3000**

### Optional: Gradio

```powershell
python main.py
```

Open: **http://127.0.0.1:7860**

---

## Incremental Ingest

```powershell
python scripts/ingest.py
```

---

## FAQ

**Ingest 401**: Check `SILICONFLOW_API_KEY`.

**No retrieval results**: Ensure `EMBED_DIM=1024`, or rerun with `--rebuild`.

**LLM errors**: `CLOSEAI_BASE_URL` must include the `/v1` suffix.

**No streaming in UI**: Ensure `run_api.py` is running on port 8001.

**Milvus fails to start**: Check ports 9002 / 9003 for conflicts.

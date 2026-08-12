# Funlab 智能问答助手

[English](README.md)

基于 `langchain==1.2.x` 的 Agent + RAG 实现的 Funlab 实验室私有知识库问答助手。

把实验室文档放进 `data/knowledge/`，入库后通过 **Next.js 聊天界面**（或可选 Gradio）用中文问答。

---

## 功能

- 多格式文档入库：`txt / md / html / docx / pdf / pptx / ppt`
- 向量检索：Milvus（BGE-M3 嵌入，1024 维，COSINE）
- Agent：`search_funlab_knowledge` 工具检索知识库后作答
- Web 对话：Next.js 前端 + FastAPI SSE 流式输出，推理过程展示在回答上方
- 短期记忆：`InMemorySaver` + `SummarizationMiddleware`
- 可选：Gradio 界面（`main.py`）
- 可选：复杂 PDF / PPT 走 MinerU API 解析（配置 `MINERU_API_TOKEN`）
- 可选：LangSmith 链路追踪

---

## 项目结构

```
funlab-assistant/
├── .env                              # API Key（不提交 git）
├── .env.example                      # 配置模板
├── run_api.py                        # 启动 FastAPI（8001）
├── main.py                           # 启动 Gradio（7860，可选）
├── requirements-extra.txt
├── milvus-standalone-docker-compose.yml
├── app/
│   ├── api.py                        # FastAPI + SSE
│   ├── agent.py                      # FunlabAssistant 核心类
│   ├── tools.py                      # search_funlab_knowledge @tool
│   ├── retriever.py                  # 向量检索封装
│   ├── vectorstore.py                # Milvus
│   ├── embeddings.py                 # SiliconFlow BGE-M3
│   ├── loaders.py                    # 多格式文档加载
│   ├── ingest_service.py             # 入库逻辑
│   └── ui.py                         # Gradio 界面（可选）
├── web/                              # Next.js 前端
├── scripts/
│   ├── ingest.py                     # 入库脚本
│   └── verify_e2e.py                 # 端到端测试
└── data/knowledge/                   # 放实验室文档（见 README.md）
```

---

## 前置条件

| 依赖 | 说明 |
|------|------|
| conda 环境 `langchain1.2` | 后端 Python 环境 |
| Node.js 18+ | Next.js 前端 |
| Docker | Milvus |
| LLM Key | `DEEPSEEK_API_KEY` 或 `CLOSEAI_API_KEY` |
| Embedding Key | `SILICONFLOW_API_KEY`（BGE-M3） |

---

## 安装

```powershell
conda activate langchain1.2
pip install -r requirements-extra.txt

cd web
npm install
```

复制并填写 `.env`：

```powershell
copy .env.example .env
```

---

## 运行

### 1. 启动 Milvus

```powershell
docker compose -f milvus-standalone-docker-compose.yml up -d
```

### 2. 入库

```powershell
python scripts/ingest.py --rebuild
```

### 3. 启动后端 + 前端

```powershell
# 终端 1：FastAPI
python run_api.py

# 终端 2：Next.js
cd web
npm run dev
```

打开：**http://localhost:3000**

### 可选：Gradio

```powershell
python main.py
```

打开：**http://127.0.0.1:7860**

---

## 增量入库

```powershell
python scripts/ingest.py
```

---

## 常见问题

**入库 401**：检查 `SILICONFLOW_API_KEY`。

**检索无结果**：确认 `EMBED_DIM=1024`，或重跑 `--rebuild`。

**LLM 失败**：`CLOSEAI_BASE_URL` 需带 `/v1` 后缀。

**前端无流式输出**：确认 `run_api.py` 在 8001 端口运行。

**Milvus 启动失败**：检查 9002 / 9003 端口占用。

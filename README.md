# Funlab 智能问答助手

基于 `langchain==1.2.x` 的 Agent + RAG 实现的 Funlab 实验室私有知识库问答助手。

把实验室文档放进 `data/knowledge/`，入库后通过 Gradio Web 界面用中文问答。

---

## 功能

- 多格式文档入库：`txt / md / html / docx / pdf / pptx / ppt`
- 向量检索：Milvus（BGE-M3 嵌入，1024 维，COSINE）
- Agent：`search_funlab_knowledge` 工具检索知识库后作答，附来源引用
- Web 对话：Gradio 聊天界面，每个会话独立 `thread_id`，支持流式输出
- 短期记忆：`InMemorySaver` + `SummarizationMiddleware`（超过 2000 token 自动摘要）
- 可选：复杂 PDF / PPT 走 MinerU API 解析（配置 `MINERU_API_TOKEN`）
- 可选：LangSmith 链路追踪

---

## 项目结构

```
funlab-assistant/
├── .env                              # API Key（不提交 git）
├── .env.example                      # 配置模板
├── main.py                           # 启动 Gradio
├── requirements-extra.txt            # 额外依赖
├── milvus-standalone-docker-compose.yml
├── app/
│   ├── config.py                     # pydantic-settings 读取 .env
│   ├── agent.py                      # FunlabAssistant 核心类
│   ├── tools.py                      # search_funlab_knowledge @tool
│   ├── retriever.py                  # 向量检索封装
│   ├── vectorstore.py                # Milvus 连接 / collection 管理
│   ├── embeddings.py                 # SiliconFlow BGE-M3 嵌入模型
│   ├── loaders.py                    # 多格式文档加载路由
│   ├── splitter.py                   # RecursiveCharacterTextSplitter
│   ├── ingest_service.py             # 入库核心逻辑（供脚本和 UI 调用）
│   └── ui.py                         # Gradio 界面
├── scripts/
│   ├── ingest.py                     # 入库脚本（CLI）
│   └── verify_e2e.py                 # 端到端冒烟测试
└── data/
    └── knowledge/                    # 放实验室文档
```

---

## 前置条件

| 依赖 | 说明 |
|------|------|
| conda 环境 `langchain1.2` | 已有，复用 |
| Docker | 用于启动 Milvus |
| LLM Key | 优先 `CLOSEAI_API_KEY`，否则 `DEEPSEEK_API_KEY` |
| Embedding Key | `SILICONFLOW_API_KEY`（BGE-M3） |

---

## 安装额外依赖

```powershell
conda activate langchain1.2
pip install -r e:\AIAgent\funlab-assistant\requirements-extra.txt
```

---

## 配置 `.env`

复制模板后填写真实 Key：

```powershell
copy .env.example .env
```

必填项：

```env
# LLM（优先 CLOSEAI；若无则用 DeepSeek）
CLOSEAI_API_KEY=<你的 Key>
CLOSEAI_BASE_URL=<接口域名，如 https://xxx.com/v1>
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=openai

# Embedding
SILICONFLOW_API_KEY=<你的 Key>

# Milvus（默认 localhost:19530，一般不用改）
MILVUS_URI=http://localhost:19530
```

---

## 运行

### 1. 启动 Milvus

```powershell
cd e:\AIAgent\funlab-assistant
docker compose -f milvus-standalone-docker-compose.yml up -d
```

### 2. 放入文档

把实验室文档（pdf / docx / txt / md 等）放进：

```
funlab-assistant/data/knowledge/
```

### 3. 入库（重建向量索引）

```powershell
D:\miniconda3\envs\langchain1.2\python.exe scripts\ingest.py --rebuild
```

输出示例：

```
Ingest done: 6 docs -> 73 vectors
```

### 4. 启动 Web

```powershell
D:\miniconda3\envs\langchain1.2\python.exe main.py
```

打开浏览器：**http://127.0.0.1:7860**

---

## 增量入库

新增文档后无需重建，直接增量入库：

```powershell
D:\miniconda3\envs\langchain1.2\python.exe scripts\ingest.py
```

或在 Gradio 界面右侧「上传文档到知识库」直接拖入文件触发增量入库。

---

## 端到端冒烟（可选）

```powershell
D:\miniconda3\envs\langchain1.2\python.exe scripts\verify_e2e.py
```

---

## 常见问题

**入库 401 / token invalid**：检查 `.env` 中 `SILICONFLOW_API_KEY` 是否填写完整（不含省略号）。

**入库成功但检索无结果**：确认 `EMBED_DIM=1024` 与 BGE-M3 一致；或重跑 `--rebuild`。

**LLM 调用失败**：若 `CLOSEAI_BASE_URL` 不含 `/v1` 后缀，手动补上后重试。

**Milvus 启动失败**：etcd / minio 端口冲突，检查宿主机 9002 / 9003 端口是否被占用；可修改 `milvus-standalone-docker-compose.yml` 中对应端口映射。

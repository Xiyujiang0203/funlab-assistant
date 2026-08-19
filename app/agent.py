import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from app.config import PROJECT_ROOT, get_settings
from app.tools import FUNLAB_TOOLS

FUNLAB_SYSTEM_PROMPT = """你是 Funlab 实验室智能问答助手。

职责：
1. 回答与 Funlab 实验室相关的问题时，优先调用 search_funlab_knowledge 工具检索知识库。
2. 仅基于检索结果作答；若知识库无相关信息，明确说明「未在知识库中找到相关信息」，不要编造。
3. 回答时尽量引用来源（文件名）；不要向用户展示 chunk_id、score 等内部字段。
4. 始终使用中文，语气专业、简洁。"""


def _init_llm():
    settings = get_settings()
    if settings.siliconflow_api_key:
        return init_chat_model(
            model=settings.llm_model or "deepseek-ai/DeepSeek-V3",
            model_provider="openai",
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
        )
    if settings.closeai_api_key:
        return init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            api_key=settings.closeai_api_key,
            base_url=settings.closeai_base_url,
        )
    if settings.deepseek_api_key:
        return init_chat_model(
            model="deepseek-chat",
            model_provider="deepseek",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    raise RuntimeError("请在 .env 中配置 SILICONFLOW_API_KEY、CLOSEAI_API_KEY 或 DEEPSEEK_API_KEY")


def _extract_ai_text(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and msg.content:
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
    return "抱歉，我无法处理这个请求。"


class FunlabAssistant:
    def __init__(self):
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        settings = get_settings()
        if settings.langsmith_tracing and settings.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

        self.model = _init_llm()
        self.agent = create_agent(
            model=self.model,
            tools=FUNLAB_TOOLS,
            system_prompt=FUNLAB_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                SummarizationMiddleware(
                    model=self.model,
                    trigger=[("tokens", 4000)],
                    keep=("messages", 10),
                    summary_prompt="对以下 Funlab 实验室对话历史做摘要，保留用户意图与关键结论：\n{messages}",
                )
            ],
        )

    def chat(self, user_input: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        return _extract_ai_text(result)

    def stream_chat(self, user_input: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        for chunk, _meta in self.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            if getattr(chunk, "type", None) == "ai" and chunk.content:
                if isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield block.get("text", "")
                        elif isinstance(block, str):
                            yield block

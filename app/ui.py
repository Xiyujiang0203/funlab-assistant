import shutil
import uuid
from pathlib import Path

import gradio as gr

from app.agent import FunlabAssistant
from app.config import get_settings
from app.vectorstore import VectorStore


def build_ui(assistant: FunlabAssistant) -> gr.Blocks:
    settings = get_settings()

    def get_doc_count() -> str:
        try:
            count = VectorStore().count()
            return f"知识库向量条目: {count}"
        except Exception as exc:
            return f"Milvus 未连接: {exc}"

    def respond(message: str, history: list, thread_id: str):
        if not message.strip():
            yield history, thread_id, ""
            return
        history = history + [{"role": "user", "content": message},
                              {"role": "assistant", "content": ""}]
        partial = ""
        for token in assistant.stream_chat(message, thread_id):
            partial += token
            history[-1]["content"] = partial
            yield history, thread_id, ""
        if not partial:
            text = assistant.chat(message, thread_id)
            history[-1]["content"] = text
            yield history, thread_id, ""

    def clear_chat():
        return [], str(uuid.uuid4()), get_doc_count()

    def handle_upload(file_obj, thread_id: str):
        if file_obj is None:
            return thread_id, get_doc_count(), "未选择文件"
        dest_dir = settings.knowledge_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        from app.ingest_service import ingest

        try:
            n = ingest(rebuild=False, file_paths=[dest])
            return thread_id, get_doc_count(), f"已上传 {src.name}，新增入库 {n} 条向量"
        except Exception as exc:
            return thread_id, get_doc_count(), f"上传成功但入库失败: {exc}"

    with gr.Blocks(title="Funlab 智能问答助手") as demo:
        gr.Markdown("# Funlab 实验室智能问答助手")
        thread_state = gr.State(str(uuid.uuid4()))
        status = gr.Textbox(value=get_doc_count(), label="知识库状态", interactive=False)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(placeholder="请输入问题...", label="消息")
                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空对话")
            with gr.Column(scale=1):
                upload = gr.File(label="上传文档到知识库")
                upload_status = gr.Textbox(label="上传状态", interactive=False)

        send.click(respond, [msg, chatbot, thread_state], [chatbot, thread_state, msg])
        msg.submit(respond, [msg, chatbot, thread_state], [chatbot, thread_state, msg])
        clear.click(clear_chat, outputs=[chatbot, thread_state, status])
        upload.change(handle_upload, [upload, thread_state], [thread_state, status, upload_status])

    return demo

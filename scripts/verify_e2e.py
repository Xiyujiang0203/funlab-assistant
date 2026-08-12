import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.agent import FunlabAssistant
from app.ingest_service import ingest as run_ingest
from app.loaders import load_knowledge_dir
from app.retriever import KnowledgeRetriever
from app.vectorstore import VectorStore

GPU_QUERY = "如何申请 GPU 账号？"


def ensure_ingested(rebuild: bool = False) -> int:
    store = VectorStore()
    if not store.ping():
        print(f"FAIL: 无法连接 Milvus: {store.uri}", file=sys.stderr)
        sys.exit(1)
    docs = load_knowledge_dir()
    if not docs:
        print("FAIL: 知识库目录为空", file=sys.stderr)
        sys.exit(1)
    count_before = store.count()
    if rebuild or count_before == 0:
        n = run_ingest(rebuild=rebuild or count_before == 0)
        print(f"ingest: {len(docs)} 个文档 -> {n} 条向量 (rebuild={rebuild or count_before == 0})")
        return n
    print(f"ingest: 跳过 (已有 {count_before} 条向量)")
    return count_before


def test_retrieve() -> None:
    print("\n=== retrieve ===")
    retriever = KnowledgeRetriever()
    hits = retriever.search(GPU_QUERY, top_k=3)
    print(KnowledgeRetriever.format_hits(hits))
    if not hits:
        print("FAIL: 检索无结果", file=sys.stderr)
        sys.exit(1)
    joined = " ".join(h["text"] for h in hits)
    if "GPU" not in joined and "gpu" not in joined.lower():
        print("FAIL: 检索结果未包含 GPU 相关内容", file=sys.stderr)
        sys.exit(1)
    print("retrieve: OK")


def test_agent_chat() -> None:
    print("\n=== agent chat ===")
    assistant = FunlabAssistant()
    thread_id = f"verify-e2e-{uuid.uuid4().hex[:8]}"
    reply = assistant.chat(GPU_QUERY, thread_id=thread_id)
    print(reply)
    if not reply or len(reply.strip()) < 10:
        print("FAIL: agent 回复过短", file=sys.stderr)
        sys.exit(1)
    print("agent chat: OK")


def main() -> None:
    print("=== verify_e2e: ingest + retrieve + agent (GPU 账号) ===")
    ensure_ingested(rebuild=False)
    test_retrieve()
    test_agent_chat()
    print("\n=== ALL PASSED ===")


if __name__ == "__main__":
    main()

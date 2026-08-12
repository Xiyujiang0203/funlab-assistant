import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.ingest_service import ingest as run_ingest
from app.loaders import load_knowledge_dir
from app.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Funlab knowledge ingest")
    parser.add_argument("--rebuild", action="store_true", help="rebuild Milvus collection")
    args = parser.parse_args()

    store = VectorStore()
    if not store.ping():
        print(f"Cannot connect Milvus: {store.uri}")
        sys.exit(1)

    docs = load_knowledge_dir()
    if not docs:
        print("Knowledge dir empty, skip ingest")
        sys.exit(0)
    n = run_ingest(rebuild=args.rebuild)
    print(f"Ingest done: {len(docs)} docs -> {n} vectors")


if __name__ == "__main__":
    main()
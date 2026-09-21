from pathlib import Path

from app.offline.pipeline import run_offline_ingest, seed_from_test_ocr


def ingest(rebuild: bool = False, file_paths: list[Path] | None = None) -> int:
    return run_offline_ingest(rebuild=rebuild, file_paths=file_paths)


def seed_test_docs(rebuild: bool = True) -> int:
    return seed_from_test_ocr(rebuild=rebuild)

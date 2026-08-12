import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.agent import FunlabAssistant
from app.config import get_settings
from app.ui import build_ui


def main():
    settings = get_settings()
    assistant = FunlabAssistant()
    demo = build_ui(assistant)
    demo.launch(server_name=settings.gradio_host, server_port=settings.gradio_port)


if __name__ == "__main__":
    main()

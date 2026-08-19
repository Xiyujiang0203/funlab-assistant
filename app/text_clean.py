import re


def clean_rag_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", text)
    return re.sub(r"\s+", "", text).strip()


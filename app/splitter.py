from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )

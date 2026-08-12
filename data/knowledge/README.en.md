# Knowledge Base Directory

[中文](README.md)

Place Funlab lab documents here. After running the ingest script, vectors are stored in Milvus.

**Documents in this folder are not committed to Git.** Maintain them locally.

## Supported Formats

| Format | Extensions |
|--------|------------|
| Plain text | `.txt` |
| Markdown | `.md` |
| HTML | `.html` |
| Word | `.docx` |
| PDF | `.pdf` |
| PowerPoint | `.pptx` / `.ppt` |

## Recommended Content

- Lab rules, policies, and workflow docs
- Equipment guides, API / tool documentation
- Project overviews, member roles, event notices
- Internal notices (network subscription, account policies, etc.)

## Ingest

```bash
python scripts/ingest.py
# Rebuild the vector store
python scripts/ingest.py --rebuild
```

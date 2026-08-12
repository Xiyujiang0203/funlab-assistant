# 知识库目录

将 Funlab 实验室文档放在此目录，运行入库脚本后写入 Milvus。

**本目录下的实际文档不会提交到 Git**，请在本机自行维护。

## 支持格式

| 格式 | 扩展名 |
|------|--------|
| 纯文本 | `.txt` |
| Markdown | `.md` |
| HTML | `.html` |
| Word | `.docx` |
| PDF | `.pdf` |
| PowerPoint | `.pptx` / `.ppt` |

## 建议放入的内容

- 实验室组规、制度与流程说明
- 设备使用说明、API / 工具文档
- 项目介绍、成员分工、活动通知
- 上网订阅、账号规范等内部须知

## 入库

```bash
python scripts/ingest.py
# 重建向量库
python scripts/ingest.py --rebuild
```

"use client";

import Link from "next/link";
import React, { useMemo, useState } from "react";
import {
  buildContext,
  extractDocumentFile,
  generateAnswer,
  processDocument,
  processQuery,
  retrieveHits,
  rerankHits,
  splitText,
  storeChunks,
  vectorizeChunks,
  type Chunk,
  type Hit,
  type VectorSummary,
} from "@/lib/admin";

type Tab = "offline" | "online";
type Status = "idle" | "running" | "done" | "error";

const sampleText = "Funlab GPU 账号申请流程：先提交申请表，说明用途、项目和预计使用时间。";
const sampleQuery = "如何申请 GPU 账号？";

function StatusPill({ status }: { status: Status }) {
  return <span className={`admin-status ${status}`}>{status}</span>;
}

function ResultBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="admin-result">
      <div className="admin-result-title">{title}</div>
      {children}
    </section>
  );
}

function TextOutput({ value }: { value: string }) {
  return <pre className="admin-pre">{value || "暂无输出"}</pre>;
}

export default function AdminConsole() {
  const [tab, setTab] = useState<Tab>("offline");
  const [busyStep, setBusyStep] = useState("");
  const [message, setMessage] = useState("");

  const [docInput, setDocInput] = useState(sampleText);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [cleanedText, setCleanedText] = useState("");
  const [chunkSize, setChunkSize] = useState(300);
  const [chunkOverlap, setChunkOverlap] = useState(60);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [vectors, setVectors] = useState<VectorSummary[]>([]);
  const [source, setSource] = useState("manual-admin");
  const [rebuild, setRebuild] = useState(false);
  const [storedCount, setStoredCount] = useState<number | null>(null);

  const [query, setQuery] = useState(sampleQuery);
  const [processedQuery, setProcessedQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [scoreThreshold, setScoreThreshold] = useState(0);
  const [retrievedHits, setRetrievedHits] = useState<Hit[]>([]);
  const [rankedHits, setRankedHits] = useState<Hit[]>([]);
  const [context, setContext] = useState("");
  const [threadId, setThreadId] = useState("");
  const [answer, setAnswer] = useState("");

  const offlineStatus = useMemo(
    () => ({
      process: cleanedText ? "done" : "idle",
      split: chunks.length ? "done" : "idle",
      vectorize: vectors.length ? "done" : "idle",
      store: storedCount !== null ? "done" : "idle",
    }),
    [cleanedText, chunks.length, vectors.length, storedCount],
  );
  const onlineStatus = useMemo(
    () => ({
      query: processedQuery ? "done" : "idle",
      retrieve: retrievedHits.length ? "done" : "idle",
      rerank: rankedHits.length ? "done" : "idle",
      context: context ? "done" : "idle",
      generation: answer ? "done" : "idle",
    }),
    [processedQuery, retrievedHits.length, rankedHits.length, context, answer],
  );

  async function runStep<T>(step: string, action: () => Promise<T>, onDone: (result: T) => void) {
    setBusyStep(step);
    setMessage(`${step}中...`);
    try {
      const result = await action();
      onDone(result);
      setMessage(`${step} 完成`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : `${step} 失败`);
    } finally {
      setBusyStep("");
    }
  }

  function currentChunks() {
    return chunks.map((chunk) => chunk.text).filter(Boolean);
  }

  function resetOfflineOutputs() {
    setCleanedText("");
    setChunks([]);
    setVectors([]);
    setStoredCount(null);
  }

  return (
    <div className="admin-root">
      <header className="admin-header">
        <div>
          <h1>Funlab RAG 管理端</h1>
          <p>离线入库与在线问答流水线控制台</p>
        </div>
        <Link className="admin-link" href="/">返回聊天</Link>
      </header>

      <main className="admin-main">
        <nav className="admin-tabs">
          <button className={tab === "offline" ? "active" : ""} onClick={() => setTab("offline")}>离线阶段</button>
          <button className={tab === "online" ? "active" : ""} onClick={() => setTab("online")}>在线阶段</button>
        </nav>

        {tab === "offline" ? (
          <section className="admin-grid">
            <aside className="admin-steps">
              <div><span>文件解析</span><StatusPill status={busyStep === "文件解析" ? "running" : selectedFileName ? "done" : "idle"} /></div>
              <div><span>文档处理</span><StatusPill status={busyStep === "文档处理" ? "running" : offlineStatus.process as Status} /></div>
              <div><span>切片</span><StatusPill status={busyStep === "切片" ? "running" : offlineStatus.split as Status} /></div>
              <div><span>向量化</span><StatusPill status={busyStep === "向量化" ? "running" : offlineStatus.vectorize as Status} /></div>
              <div><span>入库</span><StatusPill status={busyStep === "入库" ? "running" : offlineStatus.store as Status} /></div>
            </aside>

            <section className="admin-workspace">
              <div className="admin-panel">
                <label className="admin-field">
                  <span>选择 PDF / Markdown 文件</span>
                  <input
                    type="file"
                    accept=".pdf,.md,application/pdf,text/markdown,text/x-markdown"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (!file) return;
                      void runStep("文件解析", () => extractDocumentFile(file), (result) => {
                        setSelectedFileName(result.filename);
                        setSource(result.filename);
                        setDocInput(result.content);
                        resetOfflineOutputs();
                      });
                      event.currentTarget.value = "";
                    }}
                  />
                </label>
                {selectedFileName ? <p className="admin-muted">当前文件：{selectedFileName}</p> : null}
                <label className="admin-field">
                  <span>文档输入</span>
                  <textarea
                    value={docInput}
                    onChange={(e) => {
                      setDocInput(e.target.value);
                      resetOfflineOutputs();
                    }}
                  />
                </label>
                <button disabled={!!busyStep} onClick={() => void runStep("文档处理", () => processDocument(docInput), (result) => setCleanedText(result.cleaned_text))}>
                  运行文档处理
                </button>
              </div>

              <ResultBlock title="文档处理输出">
                <TextOutput value={cleanedText} />
              </ResultBlock>

              <div className="admin-panel compact">
                <label className="admin-field small">
                  <span>chunk size</span>
                  <input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
                </label>
                <label className="admin-field small">
                  <span>chunk overlap</span>
                  <input type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} />
                </label>
                <button disabled={!!busyStep} onClick={() => void runStep("切片", () => splitText(cleanedText || docInput, chunkSize, chunkOverlap), (result) => setChunks(result.chunks))}>
                  预览切片
                </button>
              </div>

              <ResultBlock title={`切片结果 (${chunks.length})`}>
                <div className="admin-list">
                  {chunks.map((chunk) => (
                    <article key={chunk.index}>
                      <strong>Chunk {chunk.index + 1}</strong>
                      <p>{chunk.text}</p>
                    </article>
                  ))}
                </div>
              </ResultBlock>

              <div className="admin-panel compact">
                <button disabled={!!busyStep || !chunks.length} onClick={() => void runStep("向量化", () => vectorizeChunks(currentChunks()), (result) => setVectors(result.vectors))}>
                  向量化当前 chunks
                </button>
                <label className="admin-field small">
                  <span>source</span>
                  <input value={source} onChange={(e) => setSource(e.target.value)} />
                </label>
                <label className="admin-check">
                  <input type="checkbox" checked={rebuild} onChange={(e) => setRebuild(e.target.checked)} />
                  重建集合
                </label>
                <button disabled={!!busyStep || !chunks.length} onClick={() => void runStep("入库", () => storeChunks(currentChunks(), source, rebuild), (result) => setStoredCount(result.chunks))}>
                  写入向量数据库
                </button>
              </div>

              <ResultBlock title="向量化 / 入库状态">
                <div className="admin-vector-grid">
                  {vectors.map((vector) => (
                    <div key={vector.index}>#{vector.index + 1} dim {vector.dimension}: [{vector.preview.join(", ")}]</div>
                  ))}
                </div>
                <p className="admin-muted">已写入：{storedCount ?? "-"}</p>
              </ResultBlock>
            </section>
          </section>
        ) : (
          <section className="admin-grid">
            <aside className="admin-steps">
              <div><span>Query 处理</span><StatusPill status={busyStep === "Query 处理" ? "running" : onlineStatus.query as Status} /></div>
              <div><span>检索</span><StatusPill status={busyStep === "检索" ? "running" : onlineStatus.retrieve as Status} /></div>
              <div><span>Rerank</span><StatusPill status={busyStep === "Rerank" ? "running" : onlineStatus.rerank as Status} /></div>
              <div><span>上下文构建</span><StatusPill status={busyStep === "上下文构建" ? "running" : onlineStatus.context as Status} /></div>
              <div><span>Generation</span><StatusPill status={busyStep === "Generation" ? "running" : onlineStatus.generation as Status} /></div>
            </aside>

            <section className="admin-workspace">
              <div className="admin-panel">
                <label className="admin-field">
                  <span>Query 输入</span>
                  <textarea value={query} onChange={(e) => setQuery(e.target.value)} />
                </label>
                <button disabled={!!busyStep} onClick={() => void runStep("Query 处理", () => processQuery(query), (result) => setProcessedQuery(result.processed_query))}>
                  运行 Query 处理
                </button>
              </div>

              <ResultBlock title="处理后的 Query">
                <TextOutput value={processedQuery} />
              </ResultBlock>

              <div className="admin-panel compact">
                <label className="admin-field small">
                  <span>topK</span>
                  <input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
                </label>
                <label className="admin-field small">
                  <span>score threshold</span>
                  <input type="number" step="0.01" value={scoreThreshold} onChange={(e) => setScoreThreshold(Number(e.target.value))} />
                </label>
                <button disabled={!!busyStep} onClick={() => void runStep("检索", () => retrieveHits(processedQuery || query, topK, scoreThreshold), (result) => setRetrievedHits(result.hits))}>
                  运行检索
                </button>
                <button disabled={!!busyStep || !retrievedHits.length} onClick={() => void runStep("Rerank", () => rerankHits(processedQuery || query, retrievedHits), (result) => setRankedHits(result.hits))}>
                  运行 Rerank
                </button>
                <button disabled={!!busyStep || (!rankedHits.length && !retrievedHits.length)} onClick={() => void runStep("上下文构建", () => buildContext(rankedHits.length ? rankedHits : retrievedHits), (result) => setContext(result.context))}>
                  构建上下文
                </button>
              </div>

              <ResultBlock title="检索 / 重排结果">
                <div className="admin-list">
                  {(rankedHits.length ? rankedHits : retrievedHits).map((hit, index) => (
                    <article key={`${hit.source}-${index}`}>
                      <strong>{index + 1}. {hit.source?.split("/").pop()}</strong>
                      <small>vec {hit.score?.toFixed?.(3) ?? "-"} / rerank {hit.rerank_score?.toFixed?.(3) ?? "-"}</small>
                      <p>{hit.text}</p>
                    </article>
                  ))}
                </div>
              </ResultBlock>

              <ResultBlock title="上下文">
                <TextOutput value={context} />
              </ResultBlock>

              <div className="admin-panel compact">
                <label className="admin-field small">
                  <span>thread id</span>
                  <input value={threadId} onChange={(e) => setThreadId(e.target.value)} />
                </label>
                <button disabled={!!busyStep} onClick={() => void runStep("Generation", () => generateAnswer(query, threadId), (result) => {
                  setThreadId(result.thread_id);
                  setAnswer(result.answer);
                })}>
                  生成回答
                </button>
              </div>

              <ResultBlock title="Generation 输出">
                <TextOutput value={answer} />
              </ResultBlock>
            </section>
          </section>
        )}
      </main>

      {message ? <div className="admin-toast">{message}</div> : null}
    </div>
  );
}

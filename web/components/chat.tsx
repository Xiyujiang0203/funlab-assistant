"use client";

import { useEffect, useRef, useState } from "react";
import { Markdown } from "./markdown";
import { streamChat } from "@/lib/sse";

type Step = { id: string; type: "tool-call" | "tool-result"; title: string; body: string };

type Message =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; steps: Step[]; streaming?: boolean };

function StepBlock({ step }: { step: Step }) {
  const [open, setOpen] = useState(true);
  return (
    <div
      className={`rounded-lg border text-xs ${
        step.type === "tool-call"
          ? "border-sky-200 bg-sky-50/80"
          : "border-emerald-200 bg-emerald-50/80"
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left font-medium text-zinc-700"
      >
        <span>{step.type === "tool-call" ? "🔍" : "📄"}</span>
        <span className="flex-1 truncate">{step.title}</span>
        <span className="text-zinc-400">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="border-t border-inherit px-3 py-2 text-zinc-600">
          {step.type === "tool-result" ? <Markdown content={step.body} /> : <pre className="whitespace-pre-wrap font-mono">{step.body}</pre>}
        </div>
      )}
    </div>
  );
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "你好！我是 Funlab 实验室智能问答助手。\n\n你可以问我关于实验室制度、项目、设备、流程、成员等相关问题，我会从知识库中为你检索答案。",
      steps: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setLoading(true);

    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();

    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", steps: [], streaming: true },
    ]);

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    let answer = "";

    try {
      await streamChat(
        text,
        threadId,
        {
          onStart: (tid) => {
            if (tid) setThreadId(tid);
          },
          onToken: (token) => {
            answer += token;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.role === "assistant"
                  ? { ...m, content: answer }
                  : m,
              ),
            );
          },
          onToolCall: (tool, inputObj) => {
            const q = String(inputObj.query ?? JSON.stringify(inputObj));
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.role === "assistant"
                  ? {
                      ...m,
                      steps: [
                        ...m.steps,
                        {
                          id: crypto.randomUUID(),
                          type: "tool-call",
                          title: `调用工具：${tool}`,
                          body: `查询：${q}`,
                        },
                      ],
                    }
                  : m,
              ),
            );
          },
          onToolResult: (content) => {
            const preview = content.slice(0, 900) + (content.length > 900 ? "\n…（已截断）" : "");
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.role === "assistant"
                  ? {
                      ...m,
                      steps: [
                        ...m.steps,
                        {
                          id: crypto.randomUUID(),
                          type: "tool-result",
                          title: "检索结果",
                          body: preview,
                        },
                      ],
                    }
                  : m,
              ),
            );
          },
          onError: (msg) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.role === "assistant"
                  ? { ...m, content: `⚠️ ${msg}` }
                  : m,
              ),
            );
          },
        },
        abortRef.current.signal,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "连接失败";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && m.role === "assistant"
            ? { ...m, content: `⚠️ ${msg}` }
            : m,
        ),
      );
    } finally {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && m.role === "assistant"
            ? { ...m, streaming: false }
            : m,
        ),
      );
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="flex h-dvh flex-col bg-[#f4f5f7]">
      <header className="shrink-0 border-b border-zinc-200/80 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-3xl items-center px-4">
          <h1 className="text-[15px] font-semibold text-zinc-900">Funlab 智能助手</h1>
          <span className="ml-auto text-xs text-zinc-500">{loading ? "思考中…" : "就绪"}</span>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-hidden px-4 py-4">
        <div className="flex-1 space-y-4 overflow-y-auto pr-1">
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-emerald-500 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={m.id} className="flex justify-start">
                <div className="max-w-[92%] rounded-2xl rounded-bl-md border border-zinc-200/80 bg-white px-4 py-3 text-sm text-zinc-800 shadow-sm">
                  {m.steps.length > 0 && (
                    <div className="mb-3 space-y-2">
                      {m.steps.map((s) => (
                        <StepBlock key={s.id} step={s} />
                      ))}
                    </div>
                  )}
                  {m.streaming ? (
                    <pre className="whitespace-pre-wrap font-[inherit] text-[inherit]">{m.content}</pre>
                  ) : (
                    <Markdown content={m.content} />
                  )}
                  {m.streaming && (
                    <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-emerald-500 align-middle" />
                  )}
                </div>
              </div>
            ),
          )}
          <div ref={bottomRef} />
        </div>

        <div className="mt-3 shrink-0 rounded-2xl border border-zinc-200/80 bg-white p-2 shadow-sm">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="输入问题，Enter 发送，Shift+Enter 换行…"
              rows={1}
              className="max-h-36 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2.5 text-sm text-zinc-900 outline-none placeholder:text-zinc-400"
            />
            <button
              type="button"
              onClick={() => void send()}
              disabled={loading || !input.trim()}
              className="mb-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-500 text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-40"
              aria-label="发送"
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M22 2 11 13" />
                <path d="m22 2-7 20-4-9-9-4z" />
              </svg>
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

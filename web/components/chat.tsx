"use client";

import Link from "next/link";
import React, { useState } from "react";
import { useChat } from "./ChatContext";
import { ChatProvider } from "./ChatContext";
import Sidebar from "./Sidebar";
import ChatWindow from "./ChatWindow";
import RagPanel from "./RagPanel";

function ChatInner() {
  const { isGenerating } = useChat();
  const [ragOpen, setRagOpen] = useState(false);

  return (
    <div className="chat-root">
      <header className="chat-header">
        <div className="chat-header-inner">
          <h1 className="chat-title">Funlab 智能助手</h1>
          <Link className="chat-rag-btn chat-admin-link" href="/admin">
            管理端
          </Link>
          <button type="button" className="chat-rag-btn" onClick={() => setRagOpen((v) => !v)}>
            RAG
          </button>
          <span className="chat-status">{isGenerating ? "思考中…" : "就绪"}</span>
        </div>
      </header>

      <main className="chat-main">
        <div className="chat-shell">
          <Sidebar />
          <ChatWindow />
        </div>
        <RagPanel open={ragOpen} onClose={() => setRagOpen(false)} />
      </main>
    </div>
  );
}

export function Chat() {
  return (
    <ChatProvider>
      <ChatInner />
    </ChatProvider>
  );
}

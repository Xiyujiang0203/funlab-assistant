"use client";

import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { streamChat } from "@/lib/sse";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
};

type Conversation = {
  id: string;
  title: string;
  threadId: string;
  messages: ChatMessage[];
};

type ChatContextValue = {
  conversations: Conversation[];
  currentId: string | null;
  currentConversation: Conversation | null;
  messages: ChatMessage[];
  isGenerating: boolean;
  createConversation: () => string;
  deleteConversation: (id: string) => void;
  switchConversation: (id: string) => void;
  sendMessage: (content: string) => void;
  stopGeneration: () => void;
};

const ChatContext = createContext<ChatContextValue | null>(null);

function createId() {
  const c = globalThis.crypto as Crypto | undefined;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `${Date.now()}_${Math.random()}`;
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const currentConversation = useMemo(
    () => conversations.find((c) => c.id === currentId) || null,
    [conversations, currentId],
  );
  const messages = currentConversation?.messages || [];

  const createConversation = useCallback(() => {
    const conv: Conversation = { id: createId(), title: "新对话", threadId: "", messages: [] };
    setConversations((prev) => [conv, ...prev]);
    setCurrentId(conv.id);
    return conv.id;
  }, []);

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (currentId === id) setCurrentId(next[0]?.id ?? null);
        return next;
      });
    },
    [currentId],
  );

  const switchConversation = useCallback((id: string) => {
    setCurrentId(id);
  }, []);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsGenerating(false);
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      const text = content.trim();
      if (!text || isGenerating) return;

      let convId = currentId;
      if (!convId) convId = createConversation();

      const userMsg: ChatMessage = { id: createId(), role: "user", content: text };
      const assistantId = createId();

      let threadId = "";
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          threadId = c.threadId;
          const title = c.messages.length === 0 ? text.slice(0, 20) : c.title;
          return {
            ...c,
            title,
            messages: [...c.messages, userMsg, { id: assistantId, role: "assistant", content: "", streaming: true }],
          };
        }),
      );

      setIsGenerating(true);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        let answer = "";
        await streamChat(
          text,
          threadId,
          {
            onStart: (tid) => {
              if (!tid) return;
              setConversations((prev) =>
                prev.map((c) => (c.id === convId ? { ...c, threadId: tid } : c)),
              );
            },
            onToken: (token) => {
              answer += token;
              setConversations((prev) =>
                prev.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantId ? { ...m, content: answer } : m,
                        ),
                      }
                    : c,
                ),
              );
            },
            onError: (msg) => {
              setConversations((prev) =>
                prev.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantId ? { ...m, content: `❌ ${msg}` } : m,
                        ),
                      }
                    : c,
                ),
              );
            },
          },
          controller.signal,
        );
      } catch (err) {
        if (err instanceof Error && err.name !== "AbortError") {
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantId ? { ...m, content: `❌ ${err.message}` } : m,
                    ),
                  }
                : c,
            ),
          );
        }
      } finally {
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantId ? { ...m, streaming: false } : m,
                  ),
                }
              : c,
          ),
        );
        abortRef.current = null;
        setIsGenerating(false);
      }
    },
    [currentId, isGenerating, createConversation],
  );

  const value: ChatContextValue = {
    conversations,
    currentId,
    currentConversation,
    messages,
    isGenerating,
    createConversation,
    deleteConversation,
    switchConversation,
    sendMessage,
    stopGeneration,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat 必须在 ChatProvider 内使用");
  return ctx;
}

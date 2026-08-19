"use client";

import React, { useEffect, useRef } from "react";
import { useChat } from "./ChatContext";
import MessageBubble from "./MessageBubble";
import LoadingIndicator from "./LoadingIndicator";

export default function MessageList() {
  const { messages, isGenerating } = useChat();
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  if (messages.length === 0 && !isGenerating) {
    return (
      <div className="chat-message-empty">
        开始一段新对话吧
      </div>
    );
  }

  return (
    <div className="chat-message-list">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      {isGenerating ? (
        <div className="chat-loading-wrap">
          <LoadingIndicator />
        </div>
      ) : null}
      <div ref={bottomRef} />
    </div>
  );
}


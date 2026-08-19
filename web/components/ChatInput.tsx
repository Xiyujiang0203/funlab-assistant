"use client";

import React, { useState } from "react";
import { useChat } from "./ChatContext";

export default function ChatInput() {
  const { sendMessage, isGenerating, stopGeneration } = useChat();
  const [text, setText] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    sendMessage(text);
    setText("");
  }

  return (
    <form
      className="chat-input-form"
      onSubmit={handleSubmit}
    >
      <div className="chat-input-row">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!text.trim()) return;
              sendMessage(text);
              setText("");
            }
          }}
          placeholder="输入消息...（Enter 发送，Shift+Enter 换行）"
          className="chat-textarea"
          disabled={isGenerating}
          rows={1}
        />
        {isGenerating ? (
          <button
            type="button"
            onClick={stopGeneration}
            className="chat-stop-btn"
          >
            停止
          </button>
        ) : (
          <button
            type="submit"
            disabled={!text.trim()}
            className="chat-send-btn"
          >
            发送
          </button>
        )}
      </div>
    </form>
  );
}


"use client";

import React from "react";
import { Markdown } from "./markdown";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`chat-bubble-row ${isUser ? "user" : "assistant"}`}>
      <div
        className={`chat-bubble ${isUser ? "user" : "assistant"}`}
      >
        {isUser ? <p>{message.content}</p> : <Markdown content={message.content} />}
      </div>
    </div>
  );
}


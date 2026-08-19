"use client";

import React from "react";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";

export default function ChatWindow() {
  return (
    <section className="chat-window">
      <MessageList />
      <ChatInput />
    </section>
  );
}


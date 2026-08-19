"use client";

import React from "react";
import ConversationItem from "./ConversationItem";
import { useChat } from "./ChatContext";

export default function Sidebar() {
  const { conversations, currentId, createConversation, deleteConversation, switchConversation } = useChat();

  return (
    <aside className="chat-sidebar">
      <button
        type="button"
        className="chat-new-btn"
        onClick={createConversation}
      >
        + 新建会话
      </button>

      <ul className="chat-conv-list">
        {conversations.length === 0 ? (
          <li className="chat-conv-empty">暂无会话</li>
        ) : (
          conversations.map((c) => (
            <ConversationItem
              key={c.id}
              conversation={c}
              active={c.id === currentId}
              onSelect={switchConversation}
              onDelete={deleteConversation}
            />
          ))
        )}
      </ul>
    </aside>
  );
}


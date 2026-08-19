"use client";

import React from "react";

type Props = {
  conversation: { id: string; title: string };
  active: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
};

export default function ConversationItem({ conversation, active, onSelect, onDelete }: Props) {
  return (
    <li
      className={`chat-conv-item${active ? " active" : ""}`}
      onClick={() => onSelect(conversation.id)}
    >
      <span className="chat-conv-title">
        {conversation.title}
      </span>
      <button
        type="button"
        className="chat-conv-delete"
        onClick={(e) => {
          e.stopPropagation();
          onDelete(conversation.id);
        }}
        aria-label="删除会话"
      >
        ✕
      </button>
    </li>
  );
}


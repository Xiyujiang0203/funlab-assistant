"use client";

import React from "react";

export default function LoadingIndicator() {
  return (
    <div className="chat-loading">
      <span className="chat-loading-dots">
        <span className="chat-loading-dot" />
        <span className="chat-loading-dot" />
        <span className="chat-loading-dot" />
      </span>
      <span>AI 正在思考...</span>
    </div>
  );
}


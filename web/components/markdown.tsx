"use client";

import ReactMarkdown from "react-markdown";

export function Markdown({ content }: { content: string }) {
  if (!content) return null;
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => <p className="md-p">{children}</p>,
        ul: ({ children }) => <ul className="md-ul">{children}</ul>,
        ol: ({ children }) => <ol className="md-ol">{children}</ol>,
        li: ({ children }) => <li className="md-li">{children}</li>,
        h1: ({ children }) => <h1 className="md-h1">{children}</h1>,
        h2: ({ children }) => <h2 className="md-h2">{children}</h2>,
        h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
        code: ({ children }) => (
          <code className="md-code">{children}</code>
        ),
        pre: ({ children }) => (
          <pre className="md-pre">{children}</pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="md-blockquote">{children}</blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

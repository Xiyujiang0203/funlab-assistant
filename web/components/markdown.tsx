"use client";

import ReactMarkdown from "react-markdown";

export function Markdown({ content }: { content: string }) {
  if (!content) return null;
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
        ul: ({ children }) => <ul className="mb-2 list-disc pl-5 space-y-1">{children}</ul>,
        ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 space-y-1">{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        h1: ({ children }) => <h1 className="mb-2 mt-3 text-lg font-semibold">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-2 mt-3 text-base font-semibold">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold">{children}</h3>,
        code: ({ children }) => (
          <code className="rounded bg-zinc-100 px-1 py-0.5 text-[0.85em]">{children}</code>
        ),
        pre: ({ children }) => (
          <pre className="mb-2 overflow-x-auto rounded-lg bg-zinc-100 p-3 text-xs">{children}</pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="mb-2 border-l-2 border-emerald-300 pl-3 text-zinc-600">
            {children}
          </blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

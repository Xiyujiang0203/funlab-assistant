import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./app.css";

export const metadata: Metadata = {
  title: "Funlab 智能助手",
  description: "Funlab 实验室智能问答助手",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="zh-CN"
      className="h-full antialiased"
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col" suppressHydrationWarning>{children}</body>
    </html>
  );
}

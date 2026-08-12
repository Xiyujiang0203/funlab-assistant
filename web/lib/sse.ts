export type SSEHandler = {
  onStart?: (threadId: string) => void;
  onToken?: (text: string) => void;
  onToolCall?: (tool: string, input: Record<string, unknown>) => void;
  onToolResult?: (content: string) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
};

export async function streamChat(
  message: string,
  threadId: string,
  handlers: SSEHandler,
  signal?: AbortSignal,
) {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId }),
    signal,
  });

  if (!resp.ok || !resp.body) {
    throw new Error(`请求失败: ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.replace(/\r$/, "");
      if (trimmed.startsWith("event: ")) {
        eventType = trimmed.slice(7).trim();
      } else if (trimmed.startsWith("data: ")) {
        const raw = trimmed.slice(6).trim();
        if (!raw) continue;
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(raw);
        } catch {
          continue;
        }

        switch (eventType) {
          case "start":
            handlers.onStart?.(String(data.thread_id ?? ""));
            break;
          case "token":
            handlers.onToken?.(String(data.text ?? ""));
            break;
          case "tool_call":
            handlers.onToolCall?.(
              String(data.tool ?? ""),
              (data.input as Record<string, unknown>) ?? {},
            );
            break;
          case "tool_result":
            handlers.onToolResult?.(String(data.content ?? ""));
            break;
          case "error":
            handlers.onError?.(String(data.message ?? "未知错误"));
            break;
          case "done":
            handlers.onDone?.();
            break;
        }
      }
    }
  }
}

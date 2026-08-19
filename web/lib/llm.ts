type ChatMsg = { role: "user" | "assistant"; content: string };

export async function chatLlm(messages: ChatMsg[], signal?: AbortSignal) {
  const res = await fetch("/api/llm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.message || "请求失败");
  return data.content as string;
}

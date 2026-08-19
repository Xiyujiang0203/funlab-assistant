const API_URL = process.env.SILICONFLOW_BASE_URL || "https://api.siliconflow.cn/v1";
const MODEL = "deepseek-ai/DeepSeek-V3";

export async function POST(req: Request) {
  const apiKey = process.env.SILICONFLOW_API_KEY;
  if (!apiKey) {
    return Response.json({ message: "未配置 SILICONFLOW_API_KEY" }, { status: 500 });
  }

  const { messages } = await req.json();
  if (!messages?.length) {
    return Response.json({ message: "消息不能为空" }, { status: 400 });
  }

  try {
    const res = await fetch(`${API_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model: MODEL, messages, stream: false }),
      signal: req.signal,
    });

    const data = await res.json();
    if (!res.ok) {
      return Response.json(
        { message: data.error?.message || data.message || "请求失败" },
        { status: res.status },
      );
    }

    return Response.json({ content: data.choices?.[0]?.message?.content || "" });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      return Response.json({ message: "已取消" }, { status: 499 });
    }
    return Response.json(
      { message: err instanceof Error ? err.message : "服务器错误" },
      { status: 500 },
    );
  }
}

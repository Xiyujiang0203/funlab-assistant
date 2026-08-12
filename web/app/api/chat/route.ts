export async function POST(req: Request) {
  const body = await req.text();
  const upstream = await fetch("http://127.0.0.1:8001/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  if (!upstream.ok || !upstream.body) {
    return new Response(JSON.stringify({ error: "upstream failed" }), {
      status: upstream.status || 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}

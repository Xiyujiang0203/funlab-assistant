const BASE = "http://127.0.0.1:8001/api/admin";

async function proxy(req: Request, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const url = new URL(`${BASE}/${path.join("/")}`);
  const incoming = new URL(req.url);
  url.search = incoming.search;

  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const body =
    req.method === "GET" || req.method === "DELETE"
      ? undefined
      : await req.arrayBuffer();

  try {
    const upstream = await fetch(url, {
      method: req.method,
      headers,
      body,
      signal: req.signal,
    });

    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/json",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "upstream unavailable";
    return Response.json(
      { detail: `后端不可用(8001): ${message}` },
      { status: 502 },
    );
  }
}

export async function GET(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function POST(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function PUT(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function DELETE(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

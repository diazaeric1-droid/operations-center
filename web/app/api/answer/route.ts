import { NextRequest, NextResponse } from "next/server";
import type { AnswerResponse, SearchRequest } from "@/lib/types";

// Proxy route: the browser calls this same-origin handler, which forwards to the
// FastAPI backend. Keeps the API base URL server-side and sidesteps CORS.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  let body: SearchRequest;
  try {
    body = (await req.json()) as SearchRequest;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (!body.query || typeof body.query !== "string" || !body.query.trim()) {
    return NextResponse.json(
      { error: "A non-empty 'query' is required." },
      { status: 400 },
    );
  }

  try {
    const upstream = await fetch(`${API_URL}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: body.query,
        top_k: body.top_k ?? 6,
        cause: body.cause ?? null,
      }),
      cache: "no-store",
    });

    const text = await upstream.text();
    if (!upstream.ok) {
      return NextResponse.json(
        { error: `Backend ${upstream.status}: ${text.slice(0, 500)}` },
        { status: upstream.status },
      );
    }

    const data = JSON.parse(text) as AnswerResponse;
    return NextResponse.json(data);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        error: `Could not reach the RAG backend at ${API_URL}. Is FastAPI running on :8000? (${detail})`,
      },
      { status: 502 },
    );
  }
}

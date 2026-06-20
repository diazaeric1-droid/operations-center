# Operator Notes — Semantic Search (Web)

A Next.js 14 (App Router) + TypeScript + Tailwind frontend for the operator-note
RAG engine. You ask a question in plain language; the FastAPI backend embeds it,
retrieves the closest notes from a **pgvector** index, and returns a grounded,
cited answer. This UI renders the answer, an LLM-vs-extractive badge, and a
ranked table of the source notes.

```
 Browser ──▶ Next.js route /api/answer ──▶ FastAPI /answer ──▶ NoteSearchEngine
                                                                   │
                                                       pgvector ◀──┘ (local embeddings)
```

## Prerequisites

- The **FastAPI backend** running on `:8000` (see `../api/`).
- The **pgvector Postgres** container up (see `../docker-compose.rag.yml`), already
  indexed with the operator-note corpus.
- Node 18+ / npm.

## Run it (two terminals)

**1. Backend — FastAPI on :8000** (from the repo root `operations-center/`):

```bash
.venv/bin/pip install -r api/requirements-api.txt   # once
.venv/bin/python -m uvicorn api.main:app --port 8000
```

Sanity check: `curl http://localhost:8000/health` →
`{"ok": true, "indexed": 384, "db": "localhost:5433/opsrag"}`

**2. Frontend — Next.js on :3000** (from `web/`):

```bash
npm install        # once
npm run dev        # http://localhost:3000
```

Open <http://localhost:3000> and search.

## Environment variables

| Variable              | Default                  | Purpose                                            |
| --------------------- | ------------------------ | -------------------------------------------------- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`  | Base URL of the FastAPI backend the proxy forwards to. |

Copy `.env.example` to `.env.local` to override. The browser never calls the
backend directly — requests go through the same-origin route handler
`app/api/answer/route.ts`, which forwards to `NEXT_PUBLIC_API_URL` (no CORS
headaches).

## Scripts

```bash
npm run dev        # dev server (:3000)
npm run build      # production build
npm run start      # serve the production build
npm run typecheck  # tsc --noEmit (strict)
```

## Deploy (Vercel)

One-line note: push `web/` to a repo and `vercel --prod` (or import in the Vercel
dashboard), then set `NEXT_PUBLIC_API_URL` in the project's environment variables
to your deployed FastAPI URL. The frontend is fully static-friendly; only the
`/api/answer` route needs the backend reachable.
```bash
vercel --prod   # set NEXT_PUBLIC_API_URL to your backend in project settings
```

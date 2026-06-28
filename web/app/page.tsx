"use client";

import { useState } from "react";
import {
  CAUSES,
  CAUSE_LABELS,
  type AnswerResponse,
  type RetrievedNote,
} from "@/lib/types";

const EXAMPLE_QUERIES = [
  "slow ESP underload failure, not an instant trip",
  "freeze-off shutdowns that lasted more than three days",
  "separator emulsion upset and what fixed it",
  "third-party gathering line curtailment",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [cause, setCause] = useState<string>("");
  const [topK, setTopK] = useState(6);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);

  async function runSearch(q: string) {
    const trimmed = q.trim();
    if (!trimmed) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          top_k: topK,
          cause: cause || null,
        }),
      });
      const data = (await res.json()) as AnswerResponse & { error?: string };
      if (!res.ok || data.error) {
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    void runSearch(query);
  }

  return (
    <main className="mx-auto w-full max-w-5xl px-5 py-10 sm:px-8 sm:py-14">
      <Header />

      <form
        onSubmit={onSubmit}
        className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-2xl shadow-black/40 backdrop-blur sm:p-6"
      >
        <label
          htmlFor="query"
          className="mb-2 block text-sm font-medium text-slate-300"
        >
          Ask the operator-note log
        </label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            id="query"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. shutdowns that read like a slow ESP failure"
            autoComplete="off"
            className="w-full rounded-xl border border-white/10 bg-ink-950/60 px-4 py-3 text-slate-100 placeholder:text-slate-500 outline-none transition focus:border-sky-400/60 focus:ring-2 focus:ring-sky-400/20"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-sky-500 px-6 py-3 font-semibold text-ink-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? (
              <>
                <Spinner /> Searching
              </>
            ) : (
              "Search"
            )}
          </button>
        </div>

        {/* Controls */}
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label
              htmlFor="cause"
              className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-slate-400"
            >
              Cause filter
            </label>
            <select
              id="cause"
              value={cause}
              onChange={(e) => setCause(e.target.value)}
              className="w-full rounded-lg border border-white/10 bg-ink-950/60 px-3 py-2.5 text-sm text-slate-200 outline-none transition focus:border-sky-400/60"
            >
              <option value="">Any cause</option>
              {CAUSES.map((c) => (
                <option key={c} value={c}>
                  {CAUSE_LABELS[c]}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label
              htmlFor="topk"
              className="mb-1.5 flex items-center justify-between text-xs font-medium uppercase tracking-wide text-slate-400"
            >
              <span>Results to retrieve</span>
              <span className="font-mono text-sky-300">{topK}</span>
            </label>
            <input
              id="topk"
              type="range"
              min={3}
              max={12}
              step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-sky-400"
            />
          </div>
        </div>

        {/* Example chips */}
        <div className="mt-4 flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setQuery(ex);
                void runSearch(ex);
              }}
              className="rounded-full border border-white/10 bg-white/[0.02] px-3 py-1 text-xs text-slate-400 transition hover:border-sky-400/40 hover:text-slate-200"
            >
              {ex}
            </button>
          ))}
        </div>
      </form>

      {/* Results region */}
      <section className="mt-8">
        {error && <ErrorBox message={error} />}
        {loading && !result && <LoadingState />}
        {!loading && !error && !result && <EmptyState />}
        {result && <Results result={result} />}
      </section>

      <Footer />
    </main>
  );
}

function Header() {
  return (
    <header>
      <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-slate-400">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        pgvector · local embeddings · grounded synthesis
      </div>
      <h1 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
        Operator Notes{" "}
        <span className="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
          Semantic Search
        </span>
      </h1>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400 sm:text-base">
        Retrieval-augmented search over the downtime-event log. Ask in plain
        language; the engine embeds your query, pulls the closest operator notes
        from a pgvector index, and synthesizes a cited answer.
      </p>
    </header>
  );
}

function Results({ result }: { result: AnswerResponse }) {
  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 sm:p-6">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
            Answer
          </h2>
          <LlmBadge usedLlm={result.used_llm} />
        </div>
        <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-slate-200">
          {result.answer}
        </pre>
      </div>

      {result.sources.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-1.5">
          <div className="px-4 pb-2 pt-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
              Source notes{" "}
              <span className="ml-1 font-normal text-slate-500">
                ({result.sources.length})
              </span>
            </h2>
          </div>
          <SourcesTable sources={result.sources} />
        </div>
      )}
    </div>
  );
}

function SourcesTable({ sources }: { sources: RetrievedNote[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="px-4 py-2.5 font-medium">Score</th>
            <th className="px-4 py-2.5 font-medium">Well</th>
            <th className="px-4 py-2.5 font-medium">Cause</th>
            <th className="px-4 py-2.5 font-medium">Start</th>
            <th className="px-4 py-2.5 text-right font-medium">Days</th>
            <th className="px-4 py-2.5 text-right font-medium">Deferred bbl</th>
            <th className="px-4 py-2.5 font-medium">Note</th>
          </tr>
        </thead>
        <tbody>
          {sources.map((s, i) => (
            <tr
              key={`${s.well_id}-${s.start_date}-${i}`}
              className="border-b border-white/5 transition hover:bg-white/[0.025]"
            >
              <td className="px-4 py-3">
                <ScorePill score={s.score} />
              </td>
              <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-slate-300">
                {s.well_id}
              </td>
              <td className="whitespace-nowrap px-4 py-3">
                <span className="rounded-md bg-indigo-400/10 px-2 py-0.5 text-xs text-indigo-300">
                  {s.cause}
                </span>
              </td>
              <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-slate-400">
                {s.start_date}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-right font-mono text-slate-300">
                {s.duration_days}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-right font-mono text-slate-300">
                {s.deferred_bbl.toLocaleString()}
              </td>
              <td className="px-4 py-3 text-slate-300">{s.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ScorePill({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score));
  return (
    <span className="inline-flex items-center gap-2">
      <span className="h-1.5 w-10 overflow-hidden rounded-full bg-white/10">
        <span
          className="block h-full rounded-full bg-gradient-to-r from-sky-400 to-emerald-400"
          style={{ width: `${pct * 100}%` }}
        />
      </span>
      <span className="font-mono text-xs text-slate-400">
        {score.toFixed(3)}
      </span>
    </span>
  );
}

function LlmBadge({ usedLlm }: { usedLlm: boolean }) {
  return usedLlm ? (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-300">
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
      LLM-synthesized
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-400/20 bg-slate-400/10 px-3 py-1 text-xs font-medium text-slate-300">
      <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
      Extractive (no key)
    </span>
  );
}

function LoadingState() {
  return (
    <div className="space-y-4">
      <div className="h-32 animate-pulse rounded-2xl border border-white/10 bg-white/[0.03]" />
      <div className="h-48 animate-pulse rounded-2xl border border-white/10 bg-white/[0.03]" />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] px-6 py-14 text-center">
      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-white/5 text-2xl">
        🔎
      </div>
      <p className="text-sm font-medium text-slate-300">
        Search the operator-note log
      </p>
      <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
        Enter a question above or pick an example. Results are ranked by semantic
        similarity, with each cited note shown below the answer.
      </p>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-red-400/30 bg-red-400/10 px-5 py-4 text-sm text-red-200">
      <p className="font-semibold">Something went wrong</p>
      <p className="mt-1 text-red-200/80">{message}</p>
    </div>
  );
}

function Footer() {
  return (
    <footer className="mt-12 border-t border-white/5 pt-6 text-xs text-slate-600">
      Next.js + TypeScript frontend over a FastAPI + pgvector RAG engine. Answers
      are grounded in retrieved notes; add an Anthropic key on the backend for
      LLM-synthesized narration.
    </footer>
  );
}

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}

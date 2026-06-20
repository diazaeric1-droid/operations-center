// TypeScript mirror of the Python RAG schema (rag/engine.py).
// Keep these in lockstep with RetrievedNote / Answer on the backend.

/** One operator note returned by the retriever. Mirrors `RetrievedNote`. */
export interface RetrievedNote {
  score: number;
  note: string;
  well_id: string;
  cause: string;
  start_date: string;
  duration_days: number;
  deferred_bbl: number;
  source: string;
}

/** Response from POST /answer. Mirrors the `Answer` dataclass payload. */
export interface AnswerResponse {
  answer: string;
  used_llm: boolean;
  sources: RetrievedNote[];
}

/** Body sent to /answer (and /search). */
export interface SearchRequest {
  query: string;
  top_k?: number;
  cause?: string | null;
}

/** The 8 valid cause-filter values understood by the corpus. */
export const CAUSES = [
  "artificial_lift",
  "surface_facility",
  "power",
  "gathering_thirdparty",
  "wellbore",
  "planned",
  "weather",
  "reservoir",
] as const;

export type Cause = (typeof CAUSES)[number];

/** Human-friendly labels for the cause <select>. */
export const CAUSE_LABELS: Record<Cause, string> = {
  artificial_lift: "Artificial lift",
  surface_facility: "Surface facility",
  power: "Power",
  gathering_thirdparty: "Gathering / third-party",
  wellbore: "Wellbore",
  planned: "Planned",
  weather: "Weather",
  reservoir: "Reservoir",
};

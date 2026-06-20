"""RAG over your Obsidian vault — ask it questions AND have it push back on your work.

Reads your Cowork-Brain notes (projects, sessions, inbox), embeds them LOCALLY
(fastembed — free, private; your notes never leave the machine except the few
retrieved snippets sent to the answer model), and serves three modes:

    python tools/vault_review.py ingest                 # build / refresh the index (run once)
    python tools/vault_review.py ask "how did we decide X on ATLAS?"
    python tools/vault_review.py review "ATLAS trading bot"   # skeptical critique of that topic
    python tools/vault_review.py recent                 # critique your most-recently-edited notes

Pick the answer model with --provider (default: best available; gpt-4o/Claude give
the sharpest review). Embeddings are local and need no key.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VAULT = Path(os.environ.get("VAULT_PATH",
                            "/Users/ericbot/.openclaw/workspace/Cowork-Brain"))
INGEST_DIRS = ["01-Projects", "02-Sessions", "00-Inbox", "04-References"]
EMB_PATH = VAULT / ".vault_rag_emb.npy"        # index lives in the vault, NOT the repo
META_PATH = VAULT / ".vault_rag_meta.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK, OVERLAP = 1200, 200


# --- embeddings (local, cached) ----------------------------------------------
_embedder = None


def _embed(texts: list[str]) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(EMBED_MODEL)
    vecs = np.array(list(_embedder.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)   # unit vectors -> dot product = cosine


# --- vault reading + chunking -------------------------------------------------
def _iter_notes():
    for d in INGEST_DIRS:
        for p in sorted((VAULT / d).rglob("*.md")):
            try:
                text = p.read_text(errors="ignore").strip()
            except Exception:  # noqa: BLE001
                continue
            if text:
                yield p, text


def _chunk(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK])
        i += CHUNK - OVERLAP
    return [c for c in out if len(c.strip()) >= 60]


# --- index lifecycle ----------------------------------------------------------
def ingest() -> None:
    chunks, meta = [], []
    t0 = time.time()
    for p, text in _iter_notes():
        rel = str(p.relative_to(VAULT))
        mtime = p.stat().st_mtime
        for c in _chunk(text):
            chunks.append(c)
            meta.append({"source": rel, "mtime": mtime})
    print(f"read {len({m['source'] for m in meta})} notes -> {len(chunks)} chunks; "
          "embedding locally…")

    embs = []
    for i in range(0, len(chunks), 256):
        embs.append(_embed([m + " :: " + c for m, c in
                            zip((mm["source"] for mm in meta[i:i + 256]), chunks[i:i + 256])]))
        print(f"  embedded {min(i + 256, len(chunks))}/{len(chunks)}", end="\r")
    matrix = np.vstack(embs)
    np.save(EMB_PATH, matrix)
    META_PATH.write_text(json.dumps([{**m, "text": c}
                                     for m, c in zip(meta, chunks)]))
    print(f"\nindexed {len(chunks)} chunks in {time.time() - t0:.0f}s -> {EMB_PATH.name}")


def _load():
    if not EMB_PATH.exists():
        sys.exit("No index yet. Run:  python tools/vault_review.py ingest")
    return np.load(EMB_PATH), json.loads(META_PATH.read_text())


def search(query: str, k: int = 10) -> list[dict]:
    matrix, meta = _load()
    q = _embed([query])[0]
    scores = matrix @ q
    top = np.argsort(-scores)[:k]
    return [{**meta[i], "score": float(scores[i])} for i in top]


def _recent_chunks(k: int = 12) -> list[dict]:
    _, meta = _load()
    return sorted(meta, key=lambda m: m["mtime"], reverse=True)[:k]


# --- LLM modes ----------------------------------------------------------------
def _provider(name: str | None) -> str:
    if name:
        return name
    from langgraph_rag.providers import available
    av = available()
    for p in ("claude", "github", "openrouter", "groq", "gemini"):
        if av.get(p):
            return p
    sys.exit("No provider key set. export GITHUB_TOKEN / GROQ_API_KEY / etc.")


def _context(hits: list[dict]) -> str:
    return "\n\n".join(f"[from {h['source']}]\n{h['text']}" for h in hits)


ASK_SYS = ("You answer questions using ONLY the user's own notes provided. Cite the "
           "note filenames you used. If the notes don't answer it, say so plainly.")
REVIEW_SYS = (
    "You are a skeptical senior engineer reviewing the user's OWN project/session "
    "notes. Based only on the notes provided:\n"
    "1. Flag anything that looks INCORRECT, RISKY, INEFFICIENT, or like a "
    "questionable decision — be direct, not a cheerleader.\n"
    "2. Give CONCRETE, actionable suggestions to improve it.\n"
    "3. Cite the note filename for each point.\n"
    "If the work is sound, say what's strong AND where the residual risks are. "
    "Be substantive and honest — the user wants real pushback, not praise.")


def ask(question: str, provider: str | None = None) -> None:
    from langgraph_rag.providers import chat
    prov = _provider(provider)
    hits = search(question, k=10)
    print(f"[answering with {prov} over {len(hits)} retrieved notes]\n")
    print(chat(f"Question: {question}\n\nNotes:\n{_context(hits)}",
               provider=prov, system=ASK_SYS, max_tokens=1024))
    print("\nSources:", ", ".join(dict.fromkeys(h["source"] for h in hits)))


def review(topic: str | None = None, provider: str | None = None) -> None:
    from langgraph_rag.providers import chat
    prov = _provider(provider)
    if topic:
        hits = search(topic, k=12)
        header = f"Review the approach/decisions around: {topic}"
    else:
        hits = _recent_chunks(12)
        header = "Review my most recently edited work for problems and improvements."
    print(f"[reviewing with {prov} over {len(hits)} notes]\n")
    print(chat(f"{header}\n\nNotes:\n{_context(hits)}",
               provider=prov, system=REVIEW_SYS, max_tokens=1500))
    print("\nReviewed:", ", ".join(dict.fromkeys(h["source"] for h in hits)))


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "help"
    prov = None
    if "--provider" in args:
        prov = args[args.index("--provider") + 1]
        args = [a for a in args if a != prov and a != "--provider"]
    rest = " ".join(a for a in args[1:] if not a.startswith("--"))

    if cmd == "ingest":
        ingest()
    elif cmd == "ask" and rest:
        ask(rest, prov)
    elif cmd == "review":
        review(rest or None, prov)
    elif cmd == "recent":
        review(None, prov)
    else:
        print(__doc__)

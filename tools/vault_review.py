"""RAG over your Obsidian vault — ask it questions AND have it push back on your work.

Reads your Cowork-Brain notes (projects, sessions, inbox), embeds them LOCALLY
(fastembed — free, private; your notes never leave the machine except the few
retrieved snippets sent to the answer model), and serves three modes:

    python tools/vault_review.py ingest                 # build / refresh the index (run once)
    python tools/vault_review.py ask "how did we decide X on ATLAS?"
    python tools/vault_review.py review "ATLAS trading bot"   # skeptical critique of that topic
    python tools/vault_review.py recent                 # critique your most-recently-edited notes
    python tools/vault_review.py sweep                  # gated weekly sweep: skip if nothing
                                                        #   changed, else re-ingest + review every
                                                        #   project + Telegram a digest
    #   sweep flags: --force (ignore the gate) · --no-notify (skip Telegram) · --budget N

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
SWEEP_STATE = VAULT / ".vault_sweep_state.json"  # last-sweep watermark (the "gate")
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


SWEEP_IDEA_SYS = (
    "You are a sharp venture/product advisor. Based on the user's project reviews "
    "and his portfolio — a model-agnostic LLM/agent platform (RAG, LangGraph, evals, "
    "observability), crypto + prediction-market trading infrastructure, an oil & gas "
    "domain + the Upstream Suite product, and Houston small-business ventures (print "
    "shop, apparel, automation) — propose 6-8 SPIN-OFF project ideas that BUILD ON "
    "what he already has. For each: title, why it fits HIM specifically, what existing "
    "asset/skill it leverages, a realistic first step, rough effort. Concrete, not "
    "generic startup fluff.")

SWEEP_PACE_S = 6   # seconds between calls — stay under free per-minute limits


def _project_folders() -> list[str]:
    base = VAULT / "01-Projects"
    folders = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if d.name.startswith("_"):
            continue
        n = sum(1 for _ in d.rglob("*.md"))
        if n:
            folders.append((d.name, n))
    return [name for name, _ in sorted(folders, key=lambda x: -x[1])]   # biggest first


# --- the gate: only sweep when the vault actually changed ---------------------
def _project_mtimes() -> dict[str, float]:
    """Newest .md mtime per project folder — the signal the gate watches."""
    base = VAULT / "01-Projects"
    out: dict[str, float] = {}
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir()
                    if p.is_dir() and not p.name.startswith("_")):
        mts = [p.stat().st_mtime for p in d.rglob("*.md")]
        if mts:
            out[d.name] = max(mts)
    return out


def _load_sweep_state() -> dict:
    if SWEEP_STATE.exists():
        try:
            return json.loads(SWEEP_STATE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _changed_projects(prev: dict, cur: dict[str, float]) -> list[str]:
    """Projects whose newest note is newer than the last sweep (or brand new)."""
    pm = prev.get("project_mtimes", {})
    return [name for name, mt in cur.items() if mt > pm.get(name, 0.0) + 1e-6]


# --- the push: a phone notification once the sweep runs -----------------------
DIGEST_SYS = (
    "You are condensing a weekly project-review report into a phone notification. "
    "Output: a one-line headline, then 3-6 terse bullets naming the projects that "
    "most need attention and the single most important issue or action for each, "
    "then one line with the top spin-off idea. Plain text, no markdown headers, no "
    "fluff — the reader is the person who owns these projects.")


def _deterministic_digest(date: str, sections: list, changed: list[str]) -> str:
    head = (f"Weekly vault sweep — {date}: {len(sections)} projects reviewed"
            + (f", {len(changed)} changed" if changed else ""))
    bullets = []
    for proj, txt, _ in sections[:6]:
        if txt.startswith("("):
            continue
        first = " ".join(txt.split())[:140]
        bullets.append(f"• {proj}: {first}")
    return head + "\n" + "\n".join(bullets)


def _telegram_html(date: str, digest: str, report_path, changed: list[str]) -> str:
    import html
    lines = digest.splitlines()
    head = html.escape(lines[0]) if lines else f"Weekly vault sweep — {date}"
    body = "\n".join(html.escape(ln) for ln in lines[1:])
    parts = [f"🗂️ <b>{head}</b>"]
    if changed:
        parts.append(f"<i>changed since last sweep: {html.escape(', '.join(changed[:10]))}</i>")
    parts.append("")
    parts.append(body)
    parts.append(f"\n<i>full report → {html.escape(report_path.name)}</i>")
    return "\n".join(parts)


def _notify_sweep(report_path, sections, ideas, changed, date, prov) -> None:
    """Push a concise digest to Telegram. Best-effort: never fails the sweep."""
    try:
        from notes_watch import config as nw_config
        from notes_watch.notify import send_telegram
    except Exception as e:  # noqa: BLE001
        print(f"[notify] notes_watch notify layer unavailable ({e}) — skipped push")
        return
    cfg = nw_config.load()
    if not cfg.telegram_ready:
        print("[notify] no Telegram creds (notes_watch/.env) — skipped push")
        return
    # Prefer an LLM-condensed digest; fall back to a deterministic one (no key).
    digest = _deterministic_digest(date, sections, changed)
    try:
        from langgraph_rag.providers import chat
        full = "\n\n".join(f"## {p}\n{t}" for p, t, _ in sections
                           if not t.startswith("(")) + f"\n\n## Ideas\n{ideas}"
        digest = chat(full[:8000], provider=prov, system=DIGEST_SYS,
                      max_tokens=500) or digest
    except Exception as e:  # noqa: BLE001 — keep the deterministic digest
        print(f"[notify] LLM digest unavailable ({type(e).__name__}) — using plain digest")
    ok = send_telegram(cfg.telegram_token, cfg.telegram_chat_id,
                       _telegram_html(date, digest, report_path, changed))
    print(f"[notify] telegram sent: {ok}")


def sweep(provider: str | None = None, budget: int = 40,
          force: bool = False, notify: bool = True) -> Path | None:
    """Loop over every project (up to `budget` LLM calls), review each, generate
    spin-off ideas, write a dated report to the vault Inbox, and push a digest.

    The gate: if no project notes changed since the last sweep (and not --force),
    skip entirely. When notes did change, re-ingest first so the review is over
    the CURRENT vault, not a stale index.
    """
    import datetime
    from langgraph_rag.providers import chat

    cur_mtimes = _project_mtimes()
    prev = _load_sweep_state()
    changed = _changed_projects(prev, cur_mtimes)
    if prev and not force and not changed:
        print(f"[gate] no project notes changed since last sweep "
              f"({prev.get('last_sweep')}) — skipping")
        return None
    if changed:
        print(f"[gate] {len(changed)} project(s) changed: {', '.join(changed[:8])}"
              f"{' …' if len(changed) > 8 else ''}")
    # Re-ingest when the vault moved (or the index is missing) so we review fresh.
    if force or changed or not EMB_PATH.exists():
        print("[gate] re-ingesting vault before review…")
        ingest()

    prov = _provider(provider)
    projects = _project_folders()
    print(f"[weekly sweep · {prov} · {len(projects)} projects · budget {budget} calls]")
    sections, used = [], 0
    for proj in projects:
        if used >= budget:
            sections.append((proj, "(skipped — call budget reached)", []))
            continue
        hits = search(f"{proj} project summary decisions risks approach", k=8)
        try:
            txt = chat(f"Review the project '{proj}'.\n\nNotes:\n{_context(hits)}",
                       provider=prov, system=REVIEW_SYS, max_tokens=900)
        except Exception as e:  # noqa: BLE001 — rate-limit/error: note it, keep going
            txt = f"(skipped — {type(e).__name__})"
        sections.append((proj, txt, [h["source"] for h in hits]))
        used += 1
        print(f"  [{used}/{budget}] {proj}")
        time.sleep(SWEEP_PACE_S)

    ideas = ""
    if used < budget:
        digest = "\n".join(f"- {p}: {t[:180]}" for p, t, _ in sections
                           if not t.startswith("("))
        try:
            ideas = chat("Project reviews:\n" + digest, provider=prov,
                         system=SWEEP_IDEA_SYS, max_tokens=1300)
        except Exception as e:  # noqa: BLE001
            ideas = f"(skipped — {type(e).__name__})"

    today = datetime.date.today().isoformat()
    out = VAULT / "00-Inbox" / f"{today} — weekly project sweep.md"
    lines = [f"# Weekly Project Sweep — {today}",
             f"_Auto-generated by vault_review · model: {prov} · {used} projects reviewed_", ""]
    for proj, txt, srcs in sections:
        lines.append(f"## {proj}\n\n{txt}\n")
        if srcs:
            lines.append(f"_sources: {', '.join(dict.fromkeys(srcs))}_\n")
    lines.append("---\n\n## Spin-off project ideas\n\n" + (ideas or "(none)") + "\n")
    out.write_text("\n".join(lines))
    print(f"\nwrote report -> {out}")

    # commit the watermark so the next tick can skip an unchanged vault
    SWEEP_STATE.write_text(json.dumps({
        "last_sweep": today, "project_mtimes": cur_mtimes,
        "changed": changed, "provider": prov,
    }, indent=2))

    if notify:
        _notify_sweep(out, sections, ideas, changed, today, prov)
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "help"
    prov, budget, positional = None, 40, []
    force, notify = False, True
    i = 1
    while i < len(argv):
        if argv[i] == "--provider" and i + 1 < len(argv):
            prov = argv[i + 1]; i += 2
        elif argv[i] == "--budget" and i + 1 < len(argv):
            budget = int(argv[i + 1]); i += 2
        elif argv[i] == "--force":
            force = True; i += 1
        elif argv[i] == "--no-notify":
            notify = False; i += 1
        else:
            positional.append(argv[i]); i += 1
    rest = " ".join(positional)

    if cmd == "ingest":
        ingest()
    elif cmd == "ask" and rest:
        ask(rest, prov)
    elif cmd == "review":
        review(rest or None, prov)
    elif cmd == "recent":
        review(None, prov)
    elif cmd == "sweep":
        sweep(prov, budget, force=force, notify=notify)
    else:
        print(__doc__)

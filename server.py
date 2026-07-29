"""GunnGPT web server.

    python server.py            (or: uvicorn server:app)

Serves the chat page and a streaming /api/chat endpoint. All inference is local
via Ollama; the index is loaded once at startup. A small concurrency queue and a
per-user rate limit let a single GPU serve a whole school politely.
"""
import datetime as dt
import json
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse

import config
from rag import Rag

app = FastAPI(title="GunnGPT")

# ---- capacity controls (tune for your GPU; override via env) ----
MAX_CONCURRENT = int(os.environ.get("GUNNGPT_MAX_CONCURRENT", "4"))  # simultaneous generations
QUEUE_WAIT = float(os.environ.get("GUNNGPT_QUEUE_WAIT", "45"))       # secs a request waits for a slot
RATE_PER_MIN = int(os.environ.get("GUNNGPT_RATE_PER_MIN", "20"))     # requests/min per user

_slots = threading.Semaphore(MAX_CONCURRENT)
_hits = defaultdict(deque)
_hits_lock = threading.Lock()
_rag = None

# ---- response cache: a school asks the same handful of questions constantly, so
# identical top-level queries are served instantly with zero GPU cost. ----
CACHE_TTL = int(os.environ.get("GUNNGPT_CACHE_TTL", "1800"))   # secs a cached answer is reused
CACHE_MAX = 500
_cache = {}
_cache_lock = threading.Lock()


def get_rag():
    global _rag
    if _rag is None:
        if not (os.path.exists(config.EMB_PATH) and os.path.exists(config.DOCS_PATH)):
            raise RuntimeError("Index not built. Run `python ingest.py` first.")
        _rag = Rag()
    return _rag


def _rate_ok(ip):
    """Sliding 60s window per user; True if under the limit."""
    now = time.time()
    with _hits_lock:
        dq = _hits[ip]
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= RATE_PER_MIN:
            return False
        dq.append(now)
        return True


def _cache_key(query):
    # Include today's date so "what's for lunch today" doesn't go stale overnight.
    return dt.date.today().isoformat() + "|" + " ".join(query.lower().split())


def _cache_get(key):
    now = time.time()
    with _cache_lock:
        v = _cache.get(key)
        if v and now - v[0] < CACHE_TTL:
            return v[1], v[2]
        if v:
            _cache.pop(key, None)
    return None


def _cache_set(key, answer, sources):
    with _cache_lock:
        if len(_cache) >= CACHE_MAX and key not in _cache:
            _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
        _cache[key] = (time.time(), answer, sources)


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
def index():
    return FileResponse(os.path.join(config.HERE, "static", "index.html"))


@app.get("/api/today")
def today():
    import bell
    return bell.today_schedule()


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    query = (body.get("message") or "").strip()
    history = body.get("history") or []
    attachments = body.get("attachments") or []
    if not query and not attachments:
        return {"error": "empty message"}

    # Behind Cloudflare Tunnel the real client IP is in this header.
    ip = req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "?")

    if not _rate_ok(ip):
        def limited():
            yield _sse({"type": "error", "text": "You're sending messages too fast — give it a few seconds."})
            yield _sse({"type": "done"})
        return StreamingResponse(limited(), media_type="text/event-stream")

    files_text = ""
    if attachments:
        import files as file_reader
        files_text = file_reader.extract_all(attachments)

    # Only cache standalone questions (no conversation context, no attachments).
    cache_key = _cache_key(query) if (not history and not attachments) else None

    def gen():
        # Cache hit → return instantly, no GPU, no queue.
        if cache_key:
            hit = _cache_get(cache_key)
            if hit:
                answer, sources = hit
                yield _sse({"type": "token", "text": answer})
                if sources:
                    yield _sse({"type": "sources", "sources": sources})
                yield _sse({"type": "done"})
                return
        # Miss → wait for a generation slot (the queue), then answer.
        if not _slots.acquire(timeout=QUEUE_WAIT):
            yield _sse({"type": "error", "text": "GunnGPT is really busy right now — try again in a moment."})
            yield _sse({"type": "done"})
            return
        acc, srcs, errored = "", None, False
        try:
            rag = get_rag()
            for ev in rag.answer_stream(query, history=history, files_text=files_text):
                if ev.get("type") == "token":
                    acc += ev.get("text", "")
                elif ev.get("type") == "sources":
                    srcs = ev.get("sources")
                yield _sse(ev)
        except Exception as e:
            errored = True
            yield _sse({"type": "error", "text": str(e)})
        finally:
            _slots.release()
        if cache_key and acc and not errored:
            _cache_set(cache_key, acc, srcs)
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("GUNNGPT_HOST", "127.0.0.1")
    port = int(os.environ.get("GUNNGPT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

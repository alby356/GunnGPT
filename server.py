"""GunnGPT web server.

    python server.py            (or: uvicorn server:app)

Serves the chat page and a streaming /api/chat endpoint. All inference is local
via Ollama; the index is loaded once at startup. A small concurrency queue and a
per-user rate limit let a single GPU serve a whole school politely.
"""
import datetime as dt
import json
import os
import re
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


@app.get("/campus-map.png")
def campus_map():
    return FileResponse(os.path.join(config.HERE, "static", "map.png"))


@app.get("/api/today")
def today(date: str = None):
    import bell
    d = None
    if date:
        try:
            d = dt.date.fromisoformat(date)
        except ValueError:
            d = None
    sc = bell.today_schedule(d)
    sc["date"] = (d or dt.date.today()).isoformat()
    return sc


_watt_cache = {}


def _watt_json(name):
    import requests
    now = time.time()
    c = _watt_cache.get(name)
    if c and now - c[0] < 21600:      # cache 6h
        return c[1]
    data = requests.get(f"{config.WATT_RAW}/{name}.json", timeout=15).json()
    _watt_cache[name] = (now, data)
    return data


@app.get("/api/clubs")
def clubs():
    out = []
    for c in _watt_json("clubs").get("data", {}).values():
        if not c.get("name"):
            continue
        out.append({
            "name": c["name"], "type": c.get("type", ""), "tier": c.get("tier", ""),
            "day": c.get("day", ""), "time": c.get("time", ""), "room": c.get("room", ""),
            "advisor": ", ".join(x for x in (c.get("advisor"), c.get("coadvisor")) if x),
            "prez": c.get("prez", ""), "desc": c.get("desc", ""),
        })
    out.sort(key=lambda x: x["name"].lower())
    return out


ACRONYMS = {"AP", "AB", "BC", "US", "IB", "CS", "SJ", "HN", "AAR", "POE", "IED",
            "PLTW", "VAPA", "CTE", "ELD", "ROTC", "PE", "ASB", "II", "III", "IV"}

# Short words that stay lowercase in a title (unless first/last word).
SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
               "nor", "of", "on", "or", "per", "the", "to", "via", "vs", "with"}

# WATT titles some courses oddly; use the official Gunn catalog names.
TITLE_FIXES = {
    "AP AB Calculus": "AP Calculus AB",
    "AP BC Calculus": "AP Calculus BC",
    "AP 2-D Art & Design Emphasis on Painting/Drawing": "AP 2-D Art & Design",
    "AP – Drawing": "AP Drawing",
}

# Courses in Gunn's official catalog that WATT's data is missing.
EXTRA_COURSES = [
    {
        "title": "AP Physics C: Electricity & Magnetism",
        "section": "Science",
        "grades": [11, 12],
        "length": "Semester",
        "credit": 'UC Approved "d"',
        "desc": ("The second semester of the traditional, calculus-based AP Physics C "
                 "course (paired with Mechanics). Equivalent to a college physics course "
                 "for majors and engineers, it covers electricity and magnetism and "
                 "prepares students for the AP Physics C: E&M exam. Students sign up for "
                 "both Mechanics (3859A) and E&M (3859E). Prerequisite: concurrent "
                 "enrollment in or completion of a calculus course (BC Calculus "
                 "recommended); a previous physics course is recommended."),
    },
]


def _cap_word(w):
    """Capitalize one word: keep acronyms/single letters uppercase, otherwise
    uppercase the first letter and lowercase the rest (punctuation-safe)."""
    core = "".join(ch for ch in w if ch.isalpha())
    if not core:
        return w
    if core.upper() in ACRONYMS or len(core) == 1:
        return w.upper()
    out, capped = [], False
    for ch in w:
        if ch.isalpha() and not capped:
            out.append(ch.upper()); capped = True
        else:
            out.append(ch.lower())
    return "".join(out)


def _titlecase(s):
    """Proper title case: acronyms stay uppercase, small words stay lowercase
    (except first/last), and hyphen/slash parts are each capitalized."""
    toks = s.split()
    n = len(toks)
    out = []
    for i, tok in enumerate(toks):
        bare = "".join(ch for ch in tok if ch.isalpha()).lower()
        if 0 < i < n - 1 and bare in SMALL_WORDS:
            out.append(tok.lower())
            continue
        # capitalize each piece around hyphens and slashes (e.g. "2-d" -> "2-D")
        out.append("".join(_cap_word(p) if p not in "-/" else p
                           for p in re.split(r"([-/])", tok)))
    return " ".join(out)


@app.get("/api/courses")
def courses():
    out, seen = [], set()
    for c in _watt_json("catalog").get("data", []):
        names = c.get("names", [])
        raw = names[0]["title"].strip().replace("*", "") if names else "Course"
        title = _titlecase(" ".join(raw.split()))
        title = TITLE_FIXES.get(title, title)
        if title.lower() in seen:                       # WATT lists some courses twice
            continue
        seen.add(title.lower())
        out.append({
            "title": title,
            "section": _titlecase(c.get("section") or ""),
            "grades": c.get("grades", []),
            "length": c.get("length", ""),
            "credit": c.get("credit", ""),
            "desc": " ".join((c.get("description") or "").split()),
        })
    for extra in EXTRA_COURSES:                          # add courses WATT is missing
        if extra["title"].lower() not in seen:
            out.append(dict(extra))
            seen.add(extra["title"].lower())
    out.sort(key=lambda x: x["title"])
    return out


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

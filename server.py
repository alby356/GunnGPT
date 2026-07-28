"""GunnGPT web server.

    uvicorn server:app --reload      (or: python server.py)

Serves the chat page and a streaming /api/chat endpoint. All inference is local
via Ollama; the index is loaded once at startup.
"""
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import config
from rag import Rag

app = FastAPI(title="GunnGPT")

_rag = None


def get_rag():
    global _rag
    if _rag is None:
        if not (os.path.exists(config.EMB_PATH) and os.path.exists(config.DOCS_PATH)):
            raise RuntimeError("Index not built. Run `python ingest.py` first.")
        _rag = Rag()
    return _rag


@app.get("/")
def index():
    return FileResponse(os.path.join(config.HERE, "static", "index.html"))


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    query = (body.get("message") or "").strip()
    history = body.get("history") or []
    attachments = body.get("attachments") or []
    if not query and not attachments:
        return {"error": "empty message"}

    files_text = ""
    if attachments:
        import files as file_reader
        files_text = file_reader.extract_all(attachments)

    def gen():
        try:
            rag = get_rag()
            for ev in rag.answer_stream(query, history=history, files_text=files_text):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

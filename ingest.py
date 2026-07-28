"""Build the GunnGPT vector index.

    python ingest.py

Fetches every source (sources.py), splits into chunks, embeds each chunk with
the local Ollama embedding model, and writes data/docs.json + data/embeddings.npy.
Re-run any time to refresh (e.g. to pull a new lunch menu during the school year).
"""
import json
import sys
import time

import numpy as np
import requests

import config
import sources


def chunk_text(text, size=config.CHUNK_CHARS, overlap=config.CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):                       # try to break on a boundary
            nl = text.rfind("\n", start + size // 2, end)
            sp = text.rfind(" ", start + size // 2, end)
            brk = nl if nl != -1 else sp
            if brk != -1:
                end = brk
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def embed(text):
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def check_ollama():
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
    except Exception:
        sys.exit("ERROR: Ollama isn't running. Start it with `ollama serve` "
                 "(or open the Ollama app), then re-run.")
    have = {m["name"].split(":")[0] for m in tags.get("models", [])}
    for model in (config.EMBED_MODEL, config.CHAT_MODEL):
        if model.split(":")[0] not in have:
            sys.exit(f"ERROR: model '{model}' not found. Run: ollama pull {model}")


def main():
    check_ollama()
    print("Fetching sources...")
    docs = sources.build_all()
    print(f"Built {len(docs)} documents. Chunking + embedding...")

    chunks, meta = [], []
    for d in docs:
        for i, ch in enumerate(chunk_text(d["text"])):
            chunks.append(ch)
            meta.append({"title": d["title"], "url": d["url"],
                         "source": d["source"], "doc_id": d["id"], "chunk": i,
                         "text": ch})

    vectors = []
    t0 = time.time()
    for i, ch in enumerate(chunks):
        vectors.append(embed(ch))
        if (i + 1) % 25 == 0 or i + 1 == len(chunks):
            rate = (i + 1) / (time.time() - t0)
            print(f"  embedded {i + 1}/{len(chunks)}  ({rate:.0f}/s)", end="\r")
    print()

    mat = np.array(vectors, dtype=np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)   # normalize

    np.save(config.EMB_PATH, mat)
    with open(config.DOCS_PATH, "w") as f:
        json.dump(meta, f)
    print(f"Saved {len(chunks)} chunks -> {config.EMB_PATH} + {config.DOCS_PATH}")


if __name__ == "__main__":
    main()

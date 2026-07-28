# GunnGPT

A fully local **RAG** (retrieval-augmented generation) chatbot that answers
questions about Henry M. Gunn High School — bell schedules, lunch menus,
teachers, courses, clubs, and everything on the Gunn Wiki.

Nothing leaves your machine: embeddings and the language model both run locally
through [Ollama](https://ollama.com). No API keys, no cloud, no cost.

## How it works

```
sources.py   fetch WATT (schedule, staff, courses, clubs) + Nutrislice lunch + Gunn Wiki  → text docs
ingest.py    chunk each doc, embed with a local model (nomic-embed-text)                   → data/*.npy
rag.py       embed your question → cosine top-k → stuff into prompt → local chat model      → answer
server.py    FastAPI web app + streaming chat UI (static/index.html)
```

It's a **plain RAG**: everything is indexed once. Re-run `ingest.py` to refresh
(e.g. to pull the current lunch menu during the school year).

## Setup

**1. Install Ollama** (one time):
```bash
brew install ollama          # or download from https://ollama.com/download
```
Then start it (leave running):
```bash
ollama serve
```

**2. Pull the two local models** (one time, ~5–9 GB):
```bash
ollama pull nomic-embed-text     # embeddings for retrieval
ollama pull qwen2.5:14b          # the chat model (see config.py to change)
```

**3. Install Python deps** (a virtual env is recommended):
```bash
cd ~/GunnGPT
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**4. Build the index** (fetches everything, embeds it — a few minutes):
```bash
python ingest.py
```

**5. Run it:**
```bash
python server.py
```
Open http://127.0.0.1:8000 and ask away.

## Quick test without the web app
```bash
python rag.py "What is Mr. Bautista like?"
```

## Configuration

Edit `config.py` (or set env vars) to change models or retrieval:

| Setting | Default | Notes |
|---|---|---|
| `CHAT_MODEL` | `qwen2.5:14b` | any Ollama chat model, e.g. `llama3.1:8b` (faster) |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `TOP_K` | `6` | how many chunks are fed to the model |

## Notes & limitations

- **Lunch** is the only date-specific source. It's snapshotted at ingest time,
  so re-run `python ingest.py` to refresh it. During summer break the menu feed
  is empty (nothing to show until the school year starts).
- Answers are grounded in the retrieved context and cite their sources, but the
  Gunn Wiki is student-written — treat teacher "personality" content as opinion.
- Data sources: [WATT](https://github.com/GunnWATT/watt) (bell schedule, staff,
  courses, clubs), PAUSD Nutrislice (lunch), and [Gunn Wiki](https://gunnwiki.org).

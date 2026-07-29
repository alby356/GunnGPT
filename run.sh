#!/usr/bin/env bash
# GunnGPT launcher — starts Ollama (with concurrency) + the web server.
# Usage:  ./run.sh          (Linux / macOS / WSL)
set -e
cd "$(dirname "$0")"

# ---- tunables (override by exporting before running) ----
export GUNNGPT_CHAT_MODEL="${GUNNGPT_CHAT_MODEL:-qwen2.5:7b}"   # smaller = more concurrent users
export GUNNGPT_MAX_CONCURRENT="${GUNNGPT_MAX_CONCURRENT:-4}"    # simultaneous generations
export GUNNGPT_RATE_PER_MIN="${GUNNGPT_RATE_PER_MIN:-20}"       # per-user requests/min
export GUNNGPT_TOP_K="${GUNNGPT_TOP_K:-4}"                      # chunks retrieved (lower = faster/less VRAM)
export GUNNGPT_MAX_TOKENS="${GUNNGPT_MAX_TOKENS:-600}"          # cap answer length
export GUNNGPT_CACHE_TTL="${GUNNGPT_CACHE_TTL:-1800}"          # secs a repeated question is cached
export GUNNGPT_HOST="${GUNNGPT_HOST:-127.0.0.1}"               # 127.0.0.1 = only Cloudflare Tunnel can reach it
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"          # let Ollama batch requests
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m

# start Ollama if it isn't already running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama..."
  ollama serve > ollama.log 2>&1 &
  for i in $(seq 1 20); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi

# make sure the models are pulled
ollama pull "$GUNNGPT_CHAT_MODEL"
ollama pull nomic-embed-text

# activate the virtualenv if present
[ -d .venv ] && source .venv/bin/activate

# build the index the first time
if [ ! -f data/embeddings.npy ]; then
  echo "No index found — building it (first run)..."
  python ingest.py
fi

echo "GunnGPT: http://$GUNNGPT_HOST:8000  |  model: $GUNNGPT_CHAT_MODEL  |  concurrency: $GUNNGPT_MAX_CONCURRENT"
python server.py

"""Central configuration for GunnGPT.

Everything is local: embeddings and generation both run through Ollama on
your own machine. Change the model names here (or via environment variables)
if you pulled different models.
"""
import os

# ---- Ollama ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Chat model that writes the final answer. Any Ollama chat model works.
# Good picks: "llama3.1:8b", "qwen2.5:7b", "qwen2.5:14b".
CHAT_MODEL = os.environ.get("GUNNGPT_CHAT_MODEL", "qwen2.5:14b")

# Embedding model used for retrieval. "nomic-embed-text" is small and good.
EMBED_MODEL = os.environ.get("GUNNGPT_EMBED_MODEL", "nomic-embed-text")

# How long Ollama keeps the chat model in RAM after a reply. Shorter = frees
# ~10GB sooner (less idle lag) at the cost of a slow first reply after idle.
KEEP_ALIVE = os.environ.get("GUNNGPT_KEEP_ALIVE", "30s")

# Lower temperature = sticks to facts, less embellishment/hallucination.
TEMPERATURE = float(os.environ.get("GUNNGPT_TEMPERATURE", "0.2"))

# Cap answer length — shorter answers free the GPU faster (more concurrent users).
MAX_TOKENS = int(os.environ.get("GUNNGPT_MAX_TOKENS", "600"))

# ---- Retrieval ----
TOP_K = int(os.environ.get("GUNNGPT_TOP_K", "8"))   # chunks fed to the model
KEYWORD_WEIGHT = 0.3                                  # hybrid: keyword vs embedding
DATE_BOOST = 0.6                                      # boost docs matching a queried date
CHUNK_CHARS = 1100                                   # ~ a few paragraphs
CHUNK_OVERLAP = 150

# ---- Data sources ----
WATT_RAW = "https://raw.githubusercontent.com/GunnWATT/watt/main/scripts/output"
WIKI_API = "https://gunnwiki.org/api.php"
WIKI_BASE = "https://gunnwiki.org"
NUTRISLICE = (
    "https://pausd.api.nutrislice.com/menu/api/weeks/school/"
    "henry-m-gunn-hs/menu-type/{meal}/{year}/{month:02d}/{day:02d}"
)

# ---- Paths ----
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DOCS_PATH = os.path.join(DATA_DIR, "docs.json")       # chunk text + metadata
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")   # matrix aligned to docs

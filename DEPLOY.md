# Deploying GunnGPT for your school

Self-host on one PC (a decent GPU like an RTX 3080 Ti handles a school of ~1,000
students, as long as you size the model for **concurrency**, not just speed).

## 1. Set up the PC (once)

```bash
git clone https://github.com/alby356/GunnGPT.git
cd GunnGPT
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install Ollama from https://ollama.com and pull the models:

```bash
ollama pull qwen2.5:7b        # the answering model (7B = lots of concurrency headroom on 12GB VRAM)
ollama pull nomic-embed-text  # embeddings for retrieval
```

> **Model choice for a school:** use **`qwen2.5:7b`** (or `llama3.1:8b`), not 14B.
> Of 1,000 students only a handful generate at the *same instant*, but lunchtime
> spikes can be 20–50. A 7–8B model lets the GPU run several in parallel; 14B
> can only do a couple before people wait. Quality loss is tiny for a grounded RAG bot.

## 2. Run it

```bash
./run.sh
```

This starts Ollama with batching enabled, builds the index on first run, and
launches the server on `http://127.0.0.1:8000`. Tunables (export before running):

| Env var | Default | Meaning |
|---|---|---|
| `GUNNGPT_CHAT_MODEL` | `qwen2.5:7b` | answering model |
| `GUNNGPT_MAX_CONCURRENT` | `4` | simultaneous generations (match `OLLAMA_NUM_PARALLEL`) |
| `OLLAMA_NUM_PARALLEL` | `4` | how many requests Ollama batches on the GPU |
| `GUNNGPT_RATE_PER_MIN` | `20` | per-user request cap (anti-spam) |

Requests beyond `MAX_CONCURRENT` **queue** and stream as slots free up; if the GPU
stays swamped for ~45s, the user gets a friendly "busy, try again" message.

**Windows (PowerShell)** — no `run.sh`; run the equivalent:
```powershell
$env:GUNNGPT_CHAT_MODEL="qwen2.5:7b"; $env:OLLAMA_NUM_PARALLEL="4"; $env:GUNNGPT_MAX_CONCURRENT="4"
ollama serve   # in one terminal
python server.py   # in another (venv activated)
```
(Or just use WSL and `./run.sh`.)

## 3. Put it online with Cloudflare Tunnel (free, no port-forwarding)

Cloudflare Tunnel exposes `localhost:8000` to the internet over HTTPS, hides your
home IP, and adds DDoS/rate protection — without opening any router ports.

**Quick test (no account, temporary URL):**
```bash
# install cloudflared (https://developers.cloudflare.com/cloudflare-tunnel/), then:
cloudflared tunnel --url http://localhost:8000
```
It prints a random `https://something.trycloudflare.com` URL you can share to test.

**Production (your own domain, stable URL):**
1. Add your domain to Cloudflare (free plan).
2. `cloudflared tunnel login`
3. `cloudflared tunnel create gunngpt`
4. Create `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: gunngpt
   credentials-file: /home/YOU/.cloudflared/<tunnel-id>.json
   ingress:
     - hostname: gunngpt.yourdomain.com
       service: http://localhost:8000
     - service: http_status:404
   ```
5. Route DNS + run:
   ```bash
   cloudflared tunnel route dns gunngpt gunngpt.yourdomain.com
   cloudflared tunnel run gunngpt
   ```
Now students go to `https://gunngpt.yourdomain.com`. Keep `run.sh` **and**
`cloudflared` both running (screen/tmux, a systemd service, or Windows startup task).

## 4. Keeping data fresh

Re-run the ingest whenever menus/pages update (e.g. weekly / when a new month of
lunch is posted):
```bash
source .venv/bin/activate && python ingest.py
```
Then restart the server so it loads the new index.

## Reality check
- **Single point of failure** — one PC. If it/your internet/power drops, it's down.
- **Peak spikes queue** — fine for a soft launch; if it goes viral, rent a cloud GPU.
- Keep the PC on during school hours (ideally 24/7). Text streaming is tiny
  bandwidth (~KB/s per user), so home upload is not the bottleneck.

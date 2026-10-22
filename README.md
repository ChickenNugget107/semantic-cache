# Semantic Caching for LLM APIs

A cloud-deployable service that caches LLM responses **by meaning**, not by exact
string match. A reworded question ("What's the capital of France?" vs. "Tell me
France's capital city") hits the cache and skips the LLM call — cutting latency
and cost.

This implements the core idea from **GPTCache: An Open-Source Semantic Cache for
LLM Applications** (Bang, 2023). The paper's claim is simple and measurable:
semantic caching raises the cache hit rate on near-duplicate queries far above an
exact-match cache, which lowers both latency and API cost. This service
reproduces that claim and shows it live.

## How it works

1. Each incoming question is embedded into a vector with a local
   sentence-transformers model (`all-MiniLM-L6-v2`).
2. We compare it to every cached question's vector using cosine similarity.
3. If the best match is above a threshold (default `0.85`), we return the cached
   answer — a **cache hit**, no LLM call.
4. Otherwise it's a **cache miss**: we call the LLM, store the answer, and return it.

```
question ──▶ embed ──▶ similarity search ──┬── hit  ──▶ cached answer   (fast, $0)
                                           └── miss ──▶ LLM API ──▶ store ──▶ answer
```

## Architecture (what maps to "cloud")

| Component            | Role                                              |
|----------------------|---------------------------------------------------|
| FastAPI service      | HTTP API + demo UI, runs in a container           |
| Embedding model      | Local vector encoder (no second API key needed)   |
| In-memory vector store | The semantic cache                              |
| LLM provider         | OpenAI-compatible endpoint (Groq by default)      |
| Cloud Run / App Service | Hosts the container, scales to zero when idle   |
| Cloud monitoring     | Collects the latency/cost numbers for the report  |

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
# open http://localhost:8080
```

It runs in **demo mode** with no API key (stubbed answers, but caching, latency,
and hit-rate are all real). To use a real LLM, set a key (Groq is free):

```bash
export LLM_API_KEY="your_groq_key"      # from https://console.groq.com
uvicorn app:app --port 8080
```

## Endpoints

| Method | Path       | Purpose                                      |
|--------|------------|----------------------------------------------|
| POST   | `/ask`     | `{"question": "...", "use_cache": true}`     |
| GET    | `/stats`   | hit rate, latency, speedup, cost saved       |
| POST   | `/reset`   | clear the cache and stats (for a clean demo) |
| GET    | `/healthz` | health check                                 |
| GET    | `/`        | demo web page                                |

## Configuration (environment variables)

| Variable               | Default                              | Meaning                              |
|------------------------|--------------------------------------|--------------------------------------|
| `LLM_API_KEY`          | *(empty → demo mode)*                | API key for the LLM provider         |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1`     | OpenAI-compatible endpoint           |
| `LLM_MODEL`            | `llama-3.1-8b-instant`               | Model name                           |
| `SIMILARITY_THRESHOLD` | `0.85`                               | Higher = stricter matching           |
| `COST_PER_CALL_USD`    | `0.0005`                             | Illustrative cost per LLM call       |

---

## Deploy to the cloud

### Option A — Google Cloud Run (recommended; one command, scales to zero)

```bash
gcloud run deploy semantic-cache \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars LLM_API_KEY=your_groq_key
```

Cloud Run builds the Dockerfile, deploys it, and prints a public HTTPS URL. The
`1Gi` memory is so the embedding model loads comfortably. `asia-south1` is Mumbai.

### Option B — Render (easiest if gcloud fights you; free tier, no card)

1. Push this folder to a GitHub repo.
2. On render.com: **New → Web Service → connect the repo**.
3. Render detects the Dockerfile automatically.
4. Add env var `LLM_API_KEY`, click deploy, get a public URL.

### Option C — Azure (if you have Azure for Students credits, no card)

```bash
az containerapp up \
  --name semantic-cache \
  --resource-group demo-rg \
  --source . \
  --ingress external \
  --env-vars LLM_API_KEY=your_groq_key
```

> First deploy builds the container with the model baked in, so the image is
> ~1 GB and the first build takes a few minutes. After that, cold starts are fast.

---

## Demo script (for the viva / presentation)

1. Open the deployed URL. Click **Reset** for a clean slate.
2. Ask **"What is the capital of France?"** → badge shows **LLM CALL** (a miss).
3. Ask **"Tell me France's capital city."** → badge shows **CACHE HIT**, with the
   similarity score and the original question it matched. No LLM call.
4. Ask **"Which city is the capital of France?"** → another **CACHE HIT**.
5. Point at the dashboard: **hit rate** climbing, **speedup** (e.g. 15–40×),
   **cost saved** rising.
6. Tick off **use semantic cache** and re-ask a duplicate → it goes back to the
   LLM every time. This is your A/B: with-cache vs. without-cache.

That contrast — hit rate and latency with caching on versus off — is the paper's
claim, reproduced on your own cloud deployment.

## What to say about scope

Implemented: semantic matching, the hit/miss decision, live latency and
cost measurement, and a cloud deployment. Honest "future work": a persistent
vector store (Redis / a managed vector DB) instead of in-memory, and cache
eviction policies. Keeping the cache in-memory is a deliberate demo choice — it
makes the before/after comparison clean and needs no extra infrastructure.

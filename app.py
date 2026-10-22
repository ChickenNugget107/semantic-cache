"""
Semantic Caching for LLM APIs
------------------------------
A cloud-deployable service that caches LLM responses by *meaning*, not by
exact string match. When a new query arrives, we embed it and compare it to
previously-seen queries. If a past query is semantically similar enough
(cosine similarity above a threshold), we return the cached answer instead of
calling the LLM again -- saving latency and cost.

This is the core idea behind GPTCache ("semantic cache for LLMs").

Design choices for a one-night build:
- Embeddings run locally (sentence-transformers), so you only need ONE API key
  (for the LLM itself), not two.
- The cache is in-memory. Simple, fast, and resets on restart -- which is fine
  for a demo and actually makes the before/after comparison clean.
- The LLM layer degrades gracefully: with no API key set, it runs in DEMO mode
  and returns a stubbed answer, so you can show the caching behavior working
  even before you have a key.
"""

import os
import time
import threading

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding

# ---------------------------------------------------------------------------
# Config (all overridable with environment variables on the cloud)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
# fastembed uses ONNX runtime (no PyTorch), so the service fits comfortably in
# a 512 MB instance -- which is what Render's free tier gives. Same MiniLM model.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM provider: defaults to Groq's OpenAI-compatible endpoint (free tier).
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Rough cost estimate used only to illustrate savings in the demo.
# (Groq's free tier is actually $0; this is a representative paid-rate figure
# so the "money saved" number is meaningful in your report.)
COST_PER_CALL_USD = float(os.getenv("COST_PER_CALL_USD", "0.0005"))

# ---------------------------------------------------------------------------
# Embedding model -- loaded once at startup
# ---------------------------------------------------------------------------
print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
print("Embedding model ready.")


def embed(text: str) -> np.ndarray:
    # fastembed returns a generator of numpy arrays, one per input text.
    vec = np.asarray(next(embedder.embed([text])), dtype=np.float32)
    # Normalize so cosine similarity is just a dot product. (MiniLM output is
    # already near-unit length; normalizing again is a harmless safeguard.)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ---------------------------------------------------------------------------
# The semantic cache
# ---------------------------------------------------------------------------
class SemanticCache:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._lock = threading.Lock()
        self.questions = []      # list[str]
        self.answers = []        # list[str]
        self.vectors = None      # np.ndarray of shape (n, dim), normalized

    def lookup(self, query_vec: np.ndarray):
        """Return (answer, similarity, matched_question) or (None, best_sim, None)."""
        with self._lock:
            if self.vectors is None or len(self.answers) == 0:
                return None, 0.0, None
            # Cosine similarity == dot product, since all vectors are normalized.
            sims = self.vectors @ query_vec
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= self.threshold:
                return self.answers[best_idx], best_sim, self.questions[best_idx]
            return None, best_sim, None

    def store(self, question: str, answer: str, query_vec: np.ndarray):
        with self._lock:
            self.questions.append(question)
            self.answers.append(answer)
            row = query_vec.reshape(1, -1)
            self.vectors = row if self.vectors is None else np.vstack([self.vectors, row])

    def clear(self):
        with self._lock:
            self.questions.clear()
            self.answers.clear()
            self.vectors = None

    def size(self):
        with self._lock:
            return len(self.answers)


cache = SemanticCache(SIMILARITY_THRESHOLD)

# ---------------------------------------------------------------------------
# Running stats (for the demo dashboard and your results section)
# ---------------------------------------------------------------------------
stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "llm_time_total": 0.0,    # seconds spent on real LLM calls
    "cache_time_total": 0.0,  # seconds spent serving from cache
}
stats_lock = threading.Lock()


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible; works with Groq, OpenAI, Together, etc.)
# ---------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    if not LLM_API_KEY:
        # DEMO mode -- no key configured. Simulate realistic latency so the
        # latency comparison in the demo still tells the right story.
        time.sleep(0.9)
        return f"[demo answer] You asked: {prompt}"

    # Import here so the app still boots in demo mode without the package.
    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Semantic LLM Cache")


class AskRequest(BaseModel):
    question: str
    use_cache: bool = True


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    query = req.question.strip()
    query_vec = embed(query)

    cached_answer, similarity, matched = (None, 0.0, None)
    if req.use_cache:
        cached_answer, similarity, matched = cache.lookup(query_vec)

    if cached_answer is not None:
        elapsed = time.time() - t0
        with stats_lock:
            stats["total_requests"] += 1
            stats["cache_hits"] += 1
            stats["cache_time_total"] += elapsed
        return {
            "answer": cached_answer,
            "source": "cache",
            "similarity": round(similarity, 4),
            "matched_question": matched,
            "latency_seconds": round(elapsed, 3),
            "cost_usd": 0.0,
        }

    # Cache miss (or caching disabled) -> real LLM call
    answer = call_llm(query)
    if req.use_cache:
        cache.store(query, answer, query_vec)
    elapsed = time.time() - t0
    with stats_lock:
        stats["total_requests"] += 1
        stats["cache_misses"] += 1
        stats["llm_time_total"] += elapsed
    return {
        "answer": answer,
        "source": "llm",
        "similarity": round(similarity, 4),
        "matched_question": None,
        "latency_seconds": round(elapsed, 3),
        "cost_usd": COST_PER_CALL_USD,
    }


@app.get("/stats")
def get_stats():
    with stats_lock:
        s = dict(stats)
    hits, total = s["cache_hits"], s["total_requests"]
    hit_rate = (hits / total) if total else 0.0
    avg_llm = (s["llm_time_total"] / s["cache_misses"]) if s["cache_misses"] else 0.0
    avg_cache = (s["cache_time_total"] / s["cache_hits"]) if s["cache_hits"] else 0.0
    return {
        "total_requests": total,
        "cache_hits": hits,
        "cache_misses": s["cache_misses"],
        "hit_rate": round(hit_rate, 4),
        "cache_entries": cache.size(),
        "avg_llm_latency_seconds": round(avg_llm, 3),
        "avg_cache_latency_seconds": round(avg_cache, 3),
        "speedup_x": round(avg_llm / avg_cache, 1) if avg_cache else None,
        "cost_saved_usd": round(hits * COST_PER_CALL_USD, 4),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "demo_mode": not bool(LLM_API_KEY),
    }


@app.post("/reset")
def reset():
    cache.clear()
    with stats_lock:
        for k in stats:
            stats[k] = 0 if isinstance(stats[k], int) else 0.0
    return {"ok": True}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html")) as f:
        return f.read()

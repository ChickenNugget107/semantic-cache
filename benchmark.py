"""
Benchmark the semantic cache against the same workload with caching OFF.
Produces the before/after numbers for your results section.

Usage (against a running server):
    python benchmark.py http://localhost:8080
    python benchmark.py https://your-cloud-url
"""
import sys
import time
import json
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8080"

# Each group is one "intent" asked several ways. The first ask in a group is a
# genuine miss; the rewordings should hit the cache.
WORKLOAD = [
    "What is the capital of France?",
    "Tell me France's capital city.",
    "Which city is the capital of France?",
    "How does photosynthesis work?",
    "Explain the process of photosynthesis.",
    "What is the boiling point of water?",
    "At what temperature does water boil?",
    "Who wrote Romeo and Juliet?",
    "Which author wrote Romeo and Juliet?",
    "What is the capital of France?",  # exact repeat
]


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.load(r)


def run(use_cache: bool, label: str):
    post("/reset", {})
    t0 = time.time()
    for q in WORKLOAD:
        post("/ask", {"question": q, "use_cache": use_cache})
    wall = time.time() - t0
    s = get("/stats")
    print(f"\n=== {label} ===")
    print(f"  requests:       {s['total_requests']}")
    print(f"  cache hits:     {s['cache_hits']}")
    print(f"  hit rate:       {s['hit_rate']*100:.0f}%")
    print(f"  total wall time:{wall:.2f}s")
    print(f"  cost incurred:  ${(s['cache_misses'])*s['similarity_threshold']*0 + s['cache_misses']*0.0005:.4f}")
    print(f"  cost saved:     ${s['cost_saved_usd']:.4f}")
    return wall


if __name__ == "__main__":
    print(f"Benchmarking {BASE} with a {len(WORKLOAD)}-query workload...")
    without = run(False, "CACHING OFF (baseline)")
    with_cache = run(True, "CACHING ON (semantic)")
    if with_cache:
        print(f"\nOverall wall-time speedup: {without/with_cache:.1f}x faster with caching\n")

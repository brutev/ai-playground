import math
import requests

OLLAMA_BASE = "http://localhost:11434"
MODEL = "nomic-embed-text"


# ── helper: embedding ─────────────────────────────────────────
def embed(text):
    text = text.lower().strip()
    response = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": MODEL, "prompt": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]


# ── cosine similarity ─────────────────────────────────────────
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ── dataset (your "database") ─────────────────────────────────
documents = [
    "car",
    "automobile",
    "bike",
    "banana",
    "software developer",
    "python programming",
    "truck",
    "fruit salad"
]

print("\n🔄 Step 1: Embedding all documents...")
doc_vectors = {doc: embed(doc) for doc in documents}


# ── search function ───────────────────────────────────────────
def search(query, top_k=3):
    print(f"\n🔍 Searching for: '{query}'")

    query_vec = embed(query)

    scores = []
    for doc, vec in doc_vectors.items():
        score = cosine_similarity(query_vec, vec)
        scores.append((doc, score))

    # sort by similarity (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 Results:")
    for doc, score in scores[:top_k]:
        bar = "█" * int((score + 1) * 10)
        print(f"{doc:<25} {score:.4f} {bar}")


# ── try queries ───────────────────────────────────────────────
search("2 wheeler")
search("IT job")
search("healthy food")
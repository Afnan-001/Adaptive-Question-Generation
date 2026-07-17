import json
import time
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# ----------------------------
# ✅ Load .env
# ----------------------------
load_dotenv()

# ----------------------------
# ✅ Files
# ----------------------------
INPUT_FILE = Path("data/dbms_chunks.json")
INDEX_FILE = Path("data/chunk_index.json")
EMBEDDINGS_FILE = Path("data/chunk_embeddings.npy")

# ----------------------------
# ✅ Internal Embedding API Config
# ----------------------------
# Base endpoint should be the /v1 endpoint, NOT /v1/chat/completions
EMBEDDING_API_BASE = os.getenv(
    "EMBEDDING_API_BASE",
    "http://10.221.0.164:4000/v1"
)

EMBEDDING_ENDPOINT = f"{EMBEDDING_API_BASE.rstrip('/')}/embeddings"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "si-rca-dds-text-embedding-3-small"
)

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# ----------------------------
# ✅ Batch Settings
# ----------------------------
BATCH_SIZE = 32
SLEEP_BETWEEN_BATCHES = 0.3
NORMALIZE_EMBEDDINGS = True


# ----------------------------
# ✅ Load Chunks
# ----------------------------
def load_chunks():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"❌ Missing file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("❌ dbms_chunks.json is empty or invalid")

    return chunks


# ----------------------------
# ✅ Normalize Vector
# ----------------------------
def normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)

    if norm == 0:
        return arr

    return arr / norm


# ----------------------------
# ✅ Call Embedding API
# ----------------------------
def get_embeddings(texts):
    """
    Calls internal OpenAI-compatible embedding endpoint.

    Input:
        texts: list[str]

    Output:
        list[list[float]]
    """

    if not EMBEDDING_API_KEY:
        raise ValueError(
            "❌ EMBEDDING_API_KEY not found. "
            "Set it in your .env file or environment variables."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDING_API_KEY}"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    response = requests.post(
        EMBEDDING_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"❌ Embedding API failed\n"
            f"Status Code: {response.status_code}\n"
            f"Response: {response.text}"
        )

    data = response.json()

    if "data" not in data:
        raise ValueError(f"❌ Invalid embedding response: {data}")

    embeddings = []

    for item in data["data"]:
        if "embedding" not in item:
            raise ValueError(f"❌ Missing embedding in response item: {item}")

        vec = item["embedding"]

        if NORMALIZE_EMBEDDINGS:
            vec = normalize_vector(vec).tolist()

        embeddings.append(vec)

    return embeddings


# ----------------------------
# ✅ Main
# ----------------------------
def main():
    print(f"📥 Loading chunks from: {INPUT_FILE}")

    chunks = load_chunks()

    print(f"✅ Loaded {len(chunks)} chunks")
    print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
    print(f"🌐 Embedding endpoint: {EMBEDDING_ENDPOINT}")

    texts = [c["text"] for c in chunks]

    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "book_name": c["book_name"],
            "page_number": c["page_number"]
        }
        for c in chunks
    ]

    all_vectors = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_no = i // BATCH_SIZE + 1
        batch_texts = texts[i:i + BATCH_SIZE]

        print(f"🧠 Processing batch {batch_no}/{total_batches} | {len(batch_texts)} chunks")

        try:
            vectors = get_embeddings(batch_texts)

            if len(vectors) != len(batch_texts):
                raise ValueError(
                    f"❌ Embedding count mismatch. "
                    f"Expected {len(batch_texts)}, got {len(vectors)}"
                )

            all_vectors.extend(vectors)

        except Exception as e:
            print(f"❌ Failed at batch {batch_no}: {e}")
            raise

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # ----------------------------
    # ✅ Save Index
    # ----------------------------
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # ----------------------------
    # ✅ Save Embeddings
    # ----------------------------
    embedding_matrix = np.array(all_vectors, dtype=np.float32)

    np.save(EMBEDDINGS_FILE, embedding_matrix)

    print("\n✅ Embedding generation complete!")
    print(f"📦 Saved chunk index: {INDEX_FILE}")
    print(f"📦 Saved embeddings: {EMBEDDINGS_FILE}")
    print(f"📐 Embedding matrix shape: {embedding_matrix.shape}")


# ----------------------------
# ✅ Run
# ----------------------------
if __name__ == "__main__":
    main()
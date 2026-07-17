import json
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ----------------------------
# ✅ Load .env
# ----------------------------
load_dotenv()

# ----------------------------
# Files
# ----------------------------
CHUNKS_FILE = Path("data/dbms_chunks.json")
CONCEPTS_FILE = Path("data/enriched_concepts.json")

CHUNK_INDEX_FILE = Path("data/chunk_index.json")
CHUNK_EMBEDDINGS_FILE = Path("data/chunk_embeddings.npy")

OUTPUT_FILE = Path("data/concept_chunk_map.json")
SCORES_OUTPUT_FILE = Path("data/concept_chunk_scores.json")

# ----------------------------
# Embedding API Config
# ----------------------------
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
# Config
# ----------------------------
TOP_K_CHUNKS_PER_CONCEPT = 25
CONCEPT_BATCH_SIZE = 16
NORMALIZE_EMBEDDINGS = True

# To force threshold filtering later, set e.g. 0.25
# For now kept 0.0 so every concept gets top-k chunks.
MIN_SIMILARITY = 0.0


# ----------------------------
# Load JSON
# ----------------------------
def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Normalize Vector / Matrix
# ----------------------------
def normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)

    if norm == 0:
        return arr

    return arr / norm


def normalize_matrix(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


# ----------------------------
# Embed Text Batch
# ----------------------------
def embed_texts(texts):
    """
    Calls internal OpenAI-compatible embedding endpoint.

    Input:
        texts: list[str]

    Output:
        np.ndarray shape = (len(texts), embedding_dim)
    """

    if not EMBEDDING_API_KEY:
        raise ValueError(
            "❌ EMBEDDING_API_KEY missing. Set it in your .env file."
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
        timeout=180
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

    # Sort by index to preserve input order
    items = sorted(data["data"], key=lambda x: x.get("index", 0))

    vectors = []

    for item in items:
        if "embedding" not in item:
            raise ValueError(f"❌ Missing embedding in response item: {item}")

        vec = item["embedding"]

        if NORMALIZE_EMBEDDINGS:
            vec = normalize_vector(vec)

        vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


# ----------------------------
# Build Concept Query Text
# ----------------------------
def build_concept_query(concept_obj):
    """
    Builds a semantic query for the concept using graph context.
    """

    concept = concept_obj.get("concept", "")

    parents = concept_obj.get("parent_concept", [])
    prereqs = concept_obj.get("prerequisites", [])

    if isinstance(parents, str):
        parents = [parents]

    if isinstance(prereqs, str):
        prereqs = [prereqs]

    query_parts = [
        f"DBMS concept: {concept}"
    ]

    if parents:
        query_parts.append(f"Parent concepts: {', '.join(parents)}")

    if prereqs:
        query_parts.append(f"Prerequisite concepts: {', '.join(prereqs)}")

    return "\n".join(query_parts)


# ----------------------------
# Main
# ----------------------------
def main():
    print("📥 Loading data...")

    chunks = load_json(CHUNKS_FILE)
    concepts = load_json(CONCEPTS_FILE)
    chunk_index = load_json(CHUNK_INDEX_FILE)

    chunk_embeddings = np.load(CHUNK_EMBEDDINGS_FILE).astype(np.float32)

    print(f"✅ Loaded {len(chunks)} chunks")
    print(f"✅ Loaded {len(concepts)} concepts")
    print(f"✅ Loaded {len(chunk_index)} chunk index records")
    print(f"✅ Chunk embedding matrix shape: {chunk_embeddings.shape}")

    if len(chunk_index) != chunk_embeddings.shape[0]:
        raise ValueError(
            f"❌ Mismatch: chunk_index has {len(chunk_index)} entries, "
            f"but embeddings matrix has {chunk_embeddings.shape[0]} rows"
        )

    # Normalize chunk embeddings again for safety
    if NORMALIZE_EMBEDDINGS:
        chunk_embeddings = normalize_matrix(chunk_embeddings)

    # ----------------------------
    # Build chunk ID list in embedding order
    # ----------------------------
    chunk_ids_by_row = [item["chunk_id"] for item in chunk_index]

    # ----------------------------
    # Build concept queries
    # ----------------------------
    concept_queries = []
    concept_names = []

    for concept_obj in concepts:
        concept_name = concept_obj.get("concept")

        if not concept_name:
            continue

        concept_names.append(concept_name)
        concept_queries.append(build_concept_query(concept_obj))

    print(f"🧠 Built {len(concept_queries)} concept semantic queries")
    print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
    print(f"🌐 Embedding endpoint: {EMBEDDING_ENDPOINT}")

    # ----------------------------
    # Embed concept queries in batches
    # ----------------------------
    all_query_embeddings = []

    total_batches = (len(concept_queries) + CONCEPT_BATCH_SIZE - 1) // CONCEPT_BATCH_SIZE

    for i in tqdm(range(0, len(concept_queries), CONCEPT_BATCH_SIZE), desc="Embedding concepts"):
        batch = concept_queries[i:i + CONCEPT_BATCH_SIZE]

        query_vectors = embed_texts(batch)

        all_query_embeddings.append(query_vectors)

    concept_embeddings = np.vstack(all_query_embeddings).astype(np.float32)

    print(f"✅ Concept embedding matrix shape: {concept_embeddings.shape}")

    # ----------------------------
    # Similarity Search
    # Since vectors are normalized, dot product = cosine similarity
    # ----------------------------
    print("🔎 Computing semantic similarities...")

    similarity_matrix = concept_embeddings @ chunk_embeddings.T

    concept_chunk_map = {}
    concept_chunk_scores = {}

    for concept_idx, concept_name in enumerate(tqdm(concept_names, desc="Mapping concepts")):
        scores = similarity_matrix[concept_idx]

        # Sort chunk indices by score descending
        top_indices = np.argsort(scores)[::-1]

        selected_chunks = []
        selected_scores = []

        for idx in top_indices:
            score = float(scores[idx])

            if score < MIN_SIMILARITY:
                continue

            chunk_id = chunk_ids_by_row[idx]

            selected_chunks.append(chunk_id)
            selected_scores.append({
                "chunk_id": chunk_id,
                "similarity": round(score, 4)
            })

            if len(selected_chunks) >= TOP_K_CHUNKS_PER_CONCEPT:
                break

        # Fallback: if threshold removed everything, force top-k
        if not selected_chunks:
            forced_indices = top_indices[:TOP_K_CHUNKS_PER_CONCEPT]

            for idx in forced_indices:
                score = float(scores[idx])
                chunk_id = chunk_ids_by_row[idx]

                selected_chunks.append(chunk_id)
                selected_scores.append({
                    "chunk_id": chunk_id,
                    "similarity": round(score, 4)
                })

        concept_chunk_map[concept_name] = selected_chunks
        concept_chunk_scores[concept_name] = selected_scores

    # ----------------------------
    # Save Outputs
    # ----------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(concept_chunk_map, f, indent=4, ensure_ascii=False)

    with open(SCORES_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(concept_chunk_scores, f, indent=4, ensure_ascii=False)

    print("\n✅ Embedding-based concept-chunk mapping complete!")
    print(f"📦 Saved concept chunk map to: {OUTPUT_FILE}")
    print(f"📦 Saved similarity scores to: {SCORES_OUTPUT_FILE}")
    print(f"🔢 Top chunks per concept: {TOP_K_CHUNKS_PER_CONCEPT}")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    main()
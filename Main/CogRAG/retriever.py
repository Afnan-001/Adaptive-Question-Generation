import json
import sys
import os
from pathlib import Path

import networkx as nx
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
MAIN_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = MAIN_DIR / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
GRAPH_FILE = OUTPUTS_DIR / "graph_model" / "knowledge_graph.gml"
CHUNK_MAP_FILE = DATA_DIR / "concept_chunk_map.json"
CHUNKS_FILE = DATA_DIR / "dbms_chunks.json"

CHUNK_INDEX_FILE = DATA_DIR / "chunk_index.json"
CHUNK_EMBEDDINGS_FILE = DATA_DIR / "chunk_embeddings.npy"

# ----------------------------
# ✅ Embedding API Config
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
# ✅ Validate Files
# ----------------------------
for file_path in [
    GRAPH_FILE,
    CHUNK_MAP_FILE,
    CHUNKS_FILE,
    CHUNK_INDEX_FILE,
    CHUNK_EMBEDDINGS_FILE
]:
    if not file_path.exists():
        raise FileNotFoundError(f"❌ Missing file: {file_path}")

# ----------------------------
# ✅ Load Graph
# ----------------------------
G = nx.read_gml(GRAPH_FILE)

# ----------------------------
# ✅ Load Concept → Chunk Map
# ----------------------------
with open(CHUNK_MAP_FILE, "r", encoding="utf-8") as f:
    concept_chunk_map = json.load(f)

# ----------------------------
# ✅ Load Chunks
# ----------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ✅ chunk_id → text lookup
chunk_lookup = {
    c["chunk_id"]: c["text"]
    for c in chunks
}

# ----------------------------
# ✅ Load Chunk Index
# ----------------------------
with open(CHUNK_INDEX_FILE, "r", encoding="utf-8") as f:
    chunk_index = json.load(f)

# ✅ chunk_id → embedding row index
chunk_id_to_embedding_index = {
    item["chunk_id"]: i
    for i, item in enumerate(chunk_index)
}

# ----------------------------
# ✅ Load Chunk Embeddings
# ----------------------------
chunk_embeddings = np.load(CHUNK_EMBEDDINGS_FILE).astype(np.float32)

if len(chunk_index) != chunk_embeddings.shape[0]:
    raise ValueError(
        f"❌ Mismatch: chunk_index has {len(chunk_index)} entries, "
        f"but embeddings matrix has {chunk_embeddings.shape[0]} rows"
    )

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
# ✅ Query Embedding Function
# ----------------------------
def embed_query(text):
    """
    Embeds query text using the same embedding model used for chunk embeddings.
    """

    if not EMBEDDING_API_KEY:
        raise ValueError(
            "❌ EMBEDDING_API_KEY missing. "
            "Set it in .env file."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDING_API_KEY}"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(
        EMBEDDING_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"❌ Query embedding failed\n"
            f"Status Code: {response.status_code}\n"
            f"Response: {response.text}"
        )

    data = response.json()

    if "data" not in data or not data["data"]:
        raise ValueError(f"❌ Invalid embedding response: {data}")

    vec = data["data"][0]["embedding"]
    return normalize_vector(vec)

# ----------------------------
# ✅ Graph Helper Functions
# ----------------------------
def get_parents(graph, concept):
    return [
        src for src, dst, data in graph.in_edges(concept, data=True)
        if data.get("relation") == "CONTAINS"
    ]


def get_prerequisites(graph, concept):
    return [
        src for src, dst, data in graph.in_edges(concept, data=True)
        if data.get("relation") == "PREREQUISITE_FOR"
    ]


def get_neighbors(graph, concept):
    return [
        dst for src, dst, data in graph.out_edges(concept, data=True)
        if data.get("relation") == "PREREQUISITE_FOR"
    ]

# ----------------------------
# ✅ Safe Chunk Fetch
# ----------------------------
def get_chunks_for_concept(concept):
    return concept_chunk_map.get(concept, [])

# ----------------------------
# ✅ Build Semantic Query
# ----------------------------
def build_query_text(concept, parents, prereqs, neighbors):
    """
    Builds a rich semantic query from graph context.
    """

    query_parts = [
        f"Concept: {concept}"
    ]

    if parents:
        query_parts.append(f"Parent concepts: {', '.join(parents)}")

    if prereqs:
        query_parts.append(f"Prerequisites: {', '.join(prereqs)}")

    return "\n".join(query_parts)

# ----------------------------
# ✅ Semantic Ranking
# ----------------------------
def rank_chunks_semantically(candidate_chunk_ids, query_text):
    """
    Ranks candidate chunks using cosine similarity between:
    query embedding and chunk embeddings.

    Since embeddings were normalized earlier, dot product = cosine similarity.
    """

    query_embedding = embed_query(query_text)

    scored_chunks = []

    for cid in candidate_chunk_ids:
        idx = chunk_id_to_embedding_index.get(cid)

        if idx is None:
            continue

        chunk_vec = chunk_embeddings[idx]

        similarity = float(np.dot(query_embedding, chunk_vec))

        scored_chunks.append((similarity, cid))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return scored_chunks

# ----------------------------
# ✅ Main Retrieval Function
# ----------------------------
def retrieve_context(concept_name, max_chunks=10, candidate_limit=100):

    if concept_name not in G:
        print(f"❌ Concept '{concept_name}' not found in graph")
        return None

    # ----------------------------
    # ✅ Get graph context
    # ----------------------------
    parents = get_parents(G, concept_name)
    prereqs = get_prerequisites(G, concept_name)
    neighbors = get_neighbors(G, concept_name)

    candidate_chunk_ids = set()

    # ----------------------------
    # ✅ 1. Own concept chunks
    # ----------------------------
    candidate_chunk_ids.update(get_chunks_for_concept(concept_name))

    # ----------------------------
    # ✅ 2. Prerequisite chunks
    # ----------------------------
    for prereq in prereqs:
        candidate_chunk_ids.update(get_chunks_for_concept(prereq))

    # ----------------------------
    # ✅ 3. Parent concept chunks
    # ----------------------------
    for parent in parents:
        candidate_chunk_ids.update(get_chunks_for_concept(parent))

    # ----------------------------
    # ✅ 4. Optional neighbor chunks
    # ----------------------------
    for neighbor in neighbors:
        candidate_chunk_ids.update(get_chunks_for_concept(neighbor))

    # ----------------------------
    # ✅ Fallback if no mapped chunks found
    # ----------------------------
    if not candidate_chunk_ids:
        print(f"⚠️ No mapped chunks found for '{concept_name}', using global corpus fallback")

        candidate_chunk_ids = set(chunk_lookup.keys())

    # ----------------------------
    # ✅ Limit candidate pool before semantic ranking
    # ----------------------------
    candidate_chunk_ids = list(candidate_chunk_ids)

    if len(candidate_chunk_ids) > candidate_limit:
        candidate_chunk_ids = candidate_chunk_ids[:candidate_limit]

    # ----------------------------
    # ✅ Build semantic query
    # ----------------------------
    query_text = build_query_text(
        concept=concept_name,
        parents=parents,
        prereqs=prereqs,
        neighbors=neighbors
    )

    # ----------------------------
    # ✅ Semantic Top-K Ranking
    # ----------------------------
    scored_chunks = rank_chunks_semantically(
        candidate_chunk_ids=candidate_chunk_ids,
        query_text=query_text
    )

    if not scored_chunks:
        print("⚠️ No semantic scores generated")
        return None

    # ✅ Select top-k
    top_chunks = scored_chunks[:max_chunks]

    selected_chunk_ids = [cid for score, cid in top_chunks]

    # ----------------------------
    # ✅ Build final context
    # ----------------------------
    texts = []

    for cid in selected_chunk_ids:
        text = chunk_lookup.get(cid)
        if text:
            texts.append(text)

    if not texts:
        print("⚠️ No valid texts found after semantic ranking")
        return None

    combined_text = "\n\n".join(texts)

    return {
        "concept": concept_name,
        "parents": parents,
        "prerequisites": prereqs,
        "neighbors": neighbors,
        "chunks": selected_chunk_ids,
        "scores": [
            {
                "chunk_id": cid,
                "similarity": round(score, 4)
            }
            for score, cid in top_chunks
        ],
        "query_text": query_text,
        "context": combined_text
    }

# ----------------------------
# ✅ CLI
# ----------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('❌ Usage: python3 retriever.py "ConceptName"')
        sys.exit(1)

    user_input = sys.argv[1]

    result = retrieve_context(user_input)

    if result:
        print("\n✅ SEMANTIC RETRIEVAL RESULT\n")

        print("Concept:", result["concept"])
        print("\nParents:", result["parents"])
        print("\nPrerequisites:", result["prerequisites"])
        print("\nNeighbors:", result["neighbors"])

        print("\nSemantic Query:")
        print(result["query_text"])

        print("\nTop Chunks:")
        for item in result["scores"]:
            print(f"- {item['chunk_id']} | similarity={item['similarity']}")

        print("\nContext Preview:\n")
        print(result["context"][:1000])

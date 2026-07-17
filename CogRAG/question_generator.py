import sys
import json
import random
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from CogRAG.retriever import retrieve_context, chunk_lookup
from CogRAG.llm import LLM

# ----------------------------
# ✅ Load .env
# ----------------------------
load_dotenv()

# ----------------------------
# ✅ File
# ----------------------------
OUTPUT_FILE = Path("data/generated_mcqs.json")

# ----------------------------
# ✅ Embedding Config
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

# Similarity threshold for detecting same-idea questions
SEMANTIC_DUPLICATE_THRESHOLD = 0.8

# If embedding API fails, continue generation instead of crashing
FAIL_OPEN_ON_EMBEDDING_ERROR = True

# ----------------------------
# ✅ LLM
# ----------------------------
llm = LLM(
    sys_msg="You are a DBMS expert generating grounded exam questions strictly from context.",
    temperature=0.55
)

# ----------------------------
# ✅ Question Style Templates
# ----------------------------
QUESTION_STYLES = {
    "easy": [
        "definition-based",
        "terminology-based",
        "basic identification",
        "simple concept recognition"
    ],
    "medium": [
        "comparison-based",
        "why/how reasoning",
        "identify the correct statement",
        "concept relationship question"
    ],
    "hard": [
        "application scenario",
        "error spotting",
        "case-based reasoning",
        "best design/action choice"
    ]
}

# ----------------------------
# ✅ Prompt
# ----------------------------
PROMPT = """
Generate ONE DBMS MCQ.

STRICT RULES:
- Use ONLY the provided context
- The question MUST be grounded in one of the provided chunk IDs
- Do NOT repeat the style or idea of the recent questions
- Return STRICT JSON only

---

CONCEPT:
{concept}

DIFFICULTY:
{difficulty}

QUESTION STYLE TO USE:
{question_style}

RECENT QUESTIONS TO AVOID:
{recent_questions}

AVAILABLE CHUNK IDS:
{available_chunk_ids}

CONTEXT:
{context}

---

OUTPUT:

{{
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "A",
  "source_chunk": "EXACT_CHUNK_ID_FROM_AVAILABLE_CHUNK_IDS"
}}

IMPORTANT:
- source_chunk MUST be one of the chunk IDs listed in AVAILABLE CHUNK IDS
- DO NOT write "chunk_id"
- DO NOT write "CONCEPT: ..."
- DO NOT invent a chunk ID
- COPY the chunk id exactly
- Make the question style DIFFERENT from the recent questions
"""

# ----------------------------
# ✅ Load Existing Questions
# ----------------------------
def load_existing():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

# ----------------------------
# ✅ Normalize Question Text
# ----------------------------
def normalize_question(text):
    return " ".join(text.strip().lower().split())

# ----------------------------
# ✅ Save Question
# ----------------------------
def save_mcq(concept, difficulty, question):
    """
    Saves MCQ only if exact duplicate does not exist.
    Returns True if saved, False if duplicate.
    """
    data = load_existing()

    if concept not in data:
        data[concept] = {"easy": [], "medium": [], "hard": []}

    if difficulty not in data[concept]:
        data[concept][difficulty] = []

    existing_questions = [
        normalize_question(q["question"])
        for q in data[concept][difficulty]
        if "question" in q
    ]

    if normalize_question(question["question"]) in existing_questions:
        print("⚠️ Exact duplicate rejected")
        return False

    data[concept][difficulty].append(question)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return True

# ----------------------------
# ✅ Generate Stable ID
# ----------------------------
def generate_id(existing, concept, difficulty):
    count = len(existing.get(concept, {}).get(difficulty, [])) + 1
    return f"{concept}_{difficulty}_{count}"

# ----------------------------
# ✅ Reuse Existing Questions
# ----------------------------
def get_existing_question(concept, difficulty, asked_ids, wrong_ids):
    data = load_existing()

    if concept not in data or difficulty not in data[concept]:
        return None

    wrong_pool = []
    fresh_pool = []

    for q in data[concept][difficulty]:
        qid = q.get("id")

        if not qid:
            continue

        # Prefer previously wrong questions
        if qid in wrong_ids:
            wrong_pool.append(q)

        # Prefer unasked questions next
        elif qid not in asked_ids:
            fresh_pool.append(q)

    if wrong_pool:
        return random.choice(wrong_pool)

    if fresh_pool:
        return random.choice(fresh_pool)

    return None

# ----------------------------
# ✅ Recent Questions / Chunks
# ----------------------------
def get_recent_metadata(concept, difficulty, limit=6):
    data = load_existing()

    if concept not in data or difficulty not in data[concept]:
        return [], []

    questions = data[concept][difficulty][-limit:]

    recent_questions = [
        q.get("question", "")
        for q in questions
        if q.get("question")
    ]

    recent_chunks = [
        q.get("source_chunk")
        for q in questions
        if q.get("source_chunk")
    ]

    return recent_questions, recent_chunks

# ----------------------------
# ✅ Existing Question Texts
# ----------------------------
def get_existing_question_texts(concept, difficulty, limit=30):
    data = load_existing()

    if concept not in data or difficulty not in data[concept]:
        return []

    questions = data[concept][difficulty]

    texts = [
        q.get("question", "")
        for q in questions
        if q.get("question")
    ]

    return texts[-limit:]

# ----------------------------
# ✅ Choose Diverse Question Style
# ----------------------------
def choose_question_style(concept, difficulty):
    existing = load_existing()
    count = len(existing.get(concept, {}).get(difficulty, []))

    styles = QUESTION_STYLES.get(difficulty, QUESTION_STYLES["medium"])

    return styles[count % len(styles)]

# ----------------------------
# ✅ Choose Diverse Chunks
# ----------------------------
def choose_diverse_chunks(all_chunks, recent_chunks, max_chunks=5):
    if not all_chunks:
        return []

    preferred = [cid for cid in all_chunks if cid not in recent_chunks]
    fallback = [cid for cid in all_chunks if cid in recent_chunks]

    random.shuffle(preferred)
    random.shuffle(fallback)

    selected = preferred[:max_chunks]

    if len(selected) < max_chunks:
        selected.extend(fallback[:max_chunks - len(selected)])

    return selected

# ----------------------------
# ✅ Build Context from Selected Chunks
# ----------------------------
def build_context(selected_chunks):
    blocks = []

    for cid in selected_chunks:
        text = chunk_lookup.get(cid, "")

        if text:
            blocks.append(f"[{cid}]\n{text}")

    return "\n\n".join(blocks)

# ----------------------------
# ✅ Choose Best Fallback Source Chunk
# ----------------------------
def choose_fallback_source_chunk(selected_chunks, recent_chunks):
    unused = [cid for cid in selected_chunks if cid not in recent_chunks]

    if unused:
        return random.choice(unused)

    if selected_chunks:
        return random.choice(selected_chunks)

    return "fallback_chunk"

# ============================================================
# ✅ EMBEDDING-BASED SEMANTIC DUPLICATE   
def normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)

    if norm == 0:
        return arr

    return arr / norm


def embed_texts(texts):
    """
    Embeds list of texts using internal embedding endpoint.
    Returns normalized numpy matrix.
    """
    if not texts:
        return np.array([], dtype=np.float32)

    if not EMBEDDING_API_KEY:
        raise ValueError("❌ EMBEDDING_API_KEY missing in .env")

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
            f"Embedding API failed: {response.status_code} | {response.text}"
        )

    data = response.json()

    if "data" not in data:
        raise ValueError(f"Invalid embedding response: {data}")

    # Sort by index to preserve input order
    items = sorted(data["data"], key=lambda x: x.get("index", 0))

    vectors = []

    for item in items:
        vec = item.get("embedding")

        if vec is None:
            raise ValueError(f"Missing embedding in response item: {item}")

        vectors.append(normalize_vector(vec))

    return np.array(vectors, dtype=np.float32)


def is_semantically_duplicate(
    new_question,
    old_questions,
    threshold=SEMANTIC_DUPLICATE_THRESHOLD
):
    """
    Returns:
    - is_duplicate: bool
    - max_similarity: float
    - most_similar_question: str or None
    """

    if not old_questions:
        return False, 0.0, None

    try:
        all_texts = [new_question] + old_questions
        embeddings = embed_texts(all_texts)

        if embeddings.shape[0] < 2:
            return False, 0.0, None

        new_vec = embeddings[0]
        old_vecs = embeddings[1:]

        similarities = old_vecs @ new_vec

        max_idx = int(np.argmax(similarities))
        max_score = float(similarities[max_idx])
        most_similar_question = old_questions[max_idx]

        if max_score >= threshold:
            return True, max_score, most_similar_question

        return False, max_score, most_similar_question

    except Exception as e:
        print("⚠️ Semantic duplicate check failed:", e)

        if FAIL_OPEN_ON_EMBEDDING_ERROR:
            print("⚠️ Continuing without semantic duplicate rejection")
            return False, 0.0, None

        raise

# ----------------------------
# ✅ Generate Question
# ----------------------------
def generate_question(concept_name, difficulty, asked_ids=None, wrong_ids=None):
    asked_ids = set(asked_ids or [])
    wrong_ids = set(wrong_ids or [])

    # ✅ 1. Try reuse first
    existing_q = get_existing_question(
        concept_name,
        difficulty,
        asked_ids,
        wrong_ids
    )

    if existing_q:
        return existing_q

    # ✅ 2. Retrieve context
    data = retrieve_context(concept_name)

    if not data or not data.get("chunks"):
        print("⚠️ Retrieval failed or empty chunks")
        return None

    # ✅ 3. Diversity metadata
    recent_questions, recent_chunks = get_recent_metadata(
        concept_name,
        difficulty,
        limit=6
    )

    question_style = choose_question_style(concept_name, difficulty)

    # ✅ 4. Select diverse chunks
    all_chunks = list(data["chunks"])

    selected_chunks = choose_diverse_chunks(
        all_chunks,
        recent_chunks,
        max_chunks=4
    )

    if not selected_chunks:
        print("⚠️ No selected chunks available")
        return None

    # ✅ 5. Build focused context
    formatted_context = build_context(selected_chunks)

    if not formatted_context.strip():
        print("⚠️ Empty formatted context")
        return None

    recent_questions_text = "\n".join(
        [f"- {q}" for q in recent_questions[-5:]]
    ) if recent_questions else "None"

    prompt = PROMPT.format(
        concept=concept_name,
        difficulty=difficulty,
        question_style=question_style,
        recent_questions=recent_questions_text,
        available_chunk_ids=", ".join(selected_chunks),
        context=formatted_context
    )

    response = llm.get_response(prompt)
    content = response.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        q = json.loads(content)

        # ----------------------------
        # ✅ Strict Validation
        # ----------------------------
        required_keys = ["question", "options", "answer"]

        for key in required_keys:
            if key not in q:
                raise ValueError(f"Missing key: {key}")

        if not isinstance(q["question"], str) or not q["question"].strip():
            raise ValueError("Question must be a non-empty string")

        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            raise ValueError("Options must be exactly 4")

        q["answer"] = q["answer"].strip().upper()

        if q["answer"] not in ["A", "B", "C", "D"]:
            raise ValueError("Answer must be A/B/C/D")

        # ----------------------------
        # ✅ Semantic Duplicate Check
        # ----------------------------
        existing_question_texts = get_existing_question_texts(
            concept_name,
            difficulty,
            limit=30
        )

        is_dup, sim_score, similar_q = is_semantically_duplicate(
            q["question"],
            existing_question_texts
        )

        if is_dup:
            print("⚠️ Semantic duplicate rejected")
            print("Similarity:", round(sim_score, 4))
            print("Similar to:", similar_q)
            return None

        # ----------------------------
        # ✅ Validate / diversify source chunk
        # ----------------------------
        if q.get("source_chunk") not in selected_chunks:
            q["source_chunk"] = choose_fallback_source_chunk(
                selected_chunks,
                recent_chunks
            )

        # ----------------------------
        # ✅ Assign ID
        # ----------------------------
        existing = load_existing()
        q["id"] = generate_id(existing, concept_name, difficulty)

        # ----------------------------
        # ✅ Save
        # ----------------------------
        saved = save_mcq(concept_name, difficulty, q)

        if not saved:
            return None

        return q

    except Exception as e:
        print("❌ Validation / Parsing Error:", str(e))
        print("RAW OUTPUT:\n", content)
        return None

# ----------------------------
# ✅ CLI
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m CogRAG.question_generator "Concept" difficulty')
        sys.exit(1)

    concept = sys.argv[1]
    difficulty = sys.argv[2]

    q = generate_question(concept, difficulty)

    if q:
        print("\n✅ QUESTION\n")
        print("ID:", q["id"])
        print(q["question"])

        for opt in q["options"]:
            print(opt)

        print("Answer:", q["answer"])
        print("Source:", q["source_chunk"])

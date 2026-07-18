import json
import sys
from pathlib import Path
import networkx as nx

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR / "CogRAG") not in sys.path:
    sys.path.append(str(MAIN_DIR / "CogRAG"))

from question_generator import generate_question

# ----------------------------
# ✅ Files
# ----------------------------
OUTPUTS_DIR = MAIN_DIR / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
MASTERY_FILE = DATA_DIR / "mastery_scores.json"
GRAPH_FILE = OUTPUTS_DIR / "graph_model" / "knowledge_graph.gml"
CURRENT_CONCEPT_FILE = DATA_DIR / "current_concept.txt"
RECENT_HISTORY_FILE = DATA_DIR / "recent_concepts.json"

G = nx.read_gml(GRAPH_FILE)

# ----------------------------
# ✅ BKT PARAMETERS
# ----------------------------
P_G = 0.2   # Guess
P_S = 0.1   # Slip
P_T = 0.1   # Learn

# ----------------------------
# ✅ Navigation Settings
# ----------------------------
MASTERY_THRESHOLD = 0.8
WEAK_THRESHOLD = 0.4
RECENT_WINDOW = 5

# ----------------------------
# ✅ Load / Save Mastery
# ----------------------------
def load_mastery():
    if not MASTERY_FILE.exists():
        return {}
    with open(MASTERY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def save_mastery(data):
    MASTERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTERY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ----------------------------
# ✅ Current Concept Tracking
# ----------------------------
def load_current_concept():
    if CURRENT_CONCEPT_FILE.exists():
        return CURRENT_CONCEPT_FILE.read_text(encoding="utf-8").strip()
    return None


def save_current_concept(concept):
    CURRENT_CONCEPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURRENT_CONCEPT_FILE, "w", encoding="utf-8") as f:
        f.write(concept)


# ----------------------------
# ✅ Recent History Tracking
# ----------------------------
def load_recent_history():
    if RECENT_HISTORY_FILE.exists():
        with open(RECENT_HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []


def save_recent_history(history):
    RECENT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECENT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-RECENT_WINDOW:], f, indent=4, ensure_ascii=False)


def push_recent_concept(concept):
    history = load_recent_history()
    history.append(concept)
    save_recent_history(history)


# ----------------------------
# ✅ Get / Init Concept (BKT)
# ----------------------------
def get_mastery(concept):
    data = load_mastery()

    if concept not in data:
        data[concept] = {
            "p_know": 0.2,
            "questions_attempted": 0,
            "correct_answers": 0,
            "asked_question_ids": [],
            "wrong_question_ids": []
        }
        save_mastery(data)

    return data[concept]


# ----------------------------
# ✅ Root Concepts (NO prerequisites)
# ----------------------------
def get_root_concepts():
    roots = []

    for node in G.nodes:
        incoming = [
            src for src, _, d in G.in_edges(node, data=True)
            if d.get("relation") == "PREREQUISITE_FOR"
        ]
        if len(incoming) == 0:
            roots.append(node)

    return roots


# ----------------------------
# ✅ Graph Helpers
# ----------------------------
def get_prerequisite_nodes(concept):
    return [
        src for src, _, d in G.in_edges(concept, data=True)
        if d.get("relation") == "PREREQUISITE_FOR"
    ]


def get_successor_nodes(concept):
    return [
        dst for _, dst, d in G.edges(concept, data=True)
        if d.get("relation") == "PREREQUISITE_FOR"
    ]


# ----------------------------
# ✅ Difficulty Selection
# ----------------------------
def get_difficulty(concept):
    p = get_mastery(concept)["p_know"]

    if p < WEAK_THRESHOLD:
        return "easy"
    elif p < MASTERY_THRESHOLD:
        return "medium"
    return "hard"


# ----------------------------
# ✅ Candidate Ranking
# ----------------------------
def weakest_concept(concepts, data, exclude_recent=True):
    recent = set(load_recent_history()) if exclude_recent else set()

    candidates = [
        c for c in concepts
        if c in data and c not in recent
    ]

    if not candidates:
        candidates = [c for c in concepts if c in data]

    if not candidates:
        return None

    return min(candidates, key=lambda c: data[c]["p_know"])


def weakest_unmastered(concepts, data, exclude_recent=True):
    recent = set(load_recent_history()) if exclude_recent else set()

    candidates = [
        c for c in concepts
        if c in data and data[c]["p_know"] < MASTERY_THRESHOLD and c not in recent
    ]

    if not candidates:
        candidates = [
            c for c in concepts
            if c in data and data[c]["p_know"] < MASTERY_THRESHOLD
        ]

    if not candidates:
        return None

    return min(candidates, key=lambda c: data[c]["p_know"])


def global_fallback_concept(data):
    """
    Pick another concept when the current branch is exhausted.
    Priority:
    1. weakest unmastered root
    2. weakest unmastered concept globally
    3. weakest concept globally
    """
    roots = get_root_concepts()

    root_choice = weakest_unmastered(roots, data, exclude_recent=True)
    if root_choice:
        return root_choice

    global_unmastered = weakest_unmastered(list(data.keys()), data, exclude_recent=True)
    if global_unmastered:
        return global_unmastered

    return weakest_concept(list(data.keys()), data, exclude_recent=False)


# ----------------------------
# ✅ GRAPH-AWARE CONCEPT SELECTION (FIXED)
# ----------------------------
def select_next_concept():
    data = load_mastery()
    if not data:
        raise ValueError("mastery_scores.json is empty")

    current = load_current_concept()

    # ----------------------------
    # ✅ FIRST RUN → start from weakest root
    # ----------------------------
    if not current or current not in data:
        start = global_fallback_concept(data)
        save_current_concept(start)
        push_recent_concept(start)
        return start

    current_p = data[current]["p_know"]

    # ----------------------------
    # ✅ WEAK → go to weaker prerequisite if possible
    # otherwise keep practicing current
    # ----------------------------
    if current_p <= WEAK_THRESHOLD:
        preds = get_prerequisite_nodes(current)

        # Only go backward if a prerequisite exists and is weaker/not mastered
        candidate_preds = [
            c for c in preds
            if c in data and data[c]["p_know"] < MASTERY_THRESHOLD
        ]

        next_concept = weakest_concept(candidate_preds, data, exclude_recent=True)

        if next_concept:
            save_current_concept(next_concept)
            push_recent_concept(next_concept)
            return next_concept

        # If no useful prerequisite, continue practicing current
        save_current_concept(current)
        push_recent_concept(current)
        return current

    # ----------------------------
    # ✅ MASTERED → move to weakest unmastered successor
    # if none exists, leave the branch
    # ----------------------------
    if current_p >= MASTERY_THRESHOLD:
        succ = get_successor_nodes(current)

        # Pick weakest unmastered successor, not just first successor
        next_concept = weakest_unmastered(succ, data, exclude_recent=True)

        if next_concept:
            save_current_concept(next_concept)
            push_recent_concept(next_concept)
            return next_concept

        # Branch exhausted → jump to another weak root/global concept
        fallback = global_fallback_concept(data)
        save_current_concept(fallback)
        push_recent_concept(fallback)
        return fallback

    # ----------------------------
    # ✅ MODERATE → continue current concept
    # ----------------------------
    save_current_concept(current)
    push_recent_concept(current)
    return current


# ----------------------------
# ✅ Adaptive Question
# ----------------------------
def get_adaptive_question():
    concept = select_next_concept()
    difficulty = get_difficulty(concept)

    mastery_data = get_mastery(concept)

    asked_ids = set(mastery_data.get("asked_question_ids", []))
    wrong_ids = set(mastery_data.get("wrong_question_ids", []))

    q = None

    for _ in range(7):
        q = generate_question(
            concept,
            difficulty,
            asked_ids,
            wrong_ids
        )

        if not q:
            continue

        qid = q.get("id")
        if not qid:
            continue

        # Ask only if not previously solved correctly
        if qid not in asked_ids or qid in wrong_ids:
            return {
                "concept": concept,
                "difficulty": difficulty,
                "question": q
            }

    return {
        "concept": concept,
        "difficulty": difficulty,
        "question": q
    }


# ----------------------------
# ✅ Evaluate Answer
# ----------------------------
def evaluate_answer(question_data, user_answer):
    return question_data.get("answer", "").strip().upper() == user_answer.strip().upper()


# ----------------------------
# ✅ BKT UPDATE
# ----------------------------
def update_mastery(concept, question_id, correct):
    data = load_mastery()
    entry = data[concept]

    p = entry["p_know"]

    if correct:
        numerator = p * (1 - P_S)
        denominator = numerator + (1 - p) * P_G
    else:
        numerator = p * P_S
        denominator = numerator + (1 - p) * (1 - P_G)

    posterior = p if denominator == 0 else numerator / denominator
    p_new = posterior + (1 - posterior) * P_T

    entry["p_know"] = max(0.0, min(1.0, p_new))

    # stats
    entry["questions_attempted"] += 1
    if correct:
        entry["correct_answers"] += 1

    # tracking
    if question_id and question_id not in entry["asked_question_ids"]:
        entry["asked_question_ids"].append(question_id)

    if question_id:
        if correct:
            if question_id in entry["wrong_question_ids"]:
                entry["wrong_question_ids"].remove(question_id)
        else:
            if question_id not in entry["wrong_question_ids"]:
                entry["wrong_question_ids"].append(question_id)

    save_mastery(data)
    return entry

# ----------------------------
# ✅ Process Answer
# ----------------------------
def process_answer(concept, question_data, user_answer):
    correct = evaluate_answer(question_data, user_answer)

    updated = update_mastery(
        concept,
        question_data.get("id"),
        correct
    )

    # recommendation = recommend_next_concept(concept)

    return {
        "correct": correct,
        "mastery": updated,
    }


# ----------------------------
# ✅ Dashboard Helpers
# ----------------------------
def get_mastered_concepts():
    return [c for c, v in load_mastery().items() if v["p_know"] >= MASTERY_THRESHOLD]


def get_weak_concepts():
    return [c for c, v in load_mastery().items() if v["p_know"] <= WEAK_THRESHOLD]


def get_overall_progress():
    data = load_mastery()
    if not data:
        return 0.0

    avg = sum(v["p_know"] for v in data.values()) / len(data)
    return round(avg * 100, 2)

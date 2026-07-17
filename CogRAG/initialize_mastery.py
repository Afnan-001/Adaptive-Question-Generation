import json
from pathlib import Path

# ----------------------------
# ✅ Files
# ----------------------------
MASTERY_FILE = Path("data/mastery_scores.json")
CONCEPTS_FILE = Path("data/enriched_concepts.json")

# ----------------------------
# ✅ BKT INITIAL PARAM
# ----------------------------
INITIAL_P_KNOW = 0.2  # initial belief

# ----------------------------
# ✅ Load Concepts (LIST FORMAT)
# ----------------------------
def load_concepts():

    if not CONCEPTS_FILE.exists():
        raise FileNotFoundError(f"{CONCEPTS_FILE} not found")

    with open(CONCEPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ✅ Extract only concept names
    concepts = []

    for item in data:
        if "concept" in item:
            concepts.append(item["concept"])

    return concepts


# ----------------------------
# ✅ Create BKT Structure
# ----------------------------
def create_initial_mastery(concepts):

    mastery_data = {}

    for concept in concepts:
        mastery_data[concept] = {
            "p_know": INITIAL_P_KNOW,   # ✅ BKT field
            "questions_attempted": 0,
            "correct_answers": 0,
            "asked_question_ids": [],
            "wrong_question_ids": []
        }

    return mastery_data


# ----------------------------
# ✅ Save JSON
# ----------------------------
def save_mastery(data):

    MASTERY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MASTERY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ mastery_scores.json initialized with {len(data)} concepts")


# ----------------------------
# ✅ Main
# ----------------------------
if __name__ == "__main__":

    concepts = load_concepts()

    print(f"📚 Loaded {len(concepts)} concepts")

    mastery_data = create_initial_mastery(concepts)

    save_mastery(mastery_data)
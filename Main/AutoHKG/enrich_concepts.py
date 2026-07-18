import json
import sys
import time
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.append(str(MAIN_DIR))

from llm import LLM

# ----------------------------
# Files
# ----------------------------
DATA_DIR = MAIN_DIR / "outputs" / "data"
INPUT_FILE = DATA_DIR / "merged_concepts.json"
OUTPUT_FILE = DATA_DIR / "enriched_concepts.json"

# ----------------------------
# LLM
# ----------------------------
llm = LLM(
    sys_msg="You are a DBMS knowledge graph expert. You create educational concept hierarchies using coarse-grained and fine-grained DBMS concepts.",
    temperature=0
)

# ----------------------------
# Prompt
# ----------------------------
PROMPT = """
You are building an educational Knowledge Graph for Database Management Systems (DBMS).

The graph should follow this structure:

Coarse-Grained Category Node
    ↓ CONTAINS
Fine-Grained Concept Node

The CURRENT CONCEPT should be treated as a fine-grained DBMS concept.

STRICT RULES:
1. Use ONLY concepts from the provided ALL CONCEPTS list.
2. Do NOT hallucinate or create new concepts.
3. The coarse_grained_category must be ONE broader DBMS concept from ALL CONCEPTS.
4. The current concept itself must NOT be used as its own category or prerequisite.
5. Add 0 to 5 strong prerequisite concepts from ALL CONCEPTS.
6. Only include widely accepted DBMS relationships.
7. Keep relationships conservative and accurate.
8. If no strong category exists, return an empty string for coarse_grained_category.
9. If no strong prerequisites exist, return an empty list.
10. Return ONLY valid JSON. No explanation.

---

CURRENT CONCEPT:
{concept}

ALL CONCEPTS:
{all_concepts}

---

TASK:
For the CURRENT CONCEPT:
1. Assign ONE best coarse_grained_category.
2. Add strong prerequisites if applicable.

Definitions:
- coarse_grained_category: a broader DBMS topic/category that contains the current concept.
- fine_grained_concept: the current specific concept being taught.
- prerequisites: concepts that should usually be understood before learning the current concept.

Examples:
- "3NF" → coarse_grained_category: "Normalization"
- "BCNF" → coarse_grained_category: "Normalization"
- "B-Tree" → coarse_grained_category: "Indexing"
- "Two-Phase Locking" → coarse_grained_category: "Concurrency Control"
- "ACID Properties" → coarse_grained_category: "Transaction"

---

OUTPUT JSON ONLY:

{{
  "concept": "{concept}",
  "node_type": "fine_grained_concept",
  "coarse_grained_category": "",
  "prerequisites": []
}}
"""

# ----------------------------
# Safe JSON Parsing
# ----------------------------
def safe_parse_json(text):
    try:
        text = text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        return json.loads(text)

    except Exception:
        return None


# ----------------------------
# Normalize List Values
# ----------------------------
def ensure_list(value):
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


# ----------------------------
# Validate LLM Output
# ----------------------------
def validate_enrichment(updated, concept_name, all_concepts_set):
    """
    Ensures:
    - No hallucinated concepts
    - No self-loops
    - Category and prerequisites are from provided concept list
    """

    if not isinstance(updated, dict):
        return {
            "concept": concept_name,
            "node_type": "fine_grained_concept",
            "coarse_grained_category": [],
            "parent_concept": [],
            "prerequisites": []
        }

    # ----------------------------
    # Coarse category
    # ----------------------------
    raw_category = updated.get("coarse_grained_category", "")

    category_list = ensure_list(raw_category)

    valid_categories = []

    for cat in category_list:
        if cat == concept_name:
            continue

        if cat in all_concepts_set:
            valid_categories.append(cat)

    # keep only one best category
    valid_categories = valid_categories[:1]

    # ----------------------------
    # Prerequisites
    # ----------------------------
    raw_prereqs = updated.get("prerequisites", [])

    prereq_list = ensure_list(raw_prereqs)

    valid_prereqs = []

    for prereq in prereq_list:
        if prereq == concept_name:
            continue

        if prereq in all_concepts_set and prereq not in valid_prereqs:
            valid_prereqs.append(prereq)

    return {
        "concept": concept_name,
        "node_type": "fine_grained_concept",
        "coarse_grained_category": valid_categories,
        "parent_concept": valid_categories,
        "prerequisites": valid_prereqs
    }


# ----------------------------
# Enrichment Function
# ----------------------------
def enrich_concept(concept_obj, all_concepts, all_concepts_set):
    concept_name = concept_obj["concept"]

    for attempt in range(3):
        try:
            prompt = PROMPT.format(
                concept=concept_name,
                all_concepts=", ".join(all_concepts)
            )

            response = llm.get_response(prompt)
            updated = safe_parse_json(response)

            if not updated:
                raise ValueError("Invalid JSON from LLM")

            validated = validate_enrichment(
                updated=updated,
                concept_name=concept_name,
                all_concepts_set=all_concepts_set
            )

            return validated

        except Exception as e:
            print(f"⚠️ Retry {attempt + 1} failed for {concept_name}: {e}")
            time.sleep(2)

    return {
        "concept": concept_name,
        "node_type": "fine_grained_concept",
        "coarse_grained_category": [],
        "parent_concept": [],
        "prerequisites": []
    }


# ----------------------------
# Main
# ----------------------------
def main():

    if not INPUT_FILE.exists():
        print(f"❌ Missing file: {INPUT_FILE}")
        return

    if INPUT_FILE.stat().st_size == 0:
        print(f"❌ Empty file: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    all_concept_names = [
        c["concept"]
        for c in concepts
        if isinstance(c, dict) and c.get("concept")
    ]

    all_concepts_set = set(all_concept_names)

    enriched = []

    print(f"✅ Loaded {len(all_concept_names)} concepts")
    print("\n--- STARTING COARSE/FINE CONCEPT ENRICHMENT ---\n")

    for i, concept in enumerate(concepts):

        if not isinstance(concept, dict) or not concept.get("concept"):
            continue

        concept_name = concept["concept"]

        print(f"[{i + 1}/{len(concepts)}] Enriching: {concept_name}")

        updated = enrich_concept(
            concept_obj=concept,
            all_concepts=all_concept_names,
            all_concepts_set=all_concepts_set
        )

        enriched.append(updated)

        time.sleep(0.5)

    # ----------------------------
    # Save Output
    # ----------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=4, ensure_ascii=False)

    print("\n✅ Enriched concepts saved!")
    print(f"📦 Output file: {OUTPUT_FILE}")
    print(f"📦 Total Concepts: {len(enriched)}")

    concepts_with_category = sum(
        1 for c in enriched if c.get("coarse_grained_category")
    )

    concepts_with_prereqs = sum(
        1 for c in enriched if c.get("prerequisites")
    )

    print(f"✅ Concepts with coarse-grained category: {concepts_with_category}")
    print(f"✅ Concepts with prerequisites: {concepts_with_prereqs}")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    main()

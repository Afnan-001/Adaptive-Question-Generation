import json
import sys
import time
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.append(str(MAIN_DIR))

from llm import LLM

# ---------------------------
# File Paths
# ---------------------------
DATA_DIR = MAIN_DIR / "outputs" / "data"
INPUT_FILE = DATA_DIR / "concepts.json"
OUTPUT_FILE = DATA_DIR / "merged_concepts.json"

# ---------------------------
# Config
# ---------------------------
BATCH_SIZE = 50

# ---------------------------
# LLM
# ---------------------------
llm = LLM(
    sys_msg="You are a DBMS ontology cleanup expert.",
    temperature=0
)

# ---------------------------
# Prompt
# ---------------------------
PROMPT = """
You are cleaning a list of extracted Database Management Systems (DBMS) concepts.

Your task:
For each concept, decide whether it should be kept as an important DBMS concept.
If kept, convert it to a single clean canonical concept name.

Rules:
- Remove generic/non-technical/weak terms like "Data", "System", "Process", "Method", "Example", "Result".
- Merge plural and singular forms.
  Example: "Entities" -> "Entity"
- Merge full-form and abbreviation variants where standard.
  Example: "First Normal Form" -> "1NF"
  Example: "Third Normal Form" -> "3NF"
  Example: "Boyce-Codd Normal Form" -> "BCNF"
- Keep only meaningful DBMS concepts.
- Do not create explanations.
- Return ONLY valid JSON.
- Do not hallucinate unrelated concepts.
- If a concept is not useful, set keep=false and canonical="".
- If useful, set keep=true and provide canonical name.

INPUT CONCEPTS:
{concepts}

OUTPUT FORMAT:
[
  {{
    "original": "...",
    "canonical": "...",
    "keep": true
  }}
]
"""

# ---------------------------
# Safe JSON Parse
# ---------------------------
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


# ---------------------------
# Clean Concepts Batch
# ---------------------------
def clean_concepts_batch(batch):

    concepts_text = json.dumps(batch, indent=2, ensure_ascii=False)

    for attempt in range(3):
        try:
            prompt = PROMPT.format(concepts=concepts_text)

            response = llm.get_response(prompt)

            parsed = safe_parse_json(response)

            if not parsed or not isinstance(parsed, list):
                raise ValueError("Invalid JSON list returned by LLM")

            cleaned = []

            for item in parsed:
                if not isinstance(item, dict):
                    continue

                original = item.get("original", "")
                canonical = item.get("canonical", "")
                keep = item.get("keep", False)

                if not isinstance(original, str):
                    continue

                if keep and isinstance(canonical, str) and canonical.strip():
                    cleaned.append({
                        "original": original.strip(),
                        "canonical": canonical.strip(),
                        "keep": True
                    })
                else:
                    cleaned.append({
                        "original": original.strip(),
                        "canonical": "",
                        "keep": False
                    })

            return cleaned

        except Exception as e:
            print(f"⚠️ Batch retry {attempt + 1} failed: {e}")
            time.sleep(2)

    return []


# ---------------------------
# Main
# ---------------------------
def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept_list = data.get("concepts", [])

    print(f"✅ Loaded {len(concept_list)} raw concepts")

    merged = {}

    print("\n--- STARTING LLM CLEAN + MERGE PROCESS ---\n")

    for i in range(0, len(concept_list), BATCH_SIZE):

        batch = concept_list[i:i + BATCH_SIZE]

        print(f"🔄 Processing batch {i // BATCH_SIZE + 1}")

        cleaned_items = clean_concepts_batch(batch)

        for item in cleaned_items:

            original = item["original"]
            canonical = item["canonical"]
            keep = item["keep"]

            if not keep:
                print(f"🗑️ Removed: {original}")
                continue

            key = canonical.lower()

            if key not in merged:
                print(f"✅ Added: {canonical}")

                merged[key] = {
                    "concept": canonical,
                    "parent_concept": [],
                    "prerequisites": [],
                }
            else:
                print(f"🔁 Merged: {original} → {canonical}")

        time.sleep(0.5)

    final_concepts = sorted(
        list(merged.values()),
        key=lambda x: x["concept"].lower()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_concepts, f, indent=4, ensure_ascii=False)

    print("\n--- MERGE COMPLETE ---")
    print("✅ Raw Input Concepts:", len(concept_list))
    print("✅ Final Clean Concepts:", len(final_concepts))
    print(f"✅ Saved to: {OUTPUT_FILE}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()

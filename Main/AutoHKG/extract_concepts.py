import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.append(str(MAIN_DIR))

from llm import LLM

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# ----------------------------
# Files
# ----------------------------
DATA_DIR = MAIN_DIR / "outputs" / "data"
INPUT_FILE = DATA_DIR / "dbms_chunks.json"
OUTPUT_FILE = DATA_DIR / "concepts.json"

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 5   # ✅ you can tune this (5–10 works well)

# ----------------------------
# LLM
# ----------------------------
llm = LLM(
    sys_msg="You are an expert in Database Management Systems (DBMS).",
    temperature=0
)

# ----------------------------
# Prompt Template (BATCH)
# ----------------------------
PROMPT = """
Extract ONLY highly important DBMS concepts from the given list of text chunks.

Rules:
- Return ONLY a JSON array
- No explanation
- No duplicates
- Use consistent singular names (e.g., "Entity", not "Entities")
- Ignore trivial, generic, or non-technical words
- Skip chunks that do NOT contain important DBMS concepts
- Extract ONLY core, high-value concepts (e.g., "Normalization", "Transaction", "Index")
- Do NOT include explanations or descriptions

TEXT CHUNKS:
{chunks}
"""

# ----------------------------
# Batch Extract Function
# ----------------------------
def extract_concepts_batch(chunk_texts):

    combined_text = "\n\n---\n\n".join(chunk_texts)

    for attempt in range(3):
        try:
            response = llm.get_response(
                usr_msg=PROMPT.format(chunks=combined_text)
            )

            content = response.strip()

            # Remove markdown
            if content.startswith("```"):
                content = content.replace("```json", "")
                content = content.replace("```", "")
                content = content.strip()

            concepts = json.loads(content)

            # ✅ normalize
            concepts = [
                c.strip() for c in concepts if isinstance(c, str)
            ]

            return concepts

        except Exception as e:
            print(f"Batch retry {attempt+1} failed:", e)
            time.sleep(2)

    return []

# ----------------------------
# Main Function
# ----------------------------
def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # ✅ TEST MODE
    chunks = chunks[:1000]

    all_concepts = set()

    # ✅ Batch loop
    for i in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[i : i + BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]

        concepts = extract_concepts_batch(batch_texts)

        for c in concepts:
            all_concepts.add(c)

        print(f"Processed batch {i//BATCH_SIZE + 1} | extracted {len(concepts)} concepts")

        time.sleep(0.3)

    # convert to sorted list
    all_concepts = sorted(list(all_concepts))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "concepts": all_concepts
        }, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_concepts)} unique concepts")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    main()

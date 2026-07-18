import json
from pathlib import Path
from collections import Counter, defaultdict

# ----------------------------
# Files
# ----------------------------
LEXICAL_MAP_FILE = Path("data/concept_chunk_map_lexical.json")
CHUNKS_FILE = Path("data/dbms_chunks.json")
OUTPUT_FILE = Path("data/lexical_repeated_chunks_report.json")


# ----------------------------
# Load JSON
# ----------------------------
def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Main
# ----------------------------
def main():
    lexical_map = load_json(LEXICAL_MAP_FILE)
    chunks = load_json(CHUNKS_FILE)

    # chunk_id -> chunk metadata/text
    chunk_lookup = {
        c["chunk_id"]: {
            "text": c.get("text", ""),
            "book_name": c.get("book_name", ""),
            "page_number": c.get("page_number", "")
        }
        for c in chunks
    }

    chunk_counter = Counter()
    chunk_to_concepts = defaultdict(list)

    # Count how often each chunk appears across concepts
    for concept, chunk_ids in lexical_map.items():
        for chunk_id in chunk_ids:
            chunk_counter[chunk_id] += 1
            chunk_to_concepts[chunk_id].append(concept)

    # Sort by reuse_count descending
    most_repeated = sorted(
        chunk_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    report = []

    for chunk_id, count in most_repeated:
        chunk_info = chunk_lookup.get(chunk_id, {})
        concepts_using_chunk = chunk_to_concepts[chunk_id]

        report.append({
            "chunk_id": chunk_id,
            "reuse_count": count,
            "book_name": chunk_info.get("book_name", ""),
            "page_number": chunk_info.get("page_number", ""),
            "concept_count": len(concepts_using_chunk),
            "concepts_using_this_chunk": concepts_using_chunk,
            "text_preview": chunk_info.get("text", "")[:500]
        })

    # Save report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # Print summary
    print("\n==============================")
    print("MOST REPEATED LEXICAL CHUNKS")
    print("==============================\n")

    print(f"Total unique repeated chunks found: {len(report)}")

    print("\nTop 30 most repeated chunks:\n")

    for item in report[:30]:
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Reuse Count: {item['reuse_count']}")
        print(f"Book: {item['book_name']}")
        print(f"Page: {item['page_number']}")
        print(f"Used by Concepts: {item['concepts_using_this_chunk'][:15]}")

        if len(item["concepts_using_this_chunk"]) > 15:
            print(f"... and {len(item['concepts_using_this_chunk']) - 15} more concepts")

        print("Preview:")
        print(item["text_preview"].replace("\n", " "))
        print("-" * 80)

    print(f"\n✅ Full repeated lexical chunk report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
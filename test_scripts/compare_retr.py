import json
from pathlib import Path
from collections import Counter
import statistics

# ----------------------------
# Files
# ----------------------------
LEXICAL_MAP_FILE = Path("data/concept_chunk_map_lexical.json")
SEMANTIC_MAP_FILE = Path("data/concept_chunk_map_semantic.json")
CHUNKS_FILE = Path("data/dbms_chunks.json")

OUTPUT_FILE = Path("data/retriever_comparison_stats.json")
CONCEPT_DETAIL_OUTPUT = Path("data/retriever_concept_level_comparison.json")


# ----------------------------
# Load JSON
# ----------------------------
def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Build Chunk Lookup
# ----------------------------
def build_chunk_lookup(chunks):
    return {
        c["chunk_id"]: {
            "text": c.get("text", ""),
            "book_name": c.get("book_name", ""),
            "page_number": c.get("page_number", "")
        }
        for c in chunks
    }


# ----------------------------
# Basic Mapping Stats
# ----------------------------
def compute_map_stats(mapping):
    total_concepts = len(mapping)

    chunk_counts = [len(chunks) for chunks in mapping.values()]

    concepts_with_chunks = sum(1 for chunks in mapping.values() if len(chunks) > 0)
    zero_match_concepts = total_concepts - concepts_with_chunks

    all_chunks = []
    for chunks in mapping.values():
        all_chunks.extend(chunks)

    unique_chunks = set(all_chunks)
    chunk_reuse_counter = Counter(all_chunks)

    avg_chunks = statistics.mean(chunk_counts) if chunk_counts else 0
    median_chunks = statistics.median(chunk_counts) if chunk_counts else 0
    min_chunks = min(chunk_counts) if chunk_counts else 0
    max_chunks = max(chunk_counts) if chunk_counts else 0

    avg_chunk_reuse = (
        statistics.mean(chunk_reuse_counter.values())
        if chunk_reuse_counter else 0
    )

    return {
        "total_concepts": total_concepts,
        "concepts_with_chunks": concepts_with_chunks,
        "zero_match_concepts": zero_match_concepts,
        "coverage_percent": round((concepts_with_chunks / total_concepts) * 100, 2) if total_concepts else 0,
        "total_concept_chunk_links": len(all_chunks),
        "unique_chunks_used": len(unique_chunks),
        "average_chunks_per_concept": round(avg_chunks, 2),
        "median_chunks_per_concept": round(median_chunks, 2),
        "min_chunks_per_concept": min_chunks,
        "max_chunks_per_concept": max_chunks,
        "average_chunk_reuse": round(avg_chunk_reuse, 2),
        "top_10_most_reused_chunks": chunk_reuse_counter.most_common(10)
    }


# ----------------------------
# Per-Concept Overlap Stats
# ----------------------------
def compute_concept_level_comparison(lexical_map, semantic_map):
    all_concepts = sorted(set(lexical_map.keys()) | set(semantic_map.keys()))

    concept_details = {}

    jaccard_scores = []
    overlap_counts = []

    lexical_more = []
    semantic_more = []
    equal_count = 0
    no_overlap = []
    high_overlap = []

    for concept in all_concepts:
        lexical_chunks = set(lexical_map.get(concept, []))
        semantic_chunks = set(semantic_map.get(concept, []))

        intersection = lexical_chunks & semantic_chunks
        union = lexical_chunks | semantic_chunks

        lexical_only = lexical_chunks - semantic_chunks
        semantic_only = semantic_chunks - lexical_chunks

        if union:
            jaccard = len(intersection) / len(union)
        else:
            jaccard = 0.0

        jaccard_scores.append(jaccard)
        overlap_counts.append(len(intersection))

        if len(semantic_chunks) > len(lexical_chunks):
            semantic_more.append(concept)
        elif len(lexical_chunks) > len(semantic_chunks):
            lexical_more.append(concept)
        else:
            equal_count += 1

        if len(intersection) == 0:
            no_overlap.append(concept)

        if jaccard >= 0.7:
            high_overlap.append(concept)

        concept_details[concept] = {
            "lexical_count": len(lexical_chunks),
            "semantic_count": len(semantic_chunks),
            "overlap_count": len(intersection),
            "jaccard_similarity": round(jaccard, 4),
            "lexical_only_count": len(lexical_only),
            "semantic_only_count": len(semantic_only),
            "common_chunks": sorted(list(intersection)),
            "lexical_only_chunks": sorted(list(lexical_only)),
            "semantic_only_chunks": sorted(list(semantic_only))
        }

    avg_jaccard = statistics.mean(jaccard_scores) if jaccard_scores else 0
    median_jaccard = statistics.median(jaccard_scores) if jaccard_scores else 0
    avg_overlap = statistics.mean(overlap_counts) if overlap_counts else 0

    summary = {
        "total_concepts_compared": len(all_concepts),
        "average_jaccard_similarity": round(avg_jaccard, 4),
        "median_jaccard_similarity": round(median_jaccard, 4),
        "average_overlap_chunks_per_concept": round(avg_overlap, 2),
        "concepts_with_no_overlap_count": len(no_overlap),
        "concepts_with_high_overlap_count": len(high_overlap),
        "semantic_has_more_chunks_count": len(semantic_more),
        "lexical_has_more_chunks_count": len(lexical_more),
        "equal_chunk_count_concepts": equal_count,
        "sample_concepts_with_no_overlap": no_overlap[:20],
        "sample_concepts_with_high_overlap": high_overlap[:20],
        "sample_semantic_more": semantic_more[:20],
        "sample_lexical_more": lexical_more[:20]
    }

    return summary, concept_details


# ----------------------------
# Chunk Reuse Comparison
# ----------------------------
def compute_chunk_reuse_stats(mapping):
    all_chunks = []

    for chunks in mapping.values():
        all_chunks.extend(chunks)

    reuse_counter = Counter(all_chunks)

    reused_more_than_5 = [
        (chunk, count)
        for chunk, count in reuse_counter.items()
        if count > 5
    ]

    reused_more_than_10 = [
        (chunk, count)
        for chunk, count in reuse_counter.items()
        if count > 10
    ]

    reused_more_than_20 = [
        (chunk, count)
        for chunk, count in reuse_counter.items()
        if count > 20
    ]

    return {
        "chunks_reused_more_than_5_concepts": len(reused_more_than_5),
        "chunks_reused_more_than_10_concepts": len(reused_more_than_10),
        "chunks_reused_more_than_20_concepts": len(reused_more_than_20),
        "top_20_most_reused_chunks": reuse_counter.most_common(20)
    }


# ----------------------------
# Add Chunk Preview for Top Reused Chunks
# ----------------------------
def add_chunk_previews(top_chunks, chunk_lookup, preview_chars=180):
    previews = []

    for chunk_id, count in top_chunks:
        info = chunk_lookup.get(chunk_id, {})
        text = info.get("text", "")

        previews.append({
            "chunk_id": chunk_id,
            "reuse_count": count,
            "book_name": info.get("book_name", ""),
            "page_number": info.get("page_number", ""),
            "preview": text[:preview_chars].replace("\n", " ")
        })

    return previews


# ----------------------------
# Print Summary
# ----------------------------
def print_summary(lexical_stats, semantic_stats, overlap_stats, lexical_reuse, semantic_reuse):
    print("\n====================================")
    print("RETRIEVER COMPARISON REPORT")
    print("====================================\n")

    print("1) LEXICAL + GRAPH MAPPING")
    print("--------------------------")
    print(f"- Total concepts: {lexical_stats['total_concepts']}")
    print(f"- Concepts with chunks: {lexical_stats['concepts_with_chunks']}")
    print(f"- Zero-match concepts: {lexical_stats['zero_match_concepts']}")
    print(f"- Coverage: {lexical_stats['coverage_percent']}%")
    print(f"- Total concept-chunk links: {lexical_stats['total_concept_chunk_links']}")
    print(f"- Unique chunks used: {lexical_stats['unique_chunks_used']}")
    print(f"- Avg chunks per concept: {lexical_stats['average_chunks_per_concept']}")
    print(f"- Median chunks per concept: {lexical_stats['median_chunks_per_concept']}")
    print(f"- Max chunks for a concept: {lexical_stats['max_chunks_per_concept']}")
    print(f"- Avg chunk reuse: {lexical_stats['average_chunk_reuse']}")

    print("\n2) SEMANTIC EMBEDDING MAPPING")
    print("-----------------------------")
    print(f"- Total concepts: {semantic_stats['total_concepts']}")
    print(f"- Concepts with chunks: {semantic_stats['concepts_with_chunks']}")
    print(f"- Zero-match concepts: {semantic_stats['zero_match_concepts']}")
    print(f"- Coverage: {semantic_stats['coverage_percent']}%")
    print(f"- Total concept-chunk links: {semantic_stats['total_concept_chunk_links']}")
    print(f"- Unique chunks used: {semantic_stats['unique_chunks_used']}")
    print(f"- Avg chunks per concept: {semantic_stats['average_chunks_per_concept']}")
    print(f"- Median chunks per concept: {semantic_stats['median_chunks_per_concept']}")
    print(f"- Max chunks for a concept: {semantic_stats['max_chunks_per_concept']}")
    print(f"- Avg chunk reuse: {semantic_stats['average_chunk_reuse']}")

    print("\n3) OVERLAP BETWEEN BOTH METHODS")
    print("--------------------------------")
    print(f"- Total concepts compared: {overlap_stats['total_concepts_compared']}")
    print(f"- Average Jaccard similarity: {overlap_stats['average_jaccard_similarity']}")
    print(f"- Median Jaccard similarity: {overlap_stats['median_jaccard_similarity']}")
    print(f"- Avg overlapping chunks per concept: {overlap_stats['average_overlap_chunks_per_concept']}")
    print(f"- Concepts with no overlap: {overlap_stats['concepts_with_no_overlap_count']}")
    print(f"- Concepts with high overlap: {overlap_stats['concepts_with_high_overlap_count']}")
    print(f"- Concepts where semantic has more chunks: {overlap_stats['semantic_has_more_chunks_count']}")
    print(f"- Concepts where lexical has more chunks: {overlap_stats['lexical_has_more_chunks_count']}")

    print("\n4) CHUNK REUSE / OVERUSE")
    print("-------------------------")
    print("Lexical:")
    print(f"- Chunks reused by >5 concepts: {lexical_reuse['chunks_reused_more_than_5_concepts']}")
    print(f"- Chunks reused by >10 concepts: {lexical_reuse['chunks_reused_more_than_10_concepts']}")
    print(f"- Chunks reused by >20 concepts: {lexical_reuse['chunks_reused_more_than_20_concepts']}")

    print("\nSemantic:")
    print(f"- Chunks reused by >5 concepts: {semantic_reuse['chunks_reused_more_than_5_concepts']}")
    print(f"- Chunks reused by >10 concepts: {semantic_reuse['chunks_reused_more_than_10_concepts']}")
    print(f"- Chunks reused by >20 concepts: {semantic_reuse['chunks_reused_more_than_20_concepts']}")

    print("\n5) PPT-FRIENDLY BULLETS")
    print("------------------------")
    print(f"- Lexical mapping coverage: {lexical_stats['coverage_percent']}%")
    print(f"- Semantic mapping coverage: {semantic_stats['coverage_percent']}%")
    print(f"- Lexical zero-match concepts: {lexical_stats['zero_match_concepts']}")
    print(f"- Semantic zero-match concepts: {semantic_stats['zero_match_concepts']}")
    print(f"- Lexical unique chunks used: {lexical_stats['unique_chunks_used']}")
    print(f"- Semantic unique chunks used: {semantic_stats['unique_chunks_used']}")
    print(f"- Average Jaccard overlap between methods: {overlap_stats['average_jaccard_similarity']}")
    print(f"- Concepts with no overlap between methods: {overlap_stats['concepts_with_no_overlap_count']}")


# ----------------------------
# Main
# ----------------------------
def main():
    print("📥 Loading files...")

    lexical_map = load_json(LEXICAL_MAP_FILE)
    semantic_map = load_json(SEMANTIC_MAP_FILE)
    chunks = load_json(CHUNKS_FILE)

    chunk_lookup = build_chunk_lookup(chunks)

    print("✅ Files loaded")

    lexical_stats = compute_map_stats(lexical_map)
    semantic_stats = compute_map_stats(semantic_map)

    overlap_stats, concept_details = compute_concept_level_comparison(
        lexical_map,
        semantic_map
    )

    lexical_reuse = compute_chunk_reuse_stats(lexical_map)
    semantic_reuse = compute_chunk_reuse_stats(semantic_map)

    # Add previews for most reused chunks
    lexical_reuse["top_20_most_reused_chunk_previews"] = add_chunk_previews(
        lexical_reuse["top_20_most_reused_chunks"],
        chunk_lookup
    )

    semantic_reuse["top_20_most_reused_chunk_previews"] = add_chunk_previews(
        semantic_reuse["top_20_most_reused_chunks"],
        chunk_lookup
    )

    report = {
        "lexical_graph_mapping_stats": lexical_stats,
        "semantic_embedding_mapping_stats": semantic_stats,
        "overlap_stats": overlap_stats,
        "lexical_chunk_reuse_stats": lexical_reuse,
        "semantic_chunk_reuse_stats": semantic_reuse
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    with open(CONCEPT_DETAIL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(concept_details, f, indent=4, ensure_ascii=False)

    print_summary(
        lexical_stats,
        semantic_stats,
        overlap_stats,
        lexical_reuse,
        semantic_reuse
    )

    print(f"\n✅ Summary saved to: {OUTPUT_FILE}")
    print(f"✅ Concept-level comparison saved to: {CONCEPT_DETAIL_OUTPUT}")


if __name__ == "__main__":
    main()
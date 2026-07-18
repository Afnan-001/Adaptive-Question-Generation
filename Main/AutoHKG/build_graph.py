import json
import networkx as nx
from pathlib import Path
from pyvis.network import Network

# ----------------------------
# Files
# ----------------------------
MAIN_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = MAIN_DIR / "outputs"
INPUT_FILE = OUTPUTS_DIR / "data" / "enriched_concepts.json"
GML_OUTPUT = OUTPUTS_DIR / "graph_model" / "knowledge_graph.gml"
HTML_OUTPUT = OUTPUTS_DIR / "data" / "knowledge_graph.html"

# ----------------------------
# Load Data
# ----------------------------
if not INPUT_FILE.exists():
    print(f"❌ File not found: {INPUT_FILE}")
    exit()

if INPUT_FILE.stat().st_size == 0:
    print(f"❌ File is empty: {INPUT_FILE}")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    concepts = json.load(f)

# ----------------------------
# Helpers
# ----------------------------
def ensure_list(value):
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def add_or_update_node(graph, node_name, node_type):
    """
    Adds node with type.
    If node already exists, preserve coarse category type as higher-level priority.
    """

    if not node_name:
        return

    if node_name not in graph:
        graph.add_node(node_name, type=node_type)
        return

    existing_type = graph.nodes[node_name].get("type")

    # Coarse category should take priority if a node appears both as category and concept
    if existing_type != "coarse_grained_category" and node_type == "coarse_grained_category":
        graph.nodes[node_name]["type"] = "coarse_grained_category"


# ----------------------------
# Build Graph
# ----------------------------
G = nx.DiGraph()

for c in concepts:

    concept_name = c.get("concept")
    if not concept_name:
        continue

    # ----------------------------
    # Fine-Grained Concept Node
    # ----------------------------
    add_or_update_node(
        G,
        concept_name,
        "fine_grained_concept"
    )

    # ----------------------------
    # Coarse-Grained Category Nodes
    # Supports both:
    # - coarse_grained_category
    # - parent_concept
    # ----------------------------
    coarse_categories = []

    coarse_categories.extend(
        ensure_list(c.get("coarse_grained_category"))
    )

    coarse_categories.extend(
        ensure_list(c.get("parent_concept"))
    )

    # remove duplicates
    coarse_categories = list(set(coarse_categories))

    for category in coarse_categories:
        if category and category != concept_name:
            add_or_update_node(
                G,
                category,
                "coarse_grained_category"
            )

            # category → fine concept
            G.add_edge(
                category,
                concept_name,
                relation="CONTAINS"
            )

    # ----------------------------
    # Prerequisite Relationships
    # prereq → concept
    # ----------------------------
    prerequisites = ensure_list(c.get("prerequisites"))

    for prereq in prerequisites:
        if prereq and prereq != concept_name:
            add_or_update_node(
                G,
                prereq,
                "fine_grained_concept"
            )

            G.add_edge(
                prereq,
                concept_name,
                relation="PREREQUISITE_FOR"
            )

# ----------------------------
# Graph Stats
# ----------------------------
node_type_counts = {}

for _, attrs in G.nodes(data=True):
    node_type = attrs.get("type", "unknown")
    node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

edge_type_counts = {}

for _, _, attrs in G.edges(data=True):
    relation = attrs.get("relation", "unknown")
    edge_type_counts[relation] = edge_type_counts.get(relation, 0) + 1

print("\n--- GRAPH STATS ---")
print(f"✅ Total Nodes: {G.number_of_nodes()}")
print(f"✅ Total Edges: {G.number_of_edges()}")

print("\n--- NODE TYPE COUNTS ---")
for node_type, count in node_type_counts.items():
    print(f"✅ {node_type}: {count}")

print("\n--- EDGE TYPE COUNTS ---")
for relation, count in edge_type_counts.items():
    print(f"✅ {relation}: {count}")

# ----------------------------
# Save Graph
# ----------------------------
GML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

nx.write_gml(G, GML_OUTPUT)

print(f"\n✅ Graph saved at {GML_OUTPUT}")

# ----------------------------
# Visualization
# ----------------------------
net = Network(
    height="800px",
    width="100%",
    directed=True,
    bgcolor="#ffffff",
    font_color="#222222"
)

# ----------------------------
# Node Styling
# ----------------------------
NODE_COLORS = {
    "coarse_grained_category": "#FF6B6B",   # red/pink
    "fine_grained_concept": "#4ECDC4",      # teal
    "unknown": "#97C2FC"
}

NODE_SIZES = {
    "coarse_grained_category": 30,
    "fine_grained_concept": 18,
    "unknown": 18
}

# Add nodes
for node, attrs in G.nodes(data=True):
    node_type = attrs.get("type", "unknown")

    net.add_node(
        node,
        label=node,
        title=f"{node}<br>Type: {node_type}",
        color=NODE_COLORS.get(node_type, "#97C2FC"),
        size=NODE_SIZES.get(node_type, 18)
    )

# ----------------------------
# Edge Styling
# ----------------------------
EDGE_COLORS = {
    "CONTAINS": "#555555",
    "PREREQUISITE_FOR": "#2B6CB0",
}

for source, target, attrs in G.edges(data=True):
    relation = attrs.get("relation", "")

    net.add_edge(
        source,
        target,
        title=relation,
        # label=relation,
        color=EDGE_COLORS.get(relation, "#999999"),
        arrows="to"
    )

# ----------------------------
# Save HTML Visualization
# ----------------------------
HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

net.write_html(str(HTML_OUTPUT))

print(f"✅ HTML visualization saved at {HTML_OUTPUT}")

# ----------------------------
# Query Functions
# ----------------------------
def get_coarse_category(graph, concept):
    return [
        src for src, dst, data in graph.in_edges(concept, data=True)
        if data.get("relation") == "CONTAINS"
    ]


def get_prerequisites(graph, concept):
    return [
        src for src, dst, data in graph.in_edges(concept, data=True)
        if data.get("relation") == "PREREQUISITE_FOR"
    ]


def get_dependents(graph, concept):
    return [
        dst for src, dst, data in graph.out_edges(concept, data=True)
        if data.get("relation") == "PREREQUISITE_FOR"
    ]

# ----------------------------
# Example Test
# ----------------------------
test_concept = "3NF"

if test_concept in G:
    print(f"\n📌 Coarse Category of {test_concept}:")
    print(get_coarse_category(G, test_concept))

    print(f"\n📌 Prerequisites of {test_concept}:")
    print(get_prerequisites(G, test_concept))

    print(f"\n📌 Concepts depending on {test_concept}:")
    print(get_dependents(G, test_concept))

else:
    print(f"\n⚠️ '{test_concept}' not found in graph")

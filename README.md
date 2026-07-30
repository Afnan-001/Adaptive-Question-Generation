# Adaptive Assessment Generation

## Project Overview
This project is a DBMS-focused adaptive question generation system. It converts textbook PDFs into a structured concept graph, links concepts back to supporting text chunks, retrieves graph-aware context, generates grounded MCQs, and updates learner mastery with a confidence-aware Bayesian Knowledge Tracing (BKT) loop.

The source code is under `Main/` and is split into two functional areas:

- `Main/AutoHKG`: **Auto Hierarchial Knowledge Graph**. This is the knowledge-graph creation pipeline, including source-grounded enrichment and optional cycle cleanup.
- `Main/CogRAG`: **Cognitive RAG**. This is the Graph+Semantic retrieval, question-generation, adaptive-engine, confidence scoring, and app layer.

Generated files are written under `Main/outputs/`.


## Project Structure
```text
.
|-- Main/
|   |-- llm.py
|   |-- log_file.py
|   |-- common_log.jsonl
|   |-- books/
|   |   |-- book1.pdf
|   |   |-- book2.pdf
|   |   |-- book3.pdf
|   |-- AutoHKG/
|   |   |-- build_graph.py
|   |   |-- clean_cycles.py
|   |   |-- embed_chunks.py
|   |   |-- enrich_concepts.py
|   |   |-- extract_and_chunk.py
|   |   |-- extract_concepts.py
|   |   |-- merge_concepts.py
|   |-- CogRAG/
|   |   |-- adaptive_engine.py
|   |   |-- app.py
|   |   |-- confidence.py
|   |   |-- concept_chunk_map.py
|   |   |-- mastery_scores.py
|   |   |-- question_generator.py
|   |   |-- retriever.py
|   |-- outputs/
|   |   |-- data/
|   |   |-- data_fullCorpus/
|   |   |-- data_halfCorpus/
|   |   |-- graph_model/
|-- desktop_assets/
|-- test_scripts/
|-- requirements.txt
|-- README.md
```

## Installation
Install dependencies using:

```bash
pip install -r requirements.txt
```

### Environment variables
The project reads configuration from `.env`:

```env
EMBEDDING_API_BASE=http://10.221.0.164:4000/v1
EMBEDDING_MODEL=si-rca-dds-text-embedding-3-small
EMBEDDING_API_KEY=your_token_here

AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_MODEL=azure/gpt-4o
```

## Configuration
### Input location
- Source PDFs: `Main/books/*.pdf`

### Generated output locations
- JSON Outputs: `Main/outputs/data`
- Graph Model: `Main/outputs/graph_model`
- Concepts data, Learner state, generated questions, navigation state, etc. are written under `Main/outputs/data`


## Architecture Diagram
![Architecture Diagram](desktop_assets/Diagrams/Architecture%20Diagram.png)


## Main Entry Points
- Knowledge-graph build: scripts in `Main/AutoHKG/`
- Retrieval and tutoring: scripts in `Main/CogRAG/`
- Interactive app: `Main/CogRAG/app.py`


### AutoHKG only
```bash
python Main/AutoHKG/extract_and_chunk.py
python Main/AutoHKG/embed_chunks.py
python Main/AutoHKG/extract_concepts.py
python Main/AutoHKG/merge_concepts.py
python Main/AutoHKG/enrich_concepts.py
python Main/AutoHKG/clean_cycles.py
python Main/AutoHKG/build_graph.py
```

### Retrieval prep
```bash
python Main/CogRAG/concept_chunk_map.py
python Main/CogRAG/mastery_scores.py
```

`concept_chunk_map.py` is also used by `enrich_concepts.py` as source evidence for category and prerequisite extraction. When regenerating source-grounded enrichment, make sure `Main/outputs/data/concept_chunk_map.json` already exists.

### Retrieval
```bash
python Main/CogRAG/retriever.py "Normalization"
```

### Question generation
```bash
python Main/CogRAG/question_generator.py "Normalization" medium
```

### App
```bash
streamlit run Main/CogRAG/app.py
```

## Execution Order
Below is the execution order of files.

### 1. `Main/AutoHKG/extract_and_chunk.py`
- **Purpose:** Extracts text from PDFs in `Main/books/`, cleans it, chunks it, and saves the chunk corpus.
- **Inputs:** `Main/books/*.pdf`
- **Outputs:** `Main/outputs/data/dbms_chunks.json`
- **Required or optional:** Required

### 2.  `Main/AutoHKG/embed_chunks.py`
- **Purpose:** Calls the embedding API for chunk text and saves chunk embeddings plus chunk metadata order.
- **Inputs:** `Main/outputs/data/dbms_chunks.json`, `EMBEDDING_*`
- **Outputs:** `Main/outputs/data/chunk_index.json`, `Main/outputs/data/chunk_embeddings.npy`
- **Required or optional:** Required

### 3.  `Main/AutoHKG/extract_concepts.py`
- **Purpose:** Uses the LLM to extract important DBMS concepts from chunk batches.
- **Inputs:** `Main/outputs/data/dbms_chunks.json`
- **Outputs:** `Main/outputs/data/concepts.json`
- **Required or optional:** Required

### 4. `Main/AutoHKG/merge_concepts.py`
- **Purpose:** Cleans, filters, and canonicalizes extracted concepts.
- **Inputs:** `Main/outputs/data/concepts.json`
- **Outputs:** `Main/outputs/data/merged_concepts.json`
- **Required or optional:** Required

### 5. `Main/AutoHKG/enrich_concepts.py`
- **Purpose:** Adds source-grounded coarse-grained category and prerequisite relationships to each concept, including confidence and evidence metadata.
- **Inputs:** `Main/outputs/data/merged_concepts.json`, `Main/outputs/data/concept_chunk_map.json`, `Main/outputs/data/dbms_chunks.json`
- **Outputs:** `Main/outputs/data/enriched_concepts.json`
- **Required or optional:** Required

### 6. `Main/AutoHKG/clean_cycles.py`
- **Purpose:** Detects prerequisite cycles and removes the lowest-confidence edge from each cycle.
- **Inputs:** `Main/outputs/data/enriched_concepts.json`
- **Outputs:** `Main/outputs/data/cleaned_enriched_concepts.json`, `Main/outputs/data/removed_prerequisite_cycle_edges.json`
- **Required or optional:** Optional cleanup/reporting step

### 7. `Main/AutoHKG/build_graph.py`
- **Purpose:** Builds the DBMS knowledge graph and writes both GML and HTML visualization outputs.
- **Inputs:** `Main/outputs/data/enriched_concepts.json`
- **Outputs:** `Main/outputs/graph_model/knowledge_graph.gml`, `Main/outputs/data/knowledge_graph.html`
- **Required or optional:** Required


### 8. `Main/CogRAG/concept_chunk_map.py`
- **Purpose:** Precomputes semantic chunk candidates and similarity scores for each concept.
- **Inputs:** `Main/outputs/data/dbms_chunks.json`, `Main/outputs/data/enriched_concepts.json`, `Main/outputs/data/chunk_index.json`, `Main/outputs/data/chunk_embeddings.npy`, `EMBEDDING_*`
- **Outputs:** `Main/outputs/data/concept_chunk_map.json`, `Main/outputs/data/concept_chunk_scores.json`
- **Required or optional:** Required for source-grounded enrichment and runtime retrieval

### 9. `Main/CogRAG/mastery_scores.py`
- **Purpose:** Creates the initial BKT mastery state for every concept.
- **Inputs:** `Main/outputs/data/enriched_concepts.json`
- **Outputs:** `Main/outputs/data/mastery_scores.json`
- **Required or optional:** Required before adaptive tutoring

### 10. `Main/CogRAG/retriever.py`
- **Purpose:** Combines graph relationships with semantic reranking to retrieve concept-grounded context. It also supports multi-concept retrieval for primary and secondary concepts.
- **Inputs:** concept CLI argument, `Main/outputs/graph_model/knowledge_graph.gml`, `Main/outputs/data/concept_chunk_map.json`, `Main/outputs/data/dbms_chunks.json`, `Main/outputs/data/chunk_index.json`, `Main/outputs/data/chunk_embeddings.npy`, `EMBEDDING_*`
- **Outputs:** console output when used as a CLI tool; in-memory retrieval result when imported
- **Required or optional:** Optional standalone tool, required indirectly by question generation
- **Example usage:** `python Main/CogRAG/retriever.py "Normalization"`

### 11. `Main/CogRAG/question_generator.py`
- **Purpose:** Retrieves context, prompts the LLM to generate an MCQ, validates it, rejects exact and semantic duplicates, assigns source chunks, and stores accepted questions.
- **Inputs:** concept and difficulty arguments, `Main/CogRAG/retriever.py`, `Main/llm.py`
- **Outputs:** `Main/outputs/data/generated_mcqs.json`
- **Required or optional:** Optional standalone tool, required by the adaptive engine
- **Example usage:** `python Main/CogRAG/question_generator.py "Normalization" medium`

### 12. `Main/CogRAG/confidence.py`
- **Purpose:** Computes learner-confidence scores from number of attempts and recent answer consistency.
- **Inputs:** question attempt counts and recent answer history from mastery state
- **Outputs:** in-memory confidence score used by `Main/CogRAG/adaptive_engine.py`
- **Required or optional:** Required by adaptive engine

### 13. `Main/CogRAG/adaptive_engine.py`
- **Purpose:** Selects the next concept, decides difficulty, evaluates answers, handles prerequisite remediation, and updates weighted multi-concept BKT mastery state.
- **Inputs:** `Main/outputs/data/mastery_scores.json`, `Main/outputs/graph_model/knowledge_graph.gml`, `Main/outputs/data/navigation_state.json`, question objects from `Main/CogRAG/question_generator.py`
- **Outputs:** updated `Main/outputs/data/mastery_scores.json` and `Main/outputs/data/navigation_state.json`
- **Required to run or optional:** Optional standalone file, used by app.py

### 14. `Main/CogRAG/app.py`
- **Purpose:** Streamlit UI for generating adaptive questions, submitting answers, showing concept weights, displaying per-concept mastery updates, and showing progress.
- **Required or optional:** Required; main user entry point
- **Example usage:** `streamlit run Main/CogRAG/app.py`

## Test Scripts
The `test_scripts/` folder contains helper scripts for validating parts of the implementation:

- `test_scripts/test_question_generation.py`: Streamlit test UI for generating graph-backed MCQs with isolated test output files.
- `test_scripts/test_semantic.py`: Utility script for inspecting repeated semantic chunk usage.

## Other Resources
The `desktop_assets/` folder contains supporting project reference material used/made during the internship:

- `desktop_assets/Diagrams/`: Visual artifacts that explain the system structure and implementation flow:
- `desktop_assets/PPTs/`: Presentation decks prepared during the internship duration:
- `desktop_assets/Adaptive_Assessment_Internship_Problem_Statement_V1.pdf`: Original internship problem statement and project scope.
- `desktop_assets/Literature Review Summary and Approaches.docx`: A document created to note and summarize the findings from the conducted literature review. Also includes some initial question generation approaches that I was thinking for the project.
- `desktop_assets/Reference Research Paper.pdf`: Research paper used as the reference for the system design and approach.

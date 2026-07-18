# Adaptive Assessment Engine

## Project Overview
This project is a DBMS-focused adaptive question generation system. It converts textbook PDFs into a structured concept graph, links concepts back to supporting text chunks, retrieves graph-aware context, generates grounded MCQs, and updates learner mastery with a Bayesian Knowledge Tracing (BKT) loop.

The source code is under `Main/` and is split into two functional areas:

- `Main/AutoHKG`: **Auto Hierarchial Knowledge Graph**. This is the knowledge-graph creation pipeline.
- `Main/CogRAG`: **Cognitive RAG**. This is the Graph+Semantic retrieval, question-generation, adaptive-engine, and app layer.

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
|   |   |-- embed_chunks.py
|   |   |-- enrich_concepts.py
|   |   |-- extract_and_chunk.py
|   |   |-- extract_concepts.py
|   |   |-- merge_concepts.py
|   |-- CogRAG/
|   |   |-- adaptive_engine.py
|   |   |-- app.py
|   |   |-- concept_chunk_map.py
|   |   |-- mastery_scores.py
|   |   |-- question_generator.py
|   |   |-- retriever.py
|   |-- outputs/
|   |   |-- data/
|   |   |-- data_fullCorpus/
|   |   |-- data_halfCorpus/
|   |   |-- graph_model/
|-- assets/
|-- lib/
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


## Architecture Diagram
![Architecture Diagram](assets/Diagrams/Architecture%20Diagram.png)


## Workflow Diagram
```mermaid
graph TD
    A[extract_and_chunk.py] --> B[embed_chunks.py]
    A --> C[extract_concepts.py]
    C --> D[merge_concepts.py]
    D --> E[enrich_concepts.py]
    E --> F[build_graph.py]
    B --> G[concept_chunk_map.py]
    E --> G
    E --> H[mastery_scores.py]
    F --> I[retriever.py]
    G --> I
    B --> I
    I --> J[question_generator.py]
    J --> K[adaptive_engine.py]
    H --> K
    K --> L[app.py]
```

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
python Main/AutoHKG/build_graph.py
```

### Retrieval prep
```bash
python Main/CogRAG/concept_chunk_map.py
python Main/CogRAG/mastery_scores.py
```

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
- **Purpose:** Adds category and prerequisite relationships to each concept.
- **Inputs:** `Main/outputs/data/merged_concepts.json`
- **Outputs:** `Main/outputs/data/enriched_concepts.json`
- **Required or optional:** Required

### 6. `Main/AutoHKG/build_graph.py`
- **Purpose:** Builds the DBMS knowledge graph and writes both GML and HTML visualization outputs.
- **Inputs:** `Main/outputs/data/enriched_concepts.json`
- **Outputs:** `Main/outputs/graph_model/knowledge_graph.gml`, `Main/outputs/data/knowledge_graph.html`
- **Required or optional:** Required


### 7. `Main/CogRAG/concept_chunk_map.py`
- **Purpose:** Precomputes semantic chunk candidates for each concept.
- **Inputs:** `Main/outputs/data/dbms_chunks.json`, `Main/outputs/data/enriched_concepts.json`, `Main/outputs/data/chunk_index.json`, `Main/outputs/data/chunk_embeddings.npy`, `EMBEDDING_*`
- **Outputs:** `Main/outputs/data/concept_chunk_map.json`, `Main/outputs/data/concept_chunk_scores.json`
- **Required or optional:** Required for runtime retrieval

### 8. `Main/CogRAG/mastery_scores.py`
- **Purpose:** Creates the initial BKT mastery state for every concept.
- **Inputs:** `Main/outputs/data/enriched_concepts.json`
- **Outputs:** `Main/outputs/data/mastery_scores.json`
- **Required or optional:** Required before adaptive tutoring

### 9. `Main/CogRAG/retriever.py`
- **Purpose:** Combines graph relationships with semantic reranking to retrieve concept-grounded context.
- **Inputs:** concept CLI argument, `Main/outputs/graph_model/knowledge_graph.gml`, `Main/outputs/data/concept_chunk_map.json`, `Main/outputs/data/dbms_chunks.json`, `Main/outputs/data/chunk_index.json`, `Main/outputs/data/chunk_embeddings.npy`, `EMBEDDING_*`
- **Outputs:** console output when used as a CLI tool; in-memory retrieval result when imported
- **Required or optional:** Optional standalone tool, required indirectly by question generation
- **Example usage:** `python Main/CogRAG/retriever.py "Normalization"`

### 10. `Main/CogRAG/question_generator.py`
- **Purpose:** Retrieves context, prompts the LLM to generate an MCQ, validates it, and stores accepted questions.
- **Inputs:** concept and difficulty arguments, `Main/CogRAG/retriever.py`, `Main/llm.py`
- **Outputs:** `Main/outputs/data/generated_mcqs.json`
- **Required or optional:** Optional standalone tool, required by the adaptive engine
- **Example usage:** `python Main/CogRAG/question_generator.py "Normalization" medium`

### 11. `Main/CogRAG/adaptive_engine.py`
- **Purpose:** Selects the next concept, decides difficulty, evaluates answers, and updates BKT mastery state.
- **Inputs:** `Main/outputs/data/mastery_scores.json`, `Main/outputs/graph_model/knowledge_graph.gml`, `Main/outputs/data/current_concept.txt`, `Main/outputs/data/recent_concepts.json`, question objects from `Main/CogRAG/question_generator.py`
- **Outputs:** updated learner-state files under `Main/outputs/data/`
- **Dependencies:** `Main/CogRAG/mastery_scores.py`, `Main/AutoHKG/build_graph.py`, `Main/CogRAG/question_generator.py`
- **Required to run or optional:** Optional standalone tool
- **Example usage:** imported by `Main/CogRAG/app.py`

### 12. `Main/CogRAG/app.py`
- **Purpose:** Streamlit UI for generating adaptive questions, submitting answers, and showing progress.
- **Required or optional:** Optional if you only need offline artifacts; main user entry point
- **Example usage:** `streamlit run Main/CogRAG/app.py`

## Other Resources
The `assets/` folder contains supporting project reference material used/made during the internship:

- `assets/Adaptive_Assessment_Internship_Problem_Statement_V1.pdf`: Original internship problem statement and project scope.
- `assets/Reference Research Paper.pdf`: Research paper used as the reference for the system design and approach.
- `assets/Diagrams/`: Visual artifacts that explain the system structure and implementation flow:
  
- `assets/PPTs/`: Presentation decks prepared during the internship duration:
  

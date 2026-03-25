# GenAI with LLMs: SEC Filings RAG and Agentic Workflows

This repository explores how retrieval-augmented generation (RAG) evolves from a simple baseline into more capable, agentic pipelines for answering questions over SEC filings.

The project centers on:

- A shared SEC filings corpus and Chroma vector store
- Baseline and advanced RAG notebooks
- Agentic implementations across LangGraph, LlamaIndex, and CrewAI
- Shared configuration and retrieval utilities for parity across frameworks

## What This Repository Contains

The notebooks in this repo compare different ways to answer financial questions using SEC 10-K and 10-Q filings:

1. Baseline retrieval and answer generation
2. Advanced retrieval with query improvement and reranking
3. Agentic orchestration with multi-step reasoning and tool use
4. Cross-framework experimentation using the same underlying data and settings

## Notebook Guide

These are the main notebooks that match the workflow progression shown in your screenshot, using the actual file names in this repository:

| Stage | Notebook | Path | Summary |
|---|---|---|---|
| 1 | Baseline RAG | `baseline_rag.ipynb` | `baseline_advanced_rag/baseline_rag.ipynb` | Dense-retrieval RAG over SEC filings with a straightforward retrieval and generation pipeline. |
| 2 | Advanced RAG | `advanced_rag.ipynb` | `baseline_advanced_rag/advanced_rag.ipynb` | Adds metadata extraction, query rewriting, hybrid retrieval, and reranking for stronger retrieval quality. |
| 3 | LangGraph Agentic RAG | `langgraph_agentic_rag_sec_v4.ipynb` | `LangGraph/langgraph_agentic_rag_sec_v4.ipynb` | Graph-based agentic workflow for SEC EDGAR question answering, built around shared config and retriever components. |
| 4 | LlamaIndex Agentic RAG | `llamaindex_agentic_rag_sec v2.ipynb` | `Framework - Llama/llamaindex_agentic_rag_sec v2.ipynb` | LlamaIndex-based agentic RAG notebook designed for parity with the shared retrieval stack. |
| 5 | CrewAI Agentic RAG | `crewai_agentic_rag_sec.ipynb` | `CrewAI/crewai_agentic_rag_sec.ipynb` | Multi-agent SEC filings QA workflow using CrewAI-style orchestration. |
| Bonus | Project Visuals | `Project_Visuals_Generator.ipynb` | `Project_Visuals_Generator.ipynb` | Architecture and workflow visuals for presenting the system design. |

## Shared Data and Assets

The notebooks use a common SEC dataset located under `sec_rag_team_share/`.

Key contents:

- `sec_rag_team_share/sec_data/raw_filings/`: raw SEC filing HTML documents
- `sec_rag_team_share/sec_data/clean_text/`: cleaned filing text
- `sec_rag_team_share/sec_data/chunks/sec_chunks.jsonl`: chunked corpus for retrieval
- `sec_rag_team_share/chroma_db/`: persisted Chroma vector database
- `sec_rag_team_share/evaluation/GenAI Eval QA.csv`: evaluation set
- `sec_rag_team_share/filings_master.csv`: filing inventory / metadata

## Repository Structure

```text
GenAI-with-LLMs/
|-- baseline_advanced_rag/
|   |-- baseline_rag.ipynb
|   `-- advanced_rag.ipynb
|-- LangGraph/
|   |-- langgraph_agentic_rag_sec_v4.ipynb
|   |-- langgraph_agentic_rag_sec_v4_lite.ipynb
|   `-- results/
|-- Framework - Llama/
|   `-- llamaindex_agentic_rag_sec v2.ipynb
|-- CrewAI/
|   `-- crewai_agentic_rag_sec.ipynb
|-- sec_rag_team_share/
|   |-- chroma_db/
|   |-- evaluation/
|   `-- sec_data/
|-- config.py
|-- shared_retriever.py
|-- requirements.txt
`-- Project_Visuals_Generator.ipynb
```

## Core Python Modules

- `config.py`: centralized configuration shared across frameworks
- `shared_retriever.py`: reusable retrieval logic and shared corpus access
- `requirements.txt`: common dependencies for notebooks and experiments

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

### 4. Start with the notebooks in this order

1. `baseline_advanced_rag/baseline_rag.ipynb`
2. `baseline_advanced_rag/advanced_rag.ipynb`
3. `LangGraph/langgraph_agentic_rag_sec_v4.ipynb`
4. `Framework - Llama/llamaindex_agentic_rag_sec v2.ipynb`
5. `CrewAI/crewai_agentic_rag_sec.ipynb`

## Configuration Notes

The repository uses `config.py` as a shared source of truth for:

- Data paths
- Chroma database location
- Evaluation dataset path
- Dense retrieval and reranking settings
- Provider and model selection
- Runtime profile settings for faster development vs. fuller runs

Default paths are already set up to look for:

- `sec_rag_team_share/sec_data/chunks/sec_chunks.jsonl`
- `sec_rag_team_share/chroma_db`
- `sec_rag_team_share/evaluation/GenAI Eval QA.csv`

## LLM and Provider Notes

The notebooks appear to support multiple providers and frameworks, including:

- Gemini
- Groq
- LangGraph
- LlamaIndex
- CrewAI
- ChromaDB
- Sentence Transformers / rerankers

Depending on which notebook you run, you may need provider API keys set in your environment before execution.

## Recommended Reading Order

If you are new to this repo, use this learning path:

1. Read `baseline_rag.ipynb` to understand the simplest SEC RAG pipeline.
2. Move to `advanced_rag.ipynb` to see retrieval quality improvements.
3. Open the LangGraph notebook to understand graph-based orchestration.
4. Compare with LlamaIndex and CrewAI to see how the same problem is modeled across frameworks.
5. Use `Project_Visuals_Generator.ipynb` for diagrams and presentation material.

## Notes

- Some notebook names in older screenshots appear to differ slightly from the current filenames in this repository.
- This repo contains both active notebooks and some superseded / experimental variants.
- The shared dataset includes SEC filings from multiple companies and reporting periods, enabling retrieval and evaluation experiments.

## License

Add your preferred license here if this repository will be shared publicly.

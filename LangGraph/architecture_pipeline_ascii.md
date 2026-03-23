# Current SEC Agentic RAG Architecture (ASCII)

This document reflects the latest flow in `langgraph_agentic_rag_sec_v4_lite.ipynb`.

Key updates in the current notebook:

- `Gemini startup model selector`
  - The notebook now starts with a model preset choice before config is built.
- `Advanced-RAG seeded agent run`
  - `run_agentic_rag()` first calls `run_advanced_rag()` and seeds the LangGraph state with its retrieved chunks.
  - If seeded retrieval is available, the graph starts at `query_planner` but skips straight to `context_evaluator` instead of re-retrieving immediately.
- `Retrieval sanity check + context retry`
  - Retrieved chunks are checked for obvious ticker/query mismatches.
  - If context is weak, the graph can retry retrieval once before generation.
- `Refined Critic + drift loop`
  - The critic can still return `insufficient_data`.
  - That path logs misses, optionally scrapes EDGAR, ingests new chunks, patches the live index, and retries retrieval.

```text
+--------------------------------------------------------------------------------------------------+
|                                      NOTEBOOK STARTUP                                            |
+--------------------------------------------------------------------------------------------------+
| User selects Gemini preset                                                                       |
| -> builds CONFIG                                                                                 |
| -> loads shared CorpusIndex / dense embedder / reranker                                          |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
|                                        SEC DATA PREP                                             |
+--------------------------------------------------------------------------------------------------+
| Raw SEC filings (.html / inline XBRL)                                                            |
| -> chunking + metadata normalization                                                             |
| -> sec_chunks.jsonl                                                                              |
| -> ChromaDB dense store                                                                          |
| -> shared_retriever.CorpusIndex                                                                  |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
|                             THREE PIPELINES USED IN THE NOTEBOOK                                 |
+--------------------------------------------------------------------------------------------------+
| 1. Simple RAG                                                                                    |
|    dense retrieval -> generate                                                                   |
|                                                                                                  |
| 2. Advanced RAG                                                                                  |
|    rewrite -> multi-query -> retrieve -> RRF merge -> rerank -> compress -> generate            |
|                                                                                                  |
| 3. Agentic RAG (LangGraph)                                                                       |
|    starts from Advanced-RAG output and adds control-flow, critique, repair, and drift handling   |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
|                           AGENTIC RAG (LATEST LANGGRAPH FLOW)                                    |
+--------------------------------------------------------------------------------------------------+
| User Question                                                                                    |
|      |                                                                                           |
|      v                                                                                           |
| run_advanced_rag(question, index)                                                                |
|   - rewrite_query()                                                                              |
|   - generate_multi_queries()                                                                     |
|   - run retrieval per query variant                                                              |
|   - RRF merge                                                                                    |
|   - rerank                                                                                       |
|   - compress retrieved context                                                                   |
|   - generate_with_citations()                                                                    |
|      |                                                                                           |
|      v                                                                                           |
| Seed LangGraph state                                                                             |
|   - rewritten_query                                                                              |
|   - retrieved_chunks                                                                             |
|   - retrieved_doc_names                                                                          |
|   - use_seeded_retrieval=True when advanced retrieval succeeded                                  |
|      |                                                                                           |
|      v                                                                                           |
| Query Planner [Agent]                                                                            |
|   - emits rewritten query + structured sub_queries                                               |
|   - can mark needs_decomposition                                                                 |
|   - if seeded retrieval exists, preserves that seeded query/sub-query state                      |
|      |                                                                                           |
|      +-------------------------------+-----------------------------------------------------------+
|                                      |                                                           |
|                                      v                                                           |
|                           Seeded retrieval available?                                            |
|                                      |                                                           |
|                     yes -------------------------------> Context Evaluator [Agent]               |
|                                      |                                                           |
|                     no                                v                                           |
|                                      +-------> Hybrid Retriever [RAG]                            |
|                                                  - may retrieve once or per sub-query            |
|                                                  - merges multi-query results                    |
|                                                  - runs retrieval sanity check                   |
|                                                  - outputs retrieved_chunks + doc names          |
|                                                          |                                       |
|                                                          v                                       |
|                                                Context Evaluator [Agent]                         |
|                                                - if sanity check failed, marks context bad       |
|                                                - otherwise judges relevance/sufficiency          |
|                                                          |                                       |
|                         +--------------------------------+-------------------+                   |
|                         |                                                    |                   |
|                         v                                                    v                   |
|                context good                                         context weak                 |
|                         |                                             and retry budget left      |
|                         v                                                    |                   |
|                  Generator [Agent]                                          v                   |
|                  - answer with citations                           Increment Context Retry        |
|                  - uses retrieved SEC evidence                              |                   |
|                         |                                                    v                   |
|                         v                                           Hybrid Retriever [RAG]       |
|                   Critic [Agent]                                                                    |
|                   - accept                                                                          |
|                   - repair                                                                          |
|                   - insufficient_data                                                               |
|                         |                                                                            |
|          +--------------+----------------------------+                                              |
|          |                                           |                                              |
|          v                                           v                                              |
|      accept -> END                           Repair [Agent]                                         |
|                                              - revises answer                                       |
|                                              - may request new retrieval                            |
|                                                       |                                             |
|                            +--------------------------+----------------------+                      |
|                            |                                                 |                      |
|                            v                                                 v                      |
|                       end -> END                             Mark Repair Retrieval                   |
|                                                              -> Hybrid Retriever                    |
|                                                              -> Generator                           |
|                                                              -> Critic                              |
|                                                              -> END after repair cycle              |
|                                                                                                     |
|      insufficient_data                                                                              |
|               |                                                                                     |
|               v                                                                                     |
|      Drift Detector [Agent/Controller]                                                              |
|      - logs miss by (ticker, filing_year, form_type)                                                |
|      - checks scrape threshold                                                                      |
|               |                                                                                     |
|         +-----+-------------------------------+                                                     |
|         |                                     |                                                     |
|         v                                     v                                                     |
|    no threshold                         threshold crossed                                           |
|         |                                     |                                                     |
|         v                                     v                                                     |
|      END                           Scrape + Ingest Missing Filing                                   |
|                                    - download from EDGAR                                            |
|                                    - chunk filing                                                   |
|                                    - upsert ChromaDB                                                |
|                                    - hot-patch CorpusIndex dataframe + BM25                         |
|                                    - clear critic state                                             |
|                                             |                                                       |
|                               +-------------+-------------+                                         |
|                               |                           |                                         |
|                               v                           v                                         |
|                      ingestion ran                  nothing ingested                                |
|                               |                           |                                         |
|                               v                           v                                         |
|                     Hybrid Retriever [RAG]              END                                         |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
|                                   EVALUATION / EXPORT                                             |
+--------------------------------------------------------------------------------------------------+
| For each question:                                                                                |
| -> run Simple RAG                                                                                 |
| -> run Advanced RAG                                                                               |
| -> run Agentic RAG                                                                                |
| -> optionally run LLM judge for correctness + faithfulness                                        |
| -> collect route trace, model snapshot, retrieval stats, repair stats, and final metrics          |
| -> export aligned result tables / eval CSVs                                                       |
+--------------------------------------------------------------------------------------------------+
```

## Boundary Summary

- `SEC Data Prep`
  - Produces chunked SEC filing text plus metadata.
  - Persists the dense store and the JSONL corpus used by `shared_retriever`.

- `RAG`
  - Lives inside `shared_retriever.CorpusIndex` plus the notebook retrieval helpers.
  - Owns dense retrieval, per-query retrieval, RRF merge, reranking, compression, and retrieved-context formatting.

- `Agent`
  - Owns planning, context evaluation, answer generation, critique, repair, retry routing, and drift-triggered ingestion.
  - In the current notebook, Agentic RAG is seeded by the Advanced-RAG retrieval output.

- `Evaluation`
  - Runs simple, advanced, and agentic pipelines side by side.
  - Can apply an LLM judge during the main eval loop.
  - Exports detailed aligned rows and summary metrics.

## Function Cheat Sheet

- `Retrieval / Advanced-RAG Layer`
  - `rewrite_query()`: rewrites the user question for cleaner retrieval.
  - `generate_multi_queries()`: produces multiple retrieval variants.
  - `rrf_merge_retrieved()`: merges ranked lists with reciprocal rank fusion.
  - `rerank_retrieved()`: reranks merged candidates with the cross-encoder reranker.
  - `compress_retrieved_context()`: trims chunks to the most relevant supporting sentences.
  - `generate_with_citations()`: generates an answer from retrieved evidence.
  - `fails_retrieval_sanity_check()`: catches obvious mismatch cases before generation.

- `Graph / Agent Layer`
  - `node_query_planner()`: produces normalized sub-queries and decomposition metadata.
  - `node_hybrid_retriever()`: runs retrieval, merges multi-query hits, and applies the sanity check.
  - `node_context_evaluator()`: judges whether the retrieved evidence is usable.
  - `node_increment_context_retry()`: increments retry count before re-retrieval.
  - `node_generator()`: drafts the answer from retrieved SEC context.
  - `node_critic()`: decides `accept`, `repair`, or `insufficient_data`.
  - `node_repair()`: revises the answer and may request one more retrieval pass.
  - `node_mark_repair_retrieval()`: tags the next retrieval as repair-driven.
  - `node_drift_detector()`: logs misses and determines whether scraping should trigger.
  - `node_scrape_and_ingest()`: fetches missing filings, chunks them, ingests them, and refreshes the live index.
  - `build_agentic_graph()`: wires the full LangGraph control flow.

## Current LangGraph Routing

- Entry: `query_planner`
- `query_planner`
  - seeded retrieval available -> `context_evaluator`
  - otherwise -> `hybrid_retriever`
- `hybrid_retriever` -> `context_evaluator`
- `context_evaluator`
  - relevant -> `generator`
  - not relevant and retry budget remains -> `increment_context_retry -> hybrid_retriever`
  - not relevant and retry budget exhausted -> `generator`
- `generator` -> `critic`
- `critic`
  - `accept` -> `END`
  - `repair` -> `repair`
  - `insufficient_data` -> `drift_detector`
- `repair`
  - `needs_new_retrieval` -> `mark_repair_retrieval -> hybrid_retriever`
  - otherwise -> `END`
- `drift_detector`
  - scrape threshold crossed -> `scrape_and_ingest`
  - otherwise -> `END`
- `scrape_and_ingest`
  - ingestion happened -> `hybrid_retriever`
  - otherwise -> `END`

## Included Proposal Concepts

- `Proposal 1: Query Decomposition`
  - Included
  - Implemented in `node_query_planner()`
  - Supports structured sub-queries for compare/contrast and multi-period questions

- `Proposal 2: Refined Critic`
  - Included
  - Implemented in `node_critic()` and `route_critic()`
  - Adds `insufficient_data` for safe stop / drift escalation

- `Proposal 3: Drift Detection + Auto-Ingestion`
  - Included
  - Implemented via `node_drift_detector()` and `node_scrape_and_ingest()`
  - Retries retrieval after successful EDGAR ingestion

- `Proposal 4: Retrieval sanity + retry guardrail`
  - Included in the latest lite notebook flow
  - Detects likely ticker/query mismatch before generation
  - Allows one context retry before continuing

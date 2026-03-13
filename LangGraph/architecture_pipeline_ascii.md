# Current SEC Agentic RAG Architecture (ASCII)

Two proposal-level upgrades are explicitly included in the current design:

- `Query Decomposition` = the `Brain` upgrade
  - The planner breaks multi-period / cross-company questions into 2 to 3 targeted sub-queries before retrieval.
- `Refined Critic` = the `Guardrail` upgrade
  - The critic can return `insufficient_data` so the agent exits safely instead of looping on unanswerable questions.

```text
+----------------------------------------------------------------------------------+
|                              SEC DATA PREP                                       |
|                        (from sec_rag_team_share)                                 |
+----------------------------------------------------------------------------------+
| Raw SEC filings (.html / inline XBRL)                                            |
| -> chunking + metadata prep                                                      |
| -> sec_chunks.jsonl                                                              |
| -> ChromaDB dense store                                                          |
+----------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                    END-TO-END AGENTIC RAG SEQUENCE                               |
+----------------------------------------------------------------------------------+
| User Question                                                                    |
|      |                                                                           |
|      v                                                                           |
| Planner [Agent]                                                                  |
|   Query Decomposition ("Brain" upgrade)                                          |
|   - figures out if the question should be split into smaller parts               |
|   - creates 1 to 3 focused search requests                                       |
|      |                                                                           |
|      v                                                                           |
| Retrieval Orchestrator [Agent]                                                   |
|   - decides how retrieval should run for this question                           |
|   - can search once or search each sub-question and combine the results          |
|      |                                                                           |
|      v                                                                           |
| Hybrid Retriever [RAG]                                                           |
|   - looks up the most useful filing snippets                                     |
|                                                                                  |
|      +--------------------------------------------------------------------+      |
|      |          ADVANCED RAG STACK (inside retrieval) [RAG]              |       |
|      | load_sec_chunks() [RAG]: load the prepared SEC filing chunks      |       |
|      | -> sec_df_to_chunk_dicts() [RAG]: shape chunks for retrieval      |       |
|      | -> BM25 [RAG]: keyword search                                     |       |
|      | -> Dense retrieval [RAG]: embedding / similarity search           |       | 
|      | -> RRF merge [RAG]: combine BM25 + dense results                  |       | 
|      | -> Reranker [RAG]: sort the best snippets                         |       | 
|      | -> Retrieved context [RAG]: final evidence sent back              |       |
|      +--------------------------------------------------------------------+      |
|      |                                                                           |
|      v                                                                           |
| Context Evaluator [Agent]                                                        |
|   - checks whether the retrieved evidence looks useful                           |
|   - if not, asks retrieval to try again                                          |
|      |                                                                           |
|      v                                                                           |
| Generator [Agent]                                                                |
|   - writes a first draft answer using the filing evidence                        |
|      |                                                                           |
|      v                                                                           |
| Critic [Agent]                                                                   |
|   Refined Critic ("Guardrail" upgrade)                                           |
|   - decides whether the draft looks good, fixable, or unsupported                |
|   - can safely stop when the filing does not contain the answer                  |
|                      |                                                           |
|                      v                                                           |
| Repair [Agent]                                                                   |
|   - fixes the draft when the problem looks fixable                               |
|   - can ask for another retrieval pass if better evidence is needed              |
+----------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                          EVALUATION / COMPARISON                                 |
+----------------------------------------------------------------------------------+
| Simple RAG | Advanced RAG | Agentic RAG                                          |
| -> compare outputs from the different pipelines                                  |
| -> use an LLM judge to score answer quality                                      |
| -> produce the final metrics table                                               |
+----------------------------------------------------------------------------------+
```

## Boundary Summary

- `SEC Data Prep`
  - Relied on `sec_rag_team_share`
  - Produced the prepared filing corpus and persisted dense store

- `RAG`
  - Starts when the notebook loads `sec_chunks.jsonl`
  - Owns indexing, hybrid retrieval, and reranking

- `Agent`
  - Owns planning, retrieval control flow, answer generation, critique, and repair
  - Includes `Query Decomposition` as the planning upgrade
  - Includes `Refined Critic` as the safe-stop guardrail

- `Evaluation`
  - Compares simple, advanced, and agentic pipelines
  - Uses an LLM judge for correctness and faithfulness

## Function Cheat Sheet

- `RAG Layer`
  - `load_sec_chunks()`: loads the prepared SEC chunk dataset from JSONL.
  - `sec_df_to_chunk_dicts()`: converts the DataFrame into the chunk format used by retrieval.
  - `CorpusIndex`: wraps BM25, dense retrieval, hybrid merge, and reranking.
  - `bm25_search()`: keyword retrieval over chunk text.
  - `dense_search()`: embedding-based retrieval over ChromaDB.
  - `hybrid_search()`: combines BM25 and dense hits, then reranks the merged set.

- `Agent Layer`
  - `node_query_planner()`: decides whether to decompose the question and emits sub-queries.
  - `node_hybrid_retriever()`: retrieves chunks for one query or multiple sub-queries.
  - `node_context_evaluator()`: checks whether retrieved context is relevant enough to continue.
  - `node_generator()`: drafts an answer using retrieved SEC context.
  - `node_critic()`: checks whether the draft is grounded, repairable, or unsupported.
  - `node_repair()`: revises the answer or asks for another retrieval pass.
  - `node_mark_repair_retrieval()`: tags the retrieval as repair-driven to avoid bad loops.

## Included Proposal Concepts

- `Proposal 1: Query Decomposition`
  - Included
  - Implemented in the planner node before retrieval
  - Addresses the single-query gap for compare/contrast, cross-company, and multi-period questions

- `Proposal 2: Refined Critic`
  - Included
  - Implemented in the critic schema and routing
  - Adds `insufficient_data` so the system can stop safely when the filing does not contain the answer

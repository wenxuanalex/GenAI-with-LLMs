"""
Centralized Configuration Module
=================================
Authoritative configuration for all agentic RAG frameworks (CrewAI, LlamaIndex, LangGraph).

Source of Truth: CrewAI notebook (crewai_agentic_rag_sec.ipynb)

This module provides a single CONFIG dict that ensures synchronized hyperparameters,
model IDs, and retrieval settings across all three framework implementations.
All environment variables are loaded with sensible defaults from CrewAI.
"""

import os
from pathlib import Path
from typing import Any, Dict, List


def _pick_first_existing_path(candidates: List[str], fallback: str) -> str:
    """Return the first existing path from candidates, else fallback."""
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).exists():
            return candidate
    return fallback


def _resolve_path_from_env(env_key: str, default_path: str) -> str:
    """Use env path only if it exists; otherwise fall back to default path."""
    env_value = os.getenv(env_key, "").strip()
    if env_value and Path(env_value).exists():
        return env_value
    return default_path


def load_config() -> Dict[str, Any]:
    """Load and validate the centralized configuration."""
    
    sec_chunks_default = _pick_first_existing_path(
        [
            r'sec_rag_team_share/sec_data/chunks/sec_chunks.jsonl',
            r'../sec_rag_team_share/sec_data/chunks/sec_chunks.jsonl',
        ],
        r'sec_rag_team_share/sec_data/chunks/sec_chunks.jsonl',
    )
    chroma_db_default = _pick_first_existing_path(
        [
            r'sec_rag_team_share/chroma_db',
            r'../sec_rag_team_share/chroma_db',
        ],
        r'sec_rag_team_share/chroma_db',
    )
    sec_eval_default = _pick_first_existing_path(
        [
            r'sec_rag_team_share/evaluation/GenAI Eval QA.csv',
            r'../sec_rag_team_share/evaluation/GenAI Eval QA.csv',
        ],
        r'sec_rag_team_share/evaluation/GenAI Eval QA.csv',
    )

    CONFIG: Dict[str, Any] = {
        # ────────────────────────────────────────────────────────────────────
        # Core Behavior
        # ────────────────────────────────────────────────────────────────────
        'random_seed': 42,

        # Pilot vs full run
        'use_pilot':             True,
        'pilot_n_questions':     10,
        'full_n_questions':      80,
        'use_llm_judge':         True,
        'use_few_shot_examples': True,
        'pilot_judge_sample_n':  1,
        'full_judge_sample_n':   2,

        # Profile: 'dev' (faster/cheaper) or 'final' (best quality)
        'execution_profile': os.getenv('EXECUTION_PROFILE', 'dev'),
        'provider':          os.getenv('LLM_PROVIDER', 'gemini'),

        # Provider fallback order and per-provider RPM caps
        'provider_fallback_order': ['gemini', 'groq'],
        'provider_rpm':            {'groq': 28, 'gemini': 10},

        # ────────────────────────────────────────────────────────────────────
        # Data Paths
        # ────────────────────────────────────────────────────────────────────
        'sec_chunks_path':   _resolve_path_from_env('SEC_CHUNKS_PATH', sec_chunks_default),
        'chroma_db_path':    _resolve_path_from_env('CHROMA_DB_PATH', chroma_db_default),
        'sec_eval_csv_path': _resolve_path_from_env('SEC_EVAL_CSV_PATH', sec_eval_default),
        'results_dir':       os.getenv('RESULTS_DIR',       r'./results'),

        # Eval split config
        'train_split':               'dev',
        'eval_split':                'test',
        'verbose_eval_logging':      True,
        'auto_export_results_input': True,

        # ────────────────────────────────────────────────────────────────────
        # Retrieval Hyperparameters (AUTHORITATIVE FROM CREWAI)
        # ────────────────────────────────────────────────────────────────────
        'bm25_top_k':                       int(os.getenv('BM25_TOP_K',                 '8')),
        'dense_top_k':                      int(os.getenv('DENSE_TOP_K',                '8')),
        'rerank_top_k':                     int(os.getenv('RERANK_TOP_K',               '5')),
        'decomposition_top_k_per_subquery': int(os.getenv('DECOMP_TOP_K_PER_SUBQUERY',  '4')),
        'adjacent_chunk_expansion_n':       int(os.getenv('ADJACENT_EXPANSION_N',       '1')),

        # ────────────────────────────────────────────────────────────────────
        # Embedding & Reranking Models (AUTHORITATIVE FROM CREWAI)
        # ────────────────────────────────────────────────────────────────────
        'dense_model_name':    os.getenv('DENSE_MODEL_NAME',
                                         'sentence-transformers/all-mpnet-base-v2'),
        'reranker_model_name': os.getenv('RERANKER_MODEL_NAME',
                                         'cross-encoder/ms-marco-MiniLM-L-6-v2'),

        # ────────────────────────────────────────────────────────────────────
        # LLM Model Selection (AUTHORITATIVE FROM CREWAI)
        # ────────────────────────────────────────────────────────────────────
        # Groq models
        'groq_dev_generator_model':   os.getenv('GROQ_DEV_GENERATOR',
                                                'llama-3.1-8b-instant'),
        'groq_dev_agent_model':       os.getenv('GROQ_DEV_AGENT',
                                                'llama-3.1-8b-instant'),
        'groq_dev_judge_model':       os.getenv('GROQ_DEV_JUDGE',
                                                'llama-3.1-8b-instant'),
        'groq_final_generator_model': os.getenv('GROQ_FINAL_GENERATOR',
                                                'meta-llama/llama-4-scout-17b-16e-instruct'),
        'groq_final_agent_model':     os.getenv('GROQ_FINAL_AGENT',
                                                'llama-3.1-8b-instant'),
        'groq_final_judge_model':     os.getenv('GROQ_FINAL_JUDGE',
                                                'llama-3.1-8b-instant'),
        'groq_fallback_agent_models':     ['llama-3.1-8b-instant'],
        'groq_fallback_generator_models': ['llama-3.1-8b-instant'],
        'groq_fallback_judge_models':     ['llama-3.1-8b-instant'],

        # Gemini models — ALL use gemini-2.5-flash-lite (CREWAI SOURCE OF TRUTH)
        'gemini_dev_generator_model':   os.getenv('GEMINI_DEV_GENERATOR',
                                                  'gemini-2.5-flash-lite'),
        'gemini_dev_agent_model':       os.getenv('GEMINI_DEV_AGENT',
                                                  'gemini-2.5-flash-lite'),
        'gemini_dev_judge_model':       os.getenv('GEMINI_DEV_JUDGE',
                                                  'gemini-2.5-flash-lite'),
        'gemini_final_generator_model': os.getenv('GEMINI_FINAL_GENERATOR',
                                                  'gemini-2.5-flash-lite'),
        'gemini_final_agent_model':     os.getenv('GEMINI_FINAL_AGENT',
                                                  'gemini-2.5-flash-lite'),
        'gemini_final_judge_model':     os.getenv('GEMINI_FINAL_JUDGE',
                                                  'gemini-2.5-flash-lite'),
        'gemini_fallback_agent_models':     [],
        'gemini_fallback_generator_models': [],
        'gemini_fallback_judge_models':     [],

        # Gemini pricing (USD per 1M tokens)
        'gemini_cost_input_per_1m':  0.10,   # gemini-2.5-flash-lite input
        'gemini_cost_output_per_1m': 0.40,   # gemini-2.5-flash-lite output

        # ────────────────────────────────────────────────────────────────────
        # Temperature Settings (AUTHORITATIVE FROM CREWAI)
        # Note: Gemini via LiteLLM returns empty candidates at exactly 0.0;
        #       use 0.1 as the effective minimum for reliability.
        # ────────────────────────────────────────────────────────────────────
        'temperature_planner':   0.1,
        'temperature_generator': 0.2,
        'temperature_critic':    0.1,
        'temperature_judge':     0.1,

        # Additional temperature options for extended workflows
        'temperature_rewriter':      0.2,
        'temperature_context_eval':  0.1,
        'temperature_repair':        0.1,

        # ────────────────────────────────────────────────────────────────────
        # Context Window Limits (AUTHORITATIVE FROM CREWAI)
        # ────────────────────────────────────────────────────────────────────
        'generator_max_context_chunks': 8,
        'generator_max_context_chars':  12000,
        'max_context_chars':            9000,
        'control_max_context_chunks':   4,
        'control_max_context_chars':    6000,
        'judge_max_context_chunks':     3,
        'judge_max_context_chars':      4000,

        # ────────────────────────────────────────────────────────────────────
        # Rate Limiting & Pacing
        # ────────────────────────────────────────────────────────────────────
        'max_rpm':                  10,  # Default; overridden by provider_rpm
        'inter_question_sleep_sec': 1.5,
        'llm_max_retries':          3,
        'llm_retry_base_delay_sec': 5,

        # ────────────────────────────────────────────────────────────────────
        # Feature Flags
        # ────────────────────────────────────────────────────────────────────
        'enable_retrieval_sanity_check': True,
        'enable_drift_and_scrape':       False,
    }

    # Compute derived values
    CONFIG['judge_sample_n'] = (
        CONFIG['pilot_judge_sample_n'] if CONFIG['use_pilot'] else CONFIG['full_judge_sample_n']
    ) if CONFIG['use_llm_judge'] else 0

    # Override max_rpm based on provider
    CONFIG['max_rpm'] = CONFIG['provider_rpm'].get(CONFIG['provider'], 10)

    return CONFIG


# Module-level singleton
CONFIG = load_config()


def get_provider_order() -> List[str]:
    """Return ordered list of LLM providers for fallback logic."""
    configured = CONFIG.get('provider_fallback_order') or [CONFIG['provider']]
    ordered = [provider for provider in configured if provider in {'groq', 'gemini'}]
    if CONFIG['provider'] in ordered:
        ordered.remove(CONFIG['provider'])
    ordered.insert(0, CONFIG['provider'])
    return list(dict.fromkeys(ordered))


def resolve_model_name(role: str, provider: str = None) -> str:
    """Resolve model name for a given role and provider."""
    provider = provider or CONFIG['provider']
    profile = CONFIG['execution_profile']
    key = f'{provider}_{profile}_{role}_model'
    if key not in CONFIG:
        raise ValueError(f'Model key not found: {key}')
    return CONFIG[key]


def resolve_fallback_model_names(role: str, provider: str = None) -> List[str]:
    """Resolve primary + fallback model names for a given role and provider."""
    provider = provider or CONFIG['provider']
    primary = resolve_model_name(role, provider=provider)
    fallback_key = f'{provider}_fallback_{role}_models'
    fallbacks = CONFIG.get(fallback_key, [])
    ordered = [primary] + list(fallbacks)
    return list(dict.fromkeys([m for m in ordered if m]))


if __name__ == '__main__':
    import json
    print('Centralized Configuration loaded:')
    print(json.dumps({k: v for k, v in CONFIG.items() if not isinstance(v, dict)}, indent=2))
    print(f'\nProvider: {CONFIG["provider"]}')
    print(f'Profile: {CONFIG["execution_profile"]}')
    print(f'Dense model: {CONFIG["dense_model_name"]}')
    print(f'Retrieval: bm25_top_k={CONFIG["bm25_top_k"]}, dense_top_k={CONFIG["dense_top_k"]}, rerank_top_k={CONFIG["rerank_top_k"]}')

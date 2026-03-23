#!/usr/bin/env python3
"""Simplify data loading to let shared_retriever handle path resolution."""
import json
from pathlib import Path

nb_path = Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# Find the data loading cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source_str = ''.join(cell['source'])
    
    # Find the data loading cell (contains "PROJECT_ROOT" and "chunks_path")
    if "PROJECT_ROOT: {PROJECT_ROOT}" in source_str:
        print(f"Found data loading cell at index {i}")
        # Simplify - let shared_retriever handle all path resolution
        cell['source'] = [
            '# ── SEC Corpus Loading ────────────────────────────────────────────────────────────────────────\n',
            '\n',
            'print("Initializing CorpusIndex from shared_retriever...")\n',
            'print("(shared_retriever handles path resolution automatically)")\n',
            '\n',
            '# Call initialize_corpus with no explicit paths - it resolves them correctly\n',
            'global_index = initialize_corpus()\n',
            'print(f"CorpusIndex ready: {len(global_index.df):,} chunks with hybrid retrieval")',
        ]
        print("✓ Simplified data loading cell")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Updated {nb_path}")

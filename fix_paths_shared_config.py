#!/usr/bin/env python3
"""Fix data loading cell to use SHARED_CONFIG."""
import json
from pathlib import Path

nb_path = Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the data loading cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source_str = ''.join(cell['source'])
    
    # Find the data loading cell (contains "global_index = initialize_corpus")
    if 'global_index = initialize_corpus' in source_str:
        print(f"Found data loading cell at index {i}")
        cell['source'] = [
            '# ── SEC Corpus Loading ────────────────────────────────────────────────────────────────────────\n',
            '\n',
            '# Resolve paths to absolute using PROJECT_ROOT + SHARED_CONFIG\n',
            'chunks_path = (PROJECT_ROOT / SHARED_CONFIG[\'sec_chunks_path\']).resolve()\n',
            'chroma_path = (PROJECT_ROOT / SHARED_CONFIG[\'chroma_db_path\']).resolve()\n',
            '\n',
            'print(f\'Loading SEC corpus from:\')\n',
            'print(f\'  chunks: {chunks_path}\')\n',
            'print(f\'  chroma_db: {chroma_path}\')\n',
            '\n',
            'print(\'Initializing CorpusIndex from shared_retriever...\')\n',
            'global_index = initialize_corpus(\n',
            '    chunks_jsonl=str(chunks_path),\n',
            '    chroma_db_path=str(chroma_path),\n',
            ')\n',
            'print(f\'CorpusIndex ready: {len(global_index.df):,} chunks with hybrid retrieval\')',
        ]
        print("✓ Updated data loading cell to use SHARED_CONFIG")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Updated {nb_path}")
